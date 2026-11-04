// Small CPU-only IRLS kernel. LAPACK is supplied by the installed SciPy ABI.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <Python.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

using Gelsd = void (*)(int*, int*, int*, double*, int*, double*, int*,
                      double*, double*, int*, double*, int*, int*, int*);

static double dot_scalar(const double* a,const double* b,int count) {
    double result=0.;
    for (int i=0;i<count;++i) result+=a[i]*b[i];
    return result;
}
#if defined(__x86_64__)
__attribute__((target("avx2")))
static double dot_avx2(const double* a,const double* b,int count) {
    __m256d first=_mm256_setzero_pd(),second=_mm256_setzero_pd();
    int i=0;
    for (;i+8<=count;i+=8) {
        first=_mm256_add_pd(first,_mm256_mul_pd(_mm256_loadu_pd(a+i),_mm256_loadu_pd(b+i)));
        second=_mm256_add_pd(second,_mm256_mul_pd(_mm256_loadu_pd(a+i+4),_mm256_loadu_pd(b+i+4)));
    }
    alignas(32) double sums[4];
    _mm256_store_pd(sums,_mm256_add_pd(first,second));
    double result=(sums[0]+sums[1])+(sums[2]+sums[3]);
    for (;i<count;++i) result+=a[i]*b[i];
    return result;
}
#endif
using Dot=double(*)(const double*,const double*,int);
static Dot dot=[]() -> Dot {
#if defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) return dot_avx2;
#endif
    return dot_scalar;
}();

static double quantile(std::vector<double>& scratch, double q) {
    const double index=(scratch.size()-1)*q;
    const size_t low=size_t(index), high=std::min(low+1,scratch.size()-1);
    std::nth_element(scratch.begin(),scratch.begin()+low,scratch.end());
    const double a=scratch[low];
    const double b=low==high ? a : *std::min_element(scratch.begin()+high,scratch.end());
    const double fraction=index-low;
    return fraction>=.5 ? b-(b-a)*(1.-fraction) : a+(b-a)*fraction;
}

static double median(std::vector<double>& scratch) {
    const size_t middle = scratch.size()/2;
    std::nth_element(scratch.begin(), scratch.begin()+middle, scratch.end());
    const double upper = scratch[middle];
    return scratch.size()%2 ? upper :
        (*std::max_element(scratch.begin(), scratch.begin()+middle)+upper)/2.;
}

// Householder QR on a copy of the weighted matrix. Unpivoted QR is used only
// when its triangular factor has a comfortably small condition estimate.
// Uncertain/rank-deficient systems are rebuilt and solved by LAPACK GELSD.
static bool qr(std::vector<double>& a, std::vector<double>& b,
               int rows, int columns, double* coefficient, std::vector<double>& inverse) {
    for (int j=0; j<columns; ++j) {
        double* column=a.data()+j*rows;
        const double squared=dot(column+j,column+j,rows-j);
        const double norm=std::sqrt(squared);
        if (!(norm>std::numeric_limits<double>::min()) || !std::isfinite(norm)) return false;
        const double alpha=column[j], beta=-std::copysign(norm,alpha);
        const double tau=(beta-alpha)/beta;
        const double denominator=alpha-beta;
        column[j]=beta;
        for (int i=j+1; i<rows; ++i) column[i]/=denominator;
        for (int k=j+1; k<columns; ++k) {
            double* next=a.data()+k*rows;
            const double projection=(next[j]+dot(column+j+1,next+j+1,rows-j-1))*tau;
            next[j]-=projection;
            for (int i=j+1; i<rows; ++i) next[i]-=column[i]*projection;
        }
        const double projection=(b[j]+dot(column+j+1,b.data()+j+1,rows-j-1))*tau;
        b[j]-=projection;
        for (int i=j+1; i<rows; ++i) b[i]-=column[i]*projection;
    }
    double norm=0., inverse_norm=0.;
    inverse.resize(columns*columns);
    for (int k=0; k<columns; ++k) {
        for (int i=columns-1; i>=0; --i) {
            double value=i==k ? 1. : 0.;
            for (int j=i+1; j<columns; ++j) value-=a[j*rows+i]*inverse[k*columns+j];
            inverse[k*columns+i]=value/a[i*rows+i];
        }
    }
    for (int i=0; i<columns; ++i) {
        double sum=0., inverse_sum=0.;
        for (int j=i; j<columns; ++j) sum+=std::abs(a[j*rows+i]);
        for (int j=0; j<columns; ++j) inverse_sum+=std::abs(inverse[j*columns+i]);
        norm=std::max(norm,sum);inverse_norm=std::max(inverse_norm,inverse_sum);
    }
    if (!std::isfinite(norm*inverse_norm) || norm*inverse_norm>1e7) return false;
    for (int i=columns-1; i>=0; --i) {
        double value=b[i];
        for (int j=i+1; j<columns; ++j) value-=a[j*rows+i]*coefficient[j];
        coefficient[i]=value/a[i*rows+i];
    }
    return true;
}

struct Workspace {
    std::vector<double> a,b,singular,work,weights,residual,scratch,inverse;
    std::vector<int> integer_work;
};
static thread_local Workspace workspace;

extern "C" int uav_robust_fit(const double* design, const double* values,
                             int rows, int columns, void* solver,
                             int work_size, int integer_size, double* coefficient) {
    if (rows < columns || columns < 1 || rows < 1 || !solver) return -100;
    auto gelsd = reinterpret_cast<Gelsd>(solver);
    auto& a=workspace.a;auto& b=workspace.b;auto& weights=workspace.weights;
    auto& residual=workspace.residual;auto& scratch=workspace.scratch;
    a.resize(rows*columns);b.resize(rows);weights.assign(rows,1.);
    residual.resize(rows);scratch.resize(rows);
    auto solve = [&]() {
        if (columns == 1) {
            double squared = 0., numerator = 0.;
            for (int i=0; i<rows; ++i) {
                const double x = design[i]*weights[i];
                squared += x*x;
                numerator += x*(values[i]*weights[i]);
            }
            if (squared > std::numeric_limits<double>::min() && std::isfinite(squared)
                    && std::isfinite(numerator)) {
                coefficient[0] = numerator/squared;
                return 0;
            }
        }
        auto fill=[&]() {
            for (int i=0; i<rows; ++i) {
                b[i] = values[i]*weights[i];
                for (int j=0; j<columns; ++j) a[j*rows+i] = design[i*columns+j]*weights[i];
            }
        };
        fill();
        if (qr(a,b,rows,columns,coefficient,workspace.inverse)) return 0;
        fill();
        auto& singular=workspace.singular;auto& work=workspace.work;
        auto& integer_work=workspace.integer_work;
        singular.resize(columns);work.resize(work_size);integer_work.resize(integer_size);
        int rhs=1, rank=0, info=0, m=rows, n=columns;
        double condition = std::numeric_limits<double>::epsilon()*rows;
        gelsd(&m, &n, &rhs, a.data(), &m, b.data(), &m, singular.data(),
              &condition, &rank, work.data(), &work_size, integer_work.data(), &info);
        if (!info) std::copy_n(b.data(), columns, coefficient);
        return info;
    };
    int info=solve();
    if (info) return info;
    for (int iteration=0; iteration<5; ++iteration) {
        for (int i=0; i<rows; ++i) {
            double predicted=0.;
            for (int j=0; j<columns; ++j) predicted += design[i*columns+j]*coefficient[j];
            residual[i] = std::abs(values[i]-predicted);
        }
        scratch=residual;
        const double scale = std::max(1e-8, median(scratch)*1.4826);
        bool same=true;
        for (int i=0; i<rows; ++i) {
            const double weight = std::min(1., 1.5*scale/std::max(residual[i], 1e-9));
            same = same && weight == weights[i];
            weights[i] = weight;
        }
        if (same) break;
        info=solve();
        if (info) return info;
    }
    return 0;
}

// Buffer protocol avoids NumPy's ctypes pointer conversion on every tiny fit.
// The extension depends on Python's C API only; all arrays remain caller-owned.
class Buffer {
public:
    Py_buffer view{};
    bool acquired=false;
    bool get(PyObject* object, bool writable=false) {
        acquired=PyObject_GetBuffer(object,&view,(writable?PyBUF_WRITABLE:0)|PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)==0;
        if (acquired && (view.itemsize!=sizeof(double) || !view.format || view.format[0]!='d')) {
            PyErr_SetString(PyExc_TypeError,"motion kernel requires float64 buffers");return false;
        }
        return acquired;
    }
    ~Buffer() { if (acquired) PyBuffer_Release(&view); }
};

static PyObject* fit(PyObject*, PyObject* args) {
    PyObject *design,*values,*output,*capsule;
    int rows,columns,work,integer_work,count=1;
    if (!PyArg_ParseTuple(args,"OOOiiOii|i",&design,&values,&output,&rows,&columns,
                          &capsule,&work,&integer_work,&count)) return nullptr;
    Buffer a,b,c;
    if (!a.get(design) || !b.get(values) || !c.get(output,true)) return nullptr;
    if (rows<columns || columns<1 || count<1 || a.view.len!=Py_ssize_t(rows)*columns*count*sizeof(double)
            || b.view.len!=Py_ssize_t(rows)*sizeof(double)
            || c.view.len!=Py_ssize_t(columns+1)*count*sizeof(double)) {
        PyErr_SetString(PyExc_ValueError,"invalid motion fit dimensions");return nullptr;
    }
    const char* name=PyCapsule_GetName(capsule);
    if (PyErr_Occurred()) return nullptr;
    void* solver=PyCapsule_GetPointer(capsule,name);
    if (!solver) return nullptr;
    const auto* x=static_cast<const double*>(a.view.buf);
    const auto* y=static_cast<const double*>(b.view.buf);
    auto* coefficient=static_cast<double*>(c.view.buf);
    int info=0;
    Py_BEGIN_ALLOW_THREADS
    for (int batch=0;batch<count;++batch) {
    const auto* matrix=x+size_t(batch)*rows*columns;
    auto* result=coefficient+size_t(batch)*(columns+1);
    info=uav_robust_fit(matrix,y,rows,columns,solver,work,integer_work,result);
    if (!info) {
        auto& residual=workspace.residual;
        residual.resize(rows);
        for (int i=0; i<rows; ++i) {
            double prediction=0.;
            for (int j=0; j<columns; ++j) prediction+=matrix[i*columns+j]*result[j];
            residual[i]=std::abs(y[i]-prediction);
        }
        result[columns]=quantile(residual,.8);
    }
    if (info) break;
    }
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(info);
}

static PyObject* discrepancy(PyObject*,PyObject* args) {
    PyObject *left,*right;
    double l,t,r,b;
    if (!PyArg_ParseTuple(args,"OOdddd",&left,&right,&l,&t,&r,&b)) return nullptr;
    Buffer a,c;
    if (!a.get(left) || !c.get(right)) return nullptr;
    if (a.view.len!=9*sizeof(double) || c.view.len!=9*sizeof(double)) {
        PyErr_SetString(PyExc_ValueError,"corner error requires two 3x3 matrices");
        return nullptr;
    }
    const auto* x=static_cast<const double*>(a.view.buf);
    const auto* y=static_cast<const double*>(c.view.buf);
    double maximum=0.;
    for (int i=0;i<4;++i) {
        double px=(i&1)?r:l, py=(i&2)?b:t;
        double dx=((px*x[0]+py*x[1])+x[2])-((px*y[0]+py*y[1])+y[2]);
        double dy=((px*x[3]+py*x[4])+x[5])-((px*y[3]+py*y[4])+y[5]);
        double error=std::sqrt(dx*dx+dy*dy);
        if (std::isnan(error)) return PyFloat_FromDouble(error);
        maximum=std::max(maximum,error);
    }
    return PyFloat_FromDouble(maximum);
}

static PyObject* bounds(PyObject*,PyObject* args) {
    PyObject *matrix;
    double l,t,r,b;
    if (!PyArg_ParseTuple(args,"Odddd",&matrix,&l,&t,&r,&b)) return nullptr;
    Buffer a;
    if (!a.get(matrix)) return nullptr;
    if (a.view.len!=9*sizeof(double)) {
        PyErr_SetString(PyExc_ValueError,"bounds requires a 3x3 matrix");return nullptr;
    }
    const auto* x=static_cast<const double*>(a.view.buf);
    double low_x=INFINITY,low_y=INFINITY,high_x=-INFINITY,high_y=-INFINITY;
    for (int i=0;i<4;++i) {
        double px=(i&1)?r:l,py=(i&2)?b:t;
        double xx=(px*x[0]+py*x[1])+x[2], yy=(px*x[3]+py*x[4])+x[5];
        if (std::isnan(xx)) low_x=high_x=xx;
        else {low_x=std::min(low_x,xx);high_x=std::max(high_x,xx);}
        if (std::isnan(yy)) low_y=high_y=yy;
        else {low_y=std::min(low_y,yy);high_y=std::max(high_y,yy);}
    }
    return Py_BuildValue("(dddd)",low_x,low_y,high_x,high_y);
}

static PyObject* components(PyObject*,PyObject* args) {
    PyObject *matrix;
    double ax,ay;
    if (!PyArg_ParseTuple(args,"Odd",&matrix,&ax,&ay)) return nullptr;
    Buffer a;
    if (!a.get(matrix)) return nullptr;
    if (a.view.len!=9*sizeof(double)) {
        PyErr_SetString(PyExc_ValueError,"components requires a 3x3 matrix");return nullptr;
    }
    const auto* x=static_cast<const double*>(a.view.buf);
    double xx=(ax*x[0]+ay*x[1])+x[2], yy=(ax*x[3]+ay*x[4])+x[5];
    double scale=std::sqrt(x[0]*x[0]+x[3]*x[3]);
    return Py_BuildValue("(dddd)",xx,yy,std::atan2(x[3],x[0]),
                         std::log(std::max(1e-9,scale)));
}

static PyObject* match_statistics(PyObject*,PyObject* args) {
    PyObject *matrix,*source,*target;
    double ex,ey;
    if (!PyArg_ParseTuple(args,"OOOdd",&matrix,&source,&target,&ex,&ey)) return nullptr;
    Buffer a,s,t;
    if (!a.get(matrix) || !s.get(source) || !t.get(target)) return nullptr;
    if (a.view.len!=9*sizeof(double) || s.view.len!=t.view.len
            || s.view.len<4*sizeof(double) || s.view.len%(2*sizeof(double))) {
        PyErr_SetString(PyExc_ValueError,"invalid match statistic dimensions");return nullptr;
    }
    const auto* m=static_cast<const double*>(a.view.buf);
    const auto* x=static_cast<const double*>(s.view.buf);
    const auto* y=static_cast<const double*>(t.view.buf);
    const size_t count=s.view.len/(2*sizeof(double));
    std::vector<double> errors(count);
    double lx=INFINITY,ly=INFINITY,hx=-INFINITY,hy=-INFINITY;
    for (size_t i=0;i<count;++i) {
        double px=x[2*i],py=x[2*i+1];
        double dx=((px*m[0]+py*m[1])+m[2])-y[2*i];
        double dy=((px*m[3]+py*m[4])+m[5])-y[2*i+1];
        errors[i]=std::sqrt(dx*dx+dy*dy);
        lx=std::min(lx,px);ly=std::min(ly,py);
        hx=std::max(hx,px);hy=std::max(hy,py);
    }
    double coverage=std::min(1.,(hx-lx)/std::max(1.,ex))
                   *std::min(1.,(hy-ly)/std::max(1.,ey));
    return Py_BuildValue("(dd)",quantile(errors,.9),coverage);
}

static PyObject* localization(PyObject*,PyObject* args) {
    PyObject* sizes;
    if (!PyArg_ParseTuple(args,"O",&sizes)) return nullptr;
    Py_buffer view{};
    if (PyObject_GetBuffer(sizes,&view,PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)) return nullptr;
    bool single=view.format && view.format[0]=='f' && view.itemsize==sizeof(float);
    bool real=view.format && view.format[0]=='d' && view.itemsize==sizeof(double);
    if ((!single && !real) || !view.len) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError,"localization requires nonempty floating sizes");return nullptr;
    }
    size_t count=view.len/view.itemsize;
    std::vector<double> values(count);
    for (size_t i=0;i<count;++i)
        values[i]=single?static_cast<const float*>(view.buf)[i]:static_cast<const double*>(view.buf)[i];
    PyBuffer_Release(&view);
    double value=median(values);
    // NumPy's float32 median rounds the sum before dividing by two.
    if (single && count%2==0) {
        size_t middle=count/2;
        double lower=*std::max_element(values.begin(),values.begin()+middle);
        value=double(float(float(lower)+float(values[middle]))/2.f);
    }
    return PyFloat_FromDouble(std::max(.7,value*.12));
}

static PyObject* distributed(PyObject*,PyObject* args) {
    PyObject* points;double threshold;
    if (!PyArg_ParseTuple(args,"Od",&points,&threshold)) return nullptr;
    Buffer a;
    if (!a.get(points)) return nullptr;
    if (a.view.len<4*sizeof(double) || a.view.len%(2*sizeof(double))) {
        PyErr_SetString(PyExc_ValueError,"distribution requires 2D points");return nullptr;
    }
    const auto* x=static_cast<const double*>(a.view.buf);
    size_t n=a.view.len/(2*sizeof(double));
    double mx=0.,my=0.;
    for (size_t i=0;i<n;++i) {mx+=x[2*i];my+=x[2*i+1];}
    mx/=n;my/=n;
    double xx=0.,xy=0.,yy=0.;
    for (size_t i=0;i<n;++i) {
        double dx=x[2*i]-mx,dy=x[2*i+1]-my;
        xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;
    }
    double trace=xx+yy;
    double small=(trace-std::hypot(xx-yy,2*xy))*.5;
    double cutoff=threshold*threshold;
    double margin=1e-10*std::max({1.,trace,cutoff});
    // Near a decision boundary (or catastrophic cancellation), use the
    // original SVD. Only well-separated decisions take the scalar fast path.
    if (!std::isfinite(small) || std::abs(small-cutoff)<=margin) return PyLong_FromLong(-1);
    return PyLong_FromLong(small>=cutoff);
}

static PyObject* visible_support(PyObject*,PyObject* args) {
    PyObject *mask,*matrix;int height,width,frame_height,frame_width;
    if (!PyArg_ParseTuple(args,"OOiiii",&mask,&matrix,&height,&width,&frame_height,&frame_width)) return nullptr;
    Buffer a;
    if (!a.get(matrix)) return nullptr;
    Py_buffer pixels{};
    if (PyObject_GetBuffer(mask,&pixels,PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)) return nullptr;
    bool single=pixels.format && pixels.format[0]=='f' && pixels.itemsize==sizeof(float);
    bool boolean=pixels.format && (pixels.format[0]=='?' || pixels.format[0]=='B') && pixels.itemsize==1;
    if ((!single && !boolean) || height<0 || width<0
            || pixels.len!=Py_ssize_t(height)*width*pixels.itemsize || a.view.len!=9*sizeof(double)) {
        PyBuffer_Release(&pixels);PyErr_SetString(PyExc_ValueError,"invalid visible support buffers");return nullptr;
    }
    const auto* m=static_cast<const double*>(a.view.buf);
    size_t count=0;
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        size_t index=size_t(y)*width+x;
        bool occupied=single?static_cast<const float*>(pixels.buf)[index]!=0:
                             static_cast<const unsigned char*>(pixels.buf)[index]!=0;
        if (!occupied) continue;
        double xx=(x*m[0]+y*m[1])+m[2], yy=(x*m[3]+y*m[4])+m[5];
        count+=xx>=0 && xx<frame_width && yy>=0 && yy<frame_height;
    }
    PyBuffer_Release(&pixels);return PyLong_FromSize_t(count);
}

static PyMethodDef methods[]={
    {"fit",fit,METH_VARARGS,"Fit and score one robust system."},
    {"discrepancy",discrepancy,METH_VARARGS,"Maximum affine corner error."},
    {"bounds",bounds,METH_VARARGS,"Projected affine bounds."},
    {"components",components,METH_VARARGS,"Affine pose components."},
    {"match_statistics",match_statistics,METH_VARARGS,"Registration residual quantile and coverage."},
    {"localization",localization,METH_VARARGS,"Median feature localization precision."},
    {"distributed",distributed,METH_VARARGS,"Well-conditioned point distribution test."},
    {"visible_support",visible_support,METH_VARARGS,"Count projected visible template pixels."},
    {nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_uav_motion_fit",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__uav_motion_fit() { return PyModule_Create(&module); }
