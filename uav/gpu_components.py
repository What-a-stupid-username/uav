"""Eight-connected components by parallel monotone union-find on CUDA."""
import torch
import triton
import triton.language as tl


@triton.jit
def _init(mask,parent,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    fg=tl.load(mask+i,i<N,other=0)
    tl.store(parent+i,tl.where(fg,i,-1),i<N)


@triton.jit
def _union(parent,H:tl.constexpr,W:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    p=tl.load(parent+i,i<N,other=-1)
    for offset in tl.static_range(4):
        delta=tl.full((),-1,tl.int32)
        valid=(i%W)>0
        if offset==1:
            delta=-W;valid=i>=W
        elif offset==2:
            delta=-W-1;valid=(i>=W)&((i%W)>0)
        elif offset==3:
            delta=-W+1;valid=(i>=W)&((i%W)<W-1)
        q=tl.load(parent+i+delta,(i<N)&valid,other=-1)
        active=(i<N)&valid&(p>=0)&(q>=0)
        a=tl.where(active,i,0);b=tl.where(active,i+delta,0)
        while tl.sum(active.to(tl.int32),0)>0:
            pa=tl.atomic_add(parent+a,0,sem='relaxed')
            pb=tl.atomic_add(parent+b,0,sem='relaxed')
            a=tl.where(active,pa,a);b=tl.where(active,pb,b)
            roots=active&(a==pa)&(b==pb)
            # CAS only links a root to a smaller id, preventing cycles.
            hi=tl.maximum(a,b);lo=tl.minimum(a,b)
            old=tl.atomic_cas(parent+tl.where(roots,hi,0),
                              tl.where(roots,hi,-2),tl.where(roots,lo,-2),sem='relaxed')
            done=(a==b)|(roots&(old==hi))
            active=active&~done


@triton.jit
def _resolve(parent,labels,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    p=tl.load(parent+i,i<N,other=-1)
    root=tl.maximum(p,0)
    following=tl.load(parent+root)
    active=(p>=0)&(following!=root)
    while tl.sum(active.to(tl.int32),0)>0:
        root=tl.where(active,following,root)
        following=tl.load(parent+root)
        active=(p>=0)&(following!=root)
    tl.store(labels+i,tl.where(p>=0,root,-1),i<N)


def components(mask):
    if not mask.is_cuda or mask.ndim!=2:raise ValueError('expected CUDA 2D mask')
    mask=mask.contiguous();h,w=mask.shape;n=h*w
    parent=torch.empty((n,),device=mask.device,dtype=torch.int32)
    labels=torch.empty_like(parent)
    _init[(triton.cdiv(n,256),)](mask,parent,n,256)
    _union[(triton.cdiv(n,128),)](parent,h,w,n,128)
    _resolve[(triton.cdiv(n,256),)](parent,labels,n,256)
    return labels.reshape(h,w)
