#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include "cudaSift.h"
#include "uav_stream.h"
thread_local cudaStream_t uav_sift_stream = nullptr;
struct Extractor {
  SiftData data{};
  float *image=nullptr,*temp=nullptr;
  size_t imageCapacity=0,tempCapacity=0;
  std::vector<int> order;
  void reserve(float*& buffer,size_t& capacity,size_t bytes){
    if(bytes<=capacity)return;
    if(buffer)cudaFree(buffer);
    cudaMalloc(reinterpret_cast<void**>(&buffer),bytes);capacity=bytes;
  }
  Extractor(){InitSiftData(data,8192,true,true);}
  ~Extractor(){FreeSiftData(data);if(image)cudaFree(image);if(temp)cudaFree(temp);}
};
extern "C" void* uav_sift_create(){return new Extractor;}
extern "C" void uav_sift_destroy(void* ptr){delete static_cast<Extractor*>(ptr);}
static int extract(void* ptr,const float* input,const unsigned char* mask,
 int width,int height,float threshold,int scale_up,float* xy,float* descriptors,int limit,
 float* geometry,bool mask_first){
  auto* ctx=static_cast<Extractor*>(ptr);
  CudaImage img;
  int pitch=iAlignUp(width,128);
  ctx->reserve(ctx->image,ctx->imageCapacity,size_t(pitch)*height*sizeof(float));
  img.Allocate(width,height,pitch,false,ctx->image,const_cast<float*>(input));
  img.Download();
  int octaves=std::min(5,std::max(1,int(std::floor(std::log2(std::min(width,height))))-3));
  int tw=width*(scale_up?2:1),th=height*(scale_up?2:1);
  size_t cells=size_t(th)*iAlignUp(tw,128);
  for(int i=0;i<octaves;i++){tw/=2;th/=2;cells+=size_t(th)*iAlignUp(tw,128);}
  // NUM_SCALES=5: one image pyramid plus eight Laplace buffers.
  ctx->reserve(ctx->temp,ctx->tempCapacity,cells*9*sizeof(float));
  ExtractSift(ctx->data,img,octaves,1.0,threshold,0.0,bool(scale_up),ctx->temp);
  auto& order=ctx->order;order.resize(ctx->data.numPts);
  for(int i=0;i<ctx->data.numPts;i++)order[i]=i;
  // OpenCV retains globally strongest candidates before applying its mask.
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return std::abs(ctx->data.h_data[a].sharpness)>std::abs(ctx->data.h_data[b].sharpness);});
  int count=0;
  int ranks=mask_first?int(order.size()):std::min(limit,int(order.size()));
  for(int rank=0;rank<ranks && count<limit;rank++){
    auto& p=ctx->data.h_data[order[rank]];
    if(geometry){
      bool valid=std::isfinite(p.scale) && p.scale>0;
      for(int i=0;i<128 && valid;i++)valid=std::isfinite(p.data[i]) && p.data[i]>=-1e-6f;
      if(!valid)continue;
    }
    int x=int(std::nearbyint(p.xpos)),y=int(std::nearbyint(p.ypos));
    if(x<0||y<0||x>=width||y>=height||!mask[y*width+x])continue;
    xy[2*count]=p.xpos;xy[2*count+1]=p.ypos;
    if(geometry){geometry[2*count]=2.0f*p.scale;geometry[2*count+1]=p.orientation;}
    std::memcpy(descriptors+128*count,p.data,128*sizeof(float));count++;
    if(geometry)for(int i=0;i<128;i++)descriptors[128*(count-1)+i]=std::max(0.0f,descriptors[128*(count-1)+i]);
  }
  return count;
}
// Preserve the production caller's existing ranking and unit-size API.
extern "C" int uav_sift_extract(void* ptr,const float* input,const unsigned char* mask,
 int width,int height,float threshold,int scale_up,float* xy,float* descriptors,int limit){
  return extract(ptr,input,mask,width,height,threshold,scale_up,xy,descriptors,limit,nullptr,false);
}
// Experimental consumer retains native scale/orientation and selects within
// its object support before limiting the keypoint count.
extern "C" int uav_sift_extract_geometry(void* ptr,const float* input,const unsigned char* mask,
 int width,int height,float threshold,int scale_up,float* xy,float* descriptors,int limit,float* geometry){
  return extract(ptr,input,mask,width,height,threshold,scale_up,xy,descriptors,limit,geometry,true);
}

extern "C" int uav_sift_extract_geometry_stream(void* ptr,const float* input,const unsigned char* mask,
 int width,int height,float threshold,int scale_up,float* xy,float* descriptors,int limit,float* geometry,
 void* stream){
  struct StreamScope {
    cudaStream_t previous = uav_sift_stream;
    explicit StreamScope(void* stream) { uav_sift_stream = static_cast<cudaStream_t>(stream); }
    ~StreamScope() { uav_sift_stream = previous; }
  } scope(stream);
  return extract(ptr,input,mask,width,height,threshold,scale_up,xy,descriptors,limit,geometry,true);
}
