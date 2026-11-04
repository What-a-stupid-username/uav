#pragma once
#include <cuda_runtime.h>

// The legacy ABI uses stream zero. The experimental entry point installs its
// caller-owned stream only for the duration of one synchronous extraction.
extern thread_local cudaStream_t uav_sift_stream;

inline cudaError_t siftReadback(void* dst, const void* src, size_t bytes) {
  cudaError_t result = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                      uav_sift_stream);
  return result == cudaSuccess ? cudaStreamSynchronize(uav_sift_stream) : result;
}

template<class Symbol>
inline cudaError_t siftUploadConstant(const Symbol& symbol, const void* src, size_t bytes) {
  cudaError_t result = cudaMemcpyToSymbolAsync(symbol, src, bytes, 0,
                                              cudaMemcpyHostToDevice, uav_sift_stream);
  // Callers use stack-backed kernels. Retain their lifetime until the upload
  // completes, without synchronizing unrelated inference streams.
  return result == cudaSuccess ? cudaStreamSynchronize(uav_sift_stream) : result;
}
