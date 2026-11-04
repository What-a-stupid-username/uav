// @nolint (improperly imported third-party code)
/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

// UAV: FP16 inference kernel; launch explicitly uses current CUDA stream.
#include <cuda_fp16.h>

// Match c10::Half arithmetic: each binary result converts float back to half.
struct H {
  __half value;
  __device__ H() {}
  __device__ H(float x): value(__float2half_rn(x)) {}
  __device__ H(int x): value(__float2half_rn(float(x))) {}
  __device__ operator float() const { return __half2float(value); }
};
__device__ H operator+(H a,H b) { return H(float(a)+float(b)); }
__device__ H operator-(H a,H b) { return H(float(a)-float(b)); }
__device__ H operator*(H a,H b) { return H(float(a)*float(b)); }
__device__ H bilinear_interpolate(
    const H* in,
    int height,
    int width,
    H h,
    H w) {
  if (float(h) <= -1 || height <= float(h) || float(w) <= -1 || width <= float(w)) {
    return 0;
  }

  int h_low = floorf(float(h));
  int w_low = floorf(float(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  H lh = h - H(h_low);
  H lw = w - H(w_low);
  H hh = H(1) - lh, hw = H(1) - lw;

  H v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  H v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  H v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  H v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  H w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  H val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

extern "C" __global__ void deformable_im2col_kernel(
    int n,
    const H* input_ptr,
    const H* offset_ptr,
    const H* mask_ptr,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int batch_sz,
    int n_in_channels,
    int n_offset_grps,
    int out_h,
    int out_w,
    bool use_mask,
    H* columns_ptr) {
  for (int index=blockIdx.x*blockDim.x+threadIdx.x; index<n; index+=blockDim.x*gridDim.x) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
          out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        H mask_value = 1;
        if (use_mask) {
          mask_value =
              mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
        }

        const H offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const H offset_w = offset_ptr
            [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const H y =
            H((out_y * stride_h - pad_h) + i * dilation_h) + offset_h;
        const H x =
            H((out_x * stride_w - pad_w) + j * dilation_w) + offset_w;
        *columns_ptr =
            mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}
