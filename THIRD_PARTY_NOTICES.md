# Third-party notices

The root UAV Non-Commercial License applies only to original UAV material.
It does not replace or restrict the original licenses of third-party material.
Third-party-derived modifications remain subject to the applicable upstream
terms. Model weights are not distributed with this repository. Permission
from UAV contributors alone does not authorize commercial use of third-party
components or weights.

## ProPainter

Source: https://github.com/sczhou/ProPainter

Copyright 2023 S-Lab. Full license: [licenses/ProPainter.txt](licenses/ProPainter.txt), also retained beside the vendored code.

ProPainter uses S-Lab License 1.0. Redistribution and use for **non-commercial purposes** are permitted subject to its conditions. Commercial use requires permission from its contributors; contact information is in the license. This repository does not relicense ProPainter.

Local inference modifications include FP16, native-resolution tiles, bounded flow/reference caching, token gathering, selective output decoding and GPU integration. Original checkpoint formats are retained. The deformable convolution wrapper uses torchvision.

## RAFT (part of ProPainter's optical-flow pipeline)

Source: https://github.com/princeton-vl/RAFT

Copyright (c) 2020, Princeton Vision & Learning Lab. BSD 3-Clause license retained in [licenses/RAFT.txt](licenses/RAFT.txt) and beside the code. Modified for inference, mixed precision, correlation reuse and compiled update blocks.

## SAM3

Source: https://github.com/facebookresearch/sam3

Upstream checkout: 7f473cfd983d5eb43040ff608f66bff0d7bdcb53 (plus the local inference modifications described below).

Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

Full SAM License, last updated November 19, 2025: [licenses/SAM3.txt](licenses/SAM3.txt), also retained under third_party/sam3/LICENSE. Vendored inference subset includes FP16/Turing compatibility, real-valued rotary embeddings, GPU preprocessing/postprocessing and compile support. Training/evaluation entry points are not provided.

## Runtime libraries

PyTorch, torchvision, NumPy, OpenCV, Pillow, SciPy, timm, einops, imageio-ffmpeg and other packages are runtime dependencies, not additional detection or repair models. Their licenses remain with their distributions. Weights are external and subject to their respective original terms.

## torchvision-derived FP16 deformable im2col

Source: https://github.com/pytorch/vision/blob/v0.24.0/torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu

The inference sampling kernel in uav/csrc/deform_im2col.cu is derived from torchvision 0.24.0. The upstream Caffe and Microsoft notices are retained in that file; torchvision BSD license is in [licenses/torchvision.txt](licenses/torchvision.txt). UAV specializes FP16 batch-one inference, preserves c10::Half operation rounding, and explicitly launches on the current PyTorch CUDA stream. CUDA runtime NVRTC compiles the kernel; this introduces no additional model. Unsupported inference layouts and training use torchvision.

## CudaSift

Source: https://github.com/Celebrandil/CudaSift

Copyright (c) 2017 Mårten Björkman. MIT license retained in licenses/CudaSift.txt and uav/csrc/cudasift/LICENSE. CUDA SIFT is a feature extraction algorithm, not an additional model. Local changes include current-CUDA compatibility, disabled diagnostic timers and reusable image/pyramid buffers. Built from the mounted source into the PyTorch extension cache; no binaries or weights are bundled.
