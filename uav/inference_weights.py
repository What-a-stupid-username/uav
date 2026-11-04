"""Store autocast GEMM/convolution parameters in their inference compute dtype."""
import torch
from torch import nn


def compact_weights(module):
    """Keep normalization/positional parameters FP32; round only autocast weights."""
    saved=0;seen=set()
    for layer in module.modules():
        names=()
        if isinstance(layer,(nn.Linear,nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.ConvTranspose2d)):
            names=('weight','bias')
        elif isinstance(layer,nn.MultiheadAttention):
            names=('in_proj_weight','in_proj_bias','q_proj_weight','k_proj_weight','v_proj_weight')
        for name in names:
            value=getattr(layer,name,None)
            if value is not None and value.is_cuda and value.dtype==torch.float32 and id(value) not in seen:
                seen.add(id(value));saved+=value.numel()*2
                value.data=value.data.half()
    return saved
