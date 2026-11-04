"""Fuse exact two-neighbour reduction and ratio test over FP32 distances.

Imported only by the CUDA matcher. Triton reuses the runtime's persistent code
cache; descriptor evidence and output indices remain run-local.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["count", "stride"])
def _nearest_two(distances, output, count, stride,
                 block:tl.constexpr, squared:tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0,block)
    values = tl.load(distances+row*stride+columns,columns<count,other=float('inf'))
    if squared:
        values = tl.maximum(values,0.)
    first = tl.min(values,0)
    index = tl.min(tl.where(values==first,columns,2147483647),0)
    second = tl.min(tl.where(columns!=index,values,float('inf')),0)
    if squared:
        # sqrt is monotone. Only the two minima need square roots. If rounding
        # collapses distinct minima into a tie the ratio fails either way, so
        # the chosen index cannot change an accepted result. Keep sqrt.rn and
        # the original FP64 ratio rather than substituting a squared ratio.
        first = libdevice.sqrt_rn(first)
        second = libdevice.sqrt_rn(second)
    # Match the original conversion to double before multiplying by .76.
    keep = first.to(tl.float64) < .76*second.to(tl.float64)
    tl.store(output+row,tl.where(keep,index,-1))


def nearest_indices(distances, *, out=None, squared=False):
    """Return one reference index per query, or -1 when its ratio fails."""
    import torch
    output = (torch.empty(distances.shape[0],dtype=torch.int64,device=distances.device)
              if out is None else out)
    if distances.shape[0]:
        _nearest_two[(distances.shape[0],)](distances,output,distances.shape[1],
            distances.stride(0),triton.next_power_of_2(distances.shape[1]),squared)
    return output
