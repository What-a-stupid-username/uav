"""Optional reduced-resolution RAFT with flows restored to source-pixel units."""
import math


def flow_size(height, width, scale):
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('RAFT scale must be finite and in (0, 1]')
    if scale == 1:
        return height, width
    # Four correlation pyramid levels need at least 16 feature cells per side.
    # Keep small crops intact and align the input to the RAFT encoder stride.
    return tuple(min(size, max(128, math.ceil(size * scale / 8) * 8))
                 for size in (height, width))


def estimate_flow(raft, frames, iters, scale=1.):
    import torch.nn.functional as F
    h, w = frames.shape[-2:]
    sh, sw = flow_size(h, w, scale)
    if (sh, sw) == (h, w):
        return raft(frames, iters=iters)
    small = F.interpolate(frames.flatten(0, 1), size=(sh, sw),
                          mode='bilinear', align_corners=False, antialias=True)
    small = small.reshape(*frames.shape[:3], sh, sw)
    flows = raft(small, iters=iters)
    restored = []
    for flow in flows:
        value = F.interpolate(flow.flatten(0, 1), size=(h, w),
                              mode='bilinear', align_corners=False)
        value[:, 0].mul_(w / sw)
        value[:, 1].mul_(h / sh)
        restored.append(value.reshape(*flow.shape[:3], h, w))
    return tuple(restored)
