"""Complete flow at reduced resolution while preserving known original flow."""
from .raft_resolution import flow_size


def resize_flow(flow, size):
    import torch.nn.functional as F
    h, w = flow.shape[-2:]
    if (h, w) == tuple(size):
        return flow
    sh, sw = size
    value = F.interpolate(flow.flatten(0, 1), size=size, mode='bilinear',
                          align_corners=False)
    value[:, 0].mul_(sw / w)
    value[:, 1].mul_(sh / h)
    return value.reshape(*flow.shape[:3], sh, sw)


def complete_flow(model, flows, masks, scale=1.):
    import torch.nn.functional as F
    original_size = flows[0].shape[-2:]
    size = flow_size(*original_size, scale)
    if size == original_size:
        prediction, _ = model.forward_bidirect_flow(flows, masks)
    else:
        small_flows = tuple(resize_flow(flow, size) for flow in flows)
        # Any masked source pixel must remain unknown to the completion model.
        small_masks = F.adaptive_max_pool2d(masks.flatten(0, 1), size)
        small_masks = small_masks.reshape(*masks.shape[:3], *size)
        prediction, _ = model.forward_bidirect_flow(small_flows, small_masks)
        prediction = tuple(resize_flow(flow, original_size) for flow in prediction)
    # Restore original known flows exactly, including at the mask boundary.
    return model.combine_flow(flows, prediction, masks)
