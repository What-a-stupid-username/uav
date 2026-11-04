"""Offset view/broadcast path matches the original cat/repeat arithmetic."""
import torch
from uav.propainter.video.model.propainter import _alignment_offset


def reference(out, flow, magnitude):
    o1, o2, mask = torch.chunk(out, 3, dim=1)
    offset = magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
    offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
    return offset, torch.sigmoid(mask)


@torch.inference_mode()
def main():
    torch.manual_seed(73)
    count = 0
    for dtype in (torch.float16, torch.float32):
        for batch, h, w in ((1, 16, 24), (1, 64, 112), (2, 32, 48)):
            out = torch.randn(batch, 432, h, w, device='cuda', dtype=dtype)
            flow = torch.randn(batch, 2, h, w, device='cuda', dtype=dtype)
            for magnitude in (0, 3, 10):
                for a, b in zip(_alignment_offset(out, flow, magnitude),
                                reference(out, flow, magnitude)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                count += 1
    print(f'{count} offset arithmetic cases bitexact', flush=True)


if __name__ == '__main__':
    main()
