"""Window partition accepts packed-QKV views without a preceding full copy."""
import torch
from uav.propainter.video.model.modules.sparse_transformer import window_partition


@torch.inference_mode()
def main():
    torch.manual_seed(95)
    cases = 0
    for dtype in (torch.float16, torch.float32):
        for batch, frames, height, width, channels in (
                (1, 7, 20, 36, 512), (1, 13, 40, 45, 512), (2, 5, 15, 27, 32)):
            packed = torch.randn(batch, frames, height, width, channels * 3,
                                 device='cuda', dtype=dtype)
            for view in packed.chunk(3, dim=-1):
                assert not view.is_contiguous()
                expected = window_partition(view.contiguous(), (5, 9), 4)
                actual = window_partition(view, (5, 9), 4)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                cases += 1
    print(f'{cases} packed QKV window layouts bitexact', flush=True)


if __name__ == '__main__':
    main()
