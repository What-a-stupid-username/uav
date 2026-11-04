"""Inference grid reuse preserves coordinate rounding and input storage."""
import unittest
import torch
from uav.propainter.video.model.modules.flow_loss_utils import flow_warp,_eager_flow_warp


class FlowWarpStorageTests(unittest.TestCase):
    @torch.inference_mode()
    def test_grid_and_warp_are_bitexact(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for dtype in [torch.float32]+([torch.float16] if device=='cuda' else []):
                for h,w in [(1,9),(8,1),(17,29)]:
                    image=torch.randn(2,3,h,w,device=device,dtype=dtype)
                    flow=torch.randn(2,2,h,w,device=device,dtype=dtype).permute(0,2,3,1)
                    saved=flow.clone();original=image.clone()
                    for mode in ['nearest','bilinear']:
                        for padding in ['zeros','border','reflection']:
                            expected=_eager_flow_warp(image,flow,mode,padding)
                            actual=flow_warp(image,flow,mode,padding)
                            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    torch.testing.assert_close(flow,saved,rtol=0,atol=0)
                    torch.testing.assert_close(image,original,rtol=0,atol=0)
