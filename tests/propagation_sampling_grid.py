"""Shared propagation grids match per-frame generation, synthetic only."""
import unittest
from unittest.mock import patch
import torch
from uav.propainter.video.model.propainter import BidirectionalPropagation
from uav.propainter.video.model.modules.flow_loss_utils import normalized_flow_grid


class SamplingGridTests(unittest.TestCase):
    @torch.inference_mode()
    def test_shared_grid_matches_original_warps(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            dtype=torch.float16 if device=='cuda' else torch.float32
            module=BidirectionalPropagation(3,learnable=False).to(device).eval()
            x=torch.randn(2,5,3,12,16,device=device,dtype=dtype)
            flow=torch.randn(2,4,2,12,16,device=device,dtype=dtype)*.5
            mask=(torch.rand(2,5,1,12,16,device=device)>.5).to(dtype)
            for interpolation in ['nearest','bilinear']:
                with patch('torch.is_grad_enabled',return_value=True):
                    expected=module(x,flow,-flow,mask,interpolation)
                with patch('uav.propainter.video.model.propainter.normalized_flow_grid',wraps=normalized_flow_grid) as grid:
                    actual=module(x,flow,-flow,mask,interpolation)
                    self.assertEqual(grid.call_count,2)
                for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
