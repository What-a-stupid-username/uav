"""Fixed correlation search geometry can be shared across pair batches."""
import unittest
import torch
from uav.propainter.video.raft.corr import CorrBlock,_search_grid


class SearchGridTests(unittest.TestCase):
    @torch.inference_mode()
    def test_original_coordinate_order_and_correlation_sampling(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for radius in [1,4]:
                axis=torch.linspace(-radius,radius,2*radius+1)
                expected=torch.stack(torch.meshgrid(axis,axis,indexing='ij'),dim=-1).to(device)
                actual=_search_grid(radius,torch.device(device))
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                x=torch.randn(1,4,8,8,device=device)
                a=CorrBlock(x,x,num_levels=2,radius=radius,cache_grid=False)
                b=CorrBlock(x,x,num_levels=2,radius=radius,cache_grid=True)
                coords=torch.randn(1,2,8,8,device=device)
                torch.testing.assert_close(a(coords),b(coords),rtol=0,atol=0)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
