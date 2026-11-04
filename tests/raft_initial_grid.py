"""RAFT coordinate initialization uses cached immutable device geometry."""
import unittest
import torch
from uav.propainter.video.raft.raft import RAFT
from uav.propainter.video.raft.utils.utils import coords_grid


class InitialGridTests(unittest.TestCase):
    def test_cached_grid_and_independent_mutable_coordinates(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for batch,h,w in [(1,128,192),(3,136,264)]:
                image=torch.empty(batch,3,h,w,device=device)
                expected=coords_grid(batch,h//8,w//8).to(device)
                a,b=RAFT.initialize_flow(None,image)
                torch.testing.assert_close(a,expected,rtol=0,atol=0)
                torch.testing.assert_close(b,expected,rtol=0,atol=0)
                b.add_(7)
                c,d=RAFT.initialize_flow(None,image)
                torch.testing.assert_close(a,c,rtol=0,atol=0)
                torch.testing.assert_close(d,expected,rtol=0,atol=0)
