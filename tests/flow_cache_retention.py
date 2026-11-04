"""Only future overlap flow pairs need owning cache copies; no media access."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from uav.propainter_flow_cache import FlowCache

class FlowRetentionTests(unittest.TestCase):
    def test_future_only_copies_preserve_current_results_and_hits(self):
        raft=SimpleNamespace(fix_raft=SimpleNamespace(args=SimpleNamespace(mixed_precision=True)),reuse_features=True)
        frames=torch.arange(8,dtype=torch.float32).view(1,8,1,1,1)
        def estimate(raft,frames,iters,scale):
            x=frames[:,:-1].expand(-1,-1,2,8,8).clone()
            return x,-x
        old=FlowCache();new=FlowCache();new.keep_from=5
        with patch('uav.raft_resolution.estimate_flow',side_effect=estimate):
            a=old.estimate(raft,frames,0,(0,0,8,8),8)
            b=new.estimate(raft,frames,0,(0,0,8,8),8)
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(len(old.items),7);self.assertEqual(len(new.items),2)
            # Raising insertion threshold must not hide existing overlap hits.
            new.keep_from=10
            with patch('uav.raft_resolution.estimate_flow',side_effect=AssertionError('unexpected recomputation')):
                hit=new.estimate(raft,frames[:,5:8],5,(0,0,8,8),8)
            for x,y in zip(hit,b):torch.testing.assert_close(x,y[:,5:7],rtol=0,atol=0)
            new.prune(8)
            self.assertEqual(new.bytes,0)

    def test_final_window_does_not_retain_any_pairs(self):
        cache=FlowCache();cache.keep_from=8
        raft=SimpleNamespace(fix_raft=SimpleNamespace(args=SimpleNamespace(mixed_precision=False)))
        frames=torch.zeros(1,8,3,8,8)
        expected=(torch.ones(1,7,2,8,8),torch.zeros(1,7,2,8,8))
        with patch('uav.raft_resolution.estimate_flow',return_value=expected):
            actual=cache.estimate(raft,frames,0,(),8)
        self.assertIs(actual[0],expected[0]);self.assertEqual(cache.bytes,0)
