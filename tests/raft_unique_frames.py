"""Unique-frame TRT routing keeps adjacent pair and context order."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from uav.propainter.video.model.modules.flow_comp_raft import RAFT_bi


class UniqueFrameTests(unittest.TestCase):
    @torch.inference_mode()
    def test_unique_and_legacy_pair_order_match(self):
        for batch in [1,2]:
            frames=torch.arange(batch*5,dtype=torch.float32).reshape(batch,5,1,1,1).expand(-1,-1,3,8,8)
            class Raft:
                args=SimpleNamespace(mixed_precision=False,alternate_corr=False)
                trt_backends={}
                def __init__(self):self.calls=[]
                def fnet(self,x):
                    self.calls.append(('fnet',len(x) if isinstance(x,list) else x.shape[0]))
                    return tuple(v.clone() for v in x) if isinstance(x,list) else x.clone()
                def cnet(self,x):self.calls.append(('cnet',x.shape[0]));return x.clone()
                def __call__(self,a,b,**kwargs):
                    f1,f2,context=kwargs['features']
                    torch.testing.assert_close(f1,a,rtol=0,atol=0)
                    torch.testing.assert_close(f2,b,rtol=0,atol=0)
                    torch.testing.assert_close(context,a,rtol=0,atol=0)
                    return None,(b-a)[:,:2]
            raft=Raft();owner=SimpleNamespace(fix_raft=raft,reuse_features=True,training=False)
            with patch('uav.raft_tensorrt.supported',return_value=True):actual=RAFT_bi.forward(owner,frames)
            self.assertEqual(raft.calls,[('fnet',batch*5),('cnet',batch*5)])
            raft.calls=[]
            with patch('uav.raft_tensorrt.supported',return_value=False):expected=RAFT_bi.forward(owner,frames)
            self.assertEqual(len(raft.calls),3)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
