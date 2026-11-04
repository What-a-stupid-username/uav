"""Synthetic normalization parity and TRT subnet routing; no media reads."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from uav.raft_tensorrt import ExplicitInstanceNorm, explicit_norms, ExportUpdate, supported, install_raft

class FakeEncoder(torch.nn.Module):
    def forward(self,x): return x

class FakeUpdate(torch.nn.Module):
    def forward(self,net,inp,corr,flow,upsample=True):
        return net,'mask' if upsample else None,flow

class RaftTests(unittest.TestCase):
    def test_explicit_statistics(self):
        torch.manual_seed(9)
        for shape in ((1,8,7,11),(3,4,16,8)):
            x=torch.randn(shape)
            expected=torch.nn.InstanceNorm2d(shape[1],eps=1e-5)(x)
            actual=ExplicitInstanceNorm(1e-5)(x)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        model=torch.nn.Sequential(torch.nn.InstanceNorm2d(4))
        self.assertIs(explicit_norms(model),model)
        self.assertIsInstance(model[0],ExplicitInstanceNorm)
        with self.assertRaises(ValueError):
            explicit_norms(torch.nn.Sequential(torch.nn.InstanceNorm2d(4,affine=True)))

    def test_export_update(self):
        x=torch.zeros(1)
        self.assertEqual(len(ExportUpdate(FakeUpdate(),True)(x,x,x,x)),3)
        a,b=ExportUpdate(FakeUpdate(),False)(x,x,x,x)
        self.assertIs(a,x);self.assertIs(b,x)

    def test_ranges(self):
        good=SimpleNamespace(is_cuda=True,ndim=4,dtype=torch.float32,shape=(24,3,128,264))
        self.assertTrue(supported((good,),(3,),128,264))
        good.shape=(25,3,128,264)
        self.assertFalse(supported((good,),(3,),128,264))
        self.assertFalse(supported((torch.zeros(1,3,128,128),),(3,),128,264))

    def test_install_routes_and_keeps_original_fallback(self):
        raft=SimpleNamespace(fnet=FakeEncoder(),cnet=FakeEncoder(),update_block=FakeUpdate())
        backend=Mock(return_value=('net','delta'))
        with patch('uav.raft_tensorrt.cached_engine',return_value='engine'),patch('uav.raft_tensorrt.Engine',return_value=backend):
            install_raft(raft,'weights')
        x=torch.zeros(1,3,4,4)
        self.assertIs(raft.fnet(x),x)
        with patch('uav.raft_tensorrt.supported',return_value=True):
            self.assertEqual(raft.update_block(x,x,x,x,upsample=False),('net',None,'delta'))
        self.assertEqual(len(raft.trt_backends),4)
