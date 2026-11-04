"""Context lifetime and shape fallback checks, with synthetic tensors only."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from uav.config import UAVConfig
from uav.propagation_tensorrt import ConvolutionEngine, install_propagation

class PropagationTests(unittest.TestCase):
    def test_supported_shapes_and_bounded_contexts(self):
        backend=ConvolutionEngine.__new__(ConvolutionEngine)
        backend.channels=261;backend.max_batch=1;backend.capacity=1;backend.contexts={}
        image=SimpleNamespace(is_cuda=True,dtype=torch.float16,ndim=4,shape=(1,261,64,128))
        self.assertTrue(backend.supports(image))
        backend.contexts[(1,261,64,128)]=object()
        self.assertTrue(backend.supports(image))
        image.shape=(1,261,66,128)
        self.assertFalse(backend.supports(image))
        backend.contexts.clear();image.shape=(1,261,64,131)
        self.assertFalse(backend.supports(image))
        image.shape=(1,258,64,128)
        self.assertFalse(backend.supports(image))

    def test_contexts_are_reused_per_shape(self):
        backend=ConvolutionEngine.__new__(ConvolutionEngine)
        backend.contexts={};backend.engine=Mock()
        def context(*args):
            result=Mock();result.get_tensor_shape.return_value=(1,128,8,8)
            return result
        backend.engine.create_execution_context.side_effect=context
        backend.workspace=Mock();backend.workspace.data_ptr.return_value=100;backend.workspace.numel.return_value=1024
        def image(shape):
            x=Mock();x.shape=shape;x.device='cuda';x.dtype=torch.float16;x.contiguous.return_value=x
            return x
        output=Mock();output.data_ptr.return_value=200
        with patch('torch.empty',return_value=output),patch('torch.cuda.current_stream',return_value=SimpleNamespace(cuda_stream=1)):
            backend(image((1,261,8,8)))
            backend(image((1,261,16,8)))
            backend(image((1,261,8,8)))
        self.assertEqual(backend.engine.create_execution_context.call_count,2)
        first=backend.contexts[(1,261,8,8)]
        self.assertEqual(first.set_input_shape.call_count,1)
        self.assertEqual(first.execute_async_v3.call_count,2)

    def test_install_fallback_and_shared_workspace(self):
        module=torch.nn.Sequential(torch.nn.Conv2d(1,1,1))
        propagation=torch.nn.Module()
        propagation.deform_align=torch.nn.ModuleDict({'backward':torch.nn.Module()})
        propagation.deform_align['backward'].conv_offset=module
        propagation.backbone=torch.nn.ModuleDict({'backward':torch.nn.Sequential(torch.nn.Conv2d(1,1,1))})
        propagation.fuse=torch.nn.Sequential(torch.nn.Conv2d(1,1,1))
        model=SimpleNamespace(feat_prop_module=propagation)
        engines=[]
        def engine(*args):
            value=Mock();value.engine.device_memory_size_v2=16;value.supports.return_value=False
            engines.append(value);return value
        with patch('uav.propagation_tensorrt.cached_engine',return_value='engine'),patch('uav.propagation_tensorrt.ConvolutionEngine',side_effect=engine):
            install_propagation(model,'checkpoint')
        x=torch.zeros(1,1,2,2)
        torch.testing.assert_close(module(x),module[0](x))
        self.assertTrue(all(e.workspace is engines[0].workspace for e in engines))
        self.assertTrue(UAVConfig().propainter_trt_propagation)
