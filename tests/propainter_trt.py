"""Synthetic export parity and backend routing without reading media."""
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
import torch
from uav.propainter_tensorrt import ExportEncoder,install_encoder
from uav.propainter.video.model.propainter import Encoder
from uav.config import UAVConfig

class EncoderTests(unittest.TestCase):
    def test_explicit_export_matches_encoder(self):
        torch.manual_seed(13)
        model=Encoder().eval()
        with torch.inference_mode():
            for shape in ((1,5,16,24),(2,5,24,16)):
                x=torch.randn(shape)
                torch.testing.assert_close(ExportEncoder(model)(x),model(x),rtol=0,atol=0)

    def test_default_and_shape_fallback(self):
        self.assertTrue(UAVConfig().propainter_trt_encoder)
        original=Mock(return_value='torch')
        backend=Mock(return_value='trt')
        encoder=SimpleNamespace(forward=original)
        with patch('uav.propainter_tensorrt.cached_engine',return_value='cached'), \
             patch('uav.propainter_tensorrt.TensorRTEncoder',return_value=backend):
            install_encoder(SimpleNamespace(encoder=encoder),'weights')
        good=SimpleNamespace(is_cuda=True,dtype=torch.float16,ndim=4,shape=(4,5,264,520))
        self.assertEqual(encoder.forward(good),'trt')
        good.shape=(5,5,264,520)
        self.assertEqual(encoder.forward(good),'torch')
