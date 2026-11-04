"""Backend selection and isolation with synthetic arrays and mocked CUDA."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

from uav.template_mask import checkpoint
from uav.cli import build_parser
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.features import SiftMatcher


class BackendTests(unittest.TestCase):
    def test_feature_margin_never_reintroduces_unknown_pixels(self):
        cache=EvidenceCache();self.addCleanup(cache.close)
        matcher=SiftMatcher(cache,TemplateConfig(sift_backend='cpu'))
        self.addCleanup(matcher.close)
        cache.put('features',xy=np.array([[5.,5.],[6.,5.]]),
            size=np.ones(2),desc=np.ones((2,128)))
        p=np.zeros((12,12),np.float32);p[5,5]=.9
        valid=np.ones_like(p,bool);valid[5,6]=False
        key=matcher.select('features','obs',p,(0,0),.23,valid=valid)
        np.testing.assert_array_equal(cache.get(key)['xy'],[[5.,5.]])

    def test_python_and_cli_default_to_cuda_and_cpu_remains_explicit(self):
        self.assertEqual(TemplateConfig().sift_backend, 'cuda')
        parser = build_parser()
        args = ['--source', 'synthetic', '--output', 'synthetic-output']
        self.assertEqual(parser.parse_args(args).sift_backend, 'cuda')
        self.assertEqual(parser.parse_args(args+['--sift-backend', 'cpu']).sift_backend, 'cpu')
        for backend in ('cuda', 'cpu'):
            self.assertEqual(TemplateConfig(sift_backend=backend).sift_backend, backend)
        with self.assertRaises(ValueError):
            TemplateConfig(sift_backend='unknown')
        with self.assertRaisesRegex(ValueError, '8192'):
            TemplateConfig(max_features=8193)

    def test_cuda_dispatch_geometry_and_cleanup(self):
        with patch('uav.cuda_sift.CudaSift') as constructor:
            matcher = SiftMatcher(EvidenceCache(), TemplateConfig())
            constructor.assert_called_once_with(threshold=.5, scale_up=True,
                nfeatures=3000, geometry=True, priority=-1)
            matcher.close()
            constructor.return_value.close.assert_called_once()

    def test_cpu_dispatch_does_not_construct_cuda(self):
        with patch('uav.cuda_sift.CudaSift', side_effect=AssertionError('CPU only')), \
             patch('cv2.SIFT_create') as constructor:
            matcher = SiftMatcher(EvidenceCache(), TemplateConfig(sift_backend='cpu'))
            constructor.assert_called_once_with(nfeatures=3000, contrastThreshold=.015)
            matcher.close()

    def test_feature_cache_is_separate_and_repeated_extraction_is_reused(self):
        image = np.zeros((40, 60, 3), np.uint8)
        fake = Mock()
        fake.detectAndCompute.return_value = ([cv2.KeyPoint(15., 20., 3.)],
                                              np.ones((1, 128), np.float32))
        fake.detect_arrays.return_value = (np.array([[15.,20.]],np.float32),
            np.array([3.],np.float32), np.ones((1,128),np.float32))
        cache = EvidenceCache()
        self.addCleanup(cache.close)
        with patch('uav.cuda_sift.CudaSift', return_value=fake), \
             patch('cv2.SIFT_create', return_value=fake):
            cpu = SiftMatcher(cache, TemplateConfig(sift_backend='cpu'))
            gpu = SiftMatcher(cache, TemplateConfig())
            cpu_key = cpu.extract(image, 1, feature_box=(5, 5, 35, 30))
            gpu_key = gpu.extract(image, 1, feature_box=(5, 5, 35, 30))
            self.assertNotEqual(cpu_key, gpu_key)
            self.assertEqual(gpu.extract(image, 1, feature_box=(5, 5, 35, 30)), gpu_key)
            self.assertEqual(fake.detectAndCompute.call_count, 1)
            self.assertEqual(fake.detect_arrays.call_count, 1)
            self.assertEqual((cpu.extractions, gpu.extractions), (1, 1))
            self.assertTrue(np.isfinite(cache.get(gpu_key)['desc']).all())
            gpu.close()

    def test_cpu_checkpoint_compatibility_and_cuda_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'synthetic-token'
            source.write_text('no media')
            frames = SimpleNamespace(start=0, count=16, source_fps=24.)
            engine = SimpleNamespace(shape=(40, 60), debug=True)
            cpu = TemplateConfig(sift_backend='cpu')
            a = checkpoint.signature(source, frames, cpu, engine)
            b = checkpoint.signature(source, frames, replace(cpu, sift_backend='cuda'), engine)
            self.assertNotIn('sift_backend', a['config'])
            self.assertEqual(b['config']['sift_backend'], 'cuda')
            self.assertNotEqual(a, b)

    def test_runner_closes_cuda_and_decoder_even_when_report_fails(self):
        from uav.template_mask.runner import run
        frames = Mock(count=1, fps=24., source_fps=24.)
        frames.return_value = np.zeros((40, 60, 3), np.uint8)
        engine = Mock(shape=(40,60))
        engine.report.side_effect = RuntimeError('synthetic report failure')
        threads = cv2.getNumThreads()
        with tempfile.TemporaryDirectory() as folder, \
             patch('uav.template_mask.runner.VideoFrames', return_value=frames), \
             patch('uav.template_mask.sam.SamDetector'), \
             patch('uav.template_mask.runner.TemplateEngine', return_value=engine), \
             patch('cv2.imwrite', return_value=True):
            engine.render.return_value = np.zeros((40, 60), bool)
            with self.assertRaisesRegex(RuntimeError, 'synthetic report failure'):
                run('synthetic', folder, model=object(), review=False)
        engine.matcher.close.assert_called_once()
        frames.close.assert_called_once()
        self.assertEqual(cv2.getNumThreads(), threads)


if __name__ == '__main__':
    unittest.main()
