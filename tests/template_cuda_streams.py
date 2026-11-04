"""Stream routing and concurrency, using in-memory synthetic data only."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event, RLock
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from uav.template_mask.sam import SamDetector


class SamStreamTests(unittest.TestCase):
    def test_sam_mutex_serializes_model_calls_on_its_own_stream(self):
        detector = SamDetector.__new__(SamDetector)
        detector.stream = object()
        detector.inference_lock = RLock()
        detector.lock_wait_seconds = 0.
        entered, release, second = Event(), Event(), Event()

        def infer(image, resolution, **kwargs):
            if image == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test release timed out')
            else:
                second.set()
            return image

        detector._infer_locked = infer
        with patch('torch.cuda.stream', side_effect=lambda stream: nullcontext()) as scope:
            with ThreadPoolExecutor(2) as pool:
                first = pool.submit(detector._infer, 1, 1008)
                self.assertTrue(entered.wait(5))
                other = pool.submit(detector._infer, 2, 1008, boxes_only=True)
                try:
                    self.assertFalse(second.wait(.02))
                finally:
                    release.set()
                self.assertEqual((first.result(), other.result()), (1, 2))
        self.assertEqual(scope.call_count, 2)
        self.assertTrue(all(call.args == (detector.stream,) for call in scope.call_args_list))
        self.assertGreater(detector.lock_wait_seconds, 0.)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime required')
class CudaStreamTests(unittest.TestCase):
    def test_native_sift_and_matching_are_stable_with_background_gpu_work(self):
        from uav.cuda_sift import CudaSift
        from uav.template_mask.neighbours import CudaNeighbours
        rng = np.random.default_rng(75)
        image = rng.integers(0, 256, (64, 96), dtype=np.uint8)
        # Avoid truncating equally ranked features: atomic append order is
        # unspecified even when repeatedly using the legacy default stream.
        legacy = CudaSift(geometry=True, nfeatures=8192)
        tracking = CudaSift(geometry=True, nfeatures=8192, priority=-1)
        search = CudaNeighbours()
        self.addCleanup(legacy.close)
        self.addCleanup(tracking.close)
        self.addCleanup(search.close)
        expected = legacy.detect_arrays(image, None)
        self.assertGreater(len(expected[0]), 10)
        reference = rng.random((80, 128), dtype=np.float32)
        target = reference[::2].copy()
        wanted = search(reference, target)
        low = torch.cuda.Stream(priority=0)
        with torch.cuda.stream(low):
            matrix = torch.randn((768, 768), device='cuda')
            # Work is enqueued independently; native readback must only wait
            # for tracking's high-priority stream, never the default stream.
            for _ in range(32):
                matrix = matrix @ matrix.T / 768
        for _ in range(3):
            for actual, previous in zip(tracking.detect_arrays(image, None), expected):
                # Descriptor reductions already vary slightly between repeated
                # legacy runs; stream routing preserves their numeric precision.
                np.testing.assert_allclose(actual, previous, rtol=0, atol=3e-4)
            for actual, previous in zip(search(reference, target), wanted):
                np.testing.assert_array_equal(actual, previous)
        low.synchronize()
        self.assertEqual(tracking.stream.priority, -1)
        self.assertEqual(search.report()['stream_priority'], -1)


if __name__ == '__main__':
    unittest.main()
