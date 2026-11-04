"""Bounded ordered CPU blending preserves evidence and propagates failures."""
import unittest
from unittest.mock import Mock
import torch
from uav.repair_accumulator import FeatureAccumulator
from uav.sam3_repair_sink import FeatureCanvas


class AccumulatorTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    @torch.inference_mode()
    def test_async_transfer_preserves_ordered_values(self):
        actual=FeatureCanvas('cpu');expected=FeatureCanvas('cpu')
        crop=(0,0,32,32)
        worker=FeatureAccumulator(capacity=2)
        stream=torch.cuda.Stream()
        try:
            for i in range(5):
                with torch.cuda.stream(stream):
                    feature=torch.full((128,8,8),i/7,device='cuda',dtype=torch.float16)
                    worker.add(actual,feature,crop,.5)
                expected.add(torch.full((128,8,8),i/7,dtype=torch.float16),crop,.5)
            worker.finish()
            torch.testing.assert_close(actual.sums,expected.sums,rtol=0,atol=0)
            torch.testing.assert_close(actual.weights,expected.weights,rtol=0,atol=0)
        finally:worker.close()

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):FeatureAccumulator(0)

    @torch.inference_mode()
    def test_ordered_accumulation_matches_synchronous(self):
        actual=FeatureCanvas('cpu');expected=FeatureCanvas('cpu')
        crops=[(0,0,32,32),(16,8,48,40),(0,0,32,32)]
        for crop in crops:actual.reserve(crop);expected.reserve(crop)
        worker=FeatureAccumulator(capacity=2)
        try:
            for i,crop in enumerate(crops*3):
                feature=torch.randn(128,8,8,dtype=torch.float16)
                weight=(i+1)/10
                expected.add(feature,crop,weight)
                worker.add(actual,feature,crop,weight)
                self.assertLessEqual(len(worker.pending),2)
            worker.finish()
            torch.testing.assert_close(actual.sums,expected.sums,rtol=0,atol=0)
            torch.testing.assert_close(actual.weights,expected.weights,rtol=0,atol=0)
            self.assertEqual(actual.crops,expected.crops)
        finally:worker.close()

    def test_worker_failure_reaches_caller(self):
        worker=FeatureAccumulator()
        canvas=Mock();canvas.add.side_effect=ValueError('synthetic accumulation error')
        worker.add(canvas,torch.zeros(1),(0,0,1,1),1.)
        try:
            with self.assertRaisesRegex(ValueError,'synthetic accumulation error'):worker.finish()
        finally:worker.close()
