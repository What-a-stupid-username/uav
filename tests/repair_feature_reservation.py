"""One allocation per window preserves ordered feature fusion exactly."""
import unittest
from types import SimpleNamespace
import torch
from uav.sam3_repair_sink import FeatureCanvas


class FeatureReservationTests(unittest.TestCase):
    def test_reserved_accumulation_matches_incremental(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            generator=torch.Generator(device=device).manual_seed(42)
            a=FeatureCanvas(device);b=FeatureCanvas(device)
            windows=[[(8,4,40,36),(24,4,56,36),(16,12,48,44)],
                     [(0,0,32,32),(32,16,64,48)]]
            for crops in windows:
                for crop in crops:b.reserve(crop)
                allocations=set()
                for crop in crops:
                    feature=torch.randn(128,8,8,device=device,dtype=torch.float16,generator=generator)
                    for canvas in (a,b):canvas.add(feature,crop,.75)
                    allocations.add(b.sums.data_ptr())
                self.assertEqual(len(allocations),1)
                self.assertEqual(a.bounds,b.bounds)
                self.assertEqual(a.crops,b.crops)
                torch.testing.assert_close(a.sums,b.sums,rtol=0,atol=0)
                torch.testing.assert_close(a.weights,b.weights,rtol=0,atol=0)
            original=torch.full((3,56,72),87,device=device,dtype=torch.uint8)
            mask=torch.ones((56,72),device=device,dtype=torch.bool)
            mask[:8]=False
            def decoder(feature):
                return torch.nn.functional.interpolate(feature[:,:3],scale_factor=4,mode='nearest')
            model=SimpleNamespace(decoder=decoder,_overlap_finite=[])
            expected,count=a.decode(model,original,mask)
            actual,actual_count=b.decode(model,original,mask)
            self.assertEqual(count,actual_count)
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
