"""Bounded compositing matches full-frame arithmetic exactly; synthetic only."""
import unittest
import torch
from uav.repair_blend import RepairBlend, edge_weights


class BoundedBlendTests(unittest.TestCase):
    def test_cached_geometry_is_exact_and_not_modified_by_blending(self):
        weight=edge_weights(12,20,'cpu')
        saved=weight.clone()
        original=torch.zeros(3,12,20,dtype=torch.uint8)
        blend=RepairBlend({0:original})
        blend.add(0,original,torch.ones(1,12,20,dtype=torch.bool),(0,0,20,12),weight)
        self.assertIs(weight,edge_weights(12,20,torch.device('cpu')))
        torch.testing.assert_close(weight,saved,rtol=0,atol=0)

    def test_pixel_parity_and_input_preservation(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            generator=torch.Generator(device=device).manual_seed(13)
            original=torch.randint(256,(3,72,96),generator=generator,device=device,dtype=torch.uint8)
            saved=original.clone()
            crops=[(8,12,48,44),(24,20,64,60),(48,12,80,36)]
            for empty in (False,True):
                full=RepairBlend({0:original})
                bounded=RepairBlend({0:original},(8,12,80,60))
                self.assertLess(bounded.sums[0].numel(),full.sums[0].numel())
                for l,t,r,b in crops:
                    patch=torch.rand((3,b-t,r-l),generator=generator,device=device)*255
                    mask=torch.rand((1,b-t,r-l),generator=generator,device=device)>.35
                    if empty:mask.zero_()
                    weight=edge_weights(b-t,r-l,device)
                    for blend in (full,bounded):blend.add(0,patch,mask,(l,t,r,b),weight)
                self.assertTrue(torch.equal(full.finish(0,original),bounded.finish(0,original)))
                self.assertTrue(torch.equal(original,saved))

    def test_full_bounds_and_invalid_crops(self):
        original=torch.zeros(3,12,20,dtype=torch.uint8)
        blend=RepairBlend({0:original},(0,0,20,12))
        self.assertTrue(torch.equal(blend.finish(0,original),original))
        with self.assertRaises(ValueError):RepairBlend({0:original},(-1,0,20,12))
        blend=RepairBlend({0:original},(4,4,12,12))
        with self.assertRaises(ValueError):blend.add(0,original,None,(0,0,20,12),None)
