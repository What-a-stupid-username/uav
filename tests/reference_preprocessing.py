"""Reference preprocessing preserves accepted half rounding and padding."""
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F
from uav.propainter_references import ReferenceBank


class ReferencePreprocessingTests(unittest.TestCase):
    def test_strided_crop_with_and_without_padding(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            rgb=torch.arange(3*24*32,device=device).remainder(256).to(torch.uint8).reshape(3,24,32)
            mask=(torch.arange(24*32,device=device).reshape(24,32)%3==0)
            for ph,pw in [(0,0),(3,5)]:
                bank=ReferenceBank(24,count=1);bank.add(0,rgb,mask)
                crop=(4,2,20,18,ph,pw)
                expected_mask=F.pad(mask[2:18,4:20],(0,pw,0,ph))[None,None,None].half()
                expected_rgb=F.pad(rgb[:,2:18,4:20],(0,pw,0,ph),mode='replicate')
                expected=(expected_rgb[None,None].half()/127.5-1)*(1-expected_mask)
                calls=[]
                def encode(image,mask_in,updated):
                    calls.append(image)
                    torch.testing.assert_close(image,expected,rtol=0,atol=0)
                    torch.testing.assert_close(mask_in,expected_mask,rtol=0,atol=0)
                    self.assertIs(updated,mask_in)
                    return image
                backend=SimpleNamespace(flow_crop=crop,model=SimpleNamespace(encode_frames=encode))
                actual=bank.get(backend,1,2,2)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                bank.get(backend,1,2,2)
                self.assertEqual(len(calls),1)
