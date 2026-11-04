"""In-place mask thresholding preserves sequential assignment semantics."""
import unittest
import torch
from uav.propainter.video.model.propainter import BidirectionalPropagation


class BinaryMaskTests(unittest.TestCase):
    def test_original_threshold_and_storage(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for dtype in [torch.float16,torch.float32]:
                for threshold in [.1,0.,1.,2.]:
                    expected=torch.tensor([float('nan'),float('inf'),-float('inf'),-.1,0.,.09,.1,.11,1.,3.],device=device,dtype=dtype)
                    actual=expected.clone()
                    expected[expected>threshold]=1
                    expected[expected<=threshold]=0
                    output=BidirectionalPropagation.binary_mask(None,actual,threshold)
                    self.assertEqual(output.data_ptr(),actual.data_ptr())
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
