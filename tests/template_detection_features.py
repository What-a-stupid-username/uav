"""Detection-only feature preparation on synthetic RAM tensors."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from PIL import Image
from uav.sam3_crop_groups import set_detection_images


class DetectionFeaturesTests(unittest.TestCase):
    def test_discarded_neck_branches_are_not_computed(self):
        image = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
        for scalp in (0, 1, 2):
            branches = [Mock(side_effect=lambda x, n=n: x+n) for n in range(3)]
            position = Mock(side_effect=lambda x: torch.zeros_like(x))
            neck = SimpleNamespace(trunk=lambda x: [x], convs=branches,
                position_encoding=position, sam2_convs=None)
            processor = SimpleNamespace(device='cpu', transform=lambda x: x,
                model=SimpleNamespace(backbone=SimpleNamespace(vision_backbone=neck, scalp=scalp)))
            result = set_detection_images(processor, [image])
            count = 3-scalp
            self.assertEqual([b.call_count for b in branches], [1]*count+[0]*scalp)
            self.assertEqual(position.call_count, count)
            self.assertEqual(len(result['backbone_fpn']), count)
            for index, feature in enumerate(result['backbone_fpn']):
                self.assertTrue(torch.equal(feature, image.unsqueeze(0)+index))
            self.assertTrue(torch.equal(result['vision_features'], image.unsqueeze(0)+count-1))

    def test_direct_rgb_tensor_preserves_pil_values_and_strides(self):
        from torchvision.transforms.v2.functional import to_image
        rgb = np.random.default_rng(51).integers(0,256,(23,37,3),dtype=np.uint8)
        previous = to_image(Image.fromarray(rgb))
        current = torch.from_numpy(rgb).permute(2,0,1)
        self.assertTrue(torch.equal(current, previous))
        self.assertEqual(current.stride(), previous.stride())


if __name__ == '__main__':
    unittest.main()
