"""Visibility shortcuts preserve native template pixels, including frame edges."""
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.geometry import from_components,points,visible_support


class VisibleSupportTests(unittest.TestCase):
    def test_random_similarity_transforms_match_all_pixel_projection(self):
        rng=np.random.default_rng(197)
        mask=rng.random((47,123))>.6
        yy,xx=np.nonzero(mask)
        for _ in range(120):
            value=[rng.uniform(-200,400),rng.uniform(-100,300),rng.uniform(-3,3),rng.uniform(-2,2)]
            matrix=from_components(value)
            projected=points(matrix,np.column_stack((xx,yy)))
            expected=int(((projected[:,0]>=0)&(projected[:,0]<400)&
                          (projected[:,1]>=0)&(projected[:,1]<300)).sum())
            self.assertEqual(visible_support(mask,matrix,(300,400)),expected)

    def test_inside_canvas_does_not_expand_support_coordinates(self):
        mask=np.ones((30,80),np.float32);mask[10:20,20:40]=0
        with patch('uav.template_mask.geometry.np.nonzero',side_effect=AssertionError('interior projection')):
            self.assertEqual(visible_support(mask,from_components([40,50,.2,0]),(300,400)),2200)

    def test_half_open_frame_edges_and_empty_support(self):
        mask=np.ones((3,4),np.float32)
        for x,y,count in ((0,0,12),(-1,0,9),(0,-1,8),(1,0,9),(0,1,8)):
            matrix=from_components([x,y,0,0])
            self.assertEqual(visible_support(mask,matrix,mask.shape),count)
            self.assertEqual(visible_support(mask*0,matrix,mask.shape),0)


if __name__=='__main__':unittest.main()
