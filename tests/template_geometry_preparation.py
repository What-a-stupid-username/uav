"""Small geometry helpers retain exact coordinates and clipping semantics."""
import math
import unittest
import numpy as np
from uav.template_mask.geometry import bounds,from_components,points,discrepancy,discrepancies,components


class GeometryPreparationTests(unittest.TestCase):
    def test_native_components_match_original_arithmetic(self):
        rng=np.random.default_rng(133)
        for _ in range(200):
            matrix=rng.normal(size=(3,3))
            anchor=tuple(rng.normal(size=2))
            xy=points(matrix,[anchor])[0]
            expected=np.array([*xy,math.atan2(matrix[1,0],matrix[0,0]),
                math.log(max(1e-9,np.linalg.norm(matrix[:2,0])))])
            np.testing.assert_allclose(components(matrix,anchor),expected,rtol=2e-15,atol=1e-12)

    def test_batched_discrepancies_equal_individual_corner_errors(self):
        rng=np.random.default_rng(122)
        for dtype in (np.float32,np.float64):
            a,b=rng.normal(size=(2,180,3,3)).astype(dtype)
            for box in ((0,0,100,40),tuple(rng.uniform(-300,300,4))):
                expected=np.array([discrepancy(x,y,box) for x,y in zip(a,b)])
                np.testing.assert_allclose(discrepancies(a,b,box),expected,rtol=2e-15,atol=1e-12)
    def test_discrepancy_preserves_original_corner_arithmetic(self):
        rng=np.random.default_rng(92)
        for dtype in (np.float32,np.float64):
            for _ in range(150):
                a,b=rng.normal(size=(2,3,3)).astype(dtype)
                box=rng.uniform(-300,300,4)
                l,t,r,bottom=box
                xy=[[l,t],[r,t],[l,bottom],[r,bottom]]
                expected=float(np.max(np.linalg.norm(points(a,xy)-points(b,xy),axis=1)))
                self.assertAlmostEqual(discrepancy(a,b,box),expected,delta=1e-12)

    def test_bounds_preserve_original_corner_reductions(self):
        rng=np.random.default_rng(193)
        for dtype in (np.float32,np.float64):
            for _ in range(150):
                matrix=rng.normal(size=(3,3)).astype(dtype)
                box=rng.uniform(-300,300,4)
                l,t,r,b=box
                xy=points(matrix,[[l,t],[r,t],[l,b],[r,b]])
                expected=np.r_[xy.min(0),xy.max(0)]
                np.testing.assert_allclose(bounds(matrix,box),expected,rtol=2e-15,atol=1e-12)
                self.assertEqual(bounds(matrix,box).dtype,expected.dtype)

    def test_scalar_scale_clipping_retains_numpy_semantics(self):
        for logscale in (-np.inf,-100.,-8.,-7.999,0.,7.999,8.,100.,np.inf,np.nan):
            actual=from_components([12.,31.,.3,logscale],(3.,7.))
            scale=math.exp(float(np.clip(logscale,-8,8)))
            expected=np.eye(3)
            expected[:2,:2]=scale*np.array([[math.cos(.3),-math.sin(.3)],[math.sin(.3),math.cos(.3)]])
            expected[:2,2]=[12.,31.]-expected[:2,:2]@np.asarray((3.,7.))
            np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
