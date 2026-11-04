"""Batched regime predictions retain scalar models and original change evidence."""
from unittest.mock import patch
import unittest
import numpy as np
from uav.template_mask.geometry import from_components
from uav.template_mask.motion import ComponentModel,MotionPrior
from uav.template_mask.records import Pose


class ModelPredictionTests(unittest.TestCase):
    def test_all_model_families_match_scalar_predictions(self):
        rng=np.random.default_rng(93)
        knots=(0.,0.,0.,0.,.3,.7,1.,1.,1.,1.)
        for mode,n,degree in [('static',1,0),('linear',2,1),('accelerating',3,2),
                              ('smooth',4,3),('smooth',6,5),('periodic',3,0),('smooth_spline',6,3)]:
            model=ComponentModel(mode,rng.normal(size=n),.3,17.,89.,3.,degree,knots)
            times=np.linspace(-13,142,169)
            expected=np.array([model.predict(t) for t in times])
            np.testing.assert_allclose(model.predict_many(times),expected,rtol=1e-13,atol=1e-13)

    def test_indexed_nearest_matches_scan_for_sparse_times_and_ties(self):
        times=[0,3,9,12,37,100,141,143,200]
        rows=[Pose(f,from_components([f,30,0,0]),.7,'measured',f,presence=True) for f in times]
        prior=MotionPrior(rows,(0,0,100,30),(50,15))
        for frame in np.arange(-10,215,.5):
            expected=min(prior.evidence,key=lambda p:abs(p.frame-frame))
            self.assertIs(prior._nearest_evidence(frame),expected)

    def test_regime_selection_and_predictions_match_scalar_path(self):
        def scalar(model,times):return np.array([model.predict(t) for t in times])
        motions=[lambda f:(20+.4*f,30+.01*f*f,.002*f,0),
                 lambda f:(20+.4*f if f<64 else 45.6-.7*(f-64),30,.01*f,.01*np.sin(f/10)),
                 lambda f:(20 if f<40 else 60,30,0,0)]
        for motion in motions:
            rows=[Pose(f,from_components(motion(f)),.7,'measured',f,presence=True) for f in range(96)]
            with patch.object(ComponentModel,'predict_many',scalar):
                expected=MotionPrior(rows,(0,0,100,30),(50,15))
            actual=MotionPrior(rows,(0,0,100,30),(50,15))
            self.assertEqual(actual.report(),expected.report())
            for frame in (-2,20,45,65,100):
                a,b=actual.predict(frame),expected.predict(frame)
                np.testing.assert_allclose(a[0],b[0],atol=1e-12,rtol=1e-12)
                self.assertEqual(a[2],b[2])
                self.assertAlmostEqual(a[1],b[1],places=12)


if __name__=='__main__':unittest.main()
