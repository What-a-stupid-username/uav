"""Whole-trajectory curve evidence, local support and independent validation."""
import json
import unittest
import numpy as np
from uav.template_mask.geometry import translation
from uav.template_mask.motion import MotionPrior, fit_component
from uav.template_mask.records import Pose


class CurveMotionTests(unittest.TestCase):
    box = (0, 0, 100, 25)
    anchor = (50, 12)

    def poses(self, motion, frames=range(360)):
        return [Pose(f, translation(*motion(f)), .25, 'measured', f, presence=True)
                for f in frames]

    def test_variable_speed_curve_beyond_one_low_degree_polynomial(self):
        times = np.arange(360, dtype=float)
        values = 40*np.sin(times/50)+.000003*times**3
        model = fit_component(times, values, .5)
        self.assertIsNotNone(model)
        self.assertEqual(model.mode, 'smooth_spline')
        self.assertLess(max(abs(model.predict(t)-v) for t, v in zip(times, values)), .5)

    def test_curve_with_almost_linear_tail_retains_whole_scope(self):
        def motion(t):
            return t, t+30*(np.sin(np.pi*t/400)**2 if t < 200 else 1.)
        prior = MotionPrior(self.poses(motion), self.box, self.anchor)
        report = prior.report()
        self.assertEqual(report['scope'], 'whole_trajectory')
        self.assertTrue(report['whole_trajectory']['strong'])
        self.assertIn('smooth_spline', report['whole_trajectory']['components'])
        self.assertEqual((report['start'], report['end']), (0, 359))
        self.assertEqual(report['measured_sources'], 360)
        json.dumps(report)

    def test_local_linear_tail_does_not_label_whole_path_linear(self):
        poses = self.poses(lambda f: (f, max(0, f-175)*2))
        prior = MotionPrior(poses, self.box, self.anchor)
        report = prior.report()
        self.assertEqual(report['scope'], 'local_window')
        self.assertFalse(report['whole_trajectory']['strong'])
        self.assertEqual(report['whole_trajectory']['components'][1], 'undetermined')
        self.assertTrue(prior.predict(50)[2])  # Earlier reliable leg is retained.
        self.assertTrue(prior.predict(300)[2])
        from uav.template_mask.gaps import trajectory_bridge
        from uav.template_mask.records import Template
        self.assertIsNone(trajectory_bridge(poses[:168], poses[184:],
            Template(0, self.box, self.anchor), range(167, 185)))

    def test_spline_does_not_certify_a_corner_or_random_path(self):
        times = np.arange(360, dtype=float)
        for values in (np.maximum(0, times-175)*2,
                       np.random.default_rng(5).normal(0, 10, len(times))):
            self.assertIsNone(fit_component(times, values, .5))

    def test_sparse_ends_and_predictions_do_not_supply_curve_evidence(self):
        times = np.r_[np.arange(30), np.arange(200, 230)]
        self.assertIsNone(fit_component(times, .01*times**2-.00002*times**3, .1))
        motion = lambda f: (f, 40*np.sin(f/50)+.000003*f**3)
        poses = self.poses(motion)
        prior = MotionPrior(poses, self.box, self.anchor)
        predictions = [Pose(f, translation(999, 999), .01, 'predicted', f+1000,
                            presence=True) for f in range(360)]
        other = MotionPrior(poses+predictions, self.box, self.anchor)
        self.assertEqual(other.report(), prior.report())
        self.assertEqual(len(other.evidence), 360)

    def test_same_source_is_not_counted_twice_and_insufficient_scope_is_explicit(self):
        poses = self.poses(lambda f: (f, 2*f), range(7))
        duplicate = Pose(7, translation(7, 14), .25, 'measured', 6, presence=True)
        prior = MotionPrior(poses+[duplicate], self.box, self.anchor)
        self.assertEqual(len(prior.evidence), 7)
        self.assertIsNone(prior.report()['whole_trajectory'])
        self.assertFalse(prior.predict(8)[2])


if __name__ == '__main__':
    unittest.main()
