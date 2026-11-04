"""Independent measured support is required before changing motion regimes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.features import Match
from uav.template_mask.geometry import components, from_components, translation
from uav.template_mask.motion import MotionPrior, fit_pose_component
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Pose


class MotionRegimeTests(unittest.TestCase):
    box = (0, 0, 120, 420)
    anchor = (60, 210)

    def rows(self, motion, frames):
        return [Pose(f, from_components(motion(f), self.anchor), .7,
                     'measured', f, presence=True) for f in frames]

    def prior(self, motion, frames):
        return MotionPrior(self.rows(motion, frames), self.box, self.anchor)

    def test_seven_contrary_originals_keep_old_static_component(self):
        motion = lambda f: (100 if f < 32 else 140, 200, 0, 0)
        prior = self.prior(motion, range(39))
        matrix, sigma, strong = prior.predict(39)
        self.assertAlmostEqual(components(matrix, self.anchor)[0], 100)
        self.assertFalse(strong)
        self.assertGreaterEqual(sigma, 40-1e-8)
        report = prior.report()
        self.assertEqual(len(report['component_regimes'][0]), 1)
        self.assertEqual(report['pending_change_sources'][0], list(range(32, 39)))

    def test_eighth_coherent_original_confirms_new_interval(self):
        motion = lambda f: (100 if f < 32 else 140, 200, 0, 0)
        prior = self.prior(motion, range(40))
        self.assertAlmostEqual(components(prior.predict(40)[0], self.anchor)[0], 140)
        self.assertTrue(prior.predict(40)[2])
        regimes = prior.report()['component_regimes'][0]
        self.assertEqual(len(regimes), 2)
        self.assertEqual(regimes[1]['start'], 32)
        self.assertEqual(regimes[1]['confirmed_at'], 39)
        self.assertEqual(regimes[1]['contradicted_sources'], 8)
        self.assertAlmostEqual(components(prior.predict(20)[0], self.anchor)[0], 100)

    def test_return_to_old_path_clears_isolated_change_evidence(self):
        motion = lambda f: (130 if f in (32, 33, 36, 38) else 100, 200, 0, 0)
        prior = self.prior(motion, range(48))
        self.assertTrue(prior.predict(49)[2])
        self.assertAlmostEqual(components(prior.predict(49)[0], self.anchor)[0], 100)
        self.assertEqual(len(prior.report()['component_regimes'][0]), 1)
        # Later return does not erase uncertainty at the earlier contradiction.
        earlier = prior.predict(33)
        self.assertFalse(earlier[2])
        self.assertGreater(earlier[1], 25)

    def test_predictions_duplicates_and_sparse_endpoints_cannot_confirm_change(self):
        motion = lambda f: (100 if f < 32 else 140, 200, 0, 0)
        rows = self.rows(motion, range(39))
        rows += [Pose(f, translation(999, 999), .01, 'predicted', f, presence=True)
                 for f in range(40, 100)]
        rows += [Pose(39, rows[-1].matrix, .01, 'measured', 38, presence=True)]
        prior = MotionPrior(rows, self.box, self.anchor)
        self.assertEqual(len(prior.evidence), 39)
        self.assertEqual(len(prior.report()['component_regimes'][0]), 1)
        # Eight independent points clustered at opposite ends do not support
        # a new regime throughout the unobserved interval.
        prior = self.prior(motion, [*range(32), 32, 33, 34, 35, 120, 121, 122, 123])
        self.assertEqual(len(prior.report()['component_regimes'][0]), 1)

    def test_short_angle_scale_wobble_cannot_turn_static_object_into_motion(self):
        def motion(f):
            jitter = f >= 32
            return (100, 200, (.012*(-1)**f if jitter else 0),
                    (.012*(-1)**f if jitter else 0))
        prior = self.prior(motion, range(64))
        matrix, sigma, _ = prior.predict(65)
        value = components(matrix, self.anchor)
        np.testing.assert_allclose(value, [100, 200, 0, 0], atol=.003)
        self.assertLess(sigma, 10)
        self.assertTrue(all(len(r) == 1 for r in prior.regimes))

    def test_constant_position_survives_unknown_rotation_component(self):
        rng = np.random.default_rng(14)
        angles = rng.normal(0, .2, 64)
        prior = self.prior(lambda f: (100, 200, angles[f], 0), range(64))
        matrix, _, _ = prior.constrain(65, from_components((999, 888, .5, .4), self.anchor), 50)
        np.testing.assert_allclose(components(matrix, self.anchor)[:2], [100, 200], atol=.01)

    def test_single_direction_rotation_unwraps_and_real_reversal_establishes_new_leg(self):
        angle = lambda f: .04*f if f < 64 else 2.56-.05*(f-64)
        prior = self.prior(lambda f: (100, 200, angle(f), 0), range(120))
        for f in (20, 50, 90, 119):
            actual = components(prior.predict(f)[0], self.anchor)[2]
            self.assertLess(abs(np.angle(np.exp(1j*(actual-angle(f))))), .02)
        self.assertGreaterEqual(len(prior.regimes[2]), 2)
        for regime in prior.regimes[2][1:]:
            self.assertGreaterEqual(regime['contradicted_sources'], 8)

    def test_slow_periodic_scale_keeps_cycle_model(self):
        times = np.arange(240.)
        values = .05*np.sin(2*np.pi*times/80)
        model = fit_pose_component(times, values, .002, 3)
        self.assertIsNotNone(model)
        self.assertEqual(model.mode, 'periodic')
        self.assertGreaterEqual(model.cycles, 2.5)
        self.assertLess(max(abs(model.predict(t)-v) for t, v in zip(times, values)), .01)


class MotionRefreshTests(unittest.TestCase):
    def test_eight_new_original_matches_refresh_existing_model(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = EvidenceCache()
            e = TemplateEngine(TemplateConfig(sift_backend='cpu'), cache, None,
                WarningLog(Path(folder)/'warnings.jsonl'), (100, 200))
            image = np.zeros((100, 200, 3), np.uint8)
            obs = e.observe(0, 0, image, Candidate(np.full((10, 20), .8, np.float32),
                (30, 40), (30, 40, 50, 50), .9, 'seed'), 1008, 'global')
            owner = e._provisional(obs, 'new_template')
            owner.poses = {f: Pose(f, translation(30, 40), .7, 'measured', f,
                                  (obs.id,), True) for f in range(16)}
            e._motion(owner)
            with (patch.object(e.matcher, 'extract', return_value=obs.feature_key),
                  patch.object(e, '_matches', return_value=[Match(translation(30, 40), .7, 20, .8, (obs.id,))])):
                for f in range(16, 24):
                    e._track(owner, f, f, image, backward=True)
            self.assertEqual(len(owner.motion.evidence), 24)
            self.assertTrue(owner.motion.predict(24)[2])
            cache.close()


if __name__ == '__main__':
    unittest.main()
