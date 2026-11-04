"""A gap inside a straight leg retains strong evidence despite other turns."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.policy import WarningLog
from uav.template_mask.geometry import discrepancy, translation
from uav.template_mask.gaps import trajectory_bridge
from uav.template_mask.motion import MotionPrior
from uav.template_mask.records import Candidate, Pose, Template


class StraightSegmentTests(unittest.TestCase):
    template = Template(0, (0, 0, 100, 25), (50, 12))

    def poses(self, motion, frames):
        return [Pose(f, translation(*motion(f)), .3, 'measured', f, presence=True)
                for f in frames]

    def bridge(self, motion, first, last, count=160):
        left = self.poses(motion, range(first))
        right = self.poses(motion, range(last, count))
        return trajectory_bridge(left, right, self.template, range(first-1, last+1))

    def assert_bridge(self, bridge, motion):
        self.assertIsNotNone(bridge)
        for frame, (matrix, sigma) in bridge[0].items():
            self.assertLessEqual(discrepancy(matrix, translation(*motion(frame)),
                                            self.template.box), 3*sigma)

    def test_gap_in_straight_leg_between_other_turns_is_strong(self):
        def motion(f):
            if f < 32:
                return 32, f-32
            if f < 120:
                return f, 0
            return 120, f-120
        self.assert_bridge(self.bridge(motion, 64, 72), motion)

    def test_later_corner_does_not_weaken_gap_inside_current_straight_leg(self):
        motion = lambda f: (f, 0) if f < 56 else (56, f-56)
        self.assert_bridge(self.bridge(motion, 40, 45), motion)

    def test_earlier_corner_does_not_weaken_gap_in_last_straight_leg(self):
        motion = lambda f: (f, 0) if f < 100 else (100, f-100)
        self.assert_bridge(self.bridge(motion, 112, 117), motion)

    def test_gap_across_corner_remains_weak(self):
        motion = lambda f: (f, 0) if f < 56 else (56, f-56)
        self.assertIsNone(self.bridge(motion, 52, 60))

    def test_partial_detection_on_early_short_leg_has_independent_continuity(self):
        motion = lambda f: (10*f, 0) if f < 12 else (120, 10*(f-12))
        prior = MotionPrior(self.poses(motion, range(100)),
                            self.template.box, self.template.anchor)
        evidence = prior.continuity(4, translation(*motion(4)), .3, 4)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence['components'][:2], ['linear', 'static'])

    def test_short_middle_leg_has_same_independent_continuity_as_long_line(self):
        def motion(f):
            if f < 40:
                return 8*f, 0
            if f < 52:
                return 320, 8*(f-40)
            return 320+8*(f-52), 96
        prior = MotionPrior(self.poses(motion, range(100)),
                            self.template.box, self.template.anchor)
        evidence = prior.continuity(46, translation(*motion(46)), .3, 46)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence['components'][:2], ['static', 'linear'])
        self.assertGreaterEqual(evidence['support_start'], 40)
        self.assertLess(evidence['support_end'], 52)


class StraightSegmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.image = np.zeros((300, 900, 3), np.uint8)
        self.engine = TemplateEngine(TemplateConfig(sift_backend='cpu'), self.cache, None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'), self.image.shape[:2])

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def owner(self, frame, motion, frames):
        x, y = map(int, motion(frame))
        obs = self.engine.observe(frame, frame, self.image,
            Candidate(np.full((20, 60), .8, np.float32), (x, y),
                      (x, y, x+60, y+20), .9, str(frame)), 1008, 'global')
        template = self.engine.bank.create(obs)
        instance = self.engine._new_instance(template, obs)
        instance.poses = {f: Pose(f, translation(*motion(f)), .3, 'measured', f,
                                (obs.id,), True) for f in frames}
        return instance

    @staticmethod
    def motion(f):
        if f < 32:
            return 62, 62+f
        if f < 128:
            return 30+f, 94
        return 158, 94+f-128

    def test_sparse_sam_fragment_inside_straight_leg_keeps_identity(self):
        owner = self.owner(0, self.motion, range(160))
        self.engine._motion(owner)
        x, y = map(int, self.motion(64))
        fragment = self.engine.observe(64, 64, self.image,
            Candidate(np.full((20, 12), .8, np.float32), (x, y),
                      (x, y, x+12, y+20), .9, 'fragment'), 1008, 'global')
        self.assertIs(self.engine.associate(fragment), owner)
        self.assertEqual(fragment.association_reason, 'registered_fragment_and_validated_trajectory')
        self.assertEqual(fragment.trajectory_evidence['components'][:2], ['linear', 'static'])
        self.assertEqual(len(self.engine.instances), 1)

    def test_second_pass_connects_identity_inside_leg_despite_other_corners(self):
        left = self.owner(0, self.motion, range(80))
        right = self.owner(88, self.motion, range(88, 160))
        before = [len(i.poses) for i in (left, right)]
        self.engine.reconciler.trajectory_identities()
        self.assertEqual(self.engine.bank.identity_members(left.template_id),
                         {left.template_id, right.template_id})
        self.assertEqual(right.continuation_of, left.id)
        self.assertEqual([len(i.poses) for i in (left, right)], before)

    def test_gap_fill_on_straight_leg_keeps_predictions_out_of_evidence(self):
        owner = self.owner(0, self.motion, [f for f in range(160) if not 80 <= f < 88])
        self.engine.source_ids = {f:f for f in range(160)}
        self.engine.gaps.resolve(lambda f: self.image)
        self.assertEqual(self.engine.stats['trajectory_gap_predictions'], 8)
        for f in range(80, 88):
            pose = owner.poses[f]
            self.assertEqual(pose.status, 'predicted')
            self.assertFalse(pose.presence)
            self.assertLess(discrepancy(pose.matrix, translation(*self.motion(f)),
                                        self.engine.bank.templates[0].box), 1.)
        self.engine._motion(owner)
        self.assertEqual(len(owner.motion.evidence), 152)


if __name__ == '__main__':
    unittest.main()
