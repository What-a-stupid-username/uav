"""Image-corroborated turns must survive an overconfident motion prior."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.features import Match
from uav.template_mask.geometry import translation
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Instance, Pose


class TurnTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.image = np.zeros((240, 500, 3), np.uint8)
        self.engine = TemplateEngine(TemplateConfig(sift_backend='cpu'), self.cache, None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'), self.image.shape[:2])
        e = self.engine
        c = Candidate(np.full((40, 180), .8, np.float32), (50, 60),
                      (50, 60, 230, 100), .9, 'a')
        first = e.observe(0, 10, self.image, c, 1008, 'global')
        self.owner = e._provisional(first, 'new_template')
        second = e.observe(1, 12, self.image, c, 1008, 'global')
        second.pose, second.template_id, second.reliable = translation(50, 60), 0, True
        e._assign(self.owner, second)
        self.refs = (first.id, second.id)
        self.target = self.cache.key('test_target')
        self.cache.put(self.target, xy=np.empty((0, 2)))

    def tearDown(self):
        self.cache.close(); self.temp.cleanup()

    def track(self, fits):
        e = self.engine
        with patch.object(e, '_prediction', return_value=(translation(50, 60), .5, True)), \
             patch.object(e.matcher, 'extract', return_value=self.target), \
             patch.object(e, '_matches', return_value=fits):
            e._track(self.owner, 2, 15, self.image)
        return self.owner.poses[2]

    def test_independent_originals_confirm_a_real_turn_outside_prior_gate(self):
        matrix = translation(60, 30)
        pose = self.track([Match(matrix, 1., 70, .7, self.refs)])
        self.assertEqual(pose.status, 'measured')
        np.testing.assert_array_equal(pose.matrix, matrix)
        self.assertEqual(pose.references, self.refs)

    def test_one_reference_cannot_overrule_prediction(self):
        pose = self.track([Match(translation(60, 30), 1., 70, .7, self.refs[:1])])
        self.assertEqual(pose.status, 'uncertain')

    def test_repeated_reference_and_current_source_are_not_corroboration(self):
        for refs in ((self.refs[0], self.refs[0]), self.refs):
            self.engine.bank.observations[self.refs[1]].source_id = 15
            pose = self.track([Match(translation(60, 30), 1., 70, .7, refs)])
            self.assertEqual(pose.status, 'uncertain')

    def test_remote_reappearance_does_not_take_over_current_instance(self):
        pose = self.track([Match(translation(300, 60), 1., 70, .7, self.refs)])
        self.assertEqual(pose.status, 'uncertain')

    def test_sparse_shared_word_cannot_overrule_prediction(self):
        pose = self.track([Match(translation(60, 30), 1., 70, .08, self.refs)])
        self.assertEqual(pose.status, 'uncertain')

    def test_other_instance_at_match_location_keeps_assignment_unresolved(self):
        e = self.engine
        e.instances[1] = Instance(1, 0, 0, poses={2:Pose(2, translation(60, 30), 1., 'measured', 15, self.refs, True)})
        pose = self.track([Match(translation(60, 30), 1., 70, .7, self.refs)])
        self.assertEqual(pose.status, 'uncertain')

    def test_invalid_nearest_fit_does_not_hide_a_corroborated_turn(self):
        matrix = translation(60, 30)
        pose = self.track([Match(translation(58, 51), .5, 10, .1, self.refs[:1]),
                           Match(matrix, 1., 70, .7, self.refs)])
        self.assertEqual(pose.status, 'measured')
        np.testing.assert_array_equal(pose.matrix, matrix)


if __name__ == '__main__':
    unittest.main()
