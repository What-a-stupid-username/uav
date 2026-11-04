"""Past/future relative motion distinguishes fragments from crossings."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.features import Match
from uav.template_mask.geometry import translation, discrepancy
from uav.template_mask.interactions import relative_motion
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Pose


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        cfg = TemplateConfig(sift_backend='cpu')
        self.engine = TemplateEngine(cfg, self.cache, None, WarningLog(Path(self.temp.name)/'warnings.jsonl'), (200, 500))
        self.image = np.zeros((200, 500, 3), np.uint8)
        self.owners = []
        for i, x in enumerate((30, 260)):
            c = Candidate(np.full((20, 30), .8, np.float32), (x, 70), (x, 70, x+30, 90), .9, str(i))
            obs = self.engine.observe(0, 0, self.image, c, 1008, 'global')
            self.owners.append(self.engine._provisional(obs, 'new_template'))

    def tearDown(self):
        self.cache.close(); self.temp.cleanup()

    def histories(self, independent=False, until=24):
        a, b = self.owners
        # Both fragments follow one accelerating, rotating, scaled curve;
        # independent=True adds a second relative path through that curve.
        for f in range(until+1):
            theta = .01*f*f
            scale = 1+.01*f
            co, si = np.cos(theta)*scale, np.sin(theta)*scale
            m = np.array([[co, -si, 60+f*1.7], [si, co, 70+.06*f*f], [0, 0, 1.]])
            offset = 90-6*f if independent else 90
            for owner, pose in ((a, m), (b, m@translation(offset, 0))):
                ref = owner.observations[0]
                owner.poses[f] = Pose(f, pose, .3, 'measured', f, (ref,), True)

    def whole(self, frame, name='whole'):
        c = Candidate(np.full((100, 240), .8, np.float32), (20, 40), (20, 40, 260, 140), .9, name)
        return self.engine.observe(frame, frame, self.image, c, 1008, 'global')

    def test_distant_fragments_on_accelerating_rotating_curve_keep_rigid_layout(self):
        self.histories()
        evidence = relative_motion(self.engine, *self.owners, 12)
        self.assertEqual(evidence['state'], 'rigid_layout')
        np.testing.assert_allclose(evidence['layout'], translation(90, 0), atol=1e-8)
        self.assertEqual(len(evidence['past_sources']), 12)
        self.assertEqual(len(evidence['future_sources']), 12)
        self.assertFalse(self.engine._independent_instances(*self.owners, 12))

    def test_relative_curve_through_other_object_is_independent_motion(self):
        self.histories(independent=True)
        evidence = relative_motion(self.engine, *self.owners, 12)
        self.assertEqual(evidence['state'], 'independent_motion')
        self.assertTrue(self.engine._independent_instances(*self.owners, 12))

    def test_future_revises_pending_record_without_replacing_identity(self):
        self.histories(independent=True, until=12)
        obs = self.whole(12)
        self.engine.interactions.hold(obs, [tuple(self.owners)])
        record = self.engine.interactions.records[obs.id]
        self.assertEqual(record['state'], 'unresolved')
        self.histories(independent=True)
        self.engine.interactions.resolve()
        self.assertEqual(record['state'], 'crossing')
        self.assertEqual(obs.crossing_instances, (0, 1))
        self.assertEqual(len(self.engine.instances), 2)
        self.assertEqual(len(self.engine.bank.templates), 2)
        self.assertFalse(self.engine.bank.links)

    def test_prediction_and_duplicate_source_do_not_create_relative_evidence(self):
        self.histories(independent=True)
        for owner in self.owners:
            for f, pose in owner.poses.items():
                if f > 1:
                    pose.status = 'predicted'
        self.assertEqual(relative_motion(self.engine, *self.owners, 12)['state'], 'unresolved')
        for owner in self.owners:
            for pose in owner.poses.values():
                pose.status = 'measured'; pose.source_id = 0
        evidence = relative_motion(self.engine, *self.owners, 12)
        self.assertEqual(evidence['state'], 'unresolved')
        self.assertEqual(evidence['sources'], [0])

    def test_single_bad_original_cannot_prove_independent_objects(self):
        self.histories()
        self.owners[1].poses[8].matrix = translation(400, 10)
        self.assertEqual(relative_motion(self.engine, *self.owners, 12)['state'], 'unresolved')

    def test_rigid_trajectory_alone_does_not_merge_templates(self):
        self.histories()
        obs = self.whole(12)
        with patch.object(self.engine, '_matches', return_value=[]):
            self.engine.interactions.hold(obs, [tuple(self.owners)])
        self.assertEqual(self.engine.interactions.records[obs.id]['state'], 'rigid_fragments')
        self.assertFalse(self.engine.bank.links)
        self.assertTrue(obs.ambiguous)

    def test_first_pass_preserves_both_paths_even_with_registered_rigid_wholes(self):
        self.histories()
        obs = self.whole(12)
        with patch.object(self.engine, '_matches', side_effect=AssertionError('wait for completed forward tracks')):
            self.engine.interactions.hold(obs, [tuple(self.owners)])
        self.assertFalse(self.engine.bank.links)
        self.assertEqual(len(self.engine.active_instances()), 2)

    def test_crossing_support_does_not_discard_distinct_image_measurements(self):
        self.histories(independent=True)
        e = self.engine
        # Move the two different-template shapes to overlapping supports while
        # keeping their reference histories independently moving.
        for owner in self.owners:
            owner.poses[12].matrix = translation(100+owner.id*5, 70)
        e._crossings(12)
        self.assertTrue(all(i.poses[12].status == 'measured' for i in self.owners))
        self.assertTrue(all(i.poses[12].presence for i in self.owners))
        self.assertTrue(all(not i.ambiguous_frames for i in self.owners))

    def test_two_wholes_registered_to_both_parts_and_future_rigid_layout_merge(self):
        self.histories()
        e = self.engine
        e.stage = 'second_pass'
        def matches(template, key):
            obs = next(o for o in e.bank.observations.values() if o.feature_key == key)
            pose = self.owners[template.id].poses[obs.frame]
            return [Match(pose.matrix.copy(), .3, 10, .8, pose.references)]
        with patch.object(e, '_matches', side_effect=matches), patch.object(e, '_shape_agreement', return_value=True):
            first, second = self.whole(10, 'whole-a'), self.whole(12, 'whole-b')
            e.interactions.hold(first, [tuple(self.owners)])
            self.assertFalse(e.bank.links[(0, 1)].active)
            e.interactions.hold(second, [tuple(self.owners)])
        self.assertTrue(e.bank.links[(0, 1)].active)
        np.testing.assert_allclose(e.bank.links[(0, 1)].b_to_a, translation(90, 0), atol=1e-8)
        self.assertTrue(first.ambiguous); self.assertTrue(second.ambiguous)

    def test_future_relative_motion_revokes_earlier_rigid_bridge(self):
        self.histories()
        e = self.engine
        for source in (10, 12):
            e.bank.propose_link(0, 1, translation(90, 0), source, 'trajectory_validated_bridge')
        obs = self.whole(12)
        with patch.object(e, '_matches', return_value=[]):
            e.interactions.hold(obs, [tuple(self.owners)])
        self.assertTrue(e.bank.links[(0, 1)].active)
        for f in range(16, 25):
            self.owners[1].poses[f].matrix = self.owners[0].poses[f].matrix@translation(90+10*(f-15), 0)
        e.interactions.resolve()
        self.assertFalse(e.bank.links[(0, 1)].active)
        self.assertTrue(e.bank.links[(0, 1)].revoked)
        self.assertEqual(e.interactions.records[obs.id]['state'], 'crossing')

    def test_batched_relative_geometry_matches_scalar_rule_with_all_originals(self):
        for independent in (False, True):
            self.histories(independent=independent, until=96)
            for outliers in ((), (8,), (0,), (0, 1), (8, 23, 61)):
                self.histories(independent=independent, until=96)
                a, b = self.owners
                for f in outliers:
                    b.poses[f].matrix = translation(400+f, 10)
                result = relative_motion(self.engine, a, b, 48)
                rows = [(a.poses[f], b.poses[f]) for f in range(97)]
                layouts = [np.linalg.inv(pa.matrix)@pb.matrix for pa, pb in rows]
                box = self.engine.bank.templates[b.template_id].box
                def errors(layout):
                    return np.array([discrepancy(pa.matrix@layout, pb.matrix, box)
                        /max(1e-6,4*(pa.uncertainty+pb.uncertainty)) for pa,pb in rows])
                choices = np.unique(np.linspace(0,96,17).astype(int))
                layout = min((layouts[i] for i in choices), key=lambda m:float(np.median(errors(m))))
                residual = errors(layout)
                expected = 'rigid_layout' if np.all(residual <= 1) else 'unresolved'
                if expected != 'rigid_layout':
                    for split in range(2,len(rows)-1):
                        early, late = rows[:split], rows[split:]
                        corroborated = all(sum(discrepancy(pa.matrix@reference,pb.matrix,box)
                            > 8*(pa.uncertainty+pb.uncertainty) for pa,pb in late) >= 2
                            for reference in layouts[:2])
                        pa0,pb0=early[0];pa1,pb1=early[1]
                        agree = discrepancy(pa1.matrix@layouts[0],pb1.matrix,box) <= \
                            8*(pa0.uncertainty+pb0.uncertainty+pa1.uncertainty+pb1.uncertainty)
                        if corroborated and agree:
                            expected='independent_motion'
                            break
                self.assertEqual(result['state'],expected)
                self.assertEqual(result['sources'],list(range(97)))
                self.assertEqual(result['incompatible_sources'],np.flatnonzero(residual>1).tolist())
                self.assertAlmostEqual(result['max_normalized_residual'],float(residual.max()),places=7)
                np.testing.assert_allclose(result['layout'],layout,atol=1e-9)

    def test_nonoverlapping_objects_do_not_need_relative_history_for_crossings(self):
        with patch.object(self.engine, '_independent_instances',
                          side_effect=AssertionError('no crossing to investigate')):
            self.engine._crossings(0)


if __name__ == '__main__':
    unittest.main()
