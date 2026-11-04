"""Verified frame-edge margins are distinct from unlocalized tracking failures."""
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.geometry import translation
from uav.template_mask.policy import SkipVideo, WarningLog
from uav.template_mask.records import Candidate, Pose


class FrameEdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.engine = TemplateEngine(TemplateConfig(sift_backend='cpu'), self.cache, None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'), (100, 160))
        self.image = np.zeros((100,160,3),np.uint8)

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def seed(self):
        e = self.engine
        obs = e.observe(0, 0, self.image, Candidate(np.full((10,20),.8,np.float32),
            (30,40), (30,40,50,50), .9, 'seed'), 1008, 'global')
        return e._provisional(obs, 'new_template')

    def setup_edge(self, edge, appearance=False):
        self.cache.close()
        self.cache = EvidenceCache()
        self.engine = TemplateEngine(TemplateConfig(sift_backend='cpu'), self.cache, None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'), (100,160))
        e = self.engine
        owner = self.seed()
        detected, frame = (13,12) if appearance else (12,13)
        e.source_ids = {f:f+f//4 for f in range(30)}
        direction = -1 if appearance else 1
        def xy(f):
            step = direction*20*(e.source_ids[f]-e.source_ids[detected])
            return dict(bottom=(40,97+step), top=(40,-7-step),
                        right=(157+step,40), left=(-17-step,40))[edge]
        measured = range(15,28) if appearance else range(11)
        owner.poses = {f:Pose(f,translation(*xy(f)),.3,'measured',e.source_ids[f],
                              presence=True) for f in measured}
        x,y = xy(detected)
        l,t,r,b = max(0,x),max(0,y),min(160,x+20),min(100,y+10)
        obs = e.observe(detected,e.source_ids[detected],self.image,
            Candidate(np.full((b-t,r-l),.8,np.float32),(l,t),(l,t,r,b),.9,'edge'),
            1008,'global')
        obs.boundary_instance_id = owner.id
        obs.association_reason = 'boundary_sam_candidate'
        owner.poses[detected] = Pose(detected,translation(x,y),25.,'uncertain',
                                     presence=True,reason='boundary_sam_candidate')
        owner.poses[frame] = Pose(frame,translation(*xy(frame)),80.,'uncertain',
                                  reason='boundary_interval_pose_pending')
        e.presence.coverage[owner.id] = {frame,detected}
        e.presence.edge_margins[(owner.id,frame)] = detected
        e.presence.checks[owner.id] = {frame:'not_detected'}
        return owner, frame, detected

    def test_all_four_exit_edges_keep_measured_pixels_but_no_offscreen_margin(self):
        for edge in ('bottom','top','left','right'):
            with self.subTest(edge=edge):
                owner,frame,detected = self.setup_edge(edge)
                measured = sum(p.status=='measured' for p in owner.poses.values())
                self.engine.presence.project(owner,frame)
                self.assertEqual(owner.poses[frame].reason,'boundary_margin_outside_frame')
                self.assertEqual(owner.poses[frame].status,'absent')
                self.assertEqual(sum(p.status=='measured' for p in owner.poses.values()),measured)
                self.assertTrue(self.engine.presence.report()['outside_frame_margins'])
                self.engine.instances.clear()
                self.engine.frame_observations.clear()
                self.engine.presence.coverage.clear()
                self.engine.presence.edge_margins.clear()
                self.engine.presence.checks.clear()

    def test_all_four_entrance_edges_handle_reverse_time(self):
        for edge in ('bottom','top','left','right'):
            with self.subTest(edge=edge):
                owner,frame,_ = self.setup_edge(edge,appearance=True)
                self.assertIsNotNone(self.engine.presence._outside_edge_margin(owner,frame))
                self.engine.instances.clear()
                self.engine.frame_observations.clear()
                self.engine.presence.edge_margins.clear()

    def test_unknown_or_positive_margin_remains_unresolved(self):
        owner,frame,_ = self.setup_edge('bottom')
        for state in (None,'sam_candidate','image_match'):
            self.engine.presence.checks[owner.id][frame]=state
            self.assertIsNone(self.engine.presence._outside_edge_margin(owner,frame))

    def test_new_local_sam_pixels_override_stale_global_negative(self):
        owner,frame,_ = self.setup_edge('bottom')
        owner.poses[frame].presence = True
        owner.poses[frame].reason = 'boundary_sam_candidate'
        self.assertIsNone(self.engine.presence._outside_edge_margin(owner,frame))

    def test_last_detected_edge_pixels_remain_in_output(self):
        owner,frame,detected = self.setup_edge('bottom')
        self.engine.presence.project(owner,frame)
        self.assertTrue(self.engine.render(detected)[99,45])
        self.assertFalse(self.engine.render(frame).any())

    def test_supported_edge_approach_uses_fitted_sigma_instead_of_search_sigma(self):
        owner,frame,detected = self.setup_edge('bottom')
        measured = sum(p.status=='measured' for p in owner.poses.values())
        self.engine.presence.project(owner,detected)
        pose = owner.poses[detected]
        self.assertEqual(pose.status,'predicted')
        self.assertEqual(pose.reason,'boundary_edge_motion_prediction')
        self.assertLess(pose.uncertainty,2.)
        self.assertAlmostEqual(pose.matrix[1,2],97.,places=5)
        self.assertEqual(sum(p.status=='measured' for p in owner.poses.values()),measured)
        self.assertTrue(self.engine.render(detected)[99,45])

    def test_weak_edge_motion_keeps_local_sam_pixels_without_full_template_prediction(self):
        owner,_,detected = self.setup_edge('bottom')
        for p in owner.poses.values():
            if p.status=='measured':
                p.matrix[0,2] += 25*np.sin(p.frame)
        self.engine.presence.project(owner,detected)
        self.assertEqual(owner.poses[detected].reason,'boundary_sam_candidate')
        self.assertTrue(self.engine.render(detected)[99,45])

    def test_explicit_disappearance_is_not_replaced_by_edge_prediction(self):
        owner,_,detected = self.setup_edge('bottom')
        pose=owner.poses[detected]
        pose.status,pose.presence,pose.reason='absent',False,'confirmed_disappearance'
        self.engine.presence.project(owner,detected)
        self.assertIs(owner.poses[detected],pose)
        self.assertEqual(pose.reason,'confirmed_disappearance')

    def test_offscreen_margin_without_edge_observation_is_silent(self):
        owner,frame,detected = self.setup_edge('bottom')
        self.engine.frame_observations[detected].clear()
        self.assertIsNone(self.engine.presence._outside_edge_margin(owner,frame))
        self.assertFalse(self.engine.render(frame).any())
        self.assertEqual(self.engine.warnings.rows, [])

    def test_uncertain_or_predicted_rows_cannot_certify_exit(self):
        owner,frame,_ = self.setup_edge('bottom')
        for pose in owner.poses.values():
            if pose.status=='measured':
                pose.status='predicted'
        self.assertIsNone(self.engine.presence._outside_edge_margin(owner,frame))

    def test_observed_strokes_still_intersecting_frame_are_not_dropped(self):
        owner,frame,detected = self.setup_edge('bottom')
        e = self.engine
        for p in owner.poses.values():
            if p.status=='measured':
                p.matrix[1,2] = 97+.5*(p.source_id-e.source_ids[detected])
        self.assertIsNone(e.presence._outside_edge_margin(owner,frame))

    def test_extra_frame_never_triggers_tracking_probing_or_template_evidence(self):
        owner, frame, detected = self.setup_edge('top', appearance=True)
        e = self.engine
        observations = set(e.bank.observations)
        sources = {r.source_id for r in e.bank.matching_references(owner.template_id)}
        e._motion(owner)
        measured_sources = {p.source_id for p in owner.motion.evidence}
        # Boundary acceptance and every later retry must avoid reading this
        # original frame for evidence, even if SIFT would have returned a fit.
        with (patch.object(e, '_track', side_effect=AssertionError('margin tracking')),
              patch.object(e.matcher, 'extract', side_effect=AssertionError('margin features'))):
            e.presence._accept_interval(owner, frame, frame,
                lambda f: self.fail('margin original image requested'))
            self.assertFalse(e.presence._probe(owner, frame,
                lambda f: self.fail('margin SAM original image requested')))
        with patch.object(e.matcher, 'extract', side_effect=AssertionError('margin features')):
            e._track(owner, frame, e.source_ids[frame], self.image, boundary=True)
            e._track(owner, frame, e.source_ids[frame], self.image, backward=True)
        e.presence.refresh()
        e._motion(owner)
        self.assertEqual(set(e.bank.observations), observations)
        self.assertEqual({r.source_id for r in e.bank.matching_references(owner.template_id)}, sources)
        self.assertEqual({p.source_id for p in owner.motion.evidence}, measured_sources)
        pose = owner.poses[frame]
        self.assertIsNone(pose.source_id)
        self.assertFalse(pose.presence)
        self.assertNotEqual(pose.status, 'measured')
        margin = e.presence.report()['output_only_margins'][0]
        self.assertFalse(margin['trajectory_evidence'])
        self.assertFalse(margin['template_evidence'])

    def output_margin_at(self, xy, sigma=.7, status='predicted'):
        owner = self.seed()
        e = self.engine
        owner.poses = {1: Pose(1, translation(*xy), sigma, status,
            reason='boundary_edge_motion_prediction')}
        e.presence.edge_margins[(owner.id, 1)] = 2
        return owner

    def test_output_margin_halo_enters_frame_from_all_four_edges(self):
        for xy, edge, interior in (
                ((40, -12), (0, 45), (10, 45)),
                ((40, 102), (99, 45), (89, 45)),
                ((-22, 40), (45, 0), (45, 10)),
                ((162, 40), (45, 159), (45, 149))):
            with self.subTest(xy=xy):
                owner = self.output_margin_at(xy)
                mask = self.engine.render(1)
                self.assertTrue(mask[edge])
                self.assertFalse(mask[interior])
                self.assertLess(mask.sum(), 400)
                self.assertFalse(owner.poses[1].presence)
                self.assertIsNone(owner.poses[1].source_id)
                self.engine.instances.clear()

    def test_localized_output_margin_may_remain_empty_without_skipping(self):
        owner = self.output_margin_at((40, -40))
        self.assertFalse(self.engine.render(1).any())
        self.assertEqual(self.engine.warnings.rows, [])
        # Without the output-only allowance, an unlocalizable ordinary pose
        # still fails. Both strong and weak optional predictions are silent.
        self.engine.presence.edge_margins.clear()
        with self.assertRaises(SkipVideo):
            self.engine.render(1)
        self.engine.presence.edge_margins[(owner.id, 1)] = 2
        owner.poses[1].status = 'uncertain'
        self.engine.warnings.rows.clear()
        self.assertFalse(self.engine.render(1).any())
        self.assertEqual(self.engine.warnings.rows, [])

    def test_optional_margin_ignores_failure_policy_and_area(self):
        owner = self.output_margin_at((40, -12))
        e = self.engine
        for policy in ('ignore', 'skip', 'conservative'):
            e.config = replace(e.config, uncertain_mask_policy=policy)
            self.assertTrue(e.render(1).any())
            self.assertEqual(e.warnings.rows, [])
        e.config = replace(e.config, uncertain_mask_policy='conservative')
        owner.poses[1].matrix = translation(40, -6)
        self.assertTrue(e.render(1).any())
        self.assertEqual(e.warnings.rows, [])

    def test_no_prediction_uses_nearest_real_observation_without_evidence_or_warning(self):
        owner = self.seed()
        e = self.engine
        obs = e.observe(8, 8, self.image, Candidate(np.full((10,20), .8, np.float32),
            (90,60), (90,60,110,70), .9, 'nearest'), 1008, 'global')
        obs.boundary_instance_id = owner.id
        owner.poses.clear()
        e.presence.edge_margins[(owner.id,9)] = 8
        observations = set(e.bank.observations)
        revision = e.bank.revision
        with patch.object(e, '_prediction', return_value=None):
            mask = e.render(9)
        self.assertTrue(mask[65,100])
        self.assertFalse(mask[45,40])
        self.assertEqual(e.render_metrics[9]['output_margins'][0]['method'], 'nearest_observation')
        self.assertFalse(owner.poses)
        self.assertEqual(set(e.bank.observations), observations)
        self.assertEqual(e.bank.revision, revision)
        self.assertEqual(e.warnings.rows, [])

    def test_no_prediction_or_observation_is_empty_and_silent(self):
        owner = self.seed()
        e = self.engine
        owner.poses.clear()
        e.bank.observations.clear()
        e.presence.edge_margins[(owner.id,1)] = 2
        with patch.object(e, '_prediction', return_value=None):
            self.assertFalse(e.render(1).any())
        self.assertEqual(e.render_metrics[1]['output_margins'][0]['method'], 'empty')
        self.assertEqual(e.warnings.rows, [])

    def test_weak_prediction_takes_precedence_without_uncertainty_growth(self):
        owner = self.output_margin_at((80, 60), sigma=40.57, status='uncertain')
        e = self.engine
        mask = e.render(1)
        self.assertTrue(mask[65,90])
        self.assertFalse(mask[45,40])
        self.assertLess(mask.sum(), 1200)
        self.assertEqual(e.render_metrics[1]['output_margins'][0]['method'], 'prediction')
        self.assertEqual(e.warnings.rows, [])
        self.assertFalse(owner.poses[1].presence)

    def test_optional_margin_cannot_invalidate_other_reliable_coverage(self):
        owner = self.output_margin_at((80, 60))
        e = self.engine
        obs = e.observe(0, 0, self.image, Candidate(np.full((10,20), .8, np.float32),
            (30,40), (30,40,50,50), .9, 'other'), 1008, 'global')
        other = e._provisional(obs, 'new_template')
        other.poses[1] = other.poses[0]
        owner.poses[1].matrix = translation(30,40)
        overlap = e.render(1)
        self.assertTrue(overlap[45,40])
        owner.poses[1].matrix = translation(80,60)
        result = e.render(1)
        # Area thresholds do not omit output; both contributions survive.
        self.assertTrue(result[45,40])
        self.assertTrue(result[65,90])
        self.assertTrue(all(row['instance'] != owner.id for row in e.warnings.rows))
        self.assertFalse(any(row['action'].startswith('skip') for row in e.warnings.rows))

    def test_video12_frame163_weak_offscreen_margin_is_silent(self):
        owner = self.output_margin_at((581.615,1281.108), sigma=40.571, status='uncertain')
        e = self.engine
        e.shape = (1280,714)
        owner.poses[163] = owner.poses.pop(1)
        e.presence.edge_margins = {(owner.id,163):162}
        before = e.bank.revision
        mask = e.render(163)
        self.assertTrue(mask[1279].any())
        self.assertLess(mask.sum(), 300)
        self.assertEqual(e.render_metrics[163]['output_margins'][0]['method'], 'prediction')
        self.assertEqual(e.warnings.rows, [])
        self.assertEqual(e.bank.revision, before)

    def test_video12_numeric_pose_history_handles_original_source_timestamps(self):
        owner=self.seed()
        e=self.engine
        xy=[(397,1116),(401.115,1124.570),(404.964,1133.142),(409.045,1142.020),
            (417.486,1159.112),(421.313,1167.751),(425.171,1176.312),(429.230,1185.199),
            (437.372,1202.594),(441.575,1211.388),(445.730,1219.687),(449.877,1228.238),
            (458.159,1245.715),(462.293,1254.531)]
        e.shape=(1280,714)
        e.source_ids={f:697+int(f*30/24) for f in range(17)}
        template=e.bank.templates[owner.template_id]
        template.box=(0,0,317,52)
        template.anchor=(158.5,26.)
        owner.poses={f:Pose(f,translation(x,y),.9,'measured',e.source_ids[f],presence=True)
                     for f,(x,y) in enumerate(xy)}
        obs=e.observe(15,e.source_ids[15],np.zeros((1280,714,3),np.uint8),
            Candidate(np.full((10,202),.8,np.float32),(478,1270),(478,1270,680,1280),.9,'edge'),1008,'global')
        obs.boundary_instance_id=owner.id
        owner.poses[16]=Pose(16,translation(474.695,1280.977),77.7,'uncertain',
                             reason='boundary_interval_pose_pending')
        e.presence.coverage[owner.id]={16}
        e.presence.edge_margins[(owner.id,16)]=15
        e.presence.checks[owner.id]={16:'not_detected'}
        e.presence.project(owner,16)
        self.assertEqual(owner.poses[16].reason,'boundary_margin_outside_frame')
        self.assertFalse(e.render(16).any())
        e.presence.refresh()
        self.assertEqual(owner.poses[16].status,'absent')
        self.assertFalse(any(w['action'].startswith('skip') for w in e.warnings.rows))
        e.presence.coverage[owner.id].update((14,15))
        for f in (14,15):
            owner.poses[f]=Pose(f,translation(470,1260),77.,'uncertain',
                presence=f==15,reason='boundary_sam_candidate' if f==15 else 'boundary_interval_pose_pending')
            e.presence.project(owner,f)
            self.assertEqual(owner.poses[f].reason,'boundary_edge_motion_prediction')
            self.assertLess(owner.poses[f].uncertainty,2.)
        self.assertAlmostEqual(owner.poses[14].matrix[1,2],1263.,delta=1.)
        e.render(14)
        sigma_rows=e.render_metrics[14]['uncertainty']
        self.assertTrue(sigma_rows)
        self.assertLessEqual(max(x['extra_radius'] for x in sigma_rows),6)


if __name__=='__main__':
    unittest.main()
