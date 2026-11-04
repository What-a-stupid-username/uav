"""Partial discovery ownership is separate from template/geometry evidence."""
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
from uav.template_mask.geometry import warp
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Instance, Pose


class FragmentContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.e = TemplateEngine(TemplateConfig(sift_backend='cpu'), self.cache, None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'), (200,400))
        self.image = np.zeros((200,400,3),np.uint8)
        self.probability = np.zeros((60,120),np.float32)
        self.probability[5:20,5:110] = .8
        self.probability[40:55,5:110] = .8
        seed = self.observation(0,self.probability,'seed')
        self.owner = self.e._provisional(seed,'new_template')
        self.seed = seed
        self.owner.poses = {f:Pose(f,translation(80,50),.7,'measured',f,(seed.id,),True)
                            for f in range(24)}
        self.e._motion(self.owner)

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def observation(self, frame, probability, label, origin=(80,50)):
        x,y=origin;h,w=probability.shape
        return self.e.observe(frame,frame,self.image,Candidate(probability,origin,
            (x,y,x+w,y+h),.9,label),1008,'global')

    def no_matches(self):
        return patch.object(self.e,'_matches',return_value=[])

    def test_same_size_box_with_missing_line_uses_visible_strokes(self):
        p=self.probability.copy();p[40:]=0
        obs=self.observation(23,p,'partial')
        original=self.owner.poses[23]
        with self.no_matches():
            self.assertIs(self.e.associate(obs),self.owner)
        self.assertTrue(obs.reliable)
        self.assertIs(self.owner.poses[23],original)
        self.assertEqual(len(self.e.instances),1)
        self.assertEqual(self.e.fragment_associations[-1]['outcome'],'unique')

    def test_predicted_pose_can_own_pixels_without_template_or_motion_votes(self):
        p=self.probability.copy();p[40:]=0
        obs=self.observation(24,p,'partial')
        self.owner.poses[24]=Pose(24,translation(80,50),50.,'uncertain')
        revision=self.e.bank.revision
        references=list(self.e.bank.templates[0].references)
        with self.no_matches():
            self.assertIs(self.e.associate(obs),self.owner)
        self.assertFalse(obs.reliable)
        self.assertEqual(self.e.bank.revision,revision)
        self.assertEqual(self.e.bank.templates[0].references,references)
        self.assertNotIn(obs.id,self.e.bank.templates[0].observations)
        pose=self.owner.poses[24]
        self.assertEqual(pose.status,'predicted')
        self.assertIsNone(pose.source_id)
        self.assertFalse(pose.presence)
        self.e._motion(self.owner)
        self.assertEqual(len(self.owner.motion.evidence),24)
        self.assertTrue(self.e.render(24)[60,100])

    def fine_fusion(self):
        mask = (self.probability > self.e.config.pixel_threshold).astype(np.float32)
        return dict(mean=self.probability, mask=mask, has_support=np.array(bool(mask.any())),
                    origin=np.zeros(2), sources=np.array([0,8]))

    def test_owned_observation_uses_fine_template_without_conservative_halo(self):
        p=self.probability.copy();p[40:]=0
        # Extra local pixels must not reintroduce a coarse union after a fine
        # shared template is available, even if they could not teach that template.
        p[22:35,5:30]=.8
        obs=self.observation(24,p,'owned')
        obs.instance_id,obs.template_id=0,0
        obs.ownership_confirmed=True
        obs.pose=translation(80,50)
        obs.association_reason='local_strokes_at_validated_trajectory'
        obs.trajectory_evidence=dict(coordinate_registration='predicted')
        self.owner.poses[24]=Pose(24,translation(80,58),80.,'predicted',
            reason='local_strokes_at_validated_trajectory')
        revision=self.e.bank.revision
        before=len(self.owner.motion.evidence)
        with patch.object(self.e.bank,'output_fusion',return_value=(self.fine_fusion(),False)):
            mask=self.e.render(24)
        raw=warp(self.probability>.23,translation(80,50),self.e.shape,nearest=True)>0
        expected=self.e.renderer.dilate(raw)
        np.testing.assert_array_equal(mask,expected)
        metrics=self.e.render_metrics[24]
        self.assertEqual(metrics['uncertainty'],[])
        self.assertEqual(metrics['conservative_pixels'],0)
        self.assertNotIn('coarse_fallback',metrics)
        self.assertEqual(metrics['dilation_radii'],[4])
        self.assertEqual(metrics['observed_presence'][0]['registration'],'predicted')
        self.assertEqual(self.e.warnings.rows,[])
        self.assertEqual(self.e.bank.revision,revision)
        self.e._motion(self.owner)
        self.assertEqual(len(self.owner.motion.evidence),before)

    def test_owned_observation_does_not_depend_on_uncertain_output_policy(self):
        from dataclasses import replace
        obs=self.observation(24,self.probability,'owned')
        obs.instance_id,obs.template_id,obs.ownership_confirmed=0,0,True
        obs.pose=translation(80,50)
        self.owner.poses[24]=Pose(24,translation(80,50),80.,'uncertain')
        for policy in ('conservative','skip','ignore'):
            self.e.config=replace(self.e.config,uncertain_mask_policy=policy)
            with patch.object(self.e.bank,'output_fusion',return_value=(self.fine_fusion(),False)):
                self.assertTrue(self.e.render(24)[60,100])
        self.assertEqual(self.e.warnings.rows,[])

    def test_missing_observation_keeps_prediction_coverage_explicit(self):
        self.owner.poses[24]=Pose(24,translation(80,50),2.,'predicted',reason='gap')
        with patch.object(self.e.bank,'output_fusion',return_value=(self.fine_fusion(),False)):
            self.e.render(24)
        metrics=self.e.render_metrics[24]
        self.assertEqual(metrics['uncertainty'][0]['extra_radius'],6)
        self.assertGreater(metrics['conservative_pixels'],0)
        self.assertNotIn('observed_presence',metrics)

    def test_ambiguous_ownership_cannot_bypass_uncertainty_policy(self):
        obs=self.observation(24,self.probability,'ambiguous')
        obs.instance_id,obs.template_id,obs.ownership_confirmed=0,0,True
        obs.pose=translation(80,50)
        obs.ambiguous=True
        self.owner.poses[24]=Pose(24,translation(80,50),2.,'predicted',reason='gap')
        with patch.object(self.e.bank,'output_fusion',return_value=(self.fine_fusion(),False)):
            self.e.render(24)
        self.assertNotIn('observed_presence',self.e.render_metrics[24])
        self.assertTrue(self.e.render_metrics[24]['uncertainty'])

    def test_low_feature_coverage_plus_independent_motion_reuses_measured_geometry(self):
        p=self.probability.copy();p[40:]=0;p[5:20,10:15]=0
        p[40:55,5:25]=.8
        # Add modest detection noise, so direct visible-overlap alone fails.
        p[22:35,5:45]=.8
        obs=self.observation(23,p,'noisy')
        before=self.e.bank.revision
        match=Match(translation(80,50),.7,8,.04,(self.seed.id,))
        with patch.object(self.e,'_matches',return_value=[match]):
            self.assertIs(self.e.associate(obs),self.owner)
        self.assertFalse(obs.reliable)
        self.assertEqual(self.e.bank.revision,before)
        self.assertEqual(len(self.e.instances),1)

    def test_actual_image_fit_at_prediction_can_add_real_pose_without_template_merge(self):
        p=self.probability.copy();p[40:]=0
        obs=self.observation(24,p,'matched')
        match=Match(translation(80,50),.7,8,.04,(self.seed.id,))
        with patch.object(self.e,'_matches',return_value=[match]):
            self.assertIs(self.e.associate(obs),self.owner)
        self.assertEqual(self.owner.poses[24].status,'measured')
        self.assertEqual(self.owner.poses[24].source_id,24)
        self.assertTrue(obs.reliable)
        self.assertFalse(self.e.bank.links)

    def test_unexplained_strokes_are_not_attached_by_location_alone(self):
        p=np.zeros_like(self.probability);p[24:36,10:100]=.8
        obs=self.observation(23,p,'different')
        with self.no_matches():
            self.assertIsNone(self.e._registered_fragment(obs))
        self.assertIsNone(obs.instance_id)

    def test_current_measured_fragment_cannot_teach_a_larger_mixed_candidate(self):
        p=np.zeros((110,170),np.float32)
        p[:60,:120]=self.probability
        p[65:105,5:165]=.8
        obs=self.observation(23,p,'larger-whole')
        before=list(self.e.bank.templates[0].references)
        with self.no_matches():
            owner=self.e.associate(obs)
        self.assertNotEqual(owner.template_id,0)
        self.assertEqual(self.e.bank.templates[0].references,before)
        self.assertNotIn(obs.id,self.e.bank.templates[0].observations)

    def test_cropped_fragment_does_not_look_like_an_equivalent_whole(self):
        from uav.template_mask.features import Match
        # Crop contains just the first line; the other line is still visible
        # elsewhere in the original frame and must count as unexplained.
        obs=self.observation(24,self.probability[:22],'cropped-line')
        evidence=self.e._shape_evidence(self.e.bank.templates[0],
            Match(translation(80,50),.7,12,.8,(self.seed.id,)),obs)
        self.assertGreater(evidence[0],.95)
        self.assertLess(evidence[1],.6)

    def test_remote_partial_image_match_does_not_spawn_a_full_template_instance(self):
        p=self.probability[:22].copy()
        obs=self.observation(24,p,'remote',origin=(240,120))
        match=Match(translation(240,120),.7,12,.8,(self.seed.id,'other'))
        with patch.object(self.e,'_matches',return_value=[match]):
            owner=self.e.associate(obs)
        self.assertNotEqual(owner.template_id,0)
        self.assertNotIn(obs.id,self.e.bank.templates[0].observations)

    def test_noisy_current_fit_does_not_block_independent_static_registration(self):
        obs=self.observation(23,self.probability,'noisy_pose')
        original=Pose(23,translation(80,58),.7,'measured',23,(self.seed.id,),True)
        self.owner.poses[23]=original
        self.e._motion(self.owner)
        revision=self.e.bank.revision
        with self.no_matches():
            self.assertIs(self.e.associate(obs),self.owner)
        self.assertFalse(obs.reliable)
        self.assertIs(self.owner.poses[23],original)
        self.assertEqual(self.e.bank.revision,revision)
        self.assertEqual(obs.trajectory_evidence['coordinate_registration'],'predicted')

    def test_candidate_segment_is_validated_in_its_own_support_coordinates(self):
        prior=self.owner.motion
        static=next(s for s in prior.segments if all(m is not None for m in s[0]))
        # A different larger window may have undetermined components. It must
        # not be the only gate for this independently validated static segment.
        with patch.object(prior,'_window',return_value=([None]*4,0,23)):
            evidence=prior.continuity(24,translation(80,50),.7,segment=static)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence['components'],['static']*4)

    def test_missing_current_match_still_checks_validated_local_segments(self):
        obs = self.observation(24, self.probability, 'missing_current_pose')
        self.owner.poses[24] = Pose(24, translation(80,58), 30., 'uncertain')
        prior = self.owner.motion
        with (self.no_matches(), patch.object(prior, '_component_prediction',
                return_value=(translation(80,58),30.,False))):
            self.assertIs(self.e.associate(obs), self.owner)
        self.assertEqual(len(self.e.instances),1)
        self.assertTrue(obs.ownership_confirmed)
        self.assertFalse(obs.reliable)
        np.testing.assert_allclose(obs.pose,translation(80,50))
        self.assertEqual(self.owner.poses[24].status,'predicted')

    def test_remote_single_line_does_not_inherit_complete_template_from_motion(self):
        p=self.probability.copy();p[40:]=0
        obs=self.observation(24,p,'remote',origin=(240,120))
        with self.no_matches():
            self.assertIsNone(self.e._registered_fragment(obs))
        self.assertIsNone(obs.template_id)

    def test_competing_overlapping_occurrences_remain_unresolved(self):
        self.e.instances[1]=Instance(1,0,0,poses={23:Pose(23,translation(80,50),.7,
            'measured',23,(self.seed.id,),True)})
        obs=self.observation(23,self.probability,'contested')
        with self.no_matches():
            self.assertIsNone(self.e._registered_fragment(obs))
        self.assertIsNone(obs.instance_id)

    def test_unrelated_later_instance_cannot_change_selected_measurement_status(self):
        # Another instance without a pose cannot turn the chosen owner's
        # original measurement into a prediction merely through loop order.
        self.e.instances[1] = Instance(1, 0, 0)
        original = self.owner.poses[23]
        obs = self.observation(23, self.probability, 'selected_measured')
        with self.no_matches():
            self.assertIs(self.e._registered_fragment(obs), self.owner)
        self.assertIs(self.owner.poses[23], original)
        self.assertEqual(original.status, 'measured')

    def test_prediction_cannot_reopen_confirmed_absence(self):
        self.owner.poses[24]=Pose(24,translation(80,50),.7,'absent',
            reason='confirmed_disappearance')
        obs=self.observation(24,self.probability,'gone')
        with self.no_matches():
            self.assertIsNone(self.e._registered_fragment(obs))
        self.assertIsNone(obs.instance_id)


if __name__=='__main__':
    unittest.main()
