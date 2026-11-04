"""Occurrence presence consumes association; box overlap cannot restart a life."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.geometry import translation
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Pose


class PresenceOwnershipTests(unittest.TestCase):
    def test_accepted_interval_does_not_erase_explicit_absence(self):
        self.owner.poses[4]=Pose(4,translation(30,40),.7,'absent',
            reason='explicit_disappearance')
        with patch.object(self.e,'_track') as track:
            self.e.presence._accept_interval(self.owner,0,8,lambda f:self.image)
        self.assertNotIn(4,[call.args[1] for call in track.call_args_list])
        self.assertEqual(self.owner.poses[4].reason,'explicit_disappearance')

    def test_boundary_negative_does_not_replace_a_stronger_absence_reason(self):
        self.owner.poses[4]=Pose(4,translation(30,40),.7,'absent',
            reason='explicit_disappearance')
        self.e.presence._absent(self.owner,4)
        self.assertEqual(self.owner.poses[4].reason,'explicit_disappearance')
        self.e.presence._check(self.owner,4)
        self.assertEqual(self.owner.poses[4].reason,'explicit_disappearance')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = EvidenceCache()
        self.image = np.zeros((160,300,3),np.uint8)
        self.e = TemplateEngine(TemplateConfig(sift_backend='cpu'),self.cache,None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'),self.image.shape[:2])
        obs = self.observe(0,'seed')
        self.owner = self.e._provisional(obs,'new_template')
        self.seed = obs
        self.e.source_ids = {f:f for f in range(65)}
        self.e.presence.appearance_checked.add(self.owner.id)
        self.e.presence.check(0,0,[self.candidate('seed')])

    def tearDown(self):
        self.cache.close()
        self.temp.cleanup()

    def candidate(self,label,box=(30,40,50,50)):
        l,t,r,b = box
        return Candidate(np.full((b-t,r-l),.8,np.float32),(l,t),box,.9,label)

    def observe(self,frame,label,box=(30,40,50,50)):
        return self.e.observe(frame,frame,self.image,self.candidate(label,box),1008,'global')

    def lose(self,frame=8):
        self.owner.poses[frame] = Pose(frame,translation(30,40),5.,'uncertain')
        self.e.presence.check(frame,frame,[])

    def test_overlapping_box_never_restarts_old_instance_regardless_gap_or_box_size(self):
        for frame in (16,64):
            for box in ((30,40,50,50),(0,0,250,140)):
                with self.subTest(frame=frame,box=box):
                    self.lose()
                    self.owner.poses[frame]=Pose(frame,translation(30,40),100.,'absent',
                                               reason='global_detection_absent')
                    self.e.presence.check(frame,frame,[self.candidate('foreign',box)])
                    self.assertEqual(self.e.presence.checks[0][frame],'not_detected')
                    self.assertEqual(self.owner.poses[frame].status,'absent')
                    self.assertFalse(self.e.presence.covers(self.owner,frame))

    def test_new_flash_same_position_shares_template_but_not_presence(self):
        self.lose()
        obs=self.observe(16,'new_flash')
        obs.pose,obs.template_id,obs.reliable=translation(30,40),0,True
        new=self.e._new_instance(self.e.bank.templates[0],obs)
        self.owner.poses[16]=Pose(16,translation(30,40),10.,'uncertain')
        self.e.presence.check(16,16,[self.candidate('new_flash')])
        self.assertEqual(self.e.presence.checks[self.owner.id][16],'not_detected')
        self.assertEqual(self.e.presence.checks[new.id][16],'image_match')
        self.assertNotEqual(self.owner.id,new.id)
        self.assertEqual(self.owner.template_id,new.template_id)

    def test_assigned_fragment_proves_presence_without_creating_pose_or_template_votes(self):
        self.lose()
        obs=self.observe(16,'fragment',(30,40,36,50))
        obs.instance_id,obs.template_id,obs.ownership_confirmed=0,0,True
        obs.pose=translation(30,40)
        self.owner.poses[16]=Pose(16,translation(30,40),2.,'predicted')
        revision=self.e.bank.revision
        self.e.presence.check(16,16,[self.candidate('fragment',(30,40,36,50))])
        self.assertEqual(self.e.presence.checks[0][16],'sam_candidate')
        self.assertEqual(self.owner.poses[16].status,'predicted')
        self.assertEqual(self.e.bank.revision,revision)
        self.assertFalse(self.e.presence.missing(self.owner,17))

    def test_independent_image_match_survives_global_sam_miss(self):
        self.owner.poses[8]=Pose(8,translation(60,70),.7,'measured',8,('ref',),True)
        self.e.presence.check(8,8,[])
        self.assertEqual(self.e.presence.checks[0][8],'image_match')
        self.assertEqual(self.owner.poses[8].status,'measured')

    def test_ambiguous_assigned_pixels_do_not_certify_presence(self):
        obs=self.observe(8,'ambiguous')
        obs.instance_id,obs.template_id,obs.ownership_confirmed,obs.ambiguous=0,0,True,True
        self.owner.poses[8]=Pose(8,translation(30,40),2.,'uncertain')
        self.e.presence.check(8,8,[self.candidate('ambiguous')])
        self.assertEqual(self.e.presence.checks[0][8],'not_detected')
        self.assertTrue(self.e._observed_candidate(obs).any())

    def test_boundary_probe_cannot_borrow_already_assigned_foreign_candidate(self):
        obs=self.observe(7,'other')
        other=self.e._provisional(obs,'new_template')
        self.owner.poses[7]=Pose(7,translation(30,40),2.,'uncertain')
        self.e.detector=type('Detector',(),{'detect':lambda *a,**k:[self.candidate('other')]})()
        with patch.object(self.e,'_track'), patch.object(self.e,'coarse_detect',return_value=[self.candidate('other')]):
            self.assertFalse(self.e.presence._probe(self.owner,7,lambda f:self.image))
        self.assertEqual(obs.instance_id,other.id)
        self.assertIsNone(obs.boundary_instance_id)
        self.assertFalse(self.owner.poses[7].presence)

    def test_negative_sequence_does_not_launch_boundary_search_on_foreign_overlap(self):
        self.lose()
        self.owner.poses[16]=Pose(16,translation(30,40),10.,'uncertain')
        self.e.presence.check(16,16,[self.candidate('unrelated')])
        # The original 0->8 disappearance is legitimate; inspect only the
        # negative->negative interval where a foreign box later appears.
        self.e.presence.global_samples.pop(0)
        self.e.presence.checks[0].pop(0)
        with patch.object(self.e.presence,'_boundary',side_effect=AssertionError('false resurrection')):
            self.e.presence.online(16,lambda f:self.image,set())
            self.e.presence.resolve(lambda f:self.image)
        self.assertFalse(self.e.presence.covers(self.owner,16))
        self.assertFalse(self.e.render(16).any())


if __name__=='__main__':
    unittest.main()
