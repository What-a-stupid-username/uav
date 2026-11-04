import unittest
from tests import template_candidate_selection as fixtures
from uav.template_mask.sam_continuity import continue_observation
from uav.template_mask.geometry import translation
from uav.template_mask.records import Pose
import numpy as np
from unittest.mock import patch


class SamContinuityTests(unittest.TestCase):
    setUp=fixtures.SelectionTests.setUp
    tearDown=fixtures.SelectionTests.tearDown
    owner=fixtures.SelectionTests.owner
    candidate=fixtures.SelectionTests.candidate

    def test_independent_similar_segmentations_continue_without_sift_votes(self):
        owner,seed=self.owner(self.small,'seed',0)
        self.e.source_ids={0:0,8:8}
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'next'),1008,'global')
        obs.origin=(41,51);obs.box=(41,51,101,91)
        chosen=continue_observation(self.e,obs)
        self.assertIs(chosen,owner)
        np.testing.assert_array_equal(obs.pose,translation(41,51))
        self.assertEqual(owner.poses[8].references,())
        self.assertEqual(owner.poses[8].reason,'continuous_sam_registration')
        self.assertEqual(len(self.e.bank.templates),1)

    def test_sampled_motion_uses_the_same_spatial_range_as_normal_tracking(self):
        owner,seed=self.owner(self.small,'seed',0)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'next'),1008,'global')
        obs.origin=(64,50);obs.box=(64,50,124,90)
        self.assertIs(continue_observation(self.e,obs),owner)
        np.testing.assert_array_equal(obs.pose,translation(64,50))

    def test_different_shape_remote_flash_and_explicit_absence_do_not_continue(self):
        for mode in ('shape','remote','absence','gap'):
            with self.subTest(mode=mode):
                owner,seed=self.owner(self.small,'seed-'+mode,len(self.e.instances))
                p=self.small.copy()
                if mode=='shape':p[:]=.9
                obs=self.e.observe(16 if mode=='gap' else 8,8,self.image,
                    self.candidate(p,'next-'+mode),1008,'global')
                if mode=='remote':obs.origin=(200,50);obs.box=(200,50,260,90)
                if mode=='absence':owner.poses[4]=Pose(4,seed.pose,1.,'absent')
                self.assertIsNone(continue_observation(self.e,obs))
                owner.ended_at=0

    def test_competing_occurrences_are_not_merged(self):
        self.owner(self.small,'a',0)
        self.owner(self.small,'b',1)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'next'),1008,'global')
        self.assertIsNone(continue_observation(self.e,obs))

    def test_two_current_candidates_cannot_claim_one_previous_occurrence(self):
        self.owner(self.small,'a',0)
        a=self.e.observe(8,8,self.image,self.candidate(self.small,'one'),1008,'global')
        b=self.e.observe(8,8,self.image,self.candidate(self.small,'two'),1008,'global')
        self.assertIsNone(continue_observation(self.e,a))
        self.assertIsNone(continue_observation(self.e,b))

    def test_actual_sam_registration_precedes_predicted_fragment_ownership(self):
        owner,seed=self.owner(self.small,'a',0)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'next'),1008,'global')
        with (patch.object(self.e,'_matches',return_value=[]),
              patch.object(self.e,'_registered_fragment',side_effect=AssertionError('prediction won'))):
            self.assertIs(self.e.associate(obs),owner)
        self.assertTrue(obs.reliable)
        self.assertEqual(owner.poses[8].reason,'continuous_sam_registration')

    def test_sam_similarity_cannot_override_a_contrary_current_measurement(self):
        owner,seed=self.owner(self.small,'a',0)
        owner.poses[8]=Pose(8,translation(120,50),.7,'measured',8,(),True)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'next'),1008,'global')
        self.assertIsNone(continue_observation(self.e,obs))
        np.testing.assert_array_equal(owner.poses[8].matrix,translation(120,50))
