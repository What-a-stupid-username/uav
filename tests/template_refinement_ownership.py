"""Independent motion separates pixel ownership from target registration."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.records import Candidate, Instance, Pose, Template
from uav.template_mask.geometry import translation
from uav.template_mask.refinement import exclude_independent_foreground
from uav.template_mask.refinement import review_refinement_ownership
from uav.template_mask.refinement import bounded_candidate
from uav.template_mask.refinement import select_originals
from uav.template_mask.ownership import OwnershipEvidence


class RefinementOwnershipTests(unittest.TestCase):
    def test_ownership_queries_reuse_one_measured_relation_per_snapshot(self):
        e,owners=self.engine()
        evidence=OwnershipEvidence(e)
        candidate=Candidate(np.ones((60,80)),(0,0),(0,0,80,60),.9,'whole')
        original=SimpleNamespace(instance_id=0,template_id=0,frame=8)
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')) as relative:
            exclude_independent_foreground(e,original,candidate,evidence=evidence)
            exclude_independent_foreground(e,original,candidate,evidence=evidence)
            self.assertEqual(relative.call_count,1)
    def test_long_registered_occurrence_can_disqualify_crossing_reference_of_short_fragment(self):
        short=Instance(0,0,8);long=Instance(1,1,0);other=Instance(2,2,0)
        other.poses[8]=Pose(8,translation(20,20),.7,'measured',8,('a',),True)
        obs=SimpleNamespace(id='obs',reliable=True,ambiguous=False,mask_role='discovery',
            instance_id=0,source_id=8,frame=8,box=(10,10,50,50),feature_key='f',score=.9)
        e=SimpleNamespace(active_instances=lambda:[short,long,other],
            config=SimpleNamespace(mask_reference_limit=8),
            bank=SimpleNamespace(observations={'obs':obs},templates={
                0:SimpleNamespace(observations=['obs']),1:SimpleNamespace(observations=[]),
                2:Template(2,(0,0,20,20))}),
            cache=SimpleNamespace(get=lambda key:dict(xy=np.zeros((10,2)))))
        def relative(engine,a,b,frame):
            return dict(state='independent_motion' if a.id==1 else 'unresolved')
        with patch('uav.template_mask.interactions.relative_motion',side_effect=relative):
            self.assertEqual(select_originals(e,{0:np.eye(3),1:np.eye(3)}),[])

    def test_evaluated_context_background_is_zero_evidence_not_unknown(self):
        p=np.ones((60,80),np.float32)
        c=Candidate(p,(0,0),(20,20,35,35),.9,'target')
        cleaned=bounded_candidate(c,np.eye(3),np.array([0,0]),np.array([80,60]))
        self.assertTrue(cleaned.valid[55,75])
        self.assertEqual(cleaned.probability[55,75],0)
        self.assertEqual(cleaned.probability[25,25],1)
        self.assertTrue(p.all())

    def engine(self):
        owners=[Instance(0,0,0),Instance(1,1,0)]
        owners[1].poses[8]=Pose(8,translation(45,20),.7,'measured',8,('a','b'),True)
        bank=SimpleNamespace(members=lambda tid:{tid:np.eye(3)},
            revision=0,
            templates={1:Template(1,(0,0,15,10))},
            observations={name:SimpleNamespace(source_id=n,reliable=True,ambiguous=False)
                for n,name in enumerate(('a','b'))},
            fuse=lambda tid,**kwargs:dict(mask=np.ones((10,15)),origin=(0,0)))
        return SimpleNamespace(instances={i.id:i for i in owners},bank=bank,
            active_instances=lambda:owners),owners

    def test_independent_pixels_are_unknown_without_changing_original_probabilities(self):
        e,owners=self.engine()
        candidate=Candidate(np.full((60,80),.9),(0,0),(0,0,80,60),.9,'union')
        original=SimpleNamespace(instance_id=0,template_id=0,frame=8)
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            result=exclude_independent_foreground(e,original,candidate)
        self.assertFalse(result.valid[25,50])
        self.assertTrue(result.valid[25,10])
        self.assertIs(result.probability,candidate.probability)
        self.assertIsNone(candidate.valid)

    def test_rigid_fragments_and_unmeasured_geometry_are_not_excluded(self):
        e,owners=self.engine()
        candidate=Candidate(np.ones((60,80)),(0,0),(0,0,80,60),.9,'whole')
        original=SimpleNamespace(instance_id=0,template_id=0,frame=8)
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='rigid_layout')):
            self.assertIs(exclude_independent_foreground(e,original,candidate),candidate)
        owners[1].poses[8].status='predicted'
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            self.assertIs(exclude_independent_foreground(e,original,candidate),candidate)

    def test_independent_measured_track_needs_no_second_reference_per_frame(self):
        e,owners=self.engine()
        owners[1].poses[8].references=('a',)
        candidate=Candidate(np.ones((60,80)),(0,0),(0,0,80,60),.9,'whole')
        original=SimpleNamespace(instance_id=0,template_id=0,frame=8)
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            result=exclude_independent_foreground(e,original,candidate)
        self.assertFalse(result.valid[25,50])

    def test_completed_batch_uses_fine_support_and_commits_after_all_checks(self):
        e,owners=self.engine()
        probability=np.ones((60,80),np.float32)
        valid=np.ones_like(probability,bool)
        obs=SimpleNamespace(id='fine',mask_role='refinement',reliable=True,ambiguous=False,
            feature_key=None,
            probability_key='p',origin=(0,0),box=(0,0,80,60),score=.9,
            candidate_id='union',instance_id=0,template_id=0,frame=8)
        e.bank.observations['fine']=obs
        for name in ('a','b'):
            e.bank.observations[name].mask_role='discovery'
        e.bank.revision=0
        from unittest.mock import Mock
        from uav.template_mask.cache import EvidenceCache
        e.cache=SimpleNamespace(get=lambda key:dict(probability=probability,valid=valid),
            put=Mock(),key=EvidenceCache.key)
        e.bank.fuse=Mock(return_value=dict(mask=np.ones((10,15)),origin=(0,0)))
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            self.assertEqual(review_refinement_ownership(e),1)
        e.bank.fuse.assert_called_once_with(1,coarse=False)
        written=e.cache.put.call_args.kwargs
        self.assertFalse(written['valid'][25,50])
        self.assertTrue(valid.all())
        self.assertIs(written['probability'],probability)
        self.assertEqual(e.bank.revision,1)
        self.assertEqual(obs.original_probability_key,'p')
        self.assertNotEqual(obs.probability_key,'p')

    def test_retracted_independence_restores_original_validity(self):
        from uav.template_mask.cache import EvidenceCache
        e,owners=self.engine()
        e.cache=EvidenceCache()
        self.addCleanup(e.cache.close)
        probability=np.ones((60,80),np.float32)
        e.cache.put('raw',probability=probability,valid=np.ones_like(probability,bool))
        obs=SimpleNamespace(mask_role='refinement',reliable=True,ambiguous=False,
            feature_key=None,
            probability_key='raw',original_probability_key=None,origin=(0,0),box=(0,0,80,60),
            score=.9,candidate_id='union',instance_id=0,template_id=0,frame=8)
        e.bank.observations={'fine':obs}
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            self.assertEqual(review_refinement_ownership(e),1)
        self.assertFalse(e.cache.get(obs.probability_key)['valid'][25,50])
        self.assertTrue(e.cache.get('raw')['valid'].all())
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='rigid_layout')):
            self.assertEqual(review_refinement_ownership(e),1)
        self.assertTrue(e.cache.get(obs.probability_key)['valid'].all())

    def test_owned_features_and_reference_keys_follow_reversible_pixel_ownership(self):
        from uav.template_mask.cache import EvidenceCache
        from uav.template_mask.records import Reference
        e,owners=self.engine()
        e.cache=EvidenceCache();self.addCleanup(e.cache.close)
        e.cache.put('raw',probability=np.ones((60,80),np.float32),valid=np.ones((60,80),bool))
        e.cache.put('features',xy=np.array([[10.,25.],[50.,25.]]),
                    size=np.ones(2),desc=np.ones((2,128)))
        obs=SimpleNamespace(id='fine',mask_role='refinement',reliable=True,ambiguous=False,
            feature_key='features',original_feature_key=None,
            probability_key='raw',original_probability_key=None,origin=(0,0),box=(0,0,80,60),
            score=.9,candidate_id='union',instance_id=0,template_id=0,frame=8)
        e.bank.observations={'fine':obs}
        e.bank.templates[0]=Template(0,(0,0,80,60),references=[Reference('fine',8,'features',1.)])
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            review_refinement_ownership(e)
        self.assertEqual(len(e.cache.get(obs.feature_key)['xy']),1)
        self.assertEqual(e.bank.templates[0].references[0].feature_key,obs.feature_key)
        self.assertEqual(len(e.cache.get('features')['xy']),2)
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='rigid_layout')):
            review_refinement_ownership(e)
        self.assertEqual(len(e.cache.get(obs.feature_key)['xy']),2)
