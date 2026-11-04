"""Whole/partial ranking, independent candidate alternatives and endpoints."""
from pathlib import Path
from types import SimpleNamespace
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
from uav.template_mask.records import Candidate, Pose
from uav.template_mask.refinement import refine_templates
from uav.template_mask.schedule import discovery_frames
from uav.template_mask.global_identity import occurrence_agreement


class SelectionTests(unittest.TestCase):
    def test_failed_partition_does_not_commit_first_child(self):
        from uav.template_mask.composite import apply_partition
        a,seed=self.owner(self.small,'a',0)
        b,other=self.owner(self.small,'b',1)
        obs=self.e.observe(8,8,self.image,self.candidate(self.double,'whole'),1008,'global')
        left=np.zeros_like(self.double,bool);left[:,:60]=self.small>.23
        empty=np.zeros_like(left)
        fit=Match(translation(40,50),.7,20,.8,(seed.id,other.id))
        best=(.9,(self.e.bank.templates[a.template_id],fit,left),
                 (self.e.bank.templates[b.template_id],fit,empty))
        before=(set(self.e.bank.observations),set(self.e.instances),self.e.bank.revision,
                tuple(self.e.frame_observations[8]))
        self.assertFalse(apply_partition(self.e,obs,best))
        self.assertEqual(before,(set(self.e.bank.observations),set(self.e.instances),
            self.e.bank.revision,tuple(self.e.frame_observations[8])))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.cache=EvidenceCache()
        self.image=np.zeros((200,400,3),np.uint8)
        self.e=TemplateEngine(TemplateConfig(sift_backend='cpu'),self.cache,None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'),self.image.shape[:2])
        self.small=np.zeros((40,60),np.float32);self.small[8:30,8:50]=.9
        self.double=np.zeros((40,140),np.float32)
        self.double[:,:60]=self.small;self.double[:,80:]=self.small

    def tearDown(self):
        self.cache.close();self.temp.cleanup()

    def candidate(self,p,label):
        return Candidate(p,(40,50),(40,50,40+p.shape[1],50+p.shape[0]),.9,label)

    def owner(self,p,label,source):
        obs=self.e.observe(0,source,self.image,self.candidate(p,label),1008,'global')
        return self.e._provisional(obs,'new_template'),obs

    def matches(self,template,key,**kwargs):
        ref=self.e.bank.templates[template.id].references[0].observation_id
        # The polluted containing template has more independent references.
        return [Match(translation(40,50),.7,20,.8,
            (ref,) if template.id==0 else (ref,'extra-a','extra-b'))]

    def test_exact_template_wins_over_measured_containing_template_with_more_references(self):
        small,seed=self.owner(self.small,'small',0)
        double,other=self.owner(self.double,'double',1)
        for owner,ref in ((small,seed),(double,other)):
            owner.poses[8]=Pose(8,translation(40,50),.7,'measured',8,(ref.id,),True)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'current'),1008,'global')
        with patch.object(self.e,'_matches',side_effect=self.matches):
            chosen=self.e.associate(obs)
        self.assertIs(chosen,small)
        self.assertEqual(obs.template_id,small.template_id)
        self.assertNotIn(obs.id,self.e.bank.templates[double.template_id].observations)

    def test_exact_template_precedes_containing_track_even_before_it_has_a_current_pose(self):
        small,seed=self.owner(self.small,'small',0)
        double,other=self.owner(self.double,'double',1)
        small.poses.clear()
        # Two independent references qualify this remote exact-template match.
        self.e.bank.observations['small-second']=SimpleNamespace(source_id=2)
        double.poses[8]=Pose(8,translation(40,50),.7,'measured',8,(other.id,),True)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'current'),1008,'global')
        def match(template,key,**kwargs):
            m=self.matches(template,key,**kwargs)[0]
            if template.id==small.template_id:m.references=(seed.id,'small-second')
            return [m]
        with patch.object(self.e,'_matches',side_effect=match):
            chosen=self.e.associate(obs)
        self.assertEqual(chosen.template_id,small.template_id)
        self.assertNotEqual(chosen.id,double.id)

    def test_fragment_still_continues_when_no_complete_template_is_available(self):
        whole,seed=self.owner(self.double,'whole',0)
        whole.poses[8]=Pose(8,translation(40,50),.7,'measured',8,(seed.id,),True)
        obs=self.e.observe(8,8,self.image,self.candidate(self.small,'fragment'),1008,'global')
        with patch.object(self.e,'_matches',side_effect=self.matches):
            chosen=self.e.associate(obs)
        self.assertIs(chosen,whole)
        self.assertTrue(obs.ownership_confirmed)

    def test_composite_discovery_uses_two_complete_registrations_without_new_big_template(self):
        a,seed=self.owner(self.small,'a',0)
        # A distinct shape at the second position.
        second=self.small.copy();second[8:18,8:50]=0
        b,other=self.owner(second,'b',1)
        self.e.bank.observations['a-ref']=SimpleNamespace(source_id=2,reliable=True,ambiguous=False)
        self.e.bank.observations['b-ref']=SimpleNamespace(source_id=3,reliable=True,ambiguous=False)
        joined=self.double.copy();joined[:,80:]=second
        obs=self.e.observe(24,24,self.image,self.candidate(joined,'joined'),1008,'global')
        def match(template,key,**kwargs):
            refs=(seed.id,'a-ref') if template.id==0 else (other.id,'b-ref')
            return [Match(translation(40 if template.id==0 else 120,50),.7,20,.8,refs)]
        with patch.object(self.e,'_matches',side_effect=match):
            self.assertIsNone(self.e.associate(obs))
        self.assertEqual(len(self.e.bank.templates),2)
        self.assertEqual(len(obs.partition_children),2)
        self.assertFalse(self.e.bank.links)
        children=[self.e.bank.observations[k] for k in obs.partition_children]
        self.assertEqual({c.template_id for c in children},{0,1})
        self.assertTrue(all(c.source_id==24 for c in children))
        for c in children:
            self.assertNotIn(obs.id,self.e.bank.templates[c.template_id].observations)
            self.assertLess(self.e._observed_candidate(c).sum(),(joined>.23).sum())
        self.assertEqual(len(self.e.composite_discoveries),1)

    def test_single_reference_composite_proposal_cannot_split_ownership(self):
        from uav.template_mask.composite import partition_discovery
        a,seed=self.owner(self.small,'a',0)
        b,other=self.owner(self.small,'b',1)
        obs=self.e.observe(24,24,self.image,self.candidate(self.double,'joined'),1008,'global')
        proposals=[(self.e.bank.templates[o.template_id],Match(translation(40+80*k,50),.7,
            20,.8,(ref.id,)),(.5,1.,2.)) for k,(o,ref) in enumerate(((a,seed),(b,other)))]
        self.assertFalse(partition_discovery(self.e,obs,proposals))
        self.assertFalse(obs.partition_children)

    def test_localized_partial_component_still_requires_independent_motion(self):
        from uav.template_mask.composite import partition_discovery
        a,seed=self.owner(self.small,'a',0)
        b,other=self.owner(self.small,'b',1)
        for label,source in (('a-ref',2),('b-ref',3)):
            self.e.bank.observations[label]=SimpleNamespace(source_id=source,reliable=True,ambiguous=False)
        obs=self.e.observe(24,24,self.image,self.candidate(self.double,'mixed'),1008,'global')
        proposals=[(self.e.bank.templates[owner.template_id],
            Match(translation(40+80*k,50),.7,20,.8,(ref.id,label)),(.5,.5,2.,True))
            for k,(owner,ref,label) in enumerate(((a,seed,'a-ref'),(b,other,'b-ref')))]
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='rigid_layout')):
            self.assertFalse(partition_discovery(self.e,obs,proposals,require_independent=True))
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            self.assertTrue(partition_discovery(self.e,obs,proposals,require_independent=True))

    def test_imprecise_large_box_cannot_hide_two_registered_complete_shapes(self):
        # A wide/tall SAM box makes each template look like an implausible
        # partial match in isolation. Partitioning must inspect geometry before
        # the single-template identity gate discards those fits.
        a,seed=self.owner(self.small,'a',0)
        second=self.small.copy();second[8:18,8:50]=0
        b,other=self.owner(second,'b',1)
        for label,source in (('a-ref',2),('b-ref',3)):
            self.e.bank.observations[label]=SimpleNamespace(source_id=source,reliable=True,ambiguous=False)
        p=np.zeros((120,180),np.float32)
        p[:40,:60]=self.small;p[75:115,100:160]=second
        obs=self.e.observe(24,24,self.image,self.candidate(p,'joined'),1008,'global')
        def match(template,key,**kwargs):
            refs=(seed.id,'a-ref') if template.id==0 else (other.id,'b-ref')
            return [Match(translation(40 if template.id==0 else 140,
                50 if template.id==0 else 125),.7,20,.8,refs)]
        with patch.object(self.e,'_matches',side_effect=match):
            self.assertIsNone(self.e.associate(obs))
        self.assertEqual(len(obs.partition_children),2)
        self.assertEqual(len(self.e.bank.templates),2)

    def test_stage2_independent_motion_rejects_retained_composite_without_discarding_originals(self):
        from uav.template_mask.composite import review_composite_templates
        second=self.small.copy();second[8:18,8:50]=0
        a,seed=self.owner(self.small,'a',0)
        b,other=self.owner(second,'b',1)
        for owner,p,source in ((a,self.small,2),(b,second,3)):
            ref=self.e.observe(1,source,self.image,self.candidate(p,f'ref-{source}'),1008,'global')
            ref.pose=translation(40,50);ref.template_id=owner.template_id;ref.reliable=True
            self.e._assign(owner,ref)
        for owner in (a,b):
            owner.poses={f:Pose(f,translation(40+f if owner.id==a.id else 120-f,50),
                .7,'measured',f+10,presence=True) for f in range(33)}
        joined=self.double.copy();joined[:,80:]=second
        raw=self.e.observe(0,100,self.image,self.candidate(joined,'old-composite'),1008,'global')
        composite=self.e._provisional(raw,'new_template')
        self.e.stage='second_pass'
        def match(template,key,**kwargs):
            if template.id>1:return []
            refs=tuple(r.observation_id for r in template.references)
            return [Match(translation(40 if template.id==0 else 120,50),.7,20,.8,refs)]
        with patch.object(self.e,'_matches',side_effect=match):
            changed=review_composite_templates(self.e)
        self.assertIn(composite.template_id,changed)
        self.assertTrue(self.e.bank.templates[composite.template_id].composite)
        self.assertFalse(self.e._renders_at(composite,0))
        self.assertEqual(len(raw.partition_children),2)
        self.assertTrue(self.e._observed_candidate(raw).any())
        self.assertNotIn(composite,self.e.active_instances())

    def test_composite_review_skips_registration_without_independent_constituents(self):
        from uav.template_mask.composite import review_composite_templates
        for number in range(3):
            owner, seed = self.owner(self.small, str(number), number)
            ref = self.e.observe(1, number+10, self.image,
                self.candidate(self.small, f'ref-{number}'), 1008, 'global')
            ref.pose=seed.pose.copy();ref.template_id=owner.template_id;ref.reliable=True
            self.e._assign(owner,ref)
        with patch.object(self.e, '_matches', side_effect=AssertionError('unnecessary registration')):
            self.assertEqual(review_composite_templates(self.e),set())

    def refine(self,original,results):
        owner,seed=self.owner(original,'seed',0)
        self.e.detector=SimpleNamespace(detect=lambda *a,**k:results)
        fit=Match(translation(40,50),.7,12,.8,(seed.id,))
        with patch.object(self.e,'_matches',return_value=[fit]):
            refine_templates(self.e,lambda f:self.image)
        return owner

    def test_independent_small_mark_disproves_whole_despite_unexplained_pixels(self):
        from uav.template_mask.composite import partition_discovery
        large=np.zeros((40,60),np.float32);large[5:35,5:55]=.9
        small=np.zeros((40,60),np.float32);small[10:22,10:25]=.9
        a,seed=self.owner(large,'large',0)
        b,other=self.owner(small,'small',1)
        for label,source in (('a-ref',2),('b-ref',3)):
            self.e.bank.observations[label]=SimpleNamespace(source_id=source,reliable=True,ambiguous=False)
        joined=np.zeros((80,160),np.float32)
        joined[:40,:60]=large;joined[:40,80:140]=small
        joined[50:70,10:150]=.9  # Unexplained SAM foreground stays local.
        obs=self.e.observe(24,24,self.image,self.candidate(joined,'joined'),1008,'global')
        proposals=[(self.e.bank.templates[owner.template_id],
            Match(translation(40+80*k,50),.7,20,.8,(ref.id,label)),(.3,1.,2.))
            for k,(owner,ref,label) in enumerate(((a,seed,'a-ref'),(b,other,'b-ref')))]
        self.assertFalse(partition_discovery(self.e,obs,proposals))
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='rigid_layout')):
            self.assertFalse(partition_discovery(self.e,obs,proposals,require_independent=True))
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            self.assertTrue(partition_discovery(self.e,obs,proposals,require_independent=True))
        self.assertEqual(len(obs.partition_children),2)
        np.testing.assert_array_equal(self.e.cache.get(obs.probability_key)['probability'],joined)

    def test_refinement_does_not_union_small_with_containing_double_candidate(self):
        self.refine(self.small,[self.candidate(self.double,'double'),self.candidate(self.small,'small')])
        view=self.e.refinement_batches[0]['views'][0]
        self.assertEqual(len(view['accepted']),1)
        self.assertEqual(self.e.bank.observations[view['accepted'][0]].candidate_id,'small')
        # The foreign containing proposal is now rejected by the same
        # candidate-area gate as legacy UAV, before pairwise selection.
        self.assertEqual(len(view['rejected_candidates']),0)
        fusion=self.e.bank.fuse(0)
        self.assertLess(fusion['mask'].sum(),self.small.size)

    def test_refinement_rejects_mostly_foreign_containing_box_without_small_alternative(self):
        self.refine(self.small,[self.candidate(self.double,'foreign-whole')])
        self.assertFalse(self.e.refinement_batches[0]['views'][0]['accepted'])

    def test_refinement_context_logits_cannot_expand_original_fusion_canvas(self):
        # The box is correct, but the returned probability tensor also contains
        # a distant high-confidence object in the shared context crop.
        candidate=self.candidate(self.double,'correct-box-foreign-logits')
        candidate.box=(40,50,100,90)
        self.refine(self.small,[candidate])
        fusion=self.e.bank.fuse(0)
        self.assertTrue(fusion['mask'].any())
        self.assertLess(self.e.bank.templates[0].box[2],80)
        self.assertLess(fusion['mask'].sum(),2*(self.small>.23).sum())

    def test_gap_motion_uses_original_sample_times(self):
        from uav.template_mask.gaps import trajectory_bridge
        times={f:int(f*30/24) for f in range(50)}
        template=SimpleNamespace(box=(0,0,200,40),anchor=(100,20))
        rows=[Pose(f,translation(20,1600-12*times[f]),.7,'measured',times[f],presence=True)
              for f in range(50)]
        bridge=trajectory_bridge(rows[:20],rows[24:],template,range(19,25),sample_times=times)
        self.assertIsNotNone(bridge)
        for f,(matrix,_) in bridge[0].items():
            np.testing.assert_allclose(matrix,rows[f].matrix,atol=1e-5)
        self.assertEqual(bridge[1]['time_basis'],'source_frames')

    def test_straight_path_allows_variable_speed_but_not_flash_or_corner(self):
        from uav.template_mask.gaps import path_continuation
        template=SimpleNamespace(box=(0,0,200,40),anchor=(100,20))
        rows=[Pose(f,translation(20,-(.15*f*f+2*f)),.7,'measured',f,presence=True)
              for f in range(40)]
        self.assertIsNotNone(path_continuation(rows[:16],rows[20:],template))
        jumped=[Pose(p.frame,p.matrix.copy(),p.uncertainty,p.status,p.source_id,presence=True) for p in rows[20:]]
        for p in jumped:p.matrix[1,2]-=2000
        self.assertIsNone(path_continuation(rows[:16],jumped,template))
        corner=[Pose(p.frame,translation(p.frame*5,-70),.7,'measured',p.source_id,presence=True) for p in rows[20:]]
        self.assertIsNone(path_continuation(rows[:16],corner,template))

    def test_partition_rechecks_parent_interval_without_copying_parent_poses(self):
        parent,raw=self.owner(self.double,'whole',0)
        child,part=self.owner(self.small,'part',1)
        parent.poses={f:Pose(f,translation(40,50),.7,'measured',f,presence=True) for f in range(16)}
        child.poses={0:Pose(0,translation(40,50),.7,'measured',0,presence=True)}
        child.birth=0
        self.e.bank.templates[parent.template_id].composite=True
        self.e.stage='second_pass'
        self.e.source_ids={f:f for f in range(16)}
        self.e.composite_discoveries=[dict(observation=raw.id,instances=[child.id])]
        jobs=self.e._unresolved_pose_jobs(set())
        self.assertTrue(all((f,child.id) in jobs for f in range(1,16)))
        self.assertEqual(set(child.poses),{0})

    def test_refinement_avoids_crossing_reference_but_keeps_rigid_fragments(self):
        from uav.template_mask.refinement import select_originals
        owner,seed=self.owner(self.small,'target',0)
        other,foreign=self.owner(self.small,'foreign',1)
        other.poses[0]=Pose(0,translation(40,50),.7,'measured',0,presence=True)
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='independent_motion')):
            self.assertFalse(select_originals(self.e,{owner.template_id:np.eye(3)}))
        with patch('uav.template_mask.interactions.relative_motion',return_value=dict(state='rigid_layout')):
            self.assertEqual(select_originals(self.e,{owner.template_id:np.eye(3)}),[seed])

    def test_complementary_refinement_fragments_are_both_retained(self):
        a=self.small.copy();a[19:]=0
        b=self.small.copy();b[:19]=0
        self.refine(self.small,[self.candidate(a,'a'),self.candidate(b,'b')])
        self.assertEqual(len(self.e.refinement_batches[0]['views'][0]['accepted']),2)

    def test_larger_refinement_is_not_rejected_without_a_better_alternative(self):
        # Additional target strokes with at least half the candidate in the
        # measured box remain allowed; an unrelated distant copy does not.
        larger=np.pad(self.small,((0,0),(0,20)))
        self.refine(self.small,[self.candidate(larger,'larger')])
        self.assertEqual(len(self.e.refinement_batches[0]['views'][0]['accepted']),1)

    def test_short_shared_motion_cannot_expand_shape_but_can_still_associate_occurrence(self):
        a,seed=self.owner(self.small,'a',0)
        b,other=self.owner(self.double,'b',1)
        for owner in (a,b):
            owner.poses={f:Pose(f,translation(40+f,50),.7,'measured',f,presence=True)
                         for f in range(12)}
        self.assertTrue(occurrence_agreement(self.e,a,b,np.eye(3))['accepted'])
        result=occurrence_agreement(self.e,a,b,np.eye(3),expanding=True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'brief_shared_motion_is_not_template_identity')
        for f in range(12,25):
            for owner in (a,b):owner.poses[f]=Pose(f,translation(40+f,50),.7,'measured',f,presence=True)
        self.assertTrue(occurrence_agreement(self.e,a,b,np.eye(3),expanding=True)['accepted'])
        for f in range(16,25):b.poses[f].matrix=translation(200+f,50)
        self.assertFalse(occurrence_agreement(self.e,a,b,np.eye(3),expanding=True)['accepted'])


class EndpointTests(unittest.TestCase):
    def test_schedule_includes_both_endpoints_without_duplicate_calls(self):
        for count,expected in ((1,[0]),(8,[0,7]),(9,[0,8]),(20,[0,8,16,19])):
            self.assertEqual(discovery_frames(count,8),expected)

    def test_watermark_only_on_last_unscheduled_frame_is_discovered(self):
        with tempfile.TemporaryDirectory() as root:
            cache=EvidenceCache();self.addCleanup(cache.close)
            image=np.zeros((100,160,3),np.uint8);calls=[]
            def detect(img,source,box,resolution):
                calls.append(source)
                return [Candidate(np.ones((10,20),np.float32),(30,40),(30,40,50,50),.9,'last')] if source==9 else []
            e=TemplateEngine(TemplateConfig(sift_backend='cpu'),cache,SimpleNamespace(detect=detect),
                WarningLog(Path(root)/'warnings.jsonl'),image.shape[:2],debug=True)
            e.frame_count=10
            for f in range(10):e.first_frame(f,f,image)
            self.assertEqual(calls,[0,8,9])
            self.assertEqual(len(e.instances),1)
            self.assertIn(9,e.discovery_groups)
            from uav.template_mask.review import stage1_mask
            self.assertTrue(stage1_mask(e,9).any())


if __name__=='__main__':unittest.main()
