"""Registered whole tracks reunite incomplete SAM hypotheses."""
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
from uav.template_mask.records import Candidate, Pose
from uav.template_mask.global_identity import evaluate_pair


class GlobalIdentityTests(unittest.TestCase):
    def test_registration_views_reuse_only_the_current_snapshot(self):
        from uav.template_mask.global_identity import registration_views
        self.owner(0)
        self.e._association_views = {}
        with patch.object(self.e.bank,'selected_views',wraps=self.e.bank.selected_views) as select:
            first=registration_views(self.e,0)
            self.assertIs(registration_views(self.e,0),first)
            self.assertEqual(select.call_count,1)
            self.e._association_views = {}
            second=registration_views(self.e,0)
            self.assertEqual(select.call_count,2)
        self.assertEqual([o.id for o,_ in first],[o.id for o,_ in second])
        for (_,a),(_,b) in zip(first,second):
            np.testing.assert_array_equal(a,b)

    def test_registered_containing_records_need_sustained_measured_layout(self):
        from uav.template_mask.association import registered_occurrence_owner
        a,oa,_=self.owner(0)
        b,ob,_=self.owner(1)
        supported=[(self.e.bank.templates[i.template_id],
            Match(i.poses[23].matrix,.7,20,.8,(o.id,)),i,{}) for i,o in ((a,oa),(b,ob))]
        for kind in ('equivalent','a_contains_b','b_contains_a'):
            with patch('uav.template_mask.global_identity.shape_agreement',return_value={'kind':kind}):
                self.assertIs(registered_occurrence_owner(self.e,supported)[2],a)
        b.poses={f:p for f,p in b.poses.items() if f>=16}
        with patch('uav.template_mask.global_identity.shape_agreement',return_value={'kind':'a_contains_b'}):
            self.assertIsNone(registered_occurrence_owner(self.e,supported))
        with patch('uav.template_mask.global_identity.shape_agreement',return_value={'kind':'unresolved'}):
            self.assertIsNone(registered_occurrence_owner(self.e,supported))

    def test_global_snapshot_does_not_materialize_redundant_coordinate_cycles(self):
        for tid in range(4):self.owner(tid)
        proposals=[dict(a=a,b=b,accepted=True,transform=translation().tolist(),sigma=.7,
            sources=[1000,1001],shape={'kind':'equivalent'},trajectories=[])
            for a in range(4) for b in range(a+1,4)]
        with (patch('uav.template_mask.association.evaluate_associations',return_value=proposals),
              patch.object(self.e.reconciler,'refresh_aliases')):
            changed=self.e.reconciler.global_review()
        self.assertEqual(set(changed),{0,1,2,3})
        self.assertEqual(len(self.e.bank.links),3)
        self.assertEqual(set(self.e.bank.members(0)),{0,1,2,3})
        self.assertEqual(len(self.e.reconciler.global_reviews),6)
        self.assertEqual(sum(r.get('coordinate_application')=='already_connected'
                             for r in proposals),3)

    def test_alias_review_reuses_roots_but_not_across_link_revocation(self):
        owners=[self.owner(t)[0] for t in range(6)]
        with patch.object(self.e.bank,'members',wraps=self.e.bank.members) as members:
            self.e.reconciler.refresh_aliases()
            self.assertLessEqual(members.call_count,len(owners))
        self.e.bank.propose_link(0,1,translation(),1000)
        self.e.bank.propose_link(0,1,translation(),1001)
        self.e.reconciler.refresh_aliases()
        self.assertEqual(owners[1].alias_of,0)
        self.e.bank.revoke(0,1,'independent_motion')
        self.e.reconciler.refresh_aliases()
        self.assertIsNone(owners[1].alias_of)

    def test_occurrence_cache_respects_coordinate_root_and_owner_subset(self):
        from uav.template_mask.global_identity import occurrence_groups
        a,_,_=self.owner(0)
        b,_,_=self.owner(1,motion=lambda f:translation(40+f*.4,50+.01*f*f))
        self.e.bank.propose_link(0,1,translation(10,0),1000)
        self.e.bank.propose_link(0,1,translation(10,0),1001)
        self.e._association_occurrences={}
        occurrence_groups(self.e,[a,b],self.e.bank.members(0))
        result=occurrence_groups(self.e,[b],self.e.bank.members(1))
        self.assertEqual(result[0][1],(b.id,))
        np.testing.assert_allclose(result[0][0].poses[8].matrix,b.poses[8].matrix)

    def test_identity_bridge_passes_original_timestamps_to_shared_validator(self):
        a,_,_=self.owner(0,frames=range(24))
        b,_,_=self.owner(1,frames=range(30,54))
        self.e.source_ids={f:int(1.25*f) for f in range(54)}
        with patch('uav.template_mask.reconciliation.trajectory_bridge',return_value=None) as bridge:
            self.e.reconciler._identity_bridge(a,b)
        self.assertIs(bridge.call_args.kwargs['sample_times'],self.e.source_ids)


    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.cache=EvidenceCache()
        self.e=TemplateEngine(TemplateConfig(sift_backend='cpu'),self.cache,None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'),(200,500))
        self.image=np.zeros((200,500,3),np.uint8)
        self.full=np.zeros((40,100),np.float32)
        self.full[4:12,4:90]=.9
        self.full[25:34,4:90]=.9

    def tearDown(self):
        self.cache.close();self.temp.cleanup()

    def owner(self,tid,p=None,source=None,reason='new_template',motion=None,frames=range(24)):
        p=self.full if p is None else p
        source=tid*100 if source is None else source
        x,y=30,50
        obs=self.e.observe(tid,source,self.image,Candidate(p,(x,y),(x,y,x+100,y+40),.9,str(tid)),1008,'global')
        owner=self.e._provisional(obs,reason)
        fine=self.e.observe(tid,source,self.image,Candidate(p,(x,y),(x,y,x+100,y+40),.9,'fine'+str(tid)),1008,('refinement',tid))
        fine.pose,fine.template_id,fine.instance_id,fine.reliable=obs.pose.copy(),tid,owner.id,True
        self.e.bank.add(fine)
        owner.observations.append(fine.id)
        motion=motion or (lambda f:translation(30+f*.4,50+.01*f*f))
        owner.poses={f:Pose(f,motion(f),.7,'measured',f,(obs.id,),True) for f in frames}
        return owner,obs,fine

    def matches(self,template,key,**kwargs):
        obs=next(o for o in self.e.bank.observations.values() if o.feature_key==key)
        ref=self.e.bank.templates[template.id].references[0].observation_id
        return [Match(obs.pose.copy(),.7,20,.8,(ref,))]

    def review(self):
        with patch.object(self.e,'_matches',side_effect=self.matches):
            return self.e.reconciler.global_review()

    def test_equivalent_templates_share_shape_but_distinct_instances_stay_separate(self):
        a,_,_=self.owner(0)
        b,_,_=self.owner(1,motion=lambda f:translation(250-f,80))
        changed=self.review()
        self.assertEqual(changed,{0,1})
        self.assertEqual(set(self.e.bank.members(0)),{0,1})
        self.assertIsNone(b.alias_of)
        self.assertEqual(len(self.e.active_instances()),2)

    def test_duplicate_occurrence_is_aliased_without_copying_or_deleting_measurements(self):
        a,_,_=self.owner(0)
        b,_,_=self.owner(1)
        original=b.poses[8]
        self.review()
        self.assertEqual(b.alias_of,a.id)
        self.assertFalse(self.e._renders_at(b,8))
        self.assertIs(b.poses[8],original)
        self.e.bank.revoke(0,1,'independent_evidence')
        self.e.reconciler.refresh_aliases()
        self.assertIsNone(b.alias_of)
        self.assertTrue(self.e._renders_at(b,8))
        self.assertIs(b.poses[8],original)

    def test_same_position_across_disappearance_shares_template_not_occurrence(self):
        a,_,_=self.owner(0,frames=range(16))
        b,_,_=self.owner(1,frames=range(32,48))
        self.review()
        self.assertIn(1,self.e.bank.members(0))
        self.assertIsNone(b.alias_of)
        self.assertIsNone(b.continuation_of)

    def test_contained_strokes_with_consistent_whole_trajectory_can_merge(self):
        partial=self.full.copy();partial[20:]=0
        self.owner(0)
        fragment,_,_=self.owner(1,partial,reason='overlap_without_identity')
        self.review()
        self.assertIn(1,self.e.bank.members(0))
        self.assertEqual(fragment.alias_of,0)

    def test_shared_single_line_at_another_position_cannot_become_whole_template(self):
        partial=self.full.copy();partial[20:]=0
        self.owner(0)
        self.owner(1,partial,motion=lambda f:translation(250-f,80))
        self.assertFalse(self.review())
        self.assertNotIn(1,self.e.bank.members(0))

    def test_one_shared_word_cannot_merge_two_different_complete_marks(self):
        other=self.full.copy();other[25:]=0;other[17:22,45:95]=.9
        self.owner(0)
        self.owner(1,other,motion=lambda f:translation(250-f,80))
        self.assertFalse(self.review())

    def test_complementary_sam_fragments_rejoin_without_any_shape_overlap(self):
        first=self.full.copy();first[20:]=0
        second=self.full.copy();second[:20]=0
        self.owner(0,first)
        b,_,_=self.owner(1,second)
        self.review()
        row=self.e.reconciler.global_reviews[0]
        self.assertEqual(row['shape']['kind'],'unresolved')
        self.assertEqual(row['reason'],'registered_occurrence_trajectory')
        self.assertTrue(row['accepted'])
        self.assertEqual(b.alias_of,0)
        # A contrary independently measured path revokes the geometry link.
        for f in range(16,24):b.poses[f].matrix=translation(200+f,50)
        self.e.reconciler.refresh_aliases()
        self.assertNotIn(1,self.e.bank.members(0))
        self.assertIsNone(b.alias_of)

    def test_fragment_of_one_instance_can_join_template_with_other_instances(self):
        a,_,_=self.owner(0)
        from uav.template_mask.records import Instance
        self.e.instances[1]=Instance(1,0,0,poses={f:Pose(f,translation(250-f,80),
            .7,'measured',f,(),True) for f in range(24)})
        partial=self.full.copy();partial[:20]=0
        fragment,_,_=self.owner(1,partial)
        self.review()
        self.assertIn(1,self.e.bank.members(0))
        self.assertEqual(fragment.alias_of,a.id)
        self.assertIsNone(self.e.instances[1].alias_of)

    def test_shape_comparison_unavailable_does_not_block_registered_occurrence(self):
        self.owner(0)
        b,_,_=self.owner(1)
        with patch('uav.template_mask.global_identity.shape_agreement',
                   return_value=dict(kind='unresolved',reason='comparison_canvas_unavailable')):
            self.review()
        self.assertEqual(b.alias_of,0)

    def test_registered_gap_uses_fixed_coordinates_despite_missing_sam_strokes(self):
        from uav.template_mask.global_identity import occurrence_agreement
        first=self.full.copy();first[20:]=0
        second=self.full.copy();second[:20]=0
        a,_,_=self.owner(0,first,frames=range(24))
        b,_,_=self.owner(1,second,frames=range(27,51))
        relation=translation(0,15)
        for pose in b.poses.values():pose.matrix=pose.matrix@relation
        result=occurrence_agreement(self.e,a,b,relation)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['kind'],'continuation')
        self.assertFalse(occurrence_agreement(self.e,a,b,np.eye(3))['accepted'])
        a.poses[25]=Pose(25,translation(30,50),.7,'absent',reason='confirmed_disappearance')
        self.assertFalse(occurrence_agreement(self.e,a,b,relation)['accepted'])

    def test_global_fragment_link_repairs_gap_without_reading_images_or_adding_votes(self):
        first=self.full.copy();first[20:]=0
        second=self.full.copy();second[:20]=0
        a,_,_=self.owner(0,first,frames=range(24))
        b,_,_=self.owner(1,second,frames=range(27,51))
        before=sum(p.status=='measured' for i in (a,b) for p in i.poses.values())
        self.review()
        self.assertIn(1,self.e.bank.members(0))
        self.e.gaps.resolve(None)
        self.assertEqual(b.continuation_of,a.id)
        self.assertEqual(a.render_end,26)
        self.assertEqual(b.render_start,27)
        for f in range(24,27):
            self.assertEqual(a.poses[f].status,'predicted')
            self.assertIsNone(a.poses[f].source_id)
            self.assertFalse(a.poses[f].presence)
        self.assertEqual(sum(p.status=='measured' for i in (a,b) for p in i.poses.values()),before)

    def test_synchronous_but_disjoint_shapes_do_not_merge_without_registration(self):
        self.owner(0)
        self.owner(1)
        with patch.object(self.e,'_matches',return_value=[]):
            self.assertFalse(self.e.reconciler.global_review())
        self.assertFalse(self.e.bank.links)

    def test_missing_fragment_edge_is_not_a_conflicting_layout(self):
        self.owner(0)
        self.owner(1)
        self.owner(2)
        def proposal(e,a,b):
            if (a,b)==(1,2):return dict(a=a,b=b,accepted=False,reason='no_independent_registration')
            return dict(a=a,b=b,accepted=True,reason='registered_occurrence_trajectory',
                transform=np.eye(3).tolist(),sigma=.7,sources=[10,20],
                shape=dict(kind='equivalent'),trajectories=[])
        with patch('uav.template_mask.global_identity.evaluate_pair',side_effect=proposal):
            self.assertEqual(self.e.reconciler.global_review(),{0,1,2})
        self.assertEqual(set(self.e.bank.members(0)),{0,1,2})

    def test_implied_fragment_edge_still_rejects_foreign_strokes(self):
        self.owner(0)
        self.owner(1)
        self.owner(2)
        def proposal(e,a,b):
            if (a,b)==(1,2):return dict(a=a,b=b,accepted=False,reason='no_independent_registration')
            return dict(a=a,b=b,accepted=True,reason='registered_occurrence_trajectory',
                transform=(translation(200,0) if b==2 else np.eye(3)).tolist(),
                sigma=.7,sources=[10,20],shape=dict(kind='equivalent'),trajectories=[])
        with patch('uav.template_mask.global_identity.evaluate_pair',side_effect=proposal):
            self.assertFalse(self.e.reconciler.global_review())
        self.assertFalse(self.e.bank.links)

    def test_mixed_hypothesis_is_reconsidered_after_fine_shapes_and_global_tracks(self):
        self.owner(0)
        mixed,_,_=self.owner(1,reason='competing_templates')
        self.assertTrue(self.e.bank.templates[1].mixed)
        self.review()
        self.assertFalse(self.e.bank.templates[1].mixed)
        self.assertEqual(mixed.alias_of,0)
        self.e.bank.revoke(0,1,'later_contradiction')
        self.e.reconciler.refresh_aliases()
        self.assertTrue(self.e.bank.templates[1].mixed)
        self.assertIsNone(mixed.alias_of)

    def test_single_original_and_duplicate_candidates_do_not_certify_coordinates(self):
        self.owner(0,source=7)
        self.owner(1,source=7)
        self.assertFalse(self.review())
        self.assertFalse(self.e.bank.links)

    def test_crossing_agreement_in_middle_does_not_alias_independent_tracks(self):
        self.owner(0)
        b,_,_=self.owner(1,motion=lambda f:translation(30+f*.4+(f-12)*3,50+.01*f*f))
        self.review()
        self.assertIn(1,self.e.bank.members(0))
        self.assertIsNone(b.alias_of)

    def test_isolated_pose_noise_does_not_prevent_alias_but_sustained_change_does(self):
        self.owner(0)
        b,_,_=self.owner(1)
        b.poses[10].matrix=translation(90,80)
        self.review()
        self.assertEqual(b.alias_of,0)
        for f in range(16,24):b.poses[f].matrix=translation(90+f,80)
        self.e.reconciler.refresh_aliases()
        self.assertIsNone(b.alias_of)

    def test_predictions_do_not_prove_containment_identity(self):
        partial=self.full.copy();partial[20:]=0
        self.owner(0)
        b,_,_=self.owner(1,partial)
        for p in b.poses.values():p.status='predicted'
        self.assertFalse(self.review())

    def test_containment_link_revokes_when_later_motion_contradicts_layout(self):
        partial=self.full.copy();partial[20:]=0
        self.owner(0)
        b,_,_=self.owner(1,partial)
        self.review()
        self.assertEqual(b.alias_of,0)
        for f in range(16,24):b.poses[f].matrix=translation(200+f,50)
        self.e.reconciler.refresh_aliases()
        self.assertTrue(self.e.bank.links[(0,1)].revoked)
        self.assertNotIn(1,self.e.bank.members(0))
        self.assertIsNone(b.alias_of)

    def test_conflicting_transitive_groups_are_not_merged_by_iteration_order(self):
        for tid in range(3):self.owner(tid)
        matrix=np.eye(3).tolist()
        def evaluate(e,a,b):
            return dict(a=a,b=b,accepted=(a,b)!=(0,2),sources=[10,11],
                sigma=.7,transform=matrix,shape=dict(kind='equivalent'),trajectories=[])
        with patch('uav.template_mask.global_identity.evaluate_pair',side_effect=evaluate):
            self.assertFalse(self.e.reconciler.global_review())
        self.assertFalse(self.e.bank.links)

    def test_association_snapshot_checks_all_pairs_before_committing_links(self):
        from uav.template_mask.association import evaluate_associations
        for tid in range(3):self.owner(tid)
        records=[]
        def evaluate(e,a,b):
            self.assertFalse(e.bank.links)
            row=dict(a=a,b=b,accepted=True,sources=[10,11],sigma=.7,
                transform=translation(200,0).tolist() if (a,b)==(0,2) else np.eye(3).tolist(),
                shape=dict(kind='equivalent'),trajectories=[])
            records.append(row)
            return row
        with patch('uav.template_mask.global_identity.evaluate_pair',side_effect=evaluate):
            decisions=evaluate_associations(self.e)
        self.assertTrue(all(r['accepted'] for r in records))
        self.assertTrue(all(not r['accepted'] for r in decisions))
        self.assertFalse(self.e.bank.links)

    def test_merged_reference_probabilities_remain_source_deduplicated_and_bounded(self):
        a,_,_=self.owner(0)
        self.owner(1)
        self.review()
        fusion=self.e.bank.output_fusion(0)[0]
        self.assertEqual(len(fusion['sources']),2)
        self.assertLessEqual(len(self.e.bank.matching_references(0)),8)
        self.assertTrue(np.all(fusion['counts']<=2))

    def test_second_pass_resolves_before_refinement_and_allows_one_feedback(self):
        from types import SimpleNamespace
        e=self.e
        e.detector=SimpleNamespace(calls=0,task_calls={})
        events=[]
        with patch('uav.template_mask.refinement.refine_templates',side_effect=lambda *a,**kw:events.append('probe' if kw.get('probe') else 'refine')), \
             patch.object(e.reconciler,'global_review',side_effect=lambda:events.append('global') or set()), \
             patch.object(e,'_finalize_poses',side_effect=lambda:events.append('finalize')):
            e.second_pass(lambda f:self.image)
        self.assertEqual(events,['global','probe','global','refine','finalize'])

    def test_shifted_fragment_coordinates_register_without_moving_template_anchors(self):
        a,_,_=self.owner(0)
        # The same partial strokes use a different template origin. Their
        # image positions still agree through the explicit registration.
        partial=np.zeros_like(self.full);partial[25:34,4:90]=.9
        b,_,fine=self.owner(1,partial)
        relation=translation(0,10)
        for pose in b.poses.values():
            pose.matrix=pose.matrix@relation
        fine.pose=fine.pose@relation
        original_anchor=self.e.bank.templates[1].anchor
        def matching(template,key,**kwargs):
            obs=next(o for o in self.e.bank.observations.values() if o.feature_key==key)
            source=obs.template_id
            ref=self.e.bank.templates[template.id].references[0].observation_id
            fit=obs.pose@np.linalg.inv(relation) if template.id==0 and source==1 else obs.pose@relation
            return [Match(fit,.7,20,.8,(ref,))]
        # Native image pixels stay unchanged; only their template gauge moves.
        self.e.bank.templates[1].box=(4,15,90,24)
        self.e.bank.templates[1].version+=1
        self.e.bank.revision+=1
        with patch.object(self.e,'_matches',side_effect=matching):
            self.e.reconciler.global_review()
        np.testing.assert_allclose(self.e.bank.members(0)[1],relation)
        self.assertEqual(b.alias_of,a.id)
        self.assertEqual(self.e.bank.templates[1].anchor,original_anchor)

    def test_sparse_pose_jobs_only_include_unresolved_records(self):
        owner,_,_=self.owner(0)
        e=self.e
        e.source_ids={f:f for f in range(10000)}
        e.presence.appearance_checked.add(owner.id)
        owner.poses[8]=Pose(8,translation(30,50),2.,'uncertain')
        owner.poses[9]=Pose(9,translation(30,50),2.,'predicted')
        owner.poses[10]=Pose(10,translation(30,50),2.,'absent')
        e.presence.coverage[owner.id]={8,9,10}
        self.assertEqual(e._unresolved_pose_jobs(set()),[(8,owner.id)])
        self.assertEqual(e._unresolved_pose_jobs({0}),[(8,owner.id),(9,owner.id)])
        e.presence.edge_margins[(owner.id,8)]=7
        self.assertEqual(e._unresolved_pose_jobs(set()),[])

    def test_resolved_second_pass_does_not_read_timeline_images(self):
        from types import SimpleNamespace
        owner,_,_=self.owner(0)
        e=self.e
        e.source_ids={f:f for f in range(24)}
        e.detector=SimpleNamespace(calls=0,task_calls={})
        e.presence.appearance_checked.add(owner.id)
        with patch('uav.template_mask.refinement.refine_templates'), \
             patch.object(e.reconciler,'global_review',return_value=set()):
            e.second_pass(lambda f:self.fail('no unresolved image job exists'))
        self.assertEqual(e.stats['second_pass_pose_jobs'],0)

    def test_multiple_image_registrations_are_resolved_by_shape_and_complete_trajectory(self):
        self.owner(0)
        b,_,_=self.owner(1,reason='competing_templates')
        def alternatives(template,key,**kwargs):
            fit=self.matches(template,key,**kwargs)[0]
            wrong=Match(fit.matrix@translation(15,0),.7,20,.8,fit.references)
            return [wrong,fit]
        with patch.object(self.e,'_matches',side_effect=alternatives):
            self.e.reconciler.global_review()
        self.assertEqual(b.alias_of,0)
        report=self.e.reconciler.global_reviews[0]
        self.assertTrue(report['accepted'])
        self.assertGreaterEqual(len(report['alternatives']),2)
        self.assertEqual(sum(x['accepted'] for x in report['alternatives']),1)

    def test_partial_fine_detection_noise_needs_trajectory_not_strict_pixel_identity(self):
        self.owner(0)
        noisy=self.full.copy();noisy[20:]=0
        noisy[15:19,4:45]=.9
        b,_,_=self.owner(1,noisy)
        self.review()
        self.assertIn(1,self.e.bank.members(0))
        self.assertEqual(b.alias_of,0)

    def test_one_query_matching_many_references_is_not_multiple_fragment_views(self):
        from uav.template_mask.global_identity import registrations
        self.owner(0)
        partial=self.full.copy();partial[:20]=0
        self.owner(1,partial)
        with patch.object(self.e,'_matches',side_effect=self.matches):
            proposals=registrations(self.e,0,1)
        proposal=proposals[0]
        proposal['sources']={1,2,3,4,5}
        with patch('uav.template_mask.global_identity.registrations',return_value=[proposal]):
            row=evaluate_pair(self.e,0,1)
        self.assertFalse(row['accepted'])
        self.assertEqual(row['alternatives'][0]['reason'],'fragment_needs_two_registered_query_originals')

    def test_online_partial_registration_records_relation_without_sharing_pixels(self):
        self.owner(0)
        partial=self.full.copy();partial[:20]=0
        b,_,_=self.owner(1,partial)
        # Two different original references in the querying template. Both
        # match a common line, but no occurrence/layout review has run yet.
        template=self.e.bank.templates[1]
        ref=template.references[0]
        from dataclasses import replace
        obs=self.e.bank.observations[ref.observation_id]
        duplicate=replace(obs,id='second-original',source_id=201,feature_key='second-key')
        self.e.bank.observations[duplicate.id]=duplicate
        template.references.append(replace(ref,observation_id=duplicate.id,source_id=201,feature_key=duplicate.feature_key))
        with patch.object(self.e,'_matches',side_effect=self.matches):
            self.e.reconciler.reconcile(1)
        self.assertTrue(self.e.bank.links)
        self.assertTrue(all(not link.active for link in self.e.bank.links.values()))
        self.assertEqual(set(self.e.bank.members(0)),{0})
        self.assertIsNone(b.alias_of)

    def test_one_complete_occurrence_cannot_expand_all_independent_partial_occurrences(self):
        from uav.template_mask.records import Instance
        self.owner(0)
        partial=self.full.copy();partial[:20]=0
        b,_,_=self.owner(1,partial)
        self.e.instances[2]=Instance(2,1,0,poses={f:Pose(f,translation(250-f,80),
            .7,'measured',f,(),True) for f in range(24)})
        self.assertFalse(self.review())
        self.assertNotIn(1,self.e.bank.members(0))
        self.assertIsNone(b.alias_of)

    def test_duplicate_complete_tracks_are_one_candidate_for_partial_identity(self):
        from uav.template_mask.global_identity import occurrence_evidence
        a,_,_=self.owner(0)
        self.owner(1)
        partial=self.full.copy();partial[:20]=0
        b,_,_=self.owner(2,partial)
        self.e.bank.propose_link(0,1,np.eye(3),10)
        self.e.bank.propose_link(0,1,np.eye(3),11)
        result=occurrence_evidence(self.e,0,2,np.eye(3),.7,dict(kind='a_contains_b'))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['required_occurrences'],[('b',b.id)])

    def test_combined_duplicate_samples_do_not_certify_remote_partial(self):
        from uav.template_mask.global_identity import occurrence_evidence
        self.owner(0)
        self.owner(1)
        partial=self.full.copy();partial[:20]=0
        self.owner(2,partial,motion=lambda f:translation(300+f,50))
        self.e.bank.propose_link(0,1,np.eye(3),10)
        self.e.bank.propose_link(0,1,np.eye(3),11)
        result=occurrence_evidence(self.e,0,2,np.eye(3),.7,dict(kind='a_contains_b'))
        self.assertFalse(result['accepted'])

    def test_group_supported_partial_link_survives_alias_refresh(self):
        from uav.template_mask.global_identity import occurrence_evidence
        self.owner(0)
        self.owner(1)
        partial=self.full.copy();partial[:20]=0
        self.owner(2,partial)
        self.e.bank.propose_link(0,1,np.eye(3),10)
        self.e.bank.propose_link(0,1,np.eye(3),11)
        groups=(self.e.bank.members(0),self.e.bank.members(2))
        evidence=occurrence_evidence(self.e,0,2,np.eye(3),.7,dict(kind='a_contains_b'))
        self.assertTrue(evidence['accepted'])
        self.e.bank.propose_link(0,2,np.eye(3),20)
        self.e.bank.propose_link(0,2,np.eye(3),21)
        self.e.reconciler.global_relations=[dict(a=0,b=2,transform=np.eye(3).tolist(),
            sigma=.7,shape=dict(kind='a_contains_b'),occurrence_groups=groups,**evidence)]
        self.e.reconciler.refresh_aliases()
        self.assertTrue(self.e.bank.links[0,2].active)
        for f in range(16,24):self.e.instances[2].poses[f].matrix=translation(300+f,50)
        self.e.reconciler.refresh_aliases()
        self.assertTrue(self.e.bank.links[0,2].revoked)

    def test_lost_support_alone_does_not_revoke_accepted_identity(self):
        self.owner(0)
        partial=self.full.copy();partial[:20]=0
        b,_,_=self.owner(1,partial)
        self.review()
        self.assertTrue(self.e.bank.links[0,1].active)
        # A later audit can quarantine measurements without new contradictions.
        for p in b.poses.values():p.status='uncertain'
        self.e.reconciler.refresh_aliases()
        self.assertTrue(self.e.bank.links[0,1].active)

    def test_short_equivalent_mixed_track_uses_measured_overlap(self):
        from uav.template_mask.global_identity import occurrence_evidence
        self.owner(0)
        self.owner(1,reason='competing_templates',frames=range(8,14))
        result=occurrence_evidence(self.e,0,1,np.eye(3),.7,dict(kind='equivalent'))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reason'],'registered_equivalent_occurrence')
        for p in self.e.instances[1].poses.values():p.matrix=translation(300,50)
        result=occurrence_evidence(self.e,0,1,np.eye(3),.7,dict(kind='equivalent'))
        self.assertFalse(result['accepted'])

    def test_equivalent_query_bridges_adjacent_noncompeting_track_fragments(self):
        from uav.template_mask.global_identity import occurrence_evidence
        self.owner(0,frames=range(8,14))
        self.owner(1,frames=range(16,22))
        self.owner(2,reason='competing_templates',frames=range(8,22))
        self.e.bank.propose_link(0,1,np.eye(3),10)
        self.e.bank.propose_link(0,1,np.eye(3),11)
        result=occurrence_evidence(self.e,0,2,np.eye(3),.7,dict(kind='equivalent'))
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reason'],'registered_equivalent_occurrence')

    def test_online_equivalent_shapes_can_share_template_before_global_review(self):
        self.owner(0)
        self.owner(1)
        template=self.e.bank.templates[1]
        ref=template.references[0]
        from dataclasses import replace
        obs=self.e.bank.observations[ref.observation_id]
        duplicate=replace(obs,id='second-original',source_id=201,feature_key='second-key')
        self.e.bank.observations[duplicate.id]=duplicate
        template.references.append(replace(ref,observation_id=duplicate.id,source_id=201,feature_key=duplicate.feature_key))
        with patch.object(self.e,'_matches',side_effect=self.matches):
            self.e.reconciler.reconcile(1)
        self.assertIn(1,self.e.bank.members(0))

    def test_partial_reference_cannot_hide_extra_strokes_in_its_own_template(self):
        from dataclasses import replace
        partial=self.full.copy();partial[:20]=0
        self.owner(0,partial)
        self.owner(1)
        template=self.e.bank.templates[1]
        template.references=[]
        for source in (201,202):
            obs=self.e.observe(source,source,self.image,Candidate(partial,(30,50),
                (30,50,130,90),.9,str(source)),1008,'global')
            obs.pose=translation(30,50);obs.template_id=1;obs.instance_id=1;obs.reliable=True
            self.e.bank.add(obs)
        with patch.object(self.e,'_matches',side_effect=self.matches):
            self.e.reconciler.reconcile(1)
        self.assertNotIn(1,self.e.bank.members(0))


if __name__=='__main__':unittest.main()
