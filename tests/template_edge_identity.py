"""Frame clipping needs corroborating originals, never proximity alone."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from tests import template_candidate_selection as fixtures
from uav.template_mask.edge_identity import resolve_edges, edge_originals, independent_sources
from dataclasses import replace
from uav.template_mask.features import Match
from uav.template_mask.geometry import translation
from uav.template_mask.records import Pose
from uav.template_mask.edge_trajectory import corroborate, boundary_prediction
from uav.template_mask.edge_trajectory import match_visible_boundary
from uav.template_mask.edge_trajectory import prepare_boundary_output
from uav.template_mask.edge_trajectory import BoundaryEvidence, validated_occurrences


class EdgeIdentityTests(unittest.TestCase):
    def test_incompatible_coordinates_in_one_target_remain_unmerged(self):
        full,seed=self.owner(self.small,'whole',10)
        edge,obs=self.owner(self.small,'edge',20)
        obs.frame=8;obs.box=(40,180,100,200);obs.pose=translation(40,180)
        self.e.source_ids={8:20}
        self.e.bank.observations['independent']=SimpleNamespace(source_id=30,
            reliable=True,ambiguous=False,id='independent',template_id=0,mask_role='discovery')
        fits=[Match(translation(x,180),.7,20,.6,(seed.id,'independent')) for x in (40,80)]
        fusion=dict(mask=np.ones((40,60),np.float32),origin=np.array([0,0]))
        with (patch.object(self.e,'_matches',side_effect=lambda t,*a,**kw: fits if t.id==0 else []),
              patch.object(self.e,'_shape_evidence',return_value=(.95,.95,.5)),
              patch.object(self.e.bank,'output_fusion',return_value=(fusion,False)),
              patch('uav.template_mask.edge_identity.match_visible_boundary',return_value=[]),
              patch('uav.template_mask.global_identity.registrations',return_value=[])):
            self.assertFalse(resolve_edges(self.e,lambda f:self.fail('ambiguous match must not probe')))
        self.assertFalse(self.e.bank.links)
        self.assertTrue(any(c.get('rejection')=='ambiguous_template_coordinates'
                            for c in self.e.edge_identity_checks))

    def test_motion_support_is_not_an_independent_coordinate_measurement(self):
        self.owner(self.small,'whole',10)
        self.owner(self.small,'fragment',20)
        support=dict(instance=0,measured_sources=[21,22,23,24])
        link=self.e.bank.propose_link(0,1,translation(),20,trajectory_support=support)
        self.assertTrue(link.active)
        self.assertEqual(set(link.evidence),{20})
        self.assertEqual(link.trajectory_support[20],support)
        self.e.bank.propose_link(0,1,translation(50,0),25,trajectory_support=support)
        self.assertTrue(link.revoked)

    def test_motion_support_needs_four_independent_originals(self):
        self.owner(self.small,'whole',10)
        self.owner(self.small,'fragment',20)
        link=self.e.bank.propose_link(0,1,translation(),20,
            trajectory_support=dict(measured_sources=[20,21,22,23]))
        self.assertFalse(link.active)
        self.assertFalse(link.trajectory_support)

    def test_boundary_occurrence_is_revalidated_after_geometry_changes(self):
        full,seed=self.owner(self.small,'whole',10)
        fragment,obs=self.owner(self.small,'fragment',20)
        self.e.bank.propose_link(0,1,translation(),100)
        self.e.bank.propose_link(0,1,translation(),101)
        record=dict(fragment=fragment.id,instance=full.id,template=0,observation=obs.id,
            evidence=dict(measured_sources=list(range(8))))
        self.e.edge_occurrences=[record]
        current=dict(instance=full.id,measured_sources=list(range(8)))
        with patch('uav.template_mask.edge_trajectory.corroborate',return_value=current):
            self.assertEqual(len(tuple(validated_occurrences(self.e))),1)
        with patch('uav.template_mask.edge_trajectory.corroborate',return_value=None):
            self.assertEqual(tuple(validated_occurrences(self.e)),())
        with patch('uav.template_mask.edge_trajectory.corroborate',return_value=dict(current,instance=99)):
            self.assertEqual(tuple(validated_occurrences(self.e)),())
        self.e.bank.revoke(0,1,'contrary_evidence')
        with patch('uav.template_mask.edge_trajectory.corroborate') as verify:
            self.assertEqual(tuple(validated_occurrences(self.e)),())
            verify.assert_not_called()

    def test_boundary_queries_share_source_time_without_caching_across_reviews(self):
        template=SimpleNamespace(id=0)
        a=SimpleNamespace(frame=8,source_id=20)
        b=SimpleNamespace(frame=8,source_id=20)
        with patch('uav.template_mask.edge_trajectory.trajectory_candidates',return_value=iter([])) as query:
            evidence=BoundaryEvidence(self.e)
            self.assertEqual(evidence.query(template,a),())
            self.assertEqual(evidence.query(template,b),())
            self.assertEqual(query.call_count,1)
            BoundaryEvidence(self.e).query(template,a)
            self.assertEqual(query.call_count,2)

    def test_unmeasured_exit_output_moves_with_complete_path_without_new_votes(self):
        full,seed=self.owner(self.small,'whole',10)
        fragment,obs=self.owner(self.small,'fragment',20)
        self.e.bank.propose_link(0,1,translation(),100)
        self.e.bank.propose_link(0,1,translation(),101)
        full.poses={f:Pose(f,translation(40,180-2*f),.7,'measured',f,(),True) for f in range(8)}
        obs.frame=8;obs.source_id=8;obs.pose=translation(40,164)
        fragment.poses={8:Pose(8,obs.pose.copy(),.7,'measured',8,(),True),
            9:Pose(9,obs.pose.copy(),4.,'uncertain')}
        self.e.source_ids={f:f for f in range(10)}
        self.e.edge_occurrences=[dict(fragment=fragment.id,instance=full.id,template=0,
            observation=obs.id,evidence=dict(side='left',measured_sources=list(range(8))))]
        before=fragment.poses[9].matrix.copy()
        with patch('uav.template_mask.edge_trajectory.validated_occurrences',
                   return_value=iter(self.e.edge_occurrences)):
            prepare_boundary_output(self.e)
        from uav.template_mask.output_geometry import selected_matrix
        actual=selected_matrix(self.e,fragment,9,observation=None)
        np.testing.assert_allclose(actual,translation(40,162),atol=1e-6)
        np.testing.assert_array_equal(fragment.poses[9].matrix,before)
        self.assertIsNone(fragment.poses[9].source_id)
        self.e.bank.revoke(0,1,'contrary_evidence')
        prepare_boundary_output(self.e)
        self.assertNotIn(9,fragment.output_constraints)
        self.assertFalse(fragment.boundary_output_frames)

        # A boundary override must restore an existing general motion filter
        # when its association is withdrawn, rather than deleting that filter.
        previous=np.array([40.,161.,np.nan,np.nan])
        fragment.output_constraints[9]=previous.copy()
        with patch('uav.template_mask.edge_trajectory.validated_occurrences',
                   return_value=iter(self.e.edge_occurrences)):
            # Recreate only the transport snapshot; coordinates remain separate
            # from the intentionally revoked bank link in the real check above.
            self.e.bank.links[0,1].active=True
            prepare_boundary_output(self.e)
        self.e.bank.links[0,1].active=False
        prepare_boundary_output(self.e)
        np.testing.assert_array_equal(fragment.output_constraints[9],previous)

    def test_visible_boundary_match_uses_trajectory_when_sift_has_no_features(self):
        full,seed=self.owner(self.small,'whole',10)
        p=np.zeros((12,60),np.float32)
        p[2:9,8:20]=.9;p[3:10,32:48]=.9
        obs=self.e.observe(8,20,self.image,
            self.candidate(p,'edge'),1008,'global')
        obs.origin=(40,188);obs.box=(40,188,100,200)
        support=np.zeros((40,60),np.float32)
        support[:12]=p>.23;support[20:30,5:50]=1
        fusion=dict(mask=support,origin=np.array([0,0]))
        prediction=translation(40,188)
        evidence=dict(instance=full.id,source=21,measured_sources=list(range(21,29)),side='right')
        before=len(self.e.bank.observations)
        with (patch.object(self.e.bank,'output_fusion',return_value=(fusion,False)),
              patch('uav.template_mask.edge_trajectory.trajectory_candidates',
                    return_value=iter([(prediction,1.,evidence)])),
              patch.object(self.e,'_shape_evidence',return_value=(.98,.98,.3))):
            fits=match_visible_boundary(self.e,self.e.bank.templates[full.template_id],obs)
        self.assertEqual(len(fits),1)
        self.assertEqual(fits[0].inliers,0)
        self.assertEqual(fits[0].references,())
        np.testing.assert_allclose(fits[0].matrix,prediction)
        self.assertEqual(len(self.e.bank.observations),before)

    def test_short_boundary_fit_requires_holdout_consistency_and_nearby_time(self):
        template=self.e.bank.templates[self.owner(self.small,'whole',10)[0].template_id]
        rows=[Pose(f,translation(40,180-2*f),.7,'measured',f,(),True) for f in range(1,8)]
        result=boundary_prediction(self.e,rows,template,'right',0)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[0],translation(40,180),atol=1e-6)
        self.assertIsNone(boundary_prediction(self.e,rows,template,'right',-10))
        rows[3].matrix=translation(100,100)
        self.assertIsNone(boundary_prediction(self.e,rows,template,'right',0))

    def test_complete_future_path_can_verify_single_entry_original(self):
        full,seed=self.owner(self.small,'whole',10)
        edge,obs=self.owner(self.small,'edge',20)
        obs.frame=8;obs.source_id=20;obs.box=(40,180,100,200)
        full.poses={f:Pose(f,translation(40,180-2*(f-8)),.7,'measured',
            20+f-8,(seed.id,),True) for f in range(9,25)}
        fit=Match(translation(40,180),.7,20,.6,(seed.id,))
        fusion=dict(mask=np.ones((40,60),np.float32),origin=np.array([0,0]))
        with (patch.object(self.e.bank,'output_fusion',return_value=(fusion,False)),
              patch.object(self.e,'_shape_evidence',return_value=(.95,.95,.5))):
            result=corroborate(self.e,self.e.bank.templates[0],obs,fit)
            self.assertEqual(result['instance'],full.id)
            self.assertGreaterEqual(len(result['measured_sources']),8)
            wrong=replace(fit,matrix=translation(90,180))
            self.assertIsNone(corroborate(self.e,self.e.bank.templates[0],obs,wrong))
            full.poses={f:p for f,p in full.poses.items() if f<12}
            self.assertIsNone(corroborate(self.e,self.e.bank.templates[0],obs,fit))

    setUp=fixtures.SelectionTests.setUp
    tearDown=fixtures.SelectionTests.tearDown
    owner=fixtures.SelectionTests.owner
    candidate=fixtures.SelectionTests.candidate

    def test_edge_identity_needs_both_neighbouring_originals(self, extra_sources=False, slight=False):
        full,seed=self.owner(self.small,'whole',10)
        edge,obs=self.owner(self.small,'edge',20)
        top=164 if slight else 180
        obs.frame=8;obs.box=(40,top,100,200)
        obs.pose=translation(40,top)
        if extra_sources:
            for source in (22,23,24):
                duplicate=replace(obs,id=f'edge-{source}',source_id=source,frame=source)
                self.e.bank.observations[duplicate.id]=duplicate
                self.e.bank.templates[edge.template_id].observations.append(duplicate.id)
        self.e.source_ids={7:19,8:20,9:21}
        self.e.bank.observations['independent']=SimpleNamespace(source_id=30,reliable=True,ambiguous=False,id="independent",template_id=0,mask_role="discovery")
        fit=Match(translation(40,top),.7,20,.6,(seed.id,'independent'))
        fusion=dict(mask=np.ones((40,60),np.float32),origin=np.array([0,0]))
        def matches(template,key,**kw):
            return [fit] if template.id==full.template_id or key=='synthetic' else []
        with (patch.object(self.e,'_matches',side_effect=matches),
              patch.object(self.e,'_shape_evidence',return_value=(.95,.95,.5)),
              patch.object(self.e.bank,'output_fusion',return_value=(fusion,False)),
              patch.object(self.e.matcher,'extract',return_value='synthetic')):
            changed=resolve_edges(self.e,lambda f:self.image)
        self.assertIn(edge.template_id,changed)
        self.assertEqual(self.e.bank.links[0,1].kind,'verified_frame_clipped_identity')
        self.assertEqual(set(self.e.bank.links[0,1].evidence),{19,20,21})

    def test_long_lived_entry_fragment_is_still_verified(self):
        self.test_edge_identity_needs_both_neighbouring_originals(extra_sources=True)

    def test_slight_clipping_does_not_need_twenty_percent_missing(self):
        self.test_edge_identity_needs_both_neighbouring_originals(slight=True)

    def test_edge_sampling_is_bounded_and_deduplicates_originals(self):
        rows=[SimpleNamespace(source_id=f,frame=f,box=(40,180,100,200)) for f in range(40)]
        result=edge_originals(self.e,rows+rows)
        self.assertEqual(len(result),8)
        self.assertEqual((result[0].frame,result[-1].frame),(0,39))

    def test_missing_neighbour_does_not_create_identity(self):
        _,seed=self.owner(self.small,'whole',10)
        _,obs=self.owner(self.small,'edge',20)
        obs.frame=8;obs.box=(40,180,100,200);obs.pose=translation(40,180)
        self.e.source_ids={8:20}
        fit=Match(translation(40,180),.7,20,.6,(seed.id,))
        with patch.object(self.e,'_matches',return_value=[fit]):
            self.assertFalse(resolve_edges(self.e,lambda f:self.fail('no eligible probes')))
        self.assertFalse(self.e.bank.links)

    def test_query_frame_and_unreliable_references_cannot_corroborate(self):
        self.e.bank.observations.update({
            'same':SimpleNamespace(source_id=20,reliable=True,ambiguous=False),
            'good':SimpleNamespace(source_id=21,reliable=True,ambiguous=False),
            'bad':SimpleNamespace(source_id=22,reliable=False,ambiguous=False),
            'ambiguous':SimpleNamespace(source_id=23,reliable=True,ambiguous=True)})
        fit=Match(translation(),.7,20,.6,('same','good','bad','ambiguous','missing'))
        self.assertEqual(independent_sources(self.e,fit,20),{21})

    def test_member_observation_registers_group_origin_instead_of_fragment_origin(self):
        full,seed=self.owner(self.small,'whole',10)
        root,_=self.owner(self.small,'root',15)
        member,obs=self.owner(self.small,'edge',20)
        obs.frame=8;obs.box=(40,180,100,200);obs.pose=translation(40,180)
        self.e.bank.propose_link(root.template_id,member.template_id,translation(10,0),31)
        self.e.bank.propose_link(root.template_id,member.template_id,translation(10,0),32)
        self.e.source_ids={8:20,9:21}
        self.e.bank.observations['independent']=SimpleNamespace(source_id=30,reliable=True,ambiguous=False,id="independent",template_id=0,mask_role="discovery")
        whole=Match(translation(40,180),.7,20,.6,(seed.id,'independent'))
        fragment=Match(translation(30,180),.7,20,.6,(seed.id,'independent'))
        def matches(template,key,**kwargs):
            if template.id==full.template_id:return [whole]
            if template.id==root.template_id and key=='synthetic':return [fragment]
            return []
        fusion=dict(mask=np.ones((40,60),np.float32),origin=np.array([0,0]))
        with (patch.object(self.e,'_matches',side_effect=matches),
              patch.object(self.e,'_shape_evidence',return_value=(.95,.95,.5)),
              patch.object(self.e.bank,'output_fusion',return_value=(fusion,False)),
              patch.object(self.e.matcher,'extract',return_value='synthetic')):
            changed=resolve_edges(self.e,lambda f:self.image)
        self.assertIn(root.template_id,changed)
        np.testing.assert_allclose(self.e.bank.members(full.template_id)[root.template_id],translation(-10,0))


if __name__=='__main__':unittest.main()
