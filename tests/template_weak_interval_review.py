"""Motion can request image review; it cannot certify its own predictions."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import numpy as np
from uav.template_mask.geometry import translation
from uav.template_mask.records import Instance, Pose, Template, Reference
from uav.template_mask.features import Match
from uav.template_mask.pose_review import review_weak_intervals, review_matches


class WeakIntervalReviewTests(unittest.TestCase):
    def test_failed_review_supplements_nearby_sources_without_mutating_references(self):
        e,owner,_,_=self.engine()
        template=e.bank.templates[0]
        for source in range(1,8):
            oid='view-'+str(source)
            e.bank.observations[oid]=SimpleNamespace(id=oid,source_id=source,
                reliable=True,ambiguous=False,pose=translation(),crossing_instances=(),
                mask_role='discovery',feature_key=oid)
            template.observations.append(oid)
        retained=[Reference('view-1',1,'view-1',1.)]
        e.bank.matching_references=lambda *a,**kw:retained
        e._matches.side_effect=None;e._matches.return_value=[]
        e.matcher.template_matches=Mock(return_value=[])
        review_matches(e,template,'target',7)
        used=e.matcher.template_matches.call_args.kwargs['references']
        self.assertEqual([r.source_id for r in used],[1,6,5,4,3])
        self.assertEqual(len(retained),1)
        self.assertEqual(template.references,[])
        self.assertEqual(len(e.bank.observations),9)

    def engine(self, reverse=False, accelerating=False):
        sources={f:100+int(f*1.25) for f in range(30)}
        def matrix(f):
            t=sources[f]-100
            return translation(100+t+(t*t*.01 if accelerating else 0),150)
        weak=range(27,30) if reverse else range(3)
        poses={f:Pose(f,translation(70,150) if f in weak else matrix(f),.7,
            'measured',sources[f],('a',) if f in weak else ('a','b'),True) for f in range(30)}
        owner=Instance(0,0,0,poses=poses)
        observations={r:SimpleNamespace(source_id=s,reliable=True,ambiguous=False)
                      for r,s in (('a',1),('b',2))}
        e=SimpleNamespace(active_instances=lambda:[owner],instances={0:owner},
            bank=SimpleNamespace(templates={0:Template(0,(0,0,120,40))},
                observations=observations,members=lambda t:{0:np.eye(3)},
                matching_references=lambda *a,**kw:[]),
            config=SimpleNamespace(global_stride=8,min_inliers=6),shape=(600,600),
            associated_observations=lambda i,f:[],presence=SimpleNamespace(output_margin=lambda i,f:False),
            matcher=SimpleNamespace(extract=Mock(side_effect=lambda image,source,roi:source)),
            pose_reviews=[],motion_fits=None)
        e._matches=Mock(side_effect=lambda t,key,**kw:[Match(matrix(next(f for f,s in sources.items()
                if s==key)),.7,20,.7,('a','b'))])
        return e,owner,matrix,list(weak)

    def test_start_and_end_weak_runs_require_new_images_in_native_time(self):
        for reverse in (False,True):
            for accelerating in (False,True):
                e,owner,matrix,weak=self.engine(reverse,accelerating)
                image_at=Mock(return_value=None)
                self.assertEqual(review_weak_intervals(e,image_at),{0})
                for f in weak:
                    np.testing.assert_allclose(owner.poses[f].matrix,matrix(f))
                    self.assertEqual(owner.poses[f].status,'measured')
                self.assertEqual(image_at.call_count,len(weak))
                self.assertEqual(len(e.pose_reviews),len(weak))

    def test_motion_alone_or_one_reference_never_replaces_measurements(self):
        for references in ((),('a',)):
            e,owner,_,weak=self.engine()
            old={f:owner.poses[f] for f in weak}
            e._matches.return_value=[]
            e._matches.side_effect=lambda *a,**kw: [] if not references else [Match(translation(100,150),.7,20,.7,references)]
            self.assertEqual(review_weak_intervals(e,lambda f:None),set())
            for f in weak:self.assertIs(owner.poses[f],old[f])

    def test_independently_supported_turn_and_competing_locations_veto_replacement(self):
        for competing in (False,True):
            e,owner,matrix,weak=self.engine()
            def matches(t,key,**kw):
                fits=[Match(translation(70,150),.7,20,.7,('a','b'))]
                if competing:fits.append(Match(translation(100,150),.7,20,.7,('a','b')))
                return fits
            e._matches.side_effect=matches
            self.assertEqual(review_weak_intervals(e,lambda f:None),set())
            self.assertFalse(e.pose_reviews)
            # Rechecking only around the prediction would hide the old
            # location, biasing the comparison against a real turn.
            for call in e.matcher.extract.call_args_list:
                roi=call.args[2]
                self.assertLessEqual(roi[0],70)
                self.assertGreaterEqual(roi[2],220)

    def test_contaminated_sparse_history_and_actual_sam_observations_are_preserved(self):
        for mode in ('sparse','ambiguous','observed','margin'):
            e,owner,_,weak=self.engine()
            if mode=='sparse':
                for f in range(5,30):owner.poses[f].references=('a',)
            elif mode=='ambiguous':owner.ambiguous_frames.update(weak)
            elif mode=='observed':e.associated_observations=lambda i,f:[object()]
            else:e.presence.output_margin=lambda i,f:True
            self.assertEqual(review_weak_intervals(e,lambda f:None),set())
            e.matcher.extract.assert_not_called()

    def test_separate_occurrence_cannot_be_stolen_by_the_motion_prior(self):
        e,owner,matrix,weak=self.engine()
        other=Instance(1,0,0,poses={f:Pose(f,matrix(f),.7,'measured',
            owner.poses[f].source_id,('a','b'),True) for f in weak})
        e.instances[1]=other
        self.assertEqual(review_weak_intervals(e,lambda f:None),set())
        self.assertFalse(e.pose_reviews)

    def test_disagreeing_temporal_sides_do_not_request_image_work(self):
        e,owner,matrix,_=self.engine()
        for f,p in owner.poses.items():
            p.references=('a',) if 12<=f<=14 else ('a','b')
        with patch('uav.template_mask.pose_review.boundary_prediction',
                   side_effect=lambda e,rows,t,side,source:
                       (translation(100 if side=='left' else 200,150),.7,True)):
            self.assertEqual(review_weak_intervals(e,lambda f:None),set())
        e.matcher.extract.assert_not_called()

    def test_short_validated_history_only_reviews_adjacent_sources(self):
        e,owner,matrix,weak=self.engine()
        owner.poses={f:p for f,p in owner.poses.items() if f<7}
        review_weak_intervals(e,lambda f:None)
        self.assertEqual(e.measurement_review_stats['accepted'],2)
        self.assertEqual(owner.poses[0].references,('a',))
        for f in (1,2):np.testing.assert_allclose(owner.poses[f].matrix,matrix(f))


if __name__=='__main__':unittest.main()
