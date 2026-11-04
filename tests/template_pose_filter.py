"""Output constraints never rewrite raw image or template evidence."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.geometry import components, discrepancy, from_components, translation
from uav.template_mask.motion import MotionPrior, evidence_signature
from uav.template_mask.pose_filter import output_matrix, prepare_output_filter
from uav.template_mask.records import Instance, Pose, Template
from uav.template_mask.rendering import MaskRenderer
from uav.template_mask.review import trajectory_segments, final_marks


class OutputFilterTests(unittest.TestCase):
    def engine(self, motion, count=96):
        template=Template(0,(0,0,120,40),anchor=(60,20))
        owner=Instance(0,0,0,poses={f:Pose(f,from_components(motion(f),template.anchor),
            .7,'measured',f,('reference',),True) for f in range(count)})
        owner.motion=MotionPrior(owner.poses.values(),template.box,template.anchor)
        return SimpleNamespace(instances={0:owner},active_instances=lambda:[owner],
            bank=SimpleNamespace(templates={0:template},members=lambda tid:{0:np.eye(3)},
                observations={'first':SimpleNamespace(source_id=-1),
                              'second':SimpleNamespace(source_id=-2)}),
            _renders_at=lambda i,f:True,associated_observations=lambda i,f:[]),owner,template

    def test_static_jitter_filters_output_without_teaching_it_back_to_evidence(self):
        e,owner,t=self.engine(lambda f:(100+.3*(-1)**f,200+.2*(-1)**f,
                                       .003*(-1)**f,.002*(-1)**f))
        before=evidence_signature(owner.poses.values(),t.box,t.anchor)
        prepare_output_filter(e)
        output=np.array([components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
                         for f,p in owner.poses.items()])
        np.testing.assert_allclose(np.ptp(output,axis=0),0,atol=1e-12)
        self.assertEqual(evidence_signature(owner.poses.values(),t.box,t.anchor),before)
        self.assertEqual(len(owner.motion.evidence),96)
        self.assertEqual(e.output_pose_filter[0]['frames'],96)
        self.assertTrue(all(p.status=='measured' and p.presence for p in owner.poses.values()))
        paths=trajectory_segments(e)[0][0]
        np.testing.assert_allclose(np.ptp(np.array(paths)[:,1:],axis=0),0,atol=1e-12)
        expected=output_matrix(owner,0,owner.poses[0].matrix,t.anchor)
        from uav.template_mask.geometry import bounds
        np.testing.assert_allclose(final_marks(e,0)[0]['box'],bounds(expected,t.box))

    def test_linear_curve_rotation_and_periodic_scale_keep_their_motion(self):
        motions=[lambda f:(100+f,200+.02*f*f,.02*f,0),
                 lambda f:(100+f,200+f,.01*f,.05*np.sin(2*np.pi*f/32))]
        for motion in motions:
            e,owner,t=self.engine(motion)
            prepare_output_filter(e)
            for f,p in owner.poses.items():
                actual=components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
                np.testing.assert_allclose(actual[:2],motion(f)[:2],atol=.8)
                self.assertLess(abs(np.angle(np.exp(1j*(actual[2]-motion(f)[2])))),.012)
                self.assertLess(abs(actual[3]-motion(f)[3]),.012)

    def test_shallow_diagonal_and_slow_curve_do_not_become_static_stairs(self):
        motions=[lambda f:(100+3*f,200-.35*f,0,0),
                 lambda f:(100+3*f,200+.0008*f*f,0,0),
                 lambda f:(100+f,200+3*np.sin(2*np.pi*f/160),0,0)]
        for motion in motions:
            with self.subTest(motion=motion):
                e,owner,t=self.engine(motion,count=160)
                prepare_output_filter(e)
                for f,p in owner.poses.items():
                    actual=components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
                    np.testing.assert_allclose(actual[:2],motion(f)[:2],atol=.8)
                if motion(159)[1]!=motion(0)[1]:
                    actual=np.array([components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
                                     for f,p in owner.poses.items()])
                    self.assertGreater(abs(actual[-1,1]-actual[0,1]),
                                       abs(motion(159)[1]-motion(0)[1])*.95)

    def test_separate_tiny_static_windows_cannot_hide_long_slow_drift(self):
        from uav.template_mask.motion import ComponentModel
        e,owner,t=self.engine(lambda f:(100+3*f,200+.05*f,0,0),count=160)
        # Even if every eight-frame search approximation accepts "static",
        # their combined record contains a clear coherent direction.
        owner.motion.regimes[1]=[dict(model=ComponentModel('static',np.array([200+.05*(start+3.5)]),
            .7,start,7),start=start,support_end=start+7,support_sources=8)
            for start in range(0,160,8)]
        prepare_output_filter(e)
        for f,p in owner.poses.items():
            actual=components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
            self.assertAlmostEqual(actual[1],200+.05*f)

    def test_noisy_long_diagonal_is_filtered_along_its_slope(self):
        motion=lambda f:(100+2.4*f,200-.35*f,0,0)
        e,owner,t=self.engine(lambda f:np.array(motion(f))+np.array(
            [.3*(-1)**f,.25*(-1)**f,.002*(-1)**f,.001*(-1)**f]),count=160)
        before={f:p.matrix.copy() for f,p in owner.poses.items()}
        prepare_output_filter(e)
        actual=np.array([components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
                         for f,p in owner.poses.items()])
        steps=np.diff(actual[:,:2],axis=0)
        np.testing.assert_allclose(steps,np.broadcast_to([2.4,-.35],steps.shape),atol=.002)
        modes=[r['modes'] for r in e.output_pose_filter[0]['intervals'] if r['components']==['x','y']]
        self.assertEqual(modes,[['linear','linear']])
        for f,p in owner.poses.items():np.testing.assert_array_equal(p.matrix,before[f])

    def test_output_reuses_existing_models_without_search_or_new_motion_evidence(self):
        e,owner,t=self.engine(lambda f:(100+2*f+.2*(-1)**f,200-.4*f,0,0))
        models=[tuple(segment[0]) for segment in owner.motion.segments]
        signature=evidence_signature(owner.poses.values(),t.box,t.anchor)
        with (patch('uav.template_mask.motion.fit_pose_component',side_effect=AssertionError('no model search')),
              patch('uav.template_mask.motion.fit_component',side_effect=AssertionError('no model search'))):
            prepare_output_filter(e)
        self.assertTrue(owner.output_constraints)
        self.assertEqual(signature,evidence_signature(owner.poses.values(),t.box,t.anchor))
        for segment,original in zip(owner.motion.segments,models):
            self.assertTrue(all(a is b for a,b in zip(segment[0],original)))

    def test_polyline_legs_keep_their_directions_and_corner(self):
        def motion(f):
            return ((100+3*f,200+2*f,0,0) if f<80 else
                    (340+2*(f-80),360-.35*(f-80),0,0))
        e,owner,t=self.engine(lambda f:np.array(motion(f))+[.2*(-1)**f,.15*(-1)**f,0,0],count=160)
        prepare_output_filter(e)
        output=np.array([components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
                         for f,p in owner.poses.items()])
        self.assertLess(np.max(np.linalg.norm(output[:,:2]-np.array([motion(f)[:2] for f in range(160)]),axis=1)),1.5)
        for start,end,slope in ((20,60,2/3),(100,140,-.35/2)):
            delta=output[end,:2]-output[start,:2]
            self.assertAlmostEqual(delta[1]/delta[0],slope,places=3)

    def test_fixed_horizontal_axis_stays_lockable_on_another_moving_axis(self):
        e,owner,t=self.engine(lambda f:(100+3*f,200+.2*(-1)**f,0,0))
        prepare_output_filter(e)
        for f,p in owner.poses.items():
            actual=components(output_matrix(owner,f,p.matrix,t.anchor),t.anchor)
            self.assertAlmostEqual(actual[0],100+3*f)
            self.assertAlmostEqual(actual[1],200,places=6)

    def test_real_step_and_ongoing_departure_are_not_flattened(self):
        e,owner,t=self.engine(lambda f:(100 if f<48 else 140,200,0,0))
        prepare_output_filter(e)
        for f in (0,47,48,49,55,95):
            actual=output_matrix(owner,f,owner.poses[f].matrix,t.anchor)
            self.assertAlmostEqual(components(actual,t.anchor)[0],100 if f<48 else 140)
        e,owner,t=self.engine(lambda f:(140 if f>=94 else 100,200,0,0))
        prepare_output_filter(e)
        for f in (94,95):
            self.assertTrue(np.isnan(owner.output_constraints[f][0]))
            self.assertAlmostEqual(components(output_matrix(owner,f,owner.poses[f].matrix,t.anchor),t.anchor)[0],140)

    def test_resolved_single_reference_jitter_is_locked_but_corroborated_excursion_is_kept(self):
        e,owner,t=self.engine(lambda f:(105 if f in (40,41) else 100,200,0,0))
        owner.poses[41].references=('first','second')
        prepare_output_filter(e)
        self.assertAlmostEqual(components(output_matrix(owner,40,owner.poses[40].matrix,t.anchor),t.anchor)[0],100)
        self.assertAlmostEqual(components(output_matrix(owner,41,owner.poses[41].matrix,t.anchor),t.anchor)[0],105)

    def test_sparse_duplicates_and_predictions_cannot_establish_lock(self):
        e,owner,t=self.engine(lambda f:(100,200,0,0),count=7)
        for f in range(7,30):
            owner.poses[f]=Pose(f,translation(100,200),.1,'predicted',presence=True)
        owner.poses[30]=Pose(30,owner.poses[0].matrix.copy(),.1,'measured',0,presence=True)
        owner.motion=MotionPrior(owner.poses.values(),t.box,t.anchor)
        prepare_output_filter(e)
        self.assertFalse(owner.output_constraints)
        self.assertFalse(e.output_pose_filter)

    def test_lock_does_not_extend_past_measured_support_or_into_absence(self):
        e,owner,t=self.engine(lambda f:(100,200,0,0))
        owner.poses[-1]=Pose(-1,translation(99,200),1,'predicted')
        owner.poses[96]=Pose(96,translation(101,200),1,'predicted')
        owner.poses[20].status='absent'
        prepare_output_filter(e)
        self.assertTrue(all(f not in owner.output_constraints for f in (-1,20,96)))


class PaddingTests(unittest.TestCase):
    def renderer(self):
        return MaskRenderer(SimpleNamespace(config=SimpleNamespace(dilation=4)))

    def test_filter_displacement_adds_one_bounded_margin_after_normal_dilation(self):
        renderer=self.renderer()
        mask=np.zeros((80,100),bool);mask[30:40,40:60]=True
        plain={'dilation_radii':[]};small={'dilation_radii':[]}
        large={'dilation_radii':[]};zero={'dilation_radii':[]}
        baseline=renderer.dilate(mask,plain)
        padded=renderer.dilate(mask,small,pose_correction=1.1)
        renderer.dilate(mask,large,pose_correction=1000.)
        unchanged=renderer.dilate(mask,zero,pose_correction=1e-10)
        self.assertEqual(plain['dilation_radii'],[4])
        self.assertEqual(small['dilation_radii'],[6])
        self.assertEqual(large['dilation_radii'],[6])
        self.assertTrue(np.all(padded[baseline]))
        np.testing.assert_array_equal(unchanged,baseline)
        self.assertNotIn('pose_filter_padding',zero)

    def test_mask_uses_same_filtered_pose_even_on_direct_observation_frame(self):
        from uav.template_mask.geometry import warp
        t=Template(0,(0,0,20,10),anchor=(10,5))
        pose=Pose(0,translation(30,30),.7,'measured',0,presence=True)
        owner=Instance(0,0,0,poses={0:pose},output_constraints={0:np.array([42.,35.,np.nan,np.nan])})
        observed=SimpleNamespace(pose=translation(29,30),reliable=True,trajectory_evidence=None,id='raw')
        config=SimpleNamespace(dilation=4,uncertain_mask_policy='conservative')
        engine=SimpleNamespace(config=config,shape=(80,100),
            presence=SimpleNamespace(output_margin=lambda i,f:False),
            associated_observations=lambda i,f:[observed],bank=SimpleNamespace(templates={0:t}))
        renderer=MaskRenderer(engine)
        mask=np.ones((10,20),bool)
        renderer.fusion=lambda i,final:(dict(mask=mask,origin=(0,0),sources=[0]),False)
        metrics={'dilation_radii':[]};result=renderer.occurrence(owner,0,True,metrics,False)
        expected=warp(mask,translation(32,30),engine.shape,nearest=True)>0
        np.testing.assert_array_equal(result.raw,expected)
        self.assertEqual(metrics['pose_filter_padding'][0]['extra_radius'],2)
        np.testing.assert_array_equal(pose.matrix,translation(30,30))

    def test_verified_boundary_output_outside_image_is_not_localization_failure(self):
        t=Template(0,(0,0,20,10),anchor=(10,5))
        owner=Instance(0,0,0,poses={0:Pose(0,translation(30,-30),2.,'uncertain')})
        owner.boundary_output_frames={0}
        engine=SimpleNamespace(config=SimpleNamespace(dilation=4,uncertain_mask_policy='conservative'),
            shape=(80,100),
            presence=SimpleNamespace(output_margin=lambda i,f:False),
            associated_observations=lambda i,f:[],bank=SimpleNamespace(templates={0:t}))
        renderer=MaskRenderer(engine)
        renderer.fusion=lambda i,final:(dict(mask=np.ones((10,20),bool),origin=(0,0),
            sources=[0],has_support=True),False)
        result=renderer.occurrence(owner,0,True,{'dilation_radii':[]},False)
        self.assertFalse(result.mask.any())

    def test_additional_margin_never_exceeds_four_source_pixels_or_accumulates(self):
        renderer=self.renderer()
        mask=np.zeros((400,500),bool);mask[100:300,100:400]=True
        for multiplier in (1,2):
            base={'dilation_radii':[]}
            renderer.dilate(mask,base,multiplier=multiplier)
            outputs=[]
            for _ in range(3):
                metrics={'dilation_radii':[]}
                outputs.append(renderer.dilate(mask,metrics,multiplier=multiplier,
                                               pose_correction=10000.))
                expected=min(4,int(np.ceil(base['dilation_radii'][0]*.5)))
                self.assertEqual(metrics['pose_filter_padding'][0]['extra_radius'],expected)
                self.assertEqual(metrics['dilation_radii'][0]-base['dilation_radii'][0],expected)
                self.assertLessEqual(expected,4)
            np.testing.assert_array_equal(outputs[0],outputs[-1])


if __name__=='__main__':unittest.main()
