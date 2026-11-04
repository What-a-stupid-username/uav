"""Whole-lifetime negative evidence and bounded output use synthetic arrays only."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.geometry import components, from_components, translation
from uav.template_mask.motion import MotionPrior, evidence_signature
from uav.template_mask.policy import WarningLog, uncertain_mask, MAX_UNCERTAINTY_PADDING
from uav.template_mask.records import Instance, Pose, Template, Candidate
from uav.template_mask.trajectory_quality import assess, prepare_trajectory_quality


class QualityTests(unittest.TestCase):
    def track(self, values, references=('ref',)):
        template = Template(0, (0, 0, 120, 40), (60, 20))
        owner = Instance(0, 0, 0, poses={f:Pose(f,from_components(v,template.anchor),
            .7,'measured',f,references,True) for f,v in enumerate(values)})
        owner.motion = MotionPrior(owner.poses.values(),template.box,template.anchor)
        observations = {ref:SimpleNamespace(source_id=-1-i) for i,ref in enumerate(references)}
        return owner,template,observations

    def chaotic(self, count=96):
        rng = np.random.default_rng(435)
        return np.column_stack((rng.uniform(0,600,(count,2)),np.zeros((count,2))))

    def test_persistent_single_reference_chaos_restricts_inference_without_rewriting_evidence(self):
        owner,t,observations=self.track(self.chaotic())
        signature=evidence_signature(owner.poses.values(),t.box,t.anchor)
        e=SimpleNamespace(instances={0:owner},active_instances=lambda:[owner],config=TemplateConfig(sift_backend='cpu'),
            bank=SimpleNamespace(templates={0:t},observations=observations),timings={})
        with patch('uav.template_mask.motion.fit_component',side_effect=AssertionError('no extra fits')):
            prepare_trajectory_quality(e)
        self.assertEqual(owner.trajectory_quality['status'],'suspected_scene')
        self.assertEqual(owner.trajectory_quality['action'],'observed_only')
        self.assertEqual(signature,evidence_signature(owner.poses.values(),t.box,t.anchor))

    def test_independent_reference_sources_preserve_irregular_motion(self):
        owner,t,observations=self.track(self.chaotic(),('a','b'))
        result=assess(owner,t,observations,8)
        self.assertEqual(result['status'],'image_supported_irregular')
        self.assertEqual(result['action'],'retain')
        observations['b'].source_id=observations['a'].source_id
        self.assertEqual(assess(owner,t,observations,8)['status'],'suspected_scene')

    def test_current_source_and_duplicate_reference_ids_do_not_corroborate(self):
        owner,t,observations=self.track(self.chaotic(),('a','b'))
        for p in owner.poses.values():
            ref=f'current-{p.frame}'
            observations[ref]=SimpleNamespace(source_id=p.source_id)
            p.references=('a', 'a', ref)
        self.assertEqual(assess(owner,t,observations,8)['status'],'suspected_scene')

    def test_static_curve_polyline_and_small_jitter_are_preserved(self):
        motions=[lambda f:(100+.3*(-1)**f,200,0,0),
            lambda f:(100+f,200+.04*f*f,.02*f,.03*np.sin(f/16)),
            lambda f:(100+3*min(f,48),200+2*max(0,f-48),0,0)]
        for motion in motions:
            with self.subTest(motion=motion):
                owner,t,observations=self.track([motion(f) for f in range(96)])
                self.assertEqual(assess(owner,t,observations,8)['action'],'retain')

    def test_chaotic_rotation_or_scale_is_not_hidden_by_static_position(self):
        for component in (2,3):
            values=np.zeros((96,4));values[:,:2]=[100,200]
            values[:,component]=np.random.default_rng(98).uniform(-.7,.7,96)
            owner,t,observations=self.track(values)
            self.assertEqual(assess(owner,t,observations,8)['status'],'suspected_scene')

    def test_short_sparse_and_duplicate_sources_do_not_become_background_evidence(self):
        owner,t,observations=self.track(self.chaotic(7))
        for f in range(7,96):
            owner.poses[f]=Pose(f,translation(f*100,0),.1,'predicted',f,presence=True)
        owner.motion=MotionPrior(owner.poses.values(),t.box,t.anchor)
        self.assertEqual(assess(owner,t,observations,8)['status'],'insufficient')
        owner,t,observations=self.track(self.chaotic())
        for p in owner.poses.values():p.source_id=0
        owner.motion=MotionPrior(owner.poses.values(),t.box,t.anchor)
        self.assertEqual(assess(owner,t,observations,8)['status'],'insufficient')
        for f,p in list(owner.poses.items()):p.frame=f*16;p.source_id=f
        owner.motion=MotionPrior(owner.poses.values(),t.box,t.anchor)
        self.assertEqual(assess(owner,t,observations,8)['status'],'insufficient')

    def test_one_bad_section_does_not_reject_entire_object(self):
        values=np.zeros((96,4));values[:,:2]=[100,200]
        values[32:64]=self.chaotic(32)
        owner,t,observations=self.track(values)
        self.assertEqual(assess(owner,t,observations,8)['action'],'retain')


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.cache=EvidenceCache()
        self.e=TemplateEngine(TemplateConfig(sift_backend='cpu'),self.cache,None,
            WarningLog(Path(self.temp.name)/'warnings.jsonl'),(100,180))
        t=Template(0,(0,0,20,10));self.e.bank.templates[0]=t
        self.owner=Instance(0,0,0,poses={f:Pose(f,translation(30+f,40),.7,
            'measured',f,presence=True) for f in range(8)})
        self.e.instances[0]=self.owner
        self.e._motion(self.owner)

    def tearDown(self):
        self.cache.close();self.temp.cleanup()

    def test_invalid_model_uses_nearby_measured_pose_and_motion_not_old_residual(self):
        with patch.object(self.owner.motion,'predict',return_value=(translation(3000,4000),196.,False)):
            matrix,sigma,strong=self.e.coverage_prediction(self.owner,8)
        np.testing.assert_array_equal(matrix,translation(37,40))
        self.assertLess(sigma,2.)
        self.assertFalse(strong)
        self.assertEqual(len(self.owner.motion.evidence),8)

    def test_validated_motion_retains_geometry_and_sigma(self):
        matrix,sigma,strong=self.e.coverage_prediction(self.owner,8)
        np.testing.assert_allclose(matrix,translation(38,40),atol=1e-6)
        self.assertTrue(strong)

    def test_large_uncertainty_has_a_visible_fixed_upper_bound(self):
        mask=np.zeros(self.e.shape,bool);mask[40:50,30:50]=True
        expanded=uncertain_mask(mask,196.,self.owner,8,self.e.config,self.e.warnings,'bad_model')
        y,x=np.nonzero(expanded)
        self.assertEqual(x.min(),30-MAX_UNCERTAINTY_PADDING)
        self.assertEqual(x.max(),49+MAX_UNCERTAINTY_PADDING)
        self.assertEqual(self.e.warnings.rows[-1]['action'],'cover_bounded_uncertainty')

    def test_render_metrics_show_requested_and_applied_radius_and_debug_reason(self):
        from uav.template_mask.review import uncertainty_banner
        metrics=dict(uncertainty=[],dilation_radii=[])
        mask=np.zeros(self.e.shape,bool);mask[40:50,30:50]=True
        layer=self.e.renderer.finish(mask,self.owner,8,True,metrics,False,
                                     reason='bad_model',sigma=196.)
        self.assertTrue(layer.conservative)
        self.assertEqual(metrics['uncertainty'][0]['requested_radius'],588)
        self.assertEqual(metrics['uncertainty'][0]['extra_radius'],12)
        self.assertIn('extra=12-12px',uncertainty_banner(metrics)[0])
        self.assertEqual(uncertainty_banner(dict(trajectory_restrictions=[{}])),
                         ['SUSPECTED SCENE / OBSERVED PIXELS ONLY'])

    def test_boundary_sam_existence_survives_final_pose_projection_without_new_votes(self):
        self.e.source_ids[8]=8
        self.e.presence.coverage[0]={8}
        self.owner.poses[8]=Pose(8,translation(99,90),196.,'uncertain',
                               presence=True,reason='boundary_sam_presence')
        count=len(self.owner.motion.evidence)
        with patch.object(self.owner.motion,'predict',return_value=(translation(3000,4000),196.,False)):
            self.e.presence.project(self.owner,8,for_output=True)
        p=self.owner.poses[8]
        self.assertTrue(p.presence);self.assertEqual(p.reason,'boundary_sam_presence')
        self.assertIsNone(p.source_id);self.assertEqual(len(self.owner.motion.evidence),count)
        np.testing.assert_array_equal(p.matrix,translation(37,40))

    def test_failed_sift_retry_does_not_erase_independent_sam_presence(self):
        self.e.source_ids[8]=8
        self.e.presence.coverage[0]={8}
        self.owner.poses[8]=Pose(8,translation(37,40),9.,'uncertain',
                               presence=True,reason='boundary_sam_presence')
        image=np.zeros((*self.e.shape,3),np.uint8)
        with (patch.object(self.e.matcher,'extract',return_value='synthetic-features'),
              patch.object(self.e,'_matches',return_value=[])):
            self.e._track(self.owner,8,8,image,backward=True,image_only=True,boundary=True)
        self.e.presence.project(self.owner,8,for_output=True)
        self.assertTrue(self.owner.poses[8].presence)
        self.assertEqual(self.owner.poses[8].reason,'boundary_sam_presence')
        self.assertIsNone(self.owner.poses[8].source_id)
        self.assertEqual(len(self.owner.motion.evidence),8)

    def test_suspected_scene_preserves_detection_pixels_but_never_propagates_template(self):
        e=self.e;image=np.zeros((*e.shape,3),np.uint8)
        obs=e.observe(0,0,image,Candidate(np.ones((10,20),np.float32),(30,40),
            (30,40,50,50),.9,'raw'),1008,'global')
        obs.instance_id=0;obs.template_id=0;obs.pose=translation(30,40)
        obs.reliable=obs.ownership_confirmed=True
        self.owner.trajectory_quality=dict(status='suspected_scene',action='observed_only')
        with patch.object(e.bank,'output_fusion',side_effect=AssertionError('no propagation')):
            self.assertTrue(e.render(0).any())
            self.assertFalse(e.render(1).any())
        self.assertTrue(e.render_metrics[1]['trajectory_restrictions'])


if __name__=='__main__':unittest.main()
