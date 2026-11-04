"""General evidence rules: search hints are not output motion measurements."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.gaps import review_measurements
from uav.template_mask.geometry import components, discrepancy, from_components, translation
from uav.template_mask.motion import MotionPrior
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Instance, Pose, Template


class MeasurementReviewTests(unittest.TestCase):
    def make_engine(self, motion, count=65, box=(0,0,120,40)):
        template = Template(0,box,anchor=(box[2]/2,box[3]/2))
        rows = {f:Pose(f,from_components(motion(f),template.anchor),.7,'measured',f,
                       ('ref',),True) for f in range(count)}
        owner = Instance(0,0,0,poses=rows)
        e = SimpleNamespace(active_instances=lambda:[owner],
            bank=SimpleNamespace(templates={0:template},observations={
                'ref':SimpleNamespace(source_id=-1), 'other':SimpleNamespace(source_id=-2)}),
            associated_observations=lambda i,f:[],pose_reviews=[])
        return e,owner,template

    def test_isolated_noise_is_reviewed_on_static_linear_and_accelerating_paths(self):
        for motion in (lambda f:(100,200,0,0),lambda f:(100+f,200-.5*f,0,0),
                       lambda f:(100+.025*f*f,200+.3*f,0,0)):
            with self.subTest(motion=motion):
                e,owner,t=self.make_engine(motion)
                value=np.array(motion(32),float)+[24,-10,.12,.03]
                original=owner.poses[32]
                original.matrix=from_components(value,t.anchor)
                original_matrix=original.matrix.copy()
                changed=review_measurements(e)
                self.assertEqual(changed,{0})
                reviewed=owner.poses[32]
                self.assertEqual(reviewed.status,'predicted')
                self.assertFalse(reviewed.presence)
                self.assertIsNone(reviewed.source_id)
                self.assertLess(discrepancy(reviewed.matrix,from_components(motion(32),t.anchor),t.box),3.)
                np.testing.assert_array_equal(original.matrix,original_matrix)
                self.assertEqual(len(e.pose_reviews),1)
                prior=MotionPrior(owner.poses.values(),t.box,t.anchor)
                self.assertEqual(len(prior.evidence),64)

    def test_real_turn_and_persistent_motion_change_are_not_removed(self):
        motions=[lambda f:(100+f,200 if f<32 else 200+2*(f-32),0,0),
                 lambda f:(100 if f<32 else 150,200,0,0),
                 lambda f:(100+f,200+.015*f*f,.01*f,0)]
        for motion in motions:
            e,owner,t=self.make_engine(motion)
            before={f:p.matrix.copy() for f,p in owner.poses.items()}
            self.assertEqual(review_measurements(e),set())
            self.assertFalse(e.pose_reviews)
            for frame,p in owner.poses.items():
                np.testing.assert_array_equal(p.matrix,before[frame])

    def test_independent_image_references_preserve_even_an_irregular_real_excursion(self):
        e,owner,t=self.make_engine(lambda f:(100,200,0,0))
        original=owner.poses[32]
        original.matrix=translation(160,200)
        original.references=('ref','other')
        self.assertEqual(review_measurements(e),set())
        self.assertIs(owner.poses[32],original)

    def test_sparse_or_one_sided_history_cannot_classify_measurement_as_noise(self):
        for count,at in ((15,7),(65,2),(65,63)):
            e,owner,t=self.make_engine(lambda f:(100,200,0,0),count=count)
            original=owner.poses[at]
            original.matrix=translation(160,200)
            self.assertEqual(review_measurements(e),set())
            self.assertIs(owner.poses[at],original)

    def test_uncertainty_normalizes_decision_for_object_size_and_localization_accuracy(self):
        for scale in (1,4):
            e,owner,t=self.make_engine(lambda f:(100*scale,200*scale,0,0),
                box=(0,0,120*scale,40*scale))
            for p in owner.poses.values():p.uncertainty*=scale
            owner.poses[32].matrix=from_components((102*scale,200*scale,0,0),t.anchor)
            self.assertEqual(review_measurements(e),set())


class PredictionPurposeTests(unittest.TestCase):
    def test_validated_cross_instance_bridge_survives_fragment_hold_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=EvidenceCache()
            e=TemplateEngine(TemplateConfig(sift_backend='cpu'),cache,None,
                WarningLog(Path(folder)/'warnings.jsonl'),(800,800))
            template=Template(0,(0,0,120,40),anchor=(60,20))
            e.bank.templates[0]=template
            owner=Instance(0,0,0,poses={f:Pose(f,translation(100,200-f),.7,
                'measured',f,(),True) for f in range(20)})
            owner.motion=SimpleNamespace(evidence=list(owner.poses.values()),
                predict=lambda f:(translation(100,181),50.,False))
            e.instances[0]=owner
            e.gaps.output_predictions={(0,21):(translation(100,179),1.2,True)}
            result=e.coverage_prediction(owner,21)
            np.testing.assert_array_equal(result[0],translation(100,179))
            self.assertEqual(result[1:],(1.2,True))
            result[0][1,2]=0
            self.assertEqual(e.gaps.output_predictions[0,21][0][1,2],179)
            self.assertNotIn(21,owner.poses)
            # Rebuilding with no valid bridges must withdraw the old estimate.
            with patch.object(e.gaps,'_check',return_value=None):e.gaps.resolve(None)
            self.assertEqual(e.gaps.output_predictions,{})
            self.assertFalse(e.coverage_prediction(owner,21)[2])
            cache.close()

    def test_interior_miss_uses_independent_sides_in_source_time_without_new_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=EvidenceCache()
            e=TemplateEngine(TemplateConfig(sift_backend='cpu'),cache,None,
                WarningLog(Path(folder)/'warnings.jsonl'),(800,800))
            template=Template(0,(0,0,120,40),anchor=(60,20))
            e.bank.templates[0]=template
            e.source_ids={f:int(f*1.25) for f in range(40)}
            poses={f:Pose(f,translation(100+3*e.source_ids[f],200),.7,
                'measured',e.source_ids[f],('ref',),True) for f in range(40) if f!=20}
            owner=Instance(0,0,0,poses=poses)
            owner.motion=SimpleNamespace(evidence=list(poses.values()),
                predict=lambda f:(translation(0,0),50.,False))
            before=len(poses)
            result=e.coverage_prediction(owner,20)
            self.assertTrue(result[2])
            np.testing.assert_allclose(result[0],translation(175,200),atol=.01)
            self.assertEqual(len(poses),before)
            self.assertNotIn(20,poses)
            cache.close()

    def test_sparse_jitter_velocity_is_only_a_search_hint_not_coverage_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=EvidenceCache()
            e=TemplateEngine(TemplateConfig(sift_backend='cpu'),cache,None,WarningLog(Path(folder)/'warnings.jsonl'),(800,800))
            image=np.zeros((800,800,3),np.uint8)
            seed=e.observe(232,987,image,Candidate(np.full((400,120),.8,np.float32),
                (500,50),(500,50,620,450),.9,'seed'),1008,'global')
            owner=e._provisional(seed,'new_template')
            anchor=e.bank.templates[0].anchor
            value=components(translation(500,50),anchor)
            owner.poses={f:Pose(f,from_components(value+delta,anchor),.7,'measured',f,('ref',),True)
                for f,delta in ((229,np.zeros(4)),(231,np.array([15,0,.075,.01])),(232,np.zeros(4)))}
            e._motion(owner)
            search=e._prediction(owner,225)
            with patch.object(e,'_prediction',side_effect=AssertionError('search leaked into coverage')):
                coverage=e.coverage_prediction(owner,225)
                e.presence.coverage[owner.id]={225}
                e.presence.project(owner,225,for_output=True)
            self.assertFalse(coverage[2])
            np.testing.assert_allclose(coverage[0],translation(500,50))
            self.assertLess(coverage[1],search[1])
            np.testing.assert_array_equal(owner.poses[225].matrix,coverage[0])
            cache.close()


if __name__=='__main__':
    unittest.main()
