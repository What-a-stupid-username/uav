"""SIFT-first full-frame presence checks and evidence isolation in RAM."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from threading import RLock
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from uav.sam3_gpu_image import GpuImage

from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.features import Match
from uav.template_mask.geometry import translation
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate, Pose
from uav.template_mask.sam import SamDetector


class RecheckTests(unittest.TestCase):
    def test_cached_global_presence_uses_the_same_score_filter_as_fresh_inference(self):
        cache=EvidenceCache();self.addCleanup(cache.close)
        detector=SamDetector.__new__(SamDetector)
        detector.config,detector.cache=self.config,cache
        detector.calls=detector.cache_hits=0;detector.task_calls={}
        boxes=np.array([[10,20,30,40],[40,20,60,40]])
        detector._infer=Mock(return_value=(np.array([.9,.1]),boxes,None))
        image=np.zeros((80,160,3),np.uint8)
        expected=detector.detect_presence(image,1)
        key=cache.key('sam-probability-v4',2,(0,0,160,80),1008,False,'watermark',.3)
        cache.put(key,boxes=boxes,scores=np.array([.9,.1]))
        self.assertEqual(detector.detect_presence(image,2),expected)

    def setUp(self):
        self.config = TemplateConfig(sift_backend='cpu')

    def test_full_recheck_caches_boxes_only_and_reuses_scheduled_evidence(self):
        cache=EvidenceCache();self.addCleanup(cache.close)
        detector=SamDetector.__new__(SamDetector)
        detector.config,detector.cache=self.config,cache
        detector.calls=detector.cache_hits=0;detector.task_calls={}
        detector._infer=Mock(return_value=(np.array([.9]),np.array([[10,20,30,40]]),None))
        image=np.zeros((80,160,3),np.uint8)
        self.assertEqual(detector.detect_presence(image,1),[(10,20,30,40)])
        self.assertEqual(detector.detect_presence(image,1),[(10,20,30,40)])
        detector._infer.assert_called_once()
        self.assertTrue(detector._infer.call_args.kwargs['boxes_only'])
        self.assertTrue(all(set(data)=={'boxes'} for data in cache.memory.values()))
        global_key=cache.key('sam-probability-v4',2,(0,0,160,80),1008,False,'watermark',.3)
        cache.put(global_key,boxes=np.array([[20,30,40,50]]),scores=np.array([.9]))
        self.assertEqual(detector.detect_presence(image,2),[(20,30,40,50)])
        self.assertEqual(detector._infer.call_count,1)

    def test_presence_inference_skips_masks_without_changing_regular_tasks(self):
        detector=SamDetector.__new__(SamDetector)
        detector.model=object();detector.config=self.config
        detector.processors={};detector.text={}
        detector.inference_lock=RLock()
        detector.stream=None;detector.lock_wait_seconds=0.
        image=np.zeros((80,160,3),np.uint8)
        processor=SimpleNamespace(device='cpu')
        module=SimpleNamespace(Sam3Processor=Mock(return_value=processor))
        output=dict(scores=torch.tensor([.9]),boxes=torch.tensor([[10.,20.,30.,40.]]))
        with (patch.dict(sys.modules,{'sam3.model.sam3_image_processor':module}),
              patch('torch.autocast',return_value=nullcontext()),
              patch('torch.cuda.stream',return_value=nullcontext()),
              patch('uav.sam3_resolution.set_resolution'),
              patch('uav.sam3_crop_groups.set_detection_images',return_value={}),
              patch('uav.sam3_crop_groups.segment_group',return_value=output) as segment):
            scores,boxes,probabilities=detector._infer(image,1008,boxes_only=True)
            self.assertIsNone(probabilities)
            self.assertTrue(segment.call_args.args[-1]['boxes_only'])
            output['masks_logits']=torch.full((1,1,80,160),.8)
            full_scores,full_boxes,probabilities=detector._infer(image,1008)
            self.assertFalse(segment.call_args.args[-1]['boxes_only'])
            self.assertEqual(probabilities.shape,(1,80,160))
            np.testing.assert_array_equal(scores,full_scores)
            np.testing.assert_array_equal(boxes,full_boxes)

    def test_positive_sam_presence_never_constructs_template_or_mask_evidence(self):
        image=np.zeros((400,600,3),np.uint8)
        cache=EvidenceCache();self.addCleanup(cache.close)
        with tempfile.TemporaryDirectory() as folder:
            detector=SimpleNamespace(detect_presence=Mock(return_value=[(40,60,120,90)]))
            engine=TemplateEngine(self.config,cache,detector,
                WarningLog(Path(folder)/'warnings.jsonl'),image.shape[:2])
            seed=Candidate(np.full((30,80),.8,np.float32),(40,60),(40,60,120,90),.9,'seed')
            obs=engine.observe(8,8,image,seed,1008,'global')
            owner=engine._provisional(obs,'new_template')
            engine.source_ids={f:f for f in range(9)}
            owner.poses[7]=Pose(7,translation(40,60),1.,'uncertain')
            evidence=set(engine.bank.observations);revision=engine.bank.revision
            with patch.object(engine,'_track',return_value=None):
                self.assertTrue(engine.presence._probe(owner,7,lambda f:image))
                self.assertTrue(engine.presence._probe(owner,7,lambda f:image))
            self.assertEqual(detector.detect_presence.call_count,1)
            self.assertEqual(owner.poses[7].reason,'boundary_sam_presence')
            self.assertIsNone(owner.poses[7].source_id)
            self.assertEqual(set(engine.bank.observations),evidence)
            self.assertEqual(engine.bank.revision,revision)
            self.assertNotIn(7,engine.frame_observations)

    def test_edge_recheck_uses_full_frame(self):
        image=np.zeros((400,600,3),np.uint8)
        cache=EvidenceCache();self.addCleanup(cache.close)
        with tempfile.TemporaryDirectory() as folder:
            detector=SimpleNamespace(detect_presence=Mock(return_value=[]))
            engine=TemplateEngine(self.config,cache,detector,
                WarningLog(Path(folder)/'w.jsonl'),image.shape[:2])
            obs=engine.observe(0,0,image,Candidate(np.ones((10,20),np.float32),
                (40,390),(40,390,60,400),.9,'seed'),1008,'global')
            owner=engine._provisional(obs,'new_template')
            engine.source_ids[1]=1
            owner.poses[1]=Pose(1,translation(40,401),1.,'uncertain')
            with patch.object(engine,'_track',return_value=None):
                self.assertFalse(engine.presence._probe(owner,1,lambda f:image))
            self.assertIs(detector.detect_presence.call_args.args[0],image)
            self.assertEqual(detector.detect_presence.call_args.args[1],1)

    def test_present_end_search_stops_at_first_absence_in_both_directions(self):
        cache=EvidenceCache();self.addCleanup(cache.close)
        with tempfile.TemporaryDirectory() as folder:
            engine=TemplateEngine(self.config,cache,None,WarningLog(Path(folder)/'w.jsonl'),(400,600))
            owner=SimpleNamespace(id=0,template_id=0,poses={})
            engine.source_ids={f:f for f in range(9)}
            for appearance,positive,expected in ((True,{5,6,7},[7,6,5,4]),
                                                (False,{1,2,3},[1,2,3,4])):
                engine.presence.completed.clear();engine.presence.events.clear()
                engine.presence.edge_margins.clear()
                with (patch.object(engine.presence,'_probe',side_effect=lambda i,f,*a,**k:f in positive) as probe,
                     patch.object(engine.presence,'_accept_interval') as accept,
                     patch.object(engine.presence,'_absent')):
                    engine.presence._boundary(owner,0,8,not appearance,appearance,lambda f:None,'first_pass')
                self.assertEqual([c.args[1] for c in probe.call_args_list],expected)
                self.assertEqual(accept.call_args.args[1:3],(4,8) if appearance else (0,4))
                self.assertEqual(engine.presence.events[0]['method'],'from_present_end_first_absence')

    def test_sift_success_skips_sam_and_never_adds_template_observations(self):
        image=np.zeros((400,600,3),np.uint8)
        cache=EvidenceCache();self.addCleanup(cache.close)
        with tempfile.TemporaryDirectory() as folder:
            detector=SimpleNamespace(detect_presence=Mock(side_effect=AssertionError('SIFT passed')))
            engine=TemplateEngine(self.config,cache,detector,
                WarningLog(Path(folder)/'w.jsonl'),image.shape[:2])
            seed=Candidate(np.ones((30,80),np.float32),(40,60),(40,60,120,90),.9,'seed')
            obs=engine.observe(8,8,image,seed,1008,'global')
            owner=engine._provisional(obs,'new_template')
            engine.source_ids={f:f for f in range(9)}
            revision=engine.bank.revision
            fit=Match(translation(40,60),.7,12,.9,(obs.id,))
            with (patch.object(engine,'_prediction',return_value=(translation(40,60),1.,True)),
                  patch.object(engine,'_matches',return_value=[fit])):
                self.assertTrue(engine.presence._probe(owner,7,lambda f:image))
            self.assertEqual(owner.poses[7].status,'measured')
            self.assertEqual(engine.bank.revision,revision)
            self.assertEqual(len(engine.bank.observations),1)
            self.assertEqual(engine.presence.probe_diagnostics[-1]['result'],'sift_present')


if __name__ == '__main__':
    unittest.main()
