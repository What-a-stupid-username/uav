"""Bounded discovery producer, tested with synthetic arrays and thread events."""
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.prefetch import DiscoveryBuffer, prefetch_discoveries
from uav.template_mask.sam import SamDetector


class BufferTests(unittest.TestCase):
    def test_ordered_take_releases_capacity_without_producing_extra_keys(self):
        second=Event();third=Event()
        def produce(value):
            if value==1:second.set()
            if value==2:third.set()
            return value*10
        with DiscoveryBuffer(((i,i) for i in range(4)),produce,capacity=2) as buffer:
            self.assertTrue(second.wait(5))
            self.assertFalse(third.is_set())
            self.assertEqual(buffer.take(0),0)
            self.assertTrue(third.wait(5))
            with self.assertRaises(ValueError):buffer.take(3)
            self.assertEqual([buffer.take(i) for i in (1,2,3)],[10,20,30])
        self.assertFalse(buffer.thread.is_alive())
        self.assertFalse(buffer.ready)
        self.assertEqual(buffer.report()['consumed'],4)
        self.assertLessEqual(buffer.report()['peak_slots'],2)

    def test_failure_is_forwarded_after_earlier_results(self):
        failure=RuntimeError('synthetic detector failure')
        def produce(value):
            if value==1:raise failure
            return value
        with DiscoveryBuffer(((i,i) for i in range(3)),produce) as buffer:
            self.assertEqual(buffer.take(0),0)
            with self.assertRaisesRegex(RuntimeError,'synthetic detector failure'):buffer.take(1)

    def test_close_joins_a_producer_waiting_on_full_buffer(self):
        produced=Event()
        def produce(value):produced.set();return value
        with DiscoveryBuffer(((i,i) for i in range(100)),produce,capacity=1) as buffer:
            self.assertTrue(produced.wait(5))
        self.assertFalse(buffer.thread.is_alive())
        self.assertEqual(buffer.produced,1)

    def test_prepare_is_detached_until_consumer_commits(self):
        cache=EvidenceCache();self.addCleanup(cache.close)
        detector=SamDetector.__new__(SamDetector)
        detector.config=TemplateConfig(sift_backend='cpu')
        detector.cache=cache;detector.calls=detector.cache_hits=0
        detector.task_calls={'global_coarse':0}
        p=np.full((1,80,160),.8,np.float32)
        detector._infer=Mock(return_value=(np.array([.9]),np.array([[10,20,30,40]]),p))
        image=np.zeros((80,160,3),np.uint8)
        prepared=detector.prepare_global(image,8)
        self.assertFalse(cache.memory)
        self.assertEqual(detector.calls,0)
        detector.accept_global(prepared)
        self.assertEqual(detector.calls,1)
        actual=detector.detect(image,8,None,1008)
        self.assertEqual(len(actual),1)
        detector._infer.assert_called_once()
        self.assertEqual(detector.calls,1)

    def test_context_shares_frames_and_removes_runtime_on_error(self):
        reader=Mock()
        reader.side_effect=lambda frame:np.full((2,3,3),frame,np.uint8)
        reader.source_id.side_effect=lambda frame:frame
        frames=reader
        frames.count=17
        detector=SimpleNamespace(prepare_global=lambda image,source:(source,int(image[0,0,0])))
        engine=SimpleNamespace(detector=detector,config=TemplateConfig())
        with self.assertRaisesRegex(RuntimeError,'consumer'):
            with prefetch_discoveries(engine,frames):
                self.assertEqual(engine.discovery_buffer.take(0),(0,0))
                self.assertEqual(engine.discovery_buffer.take(8),(8,8))
                raise RuntimeError('consumer')
        self.assertIsNone(engine.discovery_buffer)
        reader.fork.assert_not_called()
        reader.close.assert_not_called()
        self.assertEqual([c.args[0] for c in reader.call_args_list[:2]],[0,8])
        self.assertEqual(engine.discovery_prefetch['consumed'],2)

    def test_duplicate_source_at_scheduled_frame_is_consumed_without_dropping_a_check(self):
        frames=Mock(side_effect=lambda f:np.zeros((2,3,3),np.uint8))
        frames.count=5
        frames.source_id.side_effect=lambda f:f//3
        engine=SimpleNamespace(config=TemplateConfig(global_stride=2),
            detector=SimpleNamespace(prepare_global=lambda image,source:source))
        with prefetch_discoveries(engine,frames):
            self.assertEqual(engine.discovery_buffer.take(0),0)
            self.assertEqual(engine.discovery_buffer.take(2),0)
            self.assertEqual(engine.discovery_buffer.take(4),1)
        self.assertEqual(engine.discovery_prefetch['produced'],3)

    def test_last_frame_not_on_stride_is_prefetched_once(self):
        frames=Mock(side_effect=lambda f:np.zeros((2,3,3),np.uint8))
        frames.count=20
        frames.source_id.side_effect=lambda f:f
        engine=SimpleNamespace(config=TemplateConfig(global_stride=8),
            detector=SimpleNamespace(prepare_global=lambda image,source:source))
        with prefetch_discoveries(engine,frames):
            self.assertEqual([engine.discovery_buffer.take(f) for f in (0,8,16,19)],[0,8,16,19])
        self.assertEqual(engine.discovery_prefetch['consumed'],4)


if __name__=='__main__':unittest.main()
