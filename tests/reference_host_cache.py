"""Run-local host cache avoids re-encoding identical reference observations."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from uav.propainter_references import ReferenceBank


class ReferenceHostTests(unittest.TestCase):
    def test_reference_batch_keeps_frame_order_and_owns_each_entry(self):
        bank=ReferenceBank(24,count=6)
        for i in range(6):bank.add(i,torch.full((3,8,8),i,dtype=torch.uint8),torch.zeros(8,8,dtype=torch.bool))
        encode=Mock(side_effect=lambda image,*_:image.clone())
        backend=SimpleNamespace(flow_crop=(0,0,8,8,0,0),model=SimpleNamespace(encode_frames=encode))
        actual=bank._encode(backend,[4,1,5,0])
        self.assertEqual(encode.call_count,1)
        for idx,value in actual.items():
            expected=(bank.frames[idx][None,None].half()/127.5-1)
            torch.testing.assert_close(value,expected,rtol=0,atol=0)
            self.assertEqual(value.untyped_storage().nbytes(),value.numel()*value.element_size())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    @torch.inference_mode()
    def test_async_eviction_restores_on_another_stream(self):
        bank=ReferenceBank(24,count=1,capacity_bytes=0,host_capacity_bytes=1024*1024)
        producer=torch.cuda.Stream();consumer=torch.cuda.Stream()
        crop=(0,0,16,16,0,0)
        with torch.cuda.stream(producer):
            entry=torch.arange(4096,device='cuda',dtype=torch.float32).reshape(1,1,16,16,16)
            bank._retain_host((crop,0),entry)
        self.assertTrue(bank.host[(crop,0)].is_pinned())
        bank.frames[0]=torch.empty(0,device='cuda')
        bank.select=lambda *args:[0]
        backend=SimpleNamespace(flow_crop=crop)
        with torch.cuda.stream(consumer):actual=bank.get(backend,0,1,1)
        consumer.synchronize()
        torch.testing.assert_close(actual.cpu(),torch.arange(4096,dtype=torch.float32).reshape_as(actual),rtol=0,atol=0)
        bank.clear();self.assertFalse(bank.host_ready)

    def test_eviction_restore_prune_and_clear(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            encode=Mock(side_effect=lambda crop,*_:crop.clone())
            backend=SimpleNamespace(flow_crop=(0,0,8,8,0,0),model=SimpleNamespace(encode_frames=encode))
            size=3*8*8*2
            bank=ReferenceBank(24,count=1,capacity_bytes=size,host_capacity_bytes=2*size)
            for i in range(4):bank.add(i,torch.full((3,8,8),i,device=device,dtype=torch.uint8),torch.zeros(8,8,device=device,dtype=torch.bool))
            def get(i):
                bank.select=lambda *args:[i]
                return bank.get(backend,0,1,4)
            expected=get(0).clone();get(1)
            self.assertEqual(encode.call_count,2)
            torch.testing.assert_close(get(0),expected,rtol=0,atol=0)
            self.assertEqual(encode.call_count,2);self.assertEqual(bank.host_hits,1)
            get(2);get(3)
            self.assertLessEqual(bank.host_bytes,2*size)
            bank.prune(3)
            self.assertTrue(all(key[1]>=3 for key in bank.host))
            bank.clear()
            self.assertEqual(bank.host_bytes,0);self.assertFalse(bank.host)

    def test_disabled_host_cache_recomputes(self):
        bank=ReferenceBank(24,count=1,capacity_bytes=4,host_capacity_bytes=0)
        bank._retain_host(('crop',0),torch.ones(2))
        self.assertFalse(bank.host)
        with self.assertRaises(ValueError):ReferenceBank(24,host_capacity_bytes=-1)
