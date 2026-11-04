"""Shared RAM frames, tested with a synthetic decoder without media access."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from uav.template_mask.runner import VideoFrames


class Capture:
    def __init__(self,*args):
        self.position=0;self.seeks=[];self.grabs=0;self.reads=0;self.released=False
        self.thread_name=current_thread().name
    def get(self,key):return 30. if key==cv2.CAP_PROP_FPS else 100
    def isOpened(self):return True
    def set(self,key,value):self.position=value;self.seeks.append(value);return True
    def grab(self):self.position+=1;self.grabs+=1;return True
    def read(self):
        self.reads+=1
        image=np.full((4,5,3),self.position,np.uint8);self.position+=1
        return True,image
    def release(self):self.released=True


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.captures=[]
        def capture(*args):
            item=Capture();self.captures.append(item);return item
        self.patch=patch('cv2.VideoCapture',capture)
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_one_forward_decode_serves_arbitrary_repeated_requests(self):
        frames=VideoFrames('synthetic',24.,7,2)
        frames(frames.count-1)
        frames.thread.join(5)
        self.assertFalse(frames.thread.is_alive())
        cap=self.captures[0]
        self.assertEqual(cap.thread_name,'uav-frame-decode')
        self.assertEqual(cap.seeks,[7])
        self.assertTrue(cap.released)
        self.assertEqual(cap.reads,frames.count)
        self.assertEqual(cap.position,frames.end_source+1)
        self.assertGreater(cap.grabs,0)
        for f in (40,0,22,40,8,47,1):
            self.assertTrue(np.all(frames(f)==frames.source_id(f)))
            self.assertIs(frames(f),frames(f))
        self.assertEqual(len(self.captures),1)
        self.assertEqual(cap.reads,frames.count)
        frames.close()
        self.assertFalse(frames.images)
        frames.close()

    def test_random_interval_is_seeded_and_uses_single_decoder(self):
        starts=[]
        for _ in range(2):
            frames=VideoFrames('synthetic',24.,max_seconds=1,random_start=True,seed=31)
            try:
                starts.append(frames.start)
                self.assertGreaterEqual(frames.start,0)
                self.assertLessEqual(frames.start,70)
                self.assertEqual(frames.count,24)
                self.assertEqual(frames.source_id(0),frames.start)
                self.assertTrue(np.all(frames(0)==frames.start))
            finally:
                frames.close()
        self.assertEqual(starts[0],starts[1])
        self.assertEqual(len(self.captures),2)

    def test_bounded_producer_releases_completed_frames(self):
        frames=VideoFrames('synthetic',24.,buffer_limit=12)
        try:
            self.assertEqual(int(frames(11)[0,0,0]),frames.source_id(11))
            frames.release_before(8)
            self.assertEqual(int(frames(19)[0,0,0]),frames.source_id(19))
            self.assertLessEqual(len(frames.images),12)
            with self.assertRaises(IndexError):frames(0)
        finally:frames.close()

    def test_upsampling_shares_duplicates_and_accounts_unique_bytes(self):
        frames=VideoFrames('synthetic',60.,7,1)
        for f in range(frames.count):
            self.assertTrue(np.all(frames(f)==frames.source_id(f)))
        self.assertIs(frames(0),frames(1))
        self.assertIsNot(frames(1),frames(2))
        count=len(set(frames.sources))
        self.assertEqual(frames.decoded_frames,count)
        self.assertEqual(self.captures[0].reads,count)
        self.assertEqual(frames.bytes,count*4*5*3)
        self.assertGreaterEqual(frames.decode_seconds,0.)
        frames.close()

    def test_shared_images_are_read_only_and_parallel_requests_do_not_decode(self):
        frames=VideoFrames('synthetic',24.,7,2)
        with self.assertRaises(ValueError):frames(0)[0,0,0]=255
        requests=[40,0,22,8,47,1]*5
        with ThreadPoolExecutor(max_workers=2) as pool:
            images=list(pool.map(frames,requests))
        for f,image in zip(requests,images):
            self.assertIs(image,frames(f))
            self.assertFalse(image.flags.writeable)
            self.assertTrue(np.all(image==frames.source_id(f)))
        self.assertEqual(self.captures[0].reads,frames.count)
        self.assertEqual(len(self.captures),1)
        frames.close()

    def test_bounds_cannot_wrap_around_frame_table(self):
        frames=VideoFrames('synthetic',24.)
        for f in (-1,frames.count):
            with self.assertRaisesRegex(IndexError,'outside interval'):frames(f)
            with self.assertRaisesRegex(IndexError,'outside interval'):frames.source_id(f)
        frames.close()

    def test_last_output_sample_reads_actual_last_original_of_interval(self):
        frames=VideoFrames('synthetic',24.,0,1)
        self.assertEqual(frames.count,24)
        self.assertEqual(frames.source_id(22),27)
        self.assertEqual(frames.source_id(23),29)
        self.assertTrue(np.all(frames(23)==29))
        self.assertFalse(self.captures[0].seeks)
        frames.close()
        full=VideoFrames('synthetic',24.)
        self.assertEqual(full.source_id(full.count-1),99)
        self.assertTrue(np.all(full(full.count-1)==99))
        full.close()

    def test_single_sample_keeps_interval_start(self):
        frames=VideoFrames('synthetic',24.,7,.01)
        self.assertEqual(frames.count,1)
        self.assertEqual(frames.source_id(0),7)
        self.assertTrue(np.all(frames(0)==7))
        frames.close()

    def test_invalid_interval_and_decode_failures_release_capture(self):
        for start in (-1,100):
            with self.assertRaisesRegex(ValueError,'outside source'):
                VideoFrames('synthetic',24.,start)
            self.assertTrue(self.captures[-1].released)
        with patch.object(Capture,'read',return_value=(False,None)):
            frames=VideoFrames('synthetic',24.)
            with self.assertRaisesRegex(RuntimeError,'cannot decode'):
                frames(0)
            frames.close()
            self.assertTrue(self.captures[-1].released)
        with patch.object(Capture,'grab',return_value=False):
            frames=VideoFrames('synthetic',1.)
            with self.assertRaisesRegex(RuntimeError,'cannot skip'):
                frames(frames.count-1)
            frames.close()
            self.assertTrue(self.captures[-1].released)

    def test_consumers_start_before_decode_completes_and_share_waited_frame(self):
        blocked,release=Event(),Event()
        original=Capture.read
        def read(cap):
            if cap.position==10:
                blocked.set()
                if not release.wait(5):raise RuntimeError('test gate timed out')
            return original(cap)
        with patch.object(Capture,'read',read):
            frames=VideoFrames('synthetic',30.)
            try:
                self.assertTrue(blocked.wait(5))
                self.assertFalse(frames.done)
                self.assertEqual(frames(0)[0,0,0],0)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    try:
                        a,b=[pool.submit(frames,12) for _ in range(2)]
                        self.assertFalse(a.done());self.assertFalse(b.done())
                    finally:
                        release.set()
                    self.assertIs(a.result(5),b.result(5))
                frames(frames.count-1)
                frames.thread.join(5)
                report=frames.report()
                self.assertTrue(report['complete'])
                self.assertGreater(sum(report['consumer_wait_seconds'].values()),0.)
                self.assertEqual(len(self.captures),1)
            finally:
                release.set();frames.close()

    def test_decode_failure_preserves_earlier_frames_and_wakes_all_consumers(self):
        original=Capture.read
        def read(cap):
            if cap.position==10:return False,None
            return original(cap)
        with patch.object(Capture,'read',read):
            frames=VideoFrames('synthetic',30.)
            try:
                self.assertTrue(np.all(frames(5)==5))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=[pool.submit(frames,f) for f in (10,20)]
                    for result in results:
                        with self.assertRaisesRegex(RuntimeError,'cannot decode'):
                            result.result(5)
                self.assertFalse(frames.report()['complete'])
            finally:frames.close()
            self.assertTrue(self.captures[0].released)
            self.assertFalse(frames.thread.is_alive())

    def test_early_close_wakes_waiter_and_joins_decoder(self):
        blocked,release,closing=Event(),Event(),Event()
        original=Capture.read
        def read(cap):
            if cap.position==1:
                blocked.set()
                if not release.wait(5):raise RuntimeError('test gate timed out')
            return original(cap)
        with patch.object(Capture,'read',read):
            frames=VideoFrames('synthetic',30.)
            self.assertTrue(blocked.wait(5))
            original_close=frames.close
            def close():
                closing.set();original_close()
            with ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    waiter=pool.submit(frames,99)
                    closer=pool.submit(close)
                    self.assertTrue(closing.wait(5))
                    with self.assertRaisesRegex(RuntimeError,'closed'):waiter.result(5)
                finally:release.set()
                closer.result(5)
            self.assertFalse(frames.thread.is_alive())
            self.assertTrue(self.captures[0].released)
            self.assertFalse(frames.images)
            frames.close()


if __name__=='__main__':unittest.main()
