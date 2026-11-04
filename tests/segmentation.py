"""Synthetic segment planning, RGB blending and bounded decode tests."""
import unittest
from unittest.mock import Mock,patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
from uav.segments import plan_segments,LinearBlendWriter,SegmentWriter,FrameSlice
from uav import UAVConfig,run_batch


class SegmentTests(unittest.TestCase):
    def test_absorb_up_to_five_seconds_of_new_tail_frames(self):
        for seconds, expected in ((15,[(0,15)]),(20,[(0,20)]),
                                  (21,[(0,15),(14,21)]),
                                  (30,[(0,15),(14,30)]),
                                  (34,[(0,15),(14,34)]),
                                  (35,[(0,15),(14,29),(28,35)])):
            segments=plan_segments(seconds*24,360,24,120)
            self.assertEqual([(s.start/24,s.stop/24) for s in segments],expected)
        # One frame above the extension limit must remain a separate segment.
        segments=plan_segments(481,360,24,120)
        self.assertEqual([(s.start,s.stop) for s in segments],[(0,360),(336,481)])
        with self.assertRaises(ValueError):plan_segments(10,8,2,-1)

    def test_segment_lengths_coverage_and_short_tail(self):
        for count in (1,359,360,361,672,673,720,2401):
            segments=plan_segments(count,360,24)
            self.assertEqual(segments[0].start,0)
            self.assertEqual(segments[-1].stop,count)
            for a,b in zip(segments,segments[1:]):self.assertEqual(a.stop-b.start,24)
            self.assertTrue(all(0<s.stop-s.start<=360 for s in segments))

    def test_lerp_and_one_output_per_frame_without_encoded_reads(self):
        segments=plan_segments(14,8,3)
        frames=[];writer=Mock();writer.save.side_effect=lambda f:frames.append(f.clone())
        blend=LinearBlendWriter(writer,segments)
        for i,s in enumerate(segments):
            adapter=SegmentWriter(blend,i)
            for f in range(s.stop-s.start):adapter.save(torch.full((2,3,3),i*100,dtype=torch.uint8))
            adapter.close()
        blend.close()
        self.assertEqual(len(frames),14)
        self.assertEqual(blend.blended_frames,6)
        self.assertEqual([int(f[0,0,0]) for f in frames],[0]*6+[50,100,100,100,100,150,200,200])
        writer.close.assert_called_once()

    def test_slice_preserves_global_noninteger_source_timestamps(self):
        class Frames:
            fps=24.;source_fps=29.97
            def source_id(self,f):return int(f*self.source_fps/self.fps)
            def __call__(self,f):return self.source_id(f)
        frames=Frames();segments=plan_segments(750,360,24)
        a,b=[FrameSlice(frames,s) for s in segments[:2]]
        for f in range(24):self.assertEqual(a.source_id(336+f),b.source_id(f))

    def test_batch_continues_after_error_and_uses_distinct_output_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.mp4';b=root/'b.mp4';a.touch();b.touch()
            sessions=[]
            def session():
                s=Mock();sessions.append(s);return s
            with (patch('uav.settings.validate_models'),
                  patch('uav.model_session.ModelSession',side_effect=session),
                  patch('uav.api.run',side_effect=[RuntimeError('bad'),dict(complete=True,skipped=False)]) as run):
                report=run_batch([a,b],root/'out')
            self.assertTrue(report['complete']);self.assertEqual(report['successful'],1)
            self.assertEqual(run.call_count,2)
            self.assertEqual(run.call_args_list[0].args[1].name,'0001-a')
            self.assertEqual(run.call_args_list[1].args[1].name,'0002-b')
            self.assertTrue((root/'out/batch-report.json').exists())
            self.assertTrue(all(s.close.called for s in sessions))

    def test_default_and_invalid_segment_settings(self):
        self.assertEqual((UAVConfig().segment_seconds,UAVConfig().segment_overlap),(15.,1.))
        for kw in ({'segment_seconds':0},{'segment_overlap':15},{'segment_overlap':-1}):
            with self.assertRaises(ValueError):UAVConfig(**kw)

class SegmentOrchestrationTests(unittest.TestCase):
    def test_preflight_once_shared_decoder_and_complete_blended_frame_sequence(self):
        from uav.segmented_pipeline import run_segments
        from uav.template_mask.config import TemplateConfig
        calls=[];decoded=[];emitted=[]
        class Frames:
            count=23;fps=2.;source_fps=30.;start=0
            def __init__(self,*a,**kw):
                decoded.append(self);self.base=0
                self.buffer_limit=kw['buffer_limit']
            def __call__(self,f):return np.zeros((4,6,3),np.uint8)
            def source_id(self,f):return f*15
            def release_before(self,f):self.base=f
            def report(self):return dict(storage='memory')
            def close(self):self.closed=True
        def runner(source,folder,config,**kw):
            folder.mkdir(parents=True,exist_ok=True)
            calls.append(kw)
            for f in range(kw['frames'].count):
                kw['writer'].save(torch.full((4,6,3),len(calls)*50,dtype=torch.uint8))
            kw['writer'].close()
            return dict(complete=True,skipped=False,frames=kw['frames'].count)
        writer=Mock();writer.profile.return_value={}
        writer.save.side_effect=lambda rgb:emitted.append(rgb.clone())
        with tempfile.TemporaryDirectory() as folder:
            cfg=UAVConfig(source='synthetic',output=folder,fps=2,segment_seconds=3,segment_overlap=1)
            with (patch('uav.template_mask.frames.VideoFrames',Frames),
                  patch('uav.template_mask.runner.run',side_effect=runner),
                  patch('uav.async_video_writer.AsyncVideoWriter',return_value=writer)):
                result=run_segments(cfg,TemplateConfig(),SimpleNamespace(events=[]))
        self.assertEqual(len(decoded),1);self.assertTrue(decoded[0].closed)
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0]['max_watermark_coverage'],.15)
        self.assertTrue(all(c['max_watermark_coverage'] is None for c in calls[1:]))
        self.assertEqual(decoded[0].buffer_limit,16)
        self.assertEqual(calls[-1]['frames'].count,15)
        self.assertEqual(len(emitted),23)
        self.assertTrue(result['complete']);self.assertEqual(result['output_frames'],23)
        writer.close.assert_called_once()

    def test_cli_accepts_multiple_and_repeated_sources(self):
        from uav.cli import build_parser
        args=build_parser().parse_args(['--source','a.mp4','b.mp4','--source','c.mp4','--output','out'])
        self.assertEqual(args.source,[[Path('a.mp4'),Path('b.mp4')],[Path('c.mp4')]])


class ModelSessionTests(unittest.TestCase):
    def test_weights_and_graphs_remain_resident_between_phases(self):
        from uav.model_session import ModelSession
        sam=Mock();repair=Mock()
        for name in ('model','fix_flow_complete','fix_raft'):getattr(repair,name).modules.return_value=[]
        with (patch('uav.sam3_model.build',return_value=sam) as load_sam,
              patch('uav.inference_weights.compact_weights'),
              patch('uav.repair_model.build_repair_model',return_value=repair) as load_repair,
              patch('torch.cuda.empty_cache'),patch('torch.cuda.memory_allocated',return_value=0)):
            session=ModelSession()
            for _ in range(3):
                self.assertIs(session.sam(),sam)
                self.assertIs(session.repair(),repair)
            self.assertEqual([e['loaded'] for e in session.events],[True,True,False,False,False,False])
            session.close()
        load_sam.assert_called_once();load_repair.assert_called_once()
        sam.cpu.assert_not_called();sam.cuda.assert_not_called()
        repair.cpu.assert_not_called();repair.cuda.assert_not_called()


class MaskExportTests(unittest.TestCase):
    def test_overlap_masks_union_in_memory_and_write_once(self):
        from uav.segmented_pipeline import MaskExporter
        segments=plan_segments(8,5,2);writes={}
        with tempfile.TemporaryDirectory() as folder,patch('cv2.imwrite',side_effect=lambda p,m:writes.setdefault(Path(p).name,m.copy()) is not None):
            exporter=MaskExporter(Path(folder),segments)
            for index,s in enumerate(segments):
                for frame in range(s.stop-s.start):
                    mask=np.zeros((2,3),bool);mask[0,index]=True
                    exporter.save(index,frame,mask)
            self.assertFalse(exporter.pending)
        self.assertEqual(len(writes),8)
        self.assertTrue(writes['000003.png'][0,0] and writes['000003.png'][0,1])

class SegmentFailureTests(unittest.TestCase):
    def test_later_segment_skip_removes_partial_final_video(self):
        from uav.segmented_pipeline import run_segments
        from uav.template_mask.config import TemplateConfig
        class Frames:
            count=18;fps=2.;source_fps=2.
            def __init__(self,*a,**kw):pass
            def __call__(self,f):return np.zeros((4,6,3),np.uint8)
            def source_id(self,f):return f
            def release_before(self,f):pass
            def report(self):return {}
            def close(self):pass
        calls=[]
        def runner(source,folder,config,**kw):
            folder.mkdir(parents=True,exist_ok=True)
            calls.append(kw)
            if len(calls)==2:return dict(complete=True,skipped=True,reason='synthetic')
            for _ in range(kw['frames'].count):kw['writer'].save(torch.zeros((4,6,3),dtype=torch.uint8))
            kw['writer'].close()
            return dict(complete=True,skipped=False)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def writer(path,*a):path.touch();return Mock()
            cfg=UAVConfig(source='synthetic',output=folder,fps=2,segment_seconds=3,segment_overlap=1)
            with (patch('uav.template_mask.frames.VideoFrames',Frames),
                  patch('uav.template_mask.runner.run',side_effect=runner),
                  patch('uav.async_video_writer.AsyncVideoWriter',side_effect=writer)):
                result=run_segments(cfg,TemplateConfig(),SimpleNamespace(events=[]))
            self.assertTrue(result['skipped'])
            self.assertEqual(result['output_frames'],0)
            self.assertFalse((root/'clip-01/repaired.mp4').exists())

class GraphPoolTests(unittest.TestCase):
    def test_replay_owners_have_independent_allocator_pools(self):
        from contextlib import nullcontext
        from uav.cuda_graph import GraphReplay
        class Stream:
            def wait_stream(self,other):pass
        a=GraphReplay(lambda x:x+1)
        b=GraphReplay(lambda x:x+2)
        with (torch.inference_mode(),
              patch('torch.cuda.Stream',side_effect=Stream),
              patch('torch.cuda.current_stream',return_value=Stream()),
              patch('torch.cuda.stream',side_effect=lambda _:nullcontext()),
              patch('torch.cuda.graph_pool_handle',side_effect=['sam-pool','repair-pool']),
              patch('torch.cuda.CUDAGraph') as graph,
              patch('torch.cuda.graph',side_effect=lambda *a,**k:nullcontext()) as capture):
            a(torch.ones(2));b(torch.ones(2));a(torch.ones(2))
        self.assertEqual(a.pool,'sam-pool');self.assertEqual(b.pool,'repair-pool')
        self.assertEqual([c.kwargs['pool'] for c in capture.call_args_list],['sam-pool','repair-pool'])

class DetectorReuseTests(unittest.TestCase):
    def test_detector_text_processors_and_stream_survive_segment_cache_reset(self):
        from uav.model_session import ModelSession
        detector=Mock();session=ModelSession();session.sam=Mock(return_value=object())
        config=object();first_cache=object();second_cache=object()
        with patch('uav.template_mask.sam.SamDetector',return_value=detector) as build:
            self.assertIs(session.detection(first_cache,config),detector)
            self.assertIs(session.detection(second_cache,config),detector)
        build.assert_called_once()
        self.assertIs(detector.cache,second_cache)
        self.assertEqual(detector.calls,0)


class GraphResetTests(unittest.TestCase):
    def test_graph_clear_releases_pool_and_static_inputs_together(self):
        from uav.cuda_graph import GraphReplay
        replay=GraphReplay(lambda x:x)
        replay.cache[1]=object();replay.pool=object();replay.last_static=object()
        with patch('torch.cuda.synchronize'):replay.clear()
        self.assertEqual(replay.cache,{})
        self.assertIsNone(replay.pool);self.assertIsNone(replay.last_static)
