"""Debug snapshots use in-memory synthetic pixels only, never read media."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from uav.template_mask import checkpoint
from uav.template_mask.cache import EvidenceCache
from uav.template_mask.config import TemplateConfig
from uav.template_mask.engine import TemplateEngine
from uav.template_mask.policy import WarningLog
from uav.template_mask.records import Candidate


class Detector:
    def __init__(self,*args,**kwargs):
        self.calls = self.cache_hits = 0
        self.task_calls = dict(global_coarse=0,local=0,refinement=0)

    def detect(self,image,source,box,resolution,*,refinement=False):
        self.calls += 1
        self.task_calls['refinement' if refinement else 'global_coarse'] += 1
        p=np.zeros((60,140),np.float32);p[5:55,5:135]=.9
        return [Candidate(p,(20,20),(20,20,160,80),.9,'synthetic')]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'synthetic-source-token'
        self.source.write_text('test source identity; contains no media')
        self.path=self.root/'stage1.pkl'
        self.image=np.zeros((120,240,3),np.uint8)
        cv2.putText(self.image,'UAV123',(23,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,230,230),2)
        self.config=TemplateConfig(sift_backend='cpu')
        self.frames=SimpleNamespace(start=0,count=16,source_fps=24.)
        self.caches=[]

    def tearDown(self):
        for cache in self.caches:cache.close()
        self.temp.cleanup()

    def engine(self,config=None,debug=True):
        cache=EvidenceCache();self.caches.append(cache)
        return TemplateEngine(config or self.config,cache,Detector(),
            WarningLog(self.root/str(len(self.caches))/'warnings.jsonl'),self.image.shape[:2],debug=debug)

    def seed(self):
        e=self.engine()
        for f in range(self.frames.count):e.first_frame(f,f,self.image,image_at=lambda f:self.image)
        return e

    def test_restored_stage2_matches_direct_masks_and_original_evidence(self):
        original=self.seed()
        signature=checkpoint.signature(self.source,self.frames,self.config,original)
        stored=checkpoint.save(self.path,original,signature)
        restored=self.engine()
        runtime=(restored.matcher,restored.detector,restored.bank,restored.presence)
        loaded=checkpoint.load(self.path,restored,signature)
        self.assertTrue(loaded['reused']);self.assertFalse(stored['reused'])
        self.assertEqual(loaded['first_pass_seconds'],stored['first_pass_seconds'])
        self.assertEqual(restored.timings['first_pass_seconds'],0)
        self.assertEqual(runtime,(restored.matcher,restored.detector,restored.bank,restored.presence))
        self.assertIs(restored.bank.cache,restored.cache)
        self.assertIs(restored.presence.engine,restored)
        self.assertEqual(restored.first_pass_marks,original.first_pass_marks)
        self.assertEqual(restored.source_ids,original.source_ids)
        for key,arrays in original.cache.memory.items():
            for name,array in arrays.items():
                np.testing.assert_array_equal(restored.cache.get(key)[name],array)
                self.assertFalse(restored.cache.get(key)[name].flags.writeable)
        for e in (original,restored):e.second_pass(lambda f:self.image)
        for f in range(self.frames.count):
            np.testing.assert_array_equal(restored.render(f),original.render(f))
        self.assertEqual(restored.warnings.rows,original.warnings.rows)
        self.assertEqual(restored.detector.calls,original.detector.calls)
        # Repeated loads start from the same untouched Stage1, never Stage2.
        third=self.engine();checkpoint.load(self.path,third,signature)
        self.assertFalse(third.refinement_batches)
        self.assertEqual(third.stage,'first_pass')

    def test_mismatched_interval_settings_debug_engine_or_source_fail_without_mutation(self):
        e=self.seed();signature=checkpoint.signature(self.source,self.frames,self.config,e)
        checkpoint.save(self.path,e,signature)
        for key,value in [('start',1),('debug',False),('engine','other.Engine'),('mtime_ns',0),('config',{})]:
            fresh=self.engine()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'does not match'):
                checkpoint.load(self.path,fresh,dict(signature,**{key:value}))
            self.assertFalse(fresh.instances);self.assertFalse(fresh.cache.memory)

    def test_output_policy_and_dilation_changes_reuse_the_same_stage1(self):
        e=self.seed();a=checkpoint.signature(self.source,self.frames,self.config,e)
        config=replace(self.config,dilation=8,uncertain_mask_policy='ignore')
        fresh=self.engine(config)
        b=checkpoint.signature(self.source,self.frames,config,fresh)
        self.assertEqual(a,b)
        checkpoint.save(self.path,e,a);checkpoint.load(self.path,fresh,b)
        self.assertEqual(fresh.config,config)
        self.assertEqual(fresh.bank.config,config)

    def test_no_live_runtime_methods_or_models_are_serialized(self):
        e=self.seed()
        e.detector.model=lambda:None
        e.bank.custom_method=lambda:None
        signature=checkpoint.signature(self.source,self.frames,self.config,e)
        checkpoint.save(self.path,e,signature)
        fresh=self.engine();checkpoint.load(self.path,fresh,signature)
        self.assertFalse(hasattr(fresh.detector,'model'))
        self.assertFalse(hasattr(fresh.bank,'custom_method'))

    def test_stage2_cannot_overwrite_stage1_checkpoint(self):
        e=self.engine();e.stage='second_pass'
        with self.assertRaisesRegex(ValueError,'precede Stage2'):
            checkpoint.save(self.path,e,{})
        self.assertFalse(self.path.exists())

    def test_runner_skips_stage1_on_reuse_and_executes_current_stage2(self):
        from uav.template_mask.runner import run
        test=self
        class Frames:
            def __init__(self,*args,**kwargs):
                self.start=0;self.count=9;self.source_fps=self.fps=24.
            def source_id(self,f):return f
            def __call__(self,f):return test.image
            def close(self):pass
        count=[];outputs=[]
        first=TemplateEngine.first_frame
        def first_frame(e,*args,**kw):
            count.append(args[0]);return first(e,*args,**kw)
        def write(path,mask):outputs.append(mask.copy());return True
        with patch('uav.template_mask.runner.VideoFrames',Frames), \
             patch('uav.template_mask.sam.SamDetector',Detector), \
             patch.object(TemplateEngine,'first_frame',first_frame),patch('cv2.imwrite',side_effect=write):
            a=run(self.source,self.root/'first',model=object(),config=self.config,review=False,stage1_checkpoint=self.path)
            self.assertEqual(len(count),9)
            b=run(self.source,self.root/'reuse',model=object(),config=self.config,review=False,stage1_checkpoint=self.path)
            self.assertEqual(len(count),9)
            self.assertTrue(b['stage1_checkpoint']['reused'])
            self.assertTrue(b['evidence_reuse'])
            self.assertEqual(b['timing']['first_pass_seconds'],0)
            self.assertIn('stage1_checkpoint_load',b['stage_seconds'])
            self.assertTrue(a['complete'] and b['complete'])
            for x,y in zip(outputs[:9],outputs[9:]):np.testing.assert_array_equal(x,y)
            run(self.source,self.root/'refresh',model=object(),config=self.config,review=False,
                stage1_checkpoint=self.path,refresh_stage1=True)
            self.assertEqual(len(count),18)

    def test_default_runner_never_inspects_loads_or_saves_a_checkpoint(self):
        from uav.template_mask.runner import run
        from uav.cli import build_parser
        test=self
        class Frames:
            def __init__(self,*args,**kwargs):
                self.start=0;self.count=9;self.source_fps=self.fps=24.
            def source_id(self,f):return f
            def __call__(self,f):return test.image
            def close(self):pass
        # Even an existing conventional debug filename must not enable reuse.
        self.path.write_text('not a snapshot; do not open')
        with (patch('uav.template_mask.runner.VideoFrames',Frames),
              patch('uav.template_mask.sam.SamDetector',Detector),
              patch('cv2.imwrite',return_value=True),
              patch.object(checkpoint,'signature',side_effect=AssertionError('default checkpoint stat')),
              patch.object(checkpoint,'load',side_effect=AssertionError('default checkpoint read')),
              patch.object(checkpoint,'save',side_effect=AssertionError('default checkpoint write')),
              patch.object(TemplateEngine,'first_frame',autospec=True) as first):
            report=run(self.source,self.root/'normal',model=object(),config=self.config,review=False)
        self.assertTrue(report['complete']);self.assertEqual(first.call_count,9)
        self.assertNotIn('stage1_checkpoint',report)
        self.assertNotIn('stage1_checkpoint_load',report['stage_seconds'])
        self.assertNotIn('stage1_checkpoint_save',report['stage_seconds'])
        self.assertNotIn('stage1-checkpoint', build_parser().format_help())


if __name__=='__main__':unittest.main()
