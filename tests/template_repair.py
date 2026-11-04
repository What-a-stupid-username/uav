"""Memory-only final-mask handoff, using synthetic originals and a fake sink."""
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import tempfile
from unittest.mock import patch,Mock
import unittest
import numpy as np
import torch
from uav.template_mask.repair import repair_frames
from uav.cli import build_parser
from uav.config import UAVConfig,use_config


class RepairTests(unittest.TestCase):
    def setUp(self):
        # These checks exercise the original-resolution handoff explicitly.
        context=use_config(UAVConfig(repair_scale=1))
        context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def test_close_failure_does_not_finalize_sink_twice(self):
        frames=Mock(count=0,fps=24.,return_value=np.zeros((10,20,3),np.uint8))
        sink=Mock();sink.close.side_effect=RuntimeError('encoder failed')
        with patch('uav.template_mask.repair.RepairSink',return_value=sink):
            with self.assertRaisesRegex(RuntimeError,'encoder failed'):
                repair_frames(frames,lambda f:None,'/synthetic',object())
        sink.close.assert_called_once()

    def test_handoff_keeps_original_pixels_and_final_masks_without_decode(self):
        original=np.zeros((40,60,3),np.uint8);original[:]=[13,29,43]
        original.flags.writeable=False
        mask=np.zeros((40,60),bool);mask[8:12,20:24]=True
        class Frames:
            count=3;fps=24.
            def __call__(self,frame):return original
        sink=Mock();sink.profile.return_value=dict(complete=True,output_frames=3)
        def save(frame,pred,rgb,**kw):
            np.testing.assert_array_equal(pred.numpy(),mask)
            np.testing.assert_array_equal(rgb.numpy()[:,0,0],[43,29,13])
            self.assertTrue(kw['mask_is_final']);self.assertTrue(kw['boxes'])
        sink.save.side_effect=save
        real_to=torch.Tensor.to
        def to(tensor,*a,**kw):
            return tensor if a and a[0]=='cuda' else real_to(tensor,*a,**kw)
        with (patch('uav.template_mask.repair.RepairSink',return_value=sink),
              patch('torch.Tensor.to',to),patch('torch.cuda.synchronize'),
              patch('cv2.VideoCapture',side_effect=AssertionError('must reuse memory')),
              patch('cv2.imread',side_effect=AssertionError('must reuse final masks'))):
            report=repair_frames(Frames(),lambda f:mask,Path('/synthetic'),object())
        self.assertEqual(sink.save.call_count,3);sink.close.assert_called_once()
        self.assertEqual(report['original_frames'],'shared_memory')
        self.assertEqual(report['additional_mask_dilation'],0)
        self.assertFalse(original.flags.writeable)

    def test_invalid_mask_closes_sink_and_does_not_start_repair(self):
        frames=Mock(count=1,fps=24.,return_value=np.zeros((10,20,3),np.uint8))
        sink=Mock()
        with patch('uav.template_mask.repair.RepairSink',return_value=sink):
            with self.assertRaisesRegex(ValueError,'final boolean mask'):
                repair_frames(frames,lambda f:np.ones((10,20),np.uint8),'/synthetic',object())
        sink.close.assert_called_once();sink.save.assert_not_called()

    def test_production_cli_repairs_by_default(self):
        args=['--source','x','--output','y']
        self.assertTrue(build_parser().parse_args(args).repair)
        self.assertFalse(build_parser().parse_args(args+['--no-repair']).repair)

    def test_runner_hands_off_only_after_final_policy_gate(self):
        from uav.template_mask.runner import run
        from uav.template_mask.policy import SkipVideo
        mask=np.zeros((40,60),bool);mask[8:12,20:24]=True
        for skip in (False,True):
            with self.subTest(skip=skip),tempfile.TemporaryDirectory() as folder:
                frames=Mock(count=1,fps=24.,source_fps=24.,start=0)
                frames.return_value=np.zeros((40,60,3),np.uint8)
                engine=Mock(shape=mask.shape,instances={})
                engine.bank=SimpleNamespace(templates={},links={})
                engine.detector=SimpleNamespace(calls=0,cache_hits=0,task_calls={})
                engine.report.return_value={}
                if skip:engine.render.side_effect=SkipVideo('synthetic unresolved pose')
                else:engine.render.return_value=mask
                model=object()
                def handoff(actual_frames,mask_at,output,actual_model):
                    self.assertIs(actual_frames,frames)
                    self.assertIs(actual_model,model)
                    frames.close.assert_not_called()
                    np.testing.assert_array_equal(mask_at(0),mask)
                    return dict(complete=True,output_frames=1)
                with (patch('uav.template_mask.runner.VideoFrames',return_value=frames),
                      patch('uav.template_mask.runner.TemplateEngine',return_value=engine),
                      patch('uav.template_mask.sam.SamDetector',return_value=engine.detector),
                      patch('uav.template_mask.prefetch.prefetch_discoveries',return_value=nullcontext()),
                      patch('cv2.imwrite',side_effect=AssertionError('normal repair must not write mask files')),
                      patch('uav.template_mask.repair.repair_frames',side_effect=handoff) as repair):
                    report=run('synthetic',folder,model=object(),review=False,
                               repair=True,repair_model=model,write_masks=False)
                self.assertTrue(report['complete'])
                self.assertEqual(report['skipped'],skip)
                self.assertEqual(repair.call_count,0 if skip else 1)
                frames.close.assert_called_once()


if __name__=='__main__':unittest.main()
