"""Synthetic startup gate: stop before decode/tracking/repair on excessive area."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from uav.template_mask.runner import run
from uav.template_mask.sam import SamDetector


class CoveragePreflightTests(unittest.TestCase):
    def test_skip_stops_before_main_decoder_engine_and_repair(self):
        detector=Mock(calls=4,cache_hits=0,task_calls={'coverage_preflight':4})
        result=dict(skipped=True,skip_reasons=['watermark_coverage_exceeded'],samples=[])
        with tempfile.TemporaryDirectory() as folder:
            with (patch('uav.template_mask.sam.SamDetector',return_value=detector),
                  patch('uav.coverage_preflight.check_coverage',return_value=result) as check,
                  patch('uav.template_mask.runner.VideoFrames') as frames,
                  patch('uav.template_mask.runner.TemplateEngine') as engine,
                  patch('uav.template_mask.repair.repair_frames') as repair):
                report=run('synthetic',folder,model=object(),repair=True,
                           max_watermark_coverage=.15,max_single_watermark_coverage=.15)
            self.assertTrue(report['complete'] and report['skipped'])
            self.assertEqual(report['reason'],'watermark_coverage_exceeded')
            self.assertEqual(report['sam']['calls'],4)
            frames.assert_not_called();engine.assert_not_called();repair.assert_not_called()
            self.assertEqual(check.call_args.args[3],detector.coverage)
            warning=json.loads((Path(folder)/'warnings.jsonl').read_text())
            self.assertEqual(warning['action'],'skip_video')

    def test_disabled_limits_never_open_preflight_capture(self):
        detector=Mock(calls=0,cache_hits=0,task_calls={})
        for limits in ((None,None),(1.,1.)):
            with self.subTest(limits=limits),tempfile.TemporaryDirectory() as folder:
                with (patch('uav.template_mask.sam.SamDetector',return_value=detector),
                      patch('uav.coverage_preflight.check_coverage') as check,
                      patch('uav.template_mask.runner.VideoFrames',side_effect=RuntimeError('decoder sentinel'))):
                    with self.assertRaisesRegex(RuntimeError,'decoder sentinel'):
                        run('synthetic',folder,model=object(),review=False,
                            max_watermark_coverage=limits[0],max_single_watermark_coverage=limits[1])
                check.assert_not_called()

    def test_preflight_box_only_measurements_do_not_cache_probability_evidence(self):
        detector=object.__new__(SamDetector)
        detector.config=SimpleNamespace(global_resolution=1008)
        detector.calls=0;detector.task_calls={}
        detector._infer=Mock(return_value=(np.ones(2),np.array([[0,0,6,10],[4,0,10,10]]),None))
        measurement=detector.coverage(np.zeros((10,10,3),np.uint8))
        self.assertEqual(measurement,dict(coverage=1.,single_coverage=.6))
        self.assertEqual(detector._infer.call_args.kwargs,dict(boxes_only=True))
        self.assertEqual(detector.calls,1)
        self.assertEqual(detector.task_calls,{'coverage_preflight':1})


if __name__=='__main__':unittest.main()
