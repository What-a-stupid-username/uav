"""Public entry-point contract without media reads or model construction."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uav import UAVConfig, run
from uav.config import get_config
from uav.pipeline import mask_config


class ProductionPipelineTests(unittest.TestCase):
    def test_public_api_selects_new_engine_and_restores_context(self):
        initial=get_config()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.mp4';source.touch()
            result=dict(complete=True,skipped=False,engine='template',runs=[])
            with (patch('uav.settings.validate_models') as validate,
                  patch('uav.segmented_pipeline.run_segments',return_value=result.copy()) as engine,
                  patch('uav.model_session.ModelSession') as session):
                report=run(source,root/'out',UAVConfig())
            validate.assert_called_once_with(repair=True)
            config,mask,actual_session=engine.call_args.args
            self.assertEqual(config.segment_seconds,15.)
            self.assertFalse(hasattr(mask,'mask_schedule'))
            self.assertIs(actual_session,session.return_value)
            session.return_value.close.assert_called_once()
            self.assertEqual(report,json.loads((root/'out/report.json').read_text()))
            self.assertIs(get_config(),initial)

    def test_mask_settings_are_forwarded_without_legacy_options(self):
        config=UAVConfig(fps=30,sam3_global_stride=4,mask_dilation=7,
                         mask_pixel_threshold=.3,sift_backend='cpu',
                         uncertain_mask_policy='ignore',sam3_compile=False)
        actual=mask_config(config)
        self.assertEqual((actual.fps,actual.global_stride,actual.dilation),(30,4,7))
        self.assertEqual(actual.pixel_threshold,.3)
        self.assertEqual(actual.resolved_match_backend,'cpu')
        self.assertFalse(actual.sam3_compile)
        self.assertEqual(actual.uncertain_mask_policy,'ignore')
        self.assertEqual(mask_config(UAVConfig(mask_reference_limit=4)).mask_reference_limit,4)
        for value in (0,9,True,4.5):
            with self.assertRaises(ValueError):UAVConfig(mask_reference_limit=value)
        for name in ('two_pass','sam3_local_tracking'):
            with self.assertRaises(TypeError):UAVConfig(**{name:True})

    def test_mask_only_does_not_require_propainter_weights(self):
        from uav.settings import validate_models
        with patch('uav.settings.sam3_checkpoint',return_value=Path('/models/sam3.pt')):
            with patch.object(Path,'is_file',lambda p:p==Path('/models/sam3.pt')):
                validate_models(repair=False)
                with self.assertRaises(FileNotFoundError):validate_models(repair=True)

    def test_failed_run_restores_configuration(self):
        initial=get_config()
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.mp4';source.touch()
            with (patch('uav.settings.validate_models'),
                  patch('uav.segmented_pipeline.run_segments',side_effect=RuntimeError('failed')),
                  patch('uav.model_session.ModelSession')):
                with self.assertRaisesRegex(RuntimeError,'failed'):
                    run(source,Path(folder)/'out',UAVConfig(debug=True))
        self.assertIs(get_config(),initial)


if __name__=='__main__':unittest.main()
