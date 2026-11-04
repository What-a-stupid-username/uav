"""Shared initialization matches legacy choices, using synthetic modules."""
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch
import torch
from uav.sam3_inference import prepare_image_inference
from uav.template_mask.config import TemplateConfig


class InferenceSetupTests(unittest.TestCase):
    def model(self):
        rope=torch.nn.Module()
        rope.use_rope=True
        rope.use_rope_real=False
        rope.register_buffer('freqs_cis',torch.tensor([1+2j,3+4j]))
        trunk=torch.nn.Sequential(rope)
        decoder=torch.nn.Sequential(torch.nn.Linear(3,4))
        return SimpleNamespace(backbone=SimpleNamespace(vision_backbone=SimpleNamespace(trunk=trunk)),
            transformer=SimpleNamespace(decoder=decoder),forward_grounding=lambda *a,**kw:None)

    def test_shared_setup_is_idempotent_and_only_compacts_backbone(self):
        model=self.model()
        wrapped=lambda *a,**kw:None
        wrapped.replay=object()
        with patch('uav.sam3_inference.compact_weights') as compact, \
             patch('torch.compile',side_effect=lambda fn,**kw:fn) as compile, \
             patch('uav.cuda_graph.graph_grounding',return_value=wrapped) as graph:
            prepare_image_inference(model)
            prepare_image_inference(model)
            self.assertEqual(compile.call_count,1)
            graph.assert_called_once_with(model,eager_boxes_only=True)
            self.assertTrue(all(c.args==(model.backbone,) for c in compact.call_args_list))
        rope=model.backbone.vision_backbone.trunk[0]
        self.assertTrue(rope.use_rope_real)
        self.assertTrue(torch.equal(rope.freqs_cis_real,torch.tensor([1.,3.])))
        self.assertTrue(torch.equal(rope.freqs_cis_imag,torch.tensor([2.,4.])))
        self.assertEqual(model.transformer.decoder[0].weight.dtype,torch.float32)

    def test_eager_options_do_not_install_compile_rope_or_graph(self):
        model=self.model()
        with patch('uav.sam3_inference.compact_weights'), \
             patch('torch.compile',side_effect=AssertionError('disabled')), \
             patch('uav.cuda_graph.graph_grounding',side_effect=AssertionError('disabled')):
            prepare_image_inference(model,real_rope=False,compile_trunk=False,grounding_graph=False)
        self.assertFalse(model.backbone.vision_backbone.trunk[0].use_rope_real)
        self.assertTrue(TemplateConfig().sam3_compile)

    def test_tensorrt_installs_after_compile_and_preserves_grounding(self):
        model=self.model()
        with (patch('uav.sam3_inference.compact_weights'),
              patch('torch.compile',side_effect=lambda fn,**kw:fn) as compile,
              patch('uav.sam3_tensorrt.install_encoder') as install):
            prepare_image_inference(model,trt_engine='/cache/sam.engine',grounding_graph=False)
        compile.assert_called_once()
        install.assert_called_once_with(model,'/cache/sam.engine')

    def test_tensorrt_configuration_reaches_mask_pipeline(self):
        from uav.config import UAVConfig
        from uav.pipeline import mask_config
        from uav.cli import build_parser
        config=UAVConfig(sam3_trt_engine='/cache/sam.engine')
        self.assertEqual(mask_config(config).sam3_trt_engine,config.sam3_trt_engine)
        args=build_parser().parse_args(['--sam3-trt-engine','/cache/sam.engine'])
        self.assertEqual(args.sam3_trt_engine,config.sam3_trt_engine)
        with self.assertRaises(ValueError):
            UAVConfig(sam3_trt_engine='/cache/sam.engine',sam3_real_rope=False)

    def test_graph_can_be_disabled_while_retaining_compiled_backbone_and_real_rope(self):
        model=self.model()
        original=model.forward_grounding
        with (patch('uav.sam3_inference.compact_weights'),
              patch('torch.compile',side_effect=lambda fn,**kw:fn) as compile,
              patch('uav.cuda_graph.graph_grounding',side_effect=AssertionError('disabled'))):
            prepare_image_inference(model,grounding_graph=False)
        compile.assert_called_once()
        self.assertTrue(model.backbone.vision_backbone.trunk[0].use_rope_real)
        self.assertIs(model.forward_grounding,original)

    def test_box_only_calls_use_original_grounding_and_segmentation_uses_graph(self):
        from uav.cuda_graph import graph_grounding
        model=self.model()
        model.training=False
        original=Mock(return_value={'pred_boxes':'boxes'})
        model.forward_grounding=original
        with (patch('uav.sam3_resolution.use_configured_rpb_size'),
              patch.dict(sys.modules,{'sam3.model.geometry_encoders':SimpleNamespace(Prompt=type('Prompt',(),{})),
                                      'sam3.model.data_misc':SimpleNamespace(FindStage=lambda **kw:SimpleNamespace(**kw))}),
              patch('uav.cuda_graph.GraphReplay') as replay):
            forward=graph_grounding(model,eager_boxes_only=True)
        find=SimpleNamespace(img_ids='images')
        prompt=SimpleNamespace(boxes=None)
        # Boxes-only inputs need no replay signature or graph-owned state.
        self.assertEqual(forward({},find,None,prompt,boxes_only=True),{'pred_boxes':'boxes'})
        self.assertTrue(original.call_args.kwargs['boxes_only'])
        replay.return_value.assert_not_called()
        backbone={k:None for k in ('backbone_fpn','vision_pos_enc','language_features','language_mask')}
        forward(backbone,find,None,prompt)
        replay.return_value.assert_called_once()


if __name__=='__main__':unittest.main()
