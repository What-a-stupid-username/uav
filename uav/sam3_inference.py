"""Shared SAM3 inference setup used by both mask engines.

Only the backbone's autocast parameters are compacted. The decoder explicitly
disables autocast in its FFNs and must retain those FP32 parameters.
"""
import torch
from .inference_weights import compact_weights


def use_real_rope(model):
    for module in model.backbone.vision_backbone.trunk.modules():
        frequencies = getattr(module, 'freqs_cis', None)
        if (not getattr(module, 'use_rope', False) or frequencies is None
                or getattr(module, 'use_rope_real', False)):
            continue
        module.use_rope_real = True
        module.register_buffer('freqs_cis_real', frequencies.real.contiguous())
        module.register_buffer('freqs_cis_imag', frequencies.imag.contiguous())


def compile_backbone(model):
    trunk = model.backbone.vision_backbone.trunk
    if not getattr(trunk, '_uav_compiled_forward', False):
        trunk.forward = torch.compile(trunk.forward, mode='default', fullgraph=True, dynamic=False)
        trunk._uav_compiled_forward = True


def prepare_grounding(model, *, compile_decoder=False, graph_grounding=True, eager_boxes_only=False):
    decoder = model.transformer.decoder
    if compile_decoder and not getattr(decoder, '_uav_compiled_forward', False):
        decoder.forward = torch.compile(decoder.forward, mode='default', fullgraph=True, dynamic=False)
        decoder._uav_compiled_forward = True
    if graph_grounding and not hasattr(model.forward_grounding, 'replay'):
        from .cuda_graph import graph_grounding as wrap_grounding
        model.forward_grounding = (wrap_grounding(model, eager_boxes_only=True)
                                  if eager_boxes_only else wrap_grounding(model))


def prepare_image_inference(model, *, real_rope=True, compile_trunk=True, grounding_graph=True, trt_engine=''):
    """Idempotently install legacy UAV's validated image-inference options."""
    compact_weights(model.backbone)
    if real_rope:
        use_real_rope(model)
    if compile_trunk:
        compile_backbone(model)
    if trt_engine:
        if not real_rope:
            raise ValueError('TensorRT encoder requires real RoPE')
        from .sam3_tensorrt import install_encoder
        install_encoder(model, trt_engine)
    prepare_grounding(model, graph_grounding=grounding_graph, eager_boxes_only=True)
