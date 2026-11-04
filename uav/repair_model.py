"""Shared ProPainter construction for the shared-template pipeline."""
import torch
from .config import get_config
from .settings import propainter_models


def build_repair_model():
    from .propainter.propainter_inpaint import PropainterInpaint
    config = get_config()
    model = PropainterInpaint(torch.device('cuda'),str(propainter_models()),
                             sub_video_length=24,flow_mask_dilation=4,mask_dilation=0)
    if config.propainter_trt_encoder:
        from .propainter_tensorrt import install_encoder
        install_encoder(model.model, propainter_models()/'ProPainter.pth')
    if config.propainter_trt_propagation:
        from .propagation_tensorrt import install_propagation
        install_propagation(model.model, propainter_models()/'ProPainter.pth')
    model.fix_raft.fix_raft.final_upsample_only = True
    if config.propainter_graph_completion and config.propainter_stream_deform:
        from .cuda_graph import graph_flow_completion
        model.fix_flow_complete.forward = graph_flow_completion(model.fix_flow_complete)
    model.fix_raft.fix_raft.args.mixed_precision = True
    model.raft_iter = config.raft_iters
    if model.raft_iter < 1:
        raise ValueError('RAFT iterations must be positive')
    update = model.fix_raft.fix_raft.update_block
    if config.propainter_raft_channels_last:
        update.to(memory_format=torch.channels_last)
        update.channels_last_inputs = True
    if config.propainter_compile_raft:
        options = {'emulate_precision_casts':True} if config.propainter_compile_precise else None
        update.forward = torch.compile(update.forward,fullgraph=True,dynamic=True,options=options)
    if config.propainter_trt_raft:
        from .raft_tensorrt import install_raft
        install_raft(model.fix_raft.fix_raft, propainter_models()/'raft-things.pth')
    return model
