"""Hand final masks and shared original frames to the existing repair sink."""
from pathlib import Path
import numpy as np
import torch
from uav.sam3_repair_sink import RepairSink
from uav.repair_prefetch import prepare_repair as prepare_frame


@torch.inference_mode()
def repair_frames(frames, mask_at, output, model, *, writer=None):
    """Reuse decoded originals and final masks; optionally scale repair only."""
    output = Path(output)
    height,width = frames(0).shape[:2]
    from uav.config import get_config
    from uav.repair_resolution import repair_size, RepairInputs, ResolutionWriter
    scale = get_config().repair_scale
    size = repair_size(width,height,scale)
    original_writer = writer
    preparation = None
    if scale < 1:
        from uav.async_video_writer import AsyncVideoWriter
        writer = ResolutionWriter(writer if writer is not None else
            AsyncVideoWriter(output/'repaired.mp4',width,height,frames.fps))
        preparation = RepairInputs(frames,mask_at,size)
    # Full-frame accumulated decoder features dwarf the cropped model inputs.
    # Spill large overlap canvases to RAM, retaining the GPU path for small marks.
    sink = None
    closing = False
    try:
        sink = RepairSink(model,output,*size,frames.fps,frames.count,frames.count,
                          feature_device='auto', **({} if writer is None else dict(writer=writer)))
        def original_inputs():
            for frame in range(frames.count):
                mask = mask_at(frame)
                if mask.dtype != np.bool_ or mask.shape != (height,width):
                    raise ValueError('repair requires a final boolean mask at original resolution')
                rgb,boxes = prepare_frame(frames(frame),mask)
                yield frame,rgb,mask,boxes,None,None
        for frame,rgb,mask,boxes,original,full_mask in (preparation if preparation is not None else original_inputs()):
            if preparation is not None:writer.remember(frame,original,full_mask)
            sink.save(frame,torch.from_numpy(mask).to('cuda'),
                      torch.from_numpy(rgb).to('cuda').permute(2,0,1),
                      mask_is_final=True,boxes=boxes)
        closing = True
        sink.close()
        torch.cuda.synchronize()
        report = sink.profile()
        report.update(path='repaired.mp4' if original_writer is None else None,
                      output_sink='segment_blend' if original_writer is not None else 'video_file',
                      repair_scale=scale,repair_size=list(size),output_size=[width,height],
                      original_frames='shared_memory',
                      masks='final_memory',additional_mask_dilation=0)
        return report
    except BaseException:
        if sink is not None and not closing:
            try:
                sink.close()
            except Exception:
                pass
        elif sink is None and preparation is not None:
            try:writer.close()
            except Exception:pass
        raise
    finally:
        if preparation is not None:preparation.close()
