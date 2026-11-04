"""Sequential bounded-memory mask/repair segments with one final encoder."""
from pathlib import Path
import json
import time
import cv2
from .segments import FrameSlice,LinearBlendWriter,SegmentWriter,plan_segments


def run_segments(config, template_config, session):
    from .template_mask.frames import VideoFrames
    from .template_mask.runner import run
    output=Path(config.output)
    length=max(1,round(config.segment_seconds*config.fps))
    overlap=round(config.segment_overlap*config.fps)
    tail_extension=round(5*config.fps)
    frames=VideoFrames(config.source,config.fps,config.start_source_frame,config.max_seconds,
                       random_start=config.random_start,seed=config.seed,buffer_limit=length+tail_extension)
    writer=blend=None
    mask_export = None
    results=[];success=False
    started=time.monotonic()
    first_model_event=len(session.events)
    try:
        segments=plan_segments(frames.count,length,overlap,tail_extension)
        if not config.repair and len(segments)>1:
            mask_export=MaskExporter(output/'clip-01/masks',segments)
        # A single short clip keeps the established output layout.
        for index,segment in enumerate(segments):
            folder=output/'clip-01' if len(segments)==1 else output/'segments'/f'{index+1:04d}'
            view=FrameSlice(frames,segment)
            if config.repair and writer is None:
                h,w=frames(0).shape[:2]
                # Start only after final masks pass policy validation, via the lazy adapter.
                writer=_LazyWriter(output/'clip-01/repaired.mp4',w,h,config.fps,
                    audio_source=config.source,audio_start=frames.source_id(0)/frames.source_fps,
                    duration=frames.count/frames.fps)
                blend=LinearBlendWriter(writer,segments)
            result=run(config.source,folder,template_config,session=session,frames=view,
                       review=config.debug,repair=config.repair,write_masks=config.debug or (not config.repair and len(segments)==1),
                       mask_consumer=(lambda f,m:mask_export.save(index,f,m)) if mask_export else None,
                       seed=config.seed,
                       max_watermark_coverage=config.max_watermark_coverage if index==0 else None,
                       max_single_watermark_coverage=config.max_single_watermark_coverage if index==0 else None,
                       writer=SegmentWriter(blend,index) if blend else None)
            result.update(segment_start_frame=segment.start,segment_stop_frame=segment.stop)
            result['directory'] = str(folder.relative_to(output))
            (folder/'report.json').write_text(json.dumps(result,indent=2))
            results.append(result)
            if not result['complete'] or result['skipped']:break
            frames.release_before(segments[index+1].start if index+1<len(segments) else segment.stop)
        success=len(results)==len(segments) and all(r['complete'] and not r['skipped'] for r in results)
        if config.debug and len(segments)>1:
            _write_review_index(output,results)
        if success and mask_export and mask_export.pending:
            raise RuntimeError('incomplete overlap masks')
        if success and blend:
            try:blend.close()
            except BaseException:
                success=False
                raise
        return dict(complete=all(r['complete'] for r in results),
                    skipped=any(r['skipped'] for r in results),engine='template',runs=results,
                    frames=frames.count,fps=frames.fps,frame_memory=frames.report(),
                    repaired_path='clip-01/repaired.mp4' if success and config.repair else None,
                    output_frames=frames.count if success else 0,
                    output=writer.profile() if success and writer else None,
                    total_seconds=time.monotonic()-started,model_transitions=session.events[first_model_event:],
                    segmentation=dict(seconds=config.segment_seconds,overlap_seconds=config.segment_overlap,
                                      tail_extension_seconds=5,
                                      blend='linear_rgb',segments=len(segments),
                                      blended_frames=blend.blended_frames if blend else 0),
                    reason=next((r.get('reason') for r in results if r.get('skipped')),None))
    finally:
        frames.close()
        if writer is not None and not success:
            writer.abort()
        if writer is not None and success and blend is not None:
            # close() is idempotent; also handles an encoder exception during finalization.
            writer.close()


class _LazyWriter:
    def __init__(self,path,width,height,fps,**encoder_options):
        self.path,self.width,self.height,self.fps=path,width,height,fps
        self.writer=None;self.closed=False
        self.encoder_options=encoder_options

    def save(self,rgb):
        from .async_video_writer import AsyncVideoWriter
        if self.writer is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.writer=AsyncVideoWriter(self.path,self.width,self.height,self.fps,**self.encoder_options)
        self.writer.save(rgb)

    def close(self):
        if not self.closed:
            self.closed=True
            if self.writer:self.writer.close()

    def profile(self):
        return self.writer.profile() if self.writer else dict(frames=0)

    def abort(self):
        try:self.close()
        finally:self.path.unlink(missing_ok=True)


class MaskExporter:
    """Merge binary overlap in RAM, then write each final mask exactly once."""
    def __init__(self,destination,segments):
        self.destination,self.segments=destination,segments
        self.pending={}
        destination.mkdir(parents=True,exist_ok=True)

    def save(self,index,frame,mask):
        import numpy as np
        segment=self.segments[index]
        global_frame=segment.start+frame
        if global_frame in self.pending:mask=mask | self.pending.pop(global_frame)
        next_start=self.segments[index+1].start if index+1<len(self.segments) else segment.stop
        if global_frame>=next_start:self.pending[global_frame]=mask.copy()
        elif not cv2.imwrite(str(self.destination/f'{global_frame:06d}.png'),mask.astype(np.uint8)*255):
            raise RuntimeError('mask export failed')


def _write_review_index(output, results):
    import html
    items=[]
    for result in results:
        directory=result['directory']
        first,last=result['segment_start_frame'],result['segment_stop_frame']
        items.append('<li><a href="'+html.escape(directory,quote=True)+'/index.html">'+
                     f'帧 {first}–{last-1}</a>'+(' · 跳过' if result['skipped'] else '')+'</li>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>UAV 分段审核</h1>'+
        '<p>相邻段使用配置的重叠区，最终修复对 RGB 做线性混合。</p><ul>'+''.join(items)+'</ul>')
