"""Output-frame segment planning and one-pass linear blending before encoding."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int


def plan_segments(count, length, overlap, tail_extension=0):
    if count < 1 or length < 1 or not 0 <= overlap < length or tail_extension < 0:
        raise ValueError('invalid segment frame counts')
    result=[];start=0
    while start < count:
        stop=min(count,start+length)
        # Measure new frames beyond this block, excluding the shared overlap.
        if count-stop <= tail_extension:
            stop=count
        result.append(Segment(start,stop))
        if stop == count:break
        start=stop-overlap
    return result


class FrameSlice:
    """Read-only segment view on one continuous sampled timeline."""
    def __init__(self, frames, segment):
        self.frames,self.segment=frames,segment
        self.count=segment.stop-segment.start
        self.fps,self.source_fps=frames.fps,frames.source_fps
        self.start=frames.source_id(segment.start)

    def __call__(self, frame):
        if not 0 <= frame < self.count:raise IndexError('frame outside segment')
        return self.frames(self.segment.start+frame)

    def source_id(self, frame):
        if not 0 <= frame < self.count:raise IndexError('frame outside segment')
        return self.frames.source_id(self.segment.start+frame)

    def report(self):
        return dict(storage='shared_segment_memory',frames=self.count,start_source_frame=self.start)


class SegmentWriter:
    """RepairSink adapter: its close must not close the shared output encoder."""
    def __init__(self, blend, index):
        self.blend,self.index,self.frame=blend,index,0

    def save(self, rgb):
        self.blend.save(self.index,self.frame,rgb)
        self.frame+=1

    def close(self):
        expected=self.blend.segments[self.index]
        if self.frame != expected.stop-expected.start:
            raise RuntimeError('incomplete segment repair')

    def profile(self):
        return dict(frames=self.frame,storage='shared_linear_blend',segment=self.index)


class LinearBlendWriter:
    """Retain only the next overlap, in CPU RAM; never read encoded video."""
    def __init__(self, writer, segments):
        self.writer,self.segments=writer,segments
        self.blended_frames=0
        self.pending={}
        self.emitted=0

    def save(self, index, frame, rgb):
        import torch
        segment=self.segments[index]
        global_frame=segment.start+frame
        # Blend actual RGB outputs. Endpoints are exact for overlap >= 2 frames.
        if global_frame in self.pending:
            self.blended_frames+=1
            old=self.pending.pop(global_frame).to(rgb.device)
            overlap=self.segments[index-1].stop-segment.start
            alpha=frame/(overlap-1) if overlap>1 else .5
            rgb=torch.lerp(old.float(),rgb.float(),alpha).round().clamp_(0,255).to(torch.uint8)
        next_start=self.segments[index+1].start if index+1<len(self.segments) else segment.stop
        if global_frame>=next_start:
            self.pending[global_frame]=rgb.cpu()
        else:
            if global_frame != self.emitted:raise RuntimeError('non-contiguous segment output')
            self.writer.save(rgb)
            self.emitted+=1

    def close(self):
        try:
            if self.pending or self.emitted!=self.segments[-1].stop:
                raise RuntimeError('incomplete blended output')
        finally:self.writer.close()
