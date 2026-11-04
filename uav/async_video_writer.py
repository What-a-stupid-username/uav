"""Bounded asynchronous CUDA RGB transfer and background ffmpeg encoding."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import subprocess,time
import torch
from uav import ffmpeg


def encoder_command(path,width,height,fps,*,audio_source=None,audio_start=0.,duration=None):
    command=[ffmpeg.executable(),'-v','error','-nostdin','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0']
    if audio_source is not None:
        command+=['-ss',str(audio_start),'-i',str(audio_source),
                  '-map','0:v:0','-map','1:a:0?','-c:a','aac','-b:a','192k']
    else:command+=['-an']
    if duration is not None:command+=['-t',str(duration)]
    return command+['-c:v','libx264','-preset','fast','-crf','18','-threads','2',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]


class AsyncVideoWriter:
    def __init__(self,path,width,height,fps,capacity=4,*,audio_source=None,audio_start=0.,duration=None):
        self.capacity=capacity;self.slots=[];self.free=deque();self.pending=deque()
        self.executor=ThreadPoolExecutor(max_workers=1);self.stream=torch.cuda.Stream()
        self.wait_seconds=0.;self.wait_count=0;self.gaps=[];self.transfers=[];self.frames=0
        self.encode_seconds=0.;self.drain_seconds=0.
        self.process=subprocess.Popen(encoder_command(path,width,height,fps,
            audio_source=audio_source,audio_start=audio_start,duration=duration),stdin=subprocess.PIPE)

    def save(self,rgb):
        if rgb.device.type!='cuda' or rgb.dtype!=torch.uint8:raise ValueError('CUDA uint8 RGB required')
        while self.pending and self.pending[0].done():self.free.append(self.pending.popleft().result())
        if not self.free and len(self.slots)<self.capacity:
            self.slots.append(torch.empty(rgb.shape,dtype=rgb.dtype,pin_memory=True));self.free.append(len(self.slots)-1)
        if not self.free:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            tick=time.monotonic();self.free.append(self.pending.popleft().result());self.wait_seconds+=time.monotonic()-tick
            b.record();self.gaps.append((a,b));self.wait_count+=1
        slot=self.free.popleft();buffer=self.slots[slot]
        if rgb.shape!=buffer.shape:raise ValueError('video shape changed')
        self.stream.wait_stream(torch.cuda.current_stream())
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            a.record();buffer.copy_(rgb,non_blocking=True);b.record();rgb.record_stream(self.stream)
        self.transfers.append((a,b));self.frames+=1
        self.pending.append(self.executor.submit(self._encode,slot,b,rgb))

    def _encode(self,slot,done,source):
        done.synchronize();tick=time.monotonic()
        # The slot stays owned by this task until the blocking write completes.
        # Pass its contiguous buffer directly instead of copying every frame.
        self.process.stdin.write(memoryview(self.slots[slot].numpy()).cast("B"))
        self.encode_seconds+=time.monotonic()-tick
        return slot

    def close(self):
        tick=time.monotonic()
        try:
            while self.pending:self.pending.popleft().result()
        finally:
            self.executor.shutdown(wait=True);self.process.stdin.close();code=self.process.wait()
            self.drain_seconds=time.monotonic()-tick
        if code:raise RuntimeError(f'encoder exited {code}')

    def profile(self):
        return dict(frames=self.frames,buffers=len(self.slots),pinned_bytes=sum(x.numel() for x in self.slots),
            buffer_wait_count=self.wait_count,buffer_wait_seconds=self.wait_seconds,
            compute_stream_gap_seconds=sum(a.elapsed_time(b) for a,b in self.gaps)/1000,
            transfer_seconds=sum(a.elapsed_time(b) for a,b in self.transfers)/1000,
            encode_host_seconds=self.encode_seconds,drain_seconds=self.drain_seconds)
