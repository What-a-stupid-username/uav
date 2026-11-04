"""Bounded background encoding and event-guarded pinned-memory CUDA copies."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import torch

import numpy as np
from PIL import Image


class MaskWriter:
    def __init__(self, destination, compression=1, asynchronous=True, capacity=4):
        self.destination = destination
        self.compression = compression
        self.capacity = capacity
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.seconds = 0.
        self.wait_seconds = 0.
        if capacity < 1:raise ValueError('capacity must be positive')
        self.copy_stream = None
        self.slots = []
        self.free_slots = deque()
        self.buffer_wait_seconds = 0.
        self.buffer_wait_count = 0
        self.buffer_wait_max_seconds = 0.
        self.drain_seconds = 0.
        self.transfer_seconds = 0.
        self.transfer_wait_seconds = 0.
        self.stream_gaps = []
        self.transfer_count = 0

    def _write(self, frame, prediction):
        tick = time.monotonic()
        # Visualization must preserve the supplied mask exactly. Final dilation
        # belongs to the mask producer, never to serialization.
        prediction = np.asarray(prediction, dtype=bool)
        rgba = np.zeros((*prediction.shape, 4), np.uint8)
        rgba[prediction] = [40, 240, 140, 150]
        Image.fromarray(rgba).save(self.destination/f'{frame:04d}-adaptive.png',
                                   compress_level=self.compression)
        self.seconds += time.monotonic()-tick

    def save(self, frame, prediction):
        tick = time.monotonic()
        if isinstance(prediction,torch.Tensor) and prediction.is_cuda and self.executor is not None:
            self._save_cuda(frame,prediction)
        elif self.executor is None:
            if isinstance(prediction,torch.Tensor):prediction=prediction.cpu().numpy()
            self._write(frame, prediction)
        else:
            while self.pending and self.pending[0].done():
                slot=self.pending.popleft().result()
                if slot is not None:self.free_slots.append(slot)
            if len(self.pending) >= self.capacity:
                slot=self.pending.popleft().result()
                if slot is not None:self.free_slots.append(slot)
            if isinstance(prediction,torch.Tensor):prediction=prediction.cpu().numpy()
            self.pending.append(self.executor.submit(self._write, frame, prediction.copy()))
        self.wait_seconds += time.monotonic()-tick

    def _save_cuda(self,frame,prediction):
        if self.copy_stream is None:
            self.copy_stream=torch.cuda.Stream(device=prediction.device)
        # Slots are released only after D2H, CPU processing and file writing finish.
        while self.pending and self.pending[0].done():
            slot=self.pending.popleft().result()
            while slot is None:
                slot=self.pending.popleft().result()
            if slot is not None:self.free_slots.append(slot)
        if not self.free_slots and len(self.slots)<self.capacity:
            self.slots.append(torch.empty(prediction.shape,dtype=prediction.dtype,
                                          device='cpu',pin_memory=True))
            self.free_slots.append(len(self.slots)-1)
        if not self.free_slots:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(prediction.device))
            tick=time.monotonic()
            slot=self.pending.popleft().result()
            waited=time.monotonic()-tick
            end.record(torch.cuda.current_stream(prediction.device))
            self.stream_gaps.append((start,end))
            self.buffer_wait_seconds+=waited;self.buffer_wait_count+=1
            self.buffer_wait_max_seconds=max(self.buffer_wait_max_seconds,waited)
            self.free_slots.append(slot)
        slot=self.free_slots.popleft();buffer=self.slots[slot]
        if buffer.shape!=prediction.shape or buffer.dtype!=prediction.dtype:
            raise ValueError('mask shape/dtype changed within one writer')
        self.copy_stream.wait_stream(torch.cuda.current_stream(prediction.device))
        begin=torch.cuda.Event(enable_timing=True);done=torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            begin.record();buffer.copy_(prediction,non_blocking=True);done.record()
            prediction.record_stream(self.copy_stream)
        self.transfer_count+=1
        self.pending.append(self.executor.submit(self._write_cuda,frame,slot,begin,done,prediction))

    def _write_cuda(self,frame,slot,begin,done,source):
        # Keep source alive and never expose pinned bytes to NumPy before the event.
        tick=time.monotonic();done.synchronize()
        self.transfer_wait_seconds+=time.monotonic()-tick
        self.transfer_seconds+=begin.elapsed_time(done)/1000
        self._write(frame,self.slots[slot].numpy())
        return slot

    def profile(self):
        # Call after close/device synchronization. These gaps measure only stream
        # starvation at pool backpressure, not device-wide GPU idle time.
        gap=sum(start.elapsed_time(end) for start,end in self.stream_gaps)/1000
        return dict(capacity=self.capacity,pinned_slots=len(self.slots),
                    pinned_bytes=sum(s.numel()*s.element_size() for s in self.slots),
                    buffer_wait_count=self.buffer_wait_count,buffer_wait_seconds=self.buffer_wait_seconds,
                    buffer_wait_max_seconds=self.buffer_wait_max_seconds,
                    compute_stream_gap_seconds=gap,drain_seconds=self.drain_seconds,
                    transfer_count=self.transfer_count,transfer_seconds=self.transfer_seconds,
                    transfer_worker_wait_seconds=self.transfer_wait_seconds)

    def close(self):
        tick = time.monotonic()
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
            self.drain_seconds=time.monotonic()-tick
            self.wait_seconds += self.drain_seconds
