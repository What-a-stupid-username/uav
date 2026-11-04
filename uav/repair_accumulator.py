"""Ordered host feature accumulation overlapped with the next GPU crop."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch


class FeatureAccumulator:
    def __init__(self,capacity=24):
        if capacity<1:raise ValueError('feature queue capacity must be positive')
        self.capacity=capacity
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='uav-feature-blend')
        self.pending=deque()

    @staticmethod
    @torch.inference_mode()
    def _add(canvas,feature,crop,weight,ready=None):
        if ready is not None:ready.synchronize()
        canvas.add(feature,crop,weight)

    def add(self,canvas,feature,crop,weight):
        while self.pending and self.pending[0].done():self.pending.popleft().result()
        if len(self.pending)>=self.capacity:self.pending.popleft().result()
        ready=None
        if feature.is_cuda:
            host=torch.empty(feature.shape,dtype=feature.dtype,pin_memory=True)
            host.copy_(feature,non_blocking=True)
            ready=torch.cuda.Event()
            ready.record(torch.cuda.current_stream(feature.device))
            feature=host
        self.pending.append(self.executor.submit(self._add,canvas,feature,crop,weight,ready))

    def finish(self):
        while self.pending:self.pending.popleft().result()

    def close(self):
        try:self.finish()
        finally:self.executor.shutdown(wait=True)
