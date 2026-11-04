"""Bounded ordered discovery producer; only the consumer commits evidence."""
from collections import deque
from contextlib import contextmanager
from threading import Condition, Thread
import time
from .schedule import discovery_frames


class DiscoveryBuffer:
    def __init__(self, requests, produce, *, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('discovery capacity must be a positive integer')
        self.requests, self.produce = iter(requests), produce
        self.capacity = capacity
        self.condition = Condition()
        self.ready = deque()
        self.closed = self.done = False
        self.error = None
        self.inflight = 0
        self.produced = self.consumed = self.peak_slots = 0
        self.producer_seconds = self.wait_seconds = 0.
        self.full_wait_seconds = 0.
        self.thread = Thread(target=self._run,name='uav-sam-discovery',daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            for key,request in self.requests:
                with self.condition:
                    tick = time.monotonic()
                    self.condition.wait_for(lambda: self.closed or len(self.ready)<self.capacity)
                    self.full_wait_seconds += time.monotonic()-tick
                    if self.closed:
                        break
                    self.inflight = 1
                    self.peak_slots = max(self.peak_slots,len(self.ready)+1)
                tick = time.monotonic()
                result = self.produce(request)
                elapsed = time.monotonic()-tick
                with self.condition:
                    self.inflight = 0
                    if self.closed:
                        break
                    self.ready.append((key,result))
                    self.produced += 1
                    self.producer_seconds += elapsed
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
        finally:
            with self.condition:
                self.inflight = 0
                self.done = True
                self.condition.notify_all()

    def take(self,key):
        tick=time.monotonic()
        with self.condition:
            self.condition.wait_for(lambda: self.ready or self.done or self.closed)
            self.wait_seconds += time.monotonic()-tick
            if not self.ready:
                if self.error is not None:
                    raise self.error
                raise RuntimeError('discovery producer ended before the requested keyframe')
            actual,result = self.ready[0]
            if actual != key:
                raise ValueError(f'discovery order mismatch: expected {actual}, got {key}')
            self.ready.popleft()
            self.consumed += 1
            self.condition.notify_all()
            return result

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join()
        with self.condition:
            self.ready.clear()

    def __exit__(self,*exc):
        self.close()

    def report(self):
        with self.condition:
            return dict(capacity=self.capacity,produced=self.produced,consumed=self.consumed,
                peak_slots=self.peak_slots,producer_seconds=self.producer_seconds,
                consumer_wait_seconds=self.wait_seconds,
                producer_full_wait_seconds=self.full_wait_seconds,storage='memory')


@contextmanager
def prefetch_discoveries(engine,frames,*,capacity=8):
    """One SAM producer reading the shared frame table, joined before Stage2."""
    if not hasattr(engine.detector, 'prepare_global'):
        yield
        return
    def produce(frame):
        return engine.detector.prepare_global(frames(frame),frames.source_id(frame))
    stride=engine.config.global_stride
    # Every scheduled timestamp is consumed, including repeated source frames.
    # SAM's original-source cache reuses inference without dropping the check.
    requests=((f,f) for f in discovery_frames(frames.count,stride))
    buffer=DiscoveryBuffer(requests,produce,capacity=capacity)
    engine.discovery_buffer=buffer
    try:
        with buffer:
            yield
    finally:
        engine.discovery_buffer=None
        engine.discovery_prefetch=buffer.report()
