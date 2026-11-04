"""One decode producer; every consumer shares its immutable in-memory originals."""
from threading import Condition, Thread, current_thread
import time
import cv2
import numpy as np
from uav.frame_sampling import sample_count, source_index


class VideoFrames:
    def __init__(self, source, fps, start_source_frame=0, max_seconds=None, *, random_start=False, seed=20261019, buffer_limit=None):
        if buffer_limit is not None and (type(buffer_limit) is not int or buffer_limit < 1):
            raise ValueError('buffer_limit must be a positive integer')
        self.source, self.fps = source, fps
        self.start, self.max_seconds = start_source_frame, max_seconds
        self.random_start, self.seed = random_start, seed
        self.condition = Condition()
        self.images = []
        self.peak_buffered = 0
        self.base = 0
        self.buffer_limit = buffer_limit
        self.closed = self.done = self.metadata_ready = False
        self.error = None
        self.decoded_frames = self.bytes = 0
        self.buffer_wait_seconds = 0.
        self.decode_seconds = 0.
        self.wait_seconds = {}
        self.thread = Thread(target=self._decode,name='uav-frame-decode',daemon=True)
        self.thread.start()
        # Only metadata is needed to start the SAM and tracking consumers.
        with self.condition:
            self.condition.wait_for(lambda: self.metadata_ready or self.done)
            error = self.error if not self.metadata_ready else None
        if error is not None:
            self.close()
            raise error

    def _decode(self):
        tick = time.monotonic()
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.source))
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            self.source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or self.source_fps <= 0 or self.source_count <= 0:
                raise ValueError('invalid video')
            if self.random_start:
                if self.max_seconds is None:
                    raise ValueError('random_start requires max_seconds')
                import random
                last = max(0, self.source_count-int(np.ceil(self.max_seconds*self.source_fps)))
                self.start = random.Random(self.seed).randint(0, last)
            if not 0 <= self.start < self.source_count:
                raise ValueError('start_source_frame outside source')
            self.count = sample_count(self.source_count-self.start,self.source_fps,self.fps,self.max_seconds)
            # Keep the true final original of the requested interval.
            available = self.source_count-self.start
            interval_count = available if self.max_seconds is None else min(available,
                max(1,int(np.ceil(self.max_seconds*self.source_fps-1e-9))))
            self.end_source = self.start+interval_count-1
            sources = [self.start+source_index(f,self.source_fps,self.fps) for f in range(self.count)]
            if self.count > 1:
                sources[-1] = self.end_source
            self.sources = tuple(sources)
            with self.condition:
                self.metadata_ready = True
                self.condition.notify_all()
            if self.start and not cap.set(cv2.CAP_PROP_POS_FRAMES,self.start):
                raise RuntimeError('video seek failed')
            previous = self.start-1
            image = None
            for source in self.sources:
                with self.condition:
                    if self.closed:
                        return
                fresh = source != previous
                if fresh:
                    for _ in range(source-previous-1):
                        with self.condition:
                            if self.closed:
                                return
                        if not cap.grab():
                            raise RuntimeError(f'cannot skip toward source frame {source}')
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f'cannot decode source frame {source}')
                    image.flags.writeable = False
                    previous = source
                with self.condition:
                    if self.closed:
                        return
                    waiting=time.monotonic()
                    self.condition.wait_for(lambda: self.closed or self.buffer_limit is None
                                            or len(self.images) < self.buffer_limit)
                    self.buffer_wait_seconds += time.monotonic()-waiting
                    if self.closed:
                        return
                    self.images.append(image)
                    self.peak_buffered = max(self.peak_buffered,len(self.images))
                    if fresh:
                        self.decoded_frames += 1
                        self.bytes += image.nbytes
                    self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = error
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self.decode_seconds = time.monotonic()-tick
                self.done = True
                self.condition.notify_all()

    def source_id(self, frame):
        if not 0 <= frame < self.count:
            raise IndexError('frame outside interval')
        return self.sources[frame]

    def __call__(self, frame):
        if not 0 <= frame < self.count:
            raise IndexError('frame outside interval')
        with self.condition:
            if frame < self.base:
                raise IndexError('frame has been released')
            if not self.closed and self.base <= frame < self.base+len(self.images):
                return self.images[frame-self.base]
            tick = time.monotonic()
            self.condition.wait_for(lambda: self.base <= frame < self.base+len(self.images) or self.done or self.closed)
            name = current_thread().name
            self.wait_seconds[name] = self.wait_seconds.get(name,0.)+time.monotonic()-tick
            if self.closed:
                raise RuntimeError('frame reader is closed')
            if self.base <= frame < self.base+len(self.images):
                return self.images[frame-self.base]
            if self.error is not None:
                raise self.error
            raise RuntimeError('decoder ended before the requested frame')

    def release_before(self, frame):
        """Release completed segment originals; the decoder can advance again."""
        with self.condition:
            end = min(frame, self.base+len(self.images))
            if frame < self.base:raise ValueError('cannot release frames backwards')
            del self.images[:end-self.base]
            self.base = end
            self.condition.notify_all()

    def report(self):
        with self.condition:
            return dict(storage='memory',decoded_frames=self.decoded_frames,bytes=self.bytes,
                        buffer_limit=self.buffer_limit,peak_buffered_frames=self.peak_buffered,
                        retained_frames=len(self.images),
                        decode_seconds=self.decode_seconds,buffer_wait_seconds=self.buffer_wait_seconds,complete=self.done and self.error is None,
                        consumer_wait_seconds=dict(self.wait_seconds))

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()
        with self.condition:
            self.images.clear()
