"""Exercise review resampling with fake captures; no media is read or written."""
import io
from pathlib import Path
from unittest.mock import patch
import numpy as np
from uav.review import compose


class Capture:
    def __init__(self, frames, fps):
        self.frames=frames;self.fps=fps;self.index=-1
    def isOpened(self):return True
    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FRAME_WIDTH:2,cv2.CAP_PROP_FRAME_HEIGHT:2,
                cv2.CAP_PROP_FPS:self.fps}.get(prop,0)
    def set(self, prop, value):self.index=int(value)-1;return True
    def grab(self):
        self.index+=1
        return self.index<len(self.frames)
    def retrieve(self):return True,self.frames[self.index].copy()
    def read(self):
        return self.retrieve() if self.grab() else (False,None)
    def release(self):pass


class Pipe(io.BytesIO):
    def close(self):self.saved=self.getvalue();super().close()


class Encoder:
    def __init__(self):self.stdin=Pipe()
    def wait(self):return 0


def main():
    from uav.frame_sampling import source_index
    for source_fps,target_fps,offset in [(60,24,0),(12,24,0),(24,24,0),(60,24,5),(12,24,5)]:
        frames=[np.full((2,2,3),i,dtype=np.uint8) for i in range(20)]
        count=6
        result=[frames[offset+source_index(i,source_fps,target_fps)] for i in range(count)]
        encoder=Encoder()
        with patch('uav.review.cv2.VideoCapture',side_effect=[Capture(frames,source_fps),Capture(result,target_fps)]), \
             patch('uav.review.subprocess.Popen',return_value=encoder), \
             patch('uav.ffmpeg.executable',return_value='/fake/ffmpeg'):
            assert compose(Path('/fake/source'),Path('/fake/result'),Path('/tmp/fake-review'),max_frames=count,start_source_frame=offset)==count
        output=np.frombuffer(encoder.stdin.saved,dtype=np.uint8).reshape(count,2,6,3)
        for i in range(count):
            np.testing.assert_array_equal(output[i,:,:2],result[i])
            assert not output[i,:,4:].any(), 'delta misaligned'
    print('Review timestamp mapping: downsample, upsample and native-rate passed')


if __name__=='__main__':main()
