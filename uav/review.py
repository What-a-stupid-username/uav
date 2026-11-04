"""Create a source/result/RGB delta review video without model inference."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
from uav import ffmpeg
import numpy as np


def compose(source, result, output, gain=4, masks=None, opacity=.35, max_frames=None, start_source_frame=0, boxes=None, debug_overlay=None):
    if not isinstance(start_source_frame,int) or start_source_frame<0:
        raise ValueError('start source frame must be a nonnegative integer')
    if max_frames is not None and (not isinstance(max_frames,int) or max_frames<1):
        raise ValueError('max frames must be a positive integer')
    if not 0 <= opacity <= 1:
        raise ValueError('opacity must be between 0 and 1')
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('gain must be positive')
    if output.resolve() in (source.resolve(), result.resolve()):
        raise ValueError('output must differ from inputs')
    box_rows={} if boxes is None else json.loads(Path(boxes).read_text())
    original = cv2.VideoCapture(str(source))
    repaired = cv2.VideoCapture(str(result))
    process = None
    frames = 0
    try:
        if not original.isOpened() or not repaired.isOpened():
            raise ValueError('could not open both videos')
        width = int(original.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = original.get(cv2.CAP_PROP_FPS)
        fps = repaired.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or source_fps <= 0:
            raise ValueError('invalid input FPS')
        from .frame_sampling import source_index
        decoded_source=start_source_frame-1
        if start_source_frame:original.set(cv2.CAP_PROP_POS_FRAMES,start_source_frame)
        source_frame=None
        output.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen([
            ffmpeg.executable(), '-v', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width*3}x{height}',
            '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
        while max_frames is None or frames < max_frames:
            ok_b, b = repaired.read()
            if not ok_b:
                if max_frames is not None:raise ValueError('result ended before requested frame count')
                break
            target=start_source_frame+source_index(frames,source_fps,fps)
            advanced=False
            while decoded_source<target:
                if not original.grab():raise ValueError('source ended before result')
                decoded_source+=1;advanced=True
            if advanced:
                ok_a,source_frame=original.retrieve()
                if not ok_a:raise ValueError('could not decode source frame')
            a=source_frame
            if a.shape != b.shape or a.shape[:2] != (height, width):
                raise ValueError('input dimensions differ')
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            delta = np.clip(np.abs(a.astype(np.int16)-b.astype(np.int16))*gain,
                            0, 255).astype(np.uint8)
            overlay = a
            if masks is not None:
                # Review masks are RGBA PNGs with nonzero alpha on final mask pixels.
                path = masks / f'{frames:04d}-adaptive.png'
                mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.ndim != 3 or mask.shape[2] != 4 or mask.shape[:2] != (height, width):
                    raise ValueError(f'missing or invalid RGBA mask: {path}')
                active = mask[:, :, 3] > 0
                overlay = a.copy()
                color = np.array([40, 240, 140], dtype=np.float32)
                overlay[active] = np.rint(a[active]*(1-opacity)+color*opacity).astype(np.uint8)
            if str(frames) in box_rows:
                overlay=overlay.copy()
                row=box_rows[str(frames)]
                for box in row['global_boxes']:draw_box(overlay,box,(40,200,255))
                for box in row['local_regions']:draw_box(overlay,box,(255,50,50))
                for box in row['refined_boxes']:draw_box(overlay,box,(40,255,80),dashed=True)
            if debug_overlay is not None:
                overlay=debug_overlay(frames,overlay.copy())
            # Delta always compares the untouched original, never the tinted overlay.
            process.stdin.write(np.concatenate((overlay, b, delta), axis=1).tobytes())
            frames += 1
    finally:
        original.release()
        repaired.release()
        if process is not None:
            process.stdin.close()
            code = process.wait()
            if code:
                raise RuntimeError(f'encoder exited {code}')
    if not frames:
        raise ValueError('no frames decoded')
    return frames


def draw_box(image, box, color, dashed=False):
    h,w=image.shape[:2]
    l,t,r,b=[int(round(x)) for x in box]
    l,r=max(0,min(w-1,l)),max(0,min(w-1,r))
    t,b=max(0,min(h-1,t)),max(0,min(h-1,b))
    if r<=l or b<=t:return
    if not dashed:
        cv2.rectangle(image,(l,t),(r,b),color,1)
        return
    for x in range(l,r,10):
        cv2.line(image,(x,t),(min(x+5,r),t),color,1)
        cv2.line(image,(x,b),(min(x+5,r),b),color,1)
    for y in range(t,b,10):
        cv2.line(image,(l,y),(l,min(y+5,b)),color,1)
        cv2.line(image,(r,y),(r,min(y+5,b)),color,1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gain', type=float, default=4)
    parser.add_argument('--masks', type=Path, help='directory of debug RGBA mask PNGs')
    parser.add_argument('--opacity', type=float, default=.35)
    args = parser.parse_args()
    print(f'Wrote {compose(args.source, args.result, args.output, args.gain, args.masks, args.opacity)} frames')


if __name__ == '__main__':
    main()
