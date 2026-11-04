"""One combined review video; preserve both panels at native resolution."""
from contextlib import contextmanager
import subprocess
from uav import ffmpeg


def encoder_command(output, filename, layout, fps):
    return [ffmpeg.executable(),'-v','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s',f'{layout["width"]}x{layout["height"]}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','8',
        '-preset','veryfast','-crf',str(layout['crf']),'-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output/filename)]


@contextmanager
def review_writer(output,filename,layout,fps):
    process=subprocess.Popen(encoder_command(output,filename,layout,fps),stdin=subprocess.PIPE)
    try:
        yield process.stdin
    finally:
        try:
            process.stdin.close()
        finally:
            code=process.wait()
        if code:
            raise RuntimeError(f'review encoding failed: {filename} (exit {code})')
