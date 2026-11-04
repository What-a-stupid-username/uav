"""Resolve FFmpeg from an explicit override, runtime PATH, or packaged fallback."""
import os
import shutil


def executable():
    explicit=os.environ.get('IMAGEIO_FFMPEG_EXE')
    if explicit:return explicit
    installed=shutil.which('ffmpeg')
    if installed:return installed
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()
