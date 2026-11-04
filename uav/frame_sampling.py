"""Constant-rate output timestamps mapped to source frames without blending."""
import math


def sample_count(source_frames, source_fps, target_fps, max_seconds=None):
    if source_frames < 1 or any(not math.isfinite(x) or x <= 0
                                for x in (source_fps, target_fps)):
        raise ValueError('frame counts and rates must be positive and finite')
    duration = source_frames / source_fps
    if max_seconds is not None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('max seconds must be positive and finite')
        duration = min(duration, max_seconds)
    return max(1, math.ceil(duration * target_fps - 1e-9))


def source_index(frame, source_fps, target_fps):
    # Use the frame active at the output timestamp. Repeats support upsampling.
    return math.floor(frame * source_fps / target_fps + 1e-9)
