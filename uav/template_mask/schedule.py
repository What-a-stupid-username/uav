"""Discovery timestamps shared by the producer, consumer and debug display."""


def discovery_frames(count, stride):
    if count <= 0:
        return []
    frames = list(range(0, count, stride))
    if frames[-1] != count-1:
        frames.append(count-1)
    return frames


def is_discovery_frame(frame, stride, count=None):
    return frame % stride == 0 or count is not None and frame == count-1
