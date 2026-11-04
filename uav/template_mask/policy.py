"""Visible failure policy and contiguous warning records for uncertain masks."""
import json
import logging
from pathlib import Path
import numpy as np
from .geometry import dilate

MAX_UNCERTAINTY_PADDING = 12  # Source pixels in addition to normal output dilation.


def uncertainty_radius(sigma):
    return min(MAX_UNCERTAINTY_PADDING, max(0, int(np.ceil(3*sigma))))


class SkipVideo(RuntimeError):
    pass


class WarningLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add(self, instance, template, frame, reason, action):
        for row in reversed(self.rows):
            if (row['instance'], row['template'], row['reason'], row['action']) == (instance, template, reason, action):
                if row['last_frame'] + 1 == frame:
                    row['last_frame'] = frame
                    return
                if row['first_frame'] <= frame <= row['last_frame']:
                    return
                break
        row = dict(instance=instance, template=template, first_frame=frame, last_frame=frame,
                   reason=reason, action=action)
        self.rows.append(row)
        logging.getLogger(__name__).warning('Template mask: %s', json.dumps(row))

    def flush(self):
        with self.path.open('w') as stream:
            for row in self.rows:
                stream.write(json.dumps(row)+'\n')


def uncertain_mask(mask, sigma, instance, frame, config, warnings, reason, *, diagnostic=False):
    policy = config.uncertain_mask_policy
    if policy == 'ignore':
        warnings.add(instance.id, instance.template_id, frame, reason, 'omit_uncertain')
        return np.zeros(mask.shape, bool)
    if policy == 'skip':
        warnings.add(instance.id, instance.template_id, frame, reason, 'skip_video')
        if not diagnostic:
            raise SkipVideo(reason)
    if not np.isfinite(sigma) or not mask.any():
        warnings.add(instance.id, instance.template_id, frame, reason, 'skip_unlocalizable')
        if diagnostic:
            return mask
        raise SkipVideo(reason)
    radius = uncertainty_radius(sigma)
    expanded = dilate(mask, radius)
    warnings.add(instance.id, instance.template_id, frame, reason,
        'cover_bounded_uncertainty' if 3*sigma > MAX_UNCERTAINTY_PADDING else 'cover_uncertainty')
    return expanded
