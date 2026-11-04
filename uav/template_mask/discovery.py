"""Aggregate one SAM discovery before creating persistent object identities.

Reuse legacy UAV's pixel/component and competitive candidate grouping. Raw
probabilities remain immutable evidence; grouping changes ownership, not pixels.
"""
import numpy as np
from .records import Candidate


def aggregate_candidates(candidates, shape, threshold):
    from uav.sam3_pixel_objects import pixel_objects
    from uav.sam3_candidate_objects import candidate_objects
    import torch

    if not candidates:
        return []
    h, w = shape
    probability = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    supports = []
    for candidate in candidates:
        p = np.asarray(candidate.probability, np.float32)
        v = np.ones(p.shape, bool) if candidate.valid is None else candidate.valid
        x, y = map(int, candidate.origin)
        l, t, r, b = max(0, x), max(0, y), min(w, x+p.shape[1]), min(h, y+p.shape[0])
        if r <= l or b <= t:
            supports.append(None)
            continue
        local = p[t-y:b-y, l-x:r-x]
        known = v[t-y:b-y, l-x:r-x]
        probability[t:b, l:r] = np.maximum(probability[t:b, l:r], np.where(known, local, 0.))
        valid[t:b, l:r] |= known
        support = (local > threshold) & known
        supports.append(((l, t, r, b), torch.from_numpy(support)))
    union = (probability > threshold) & valid
    if not union.any():
        return []
    boxes, objects = pixel_objects(torch.from_numpy(union))
    boxes, objects = candidate_objects(boxes, objects, [s for s in supports if s is not None])
    result = []
    for index, (l, t, r, b) in enumerate(boxes):
        member = objects[index].numpy()
        sources = []
        scores = []
        for candidate, support in zip(candidates, supports):
            if support is None:
                continue
            (x, y, z, v), pixels = support
            if np.any(member[y:v, x:z] & pixels.numpy()):
                sources.extend(candidate.source_candidates or (candidate.candidate_id,))
                scores.append(candidate.score)
        # Keep background probabilities and a small interpolation fringe, while
        # making pixels owned by a different object unknown for this group.
        x, y, z, v = max(0, l-2), max(0, t-2), min(w, r+2), min(h, b+2)
        p = probability[y:v, x:z].copy()
        known = valid[y:v, x:z].copy()
        foreign = union[y:v, x:z] & ~member[y:v, x:z]
        p[foreign] = 0.
        known[foreign] = False
        result.append(Candidate(p, (x, y), (l, t, r, b), max(scores, default=0.),
            'group:'+str(index), known, tuple(dict.fromkeys(sources))))
    return result
