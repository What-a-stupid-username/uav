"""Cheap, measured-only negative evidence for persistently incoherent tracks.

A failed global fit is not negative evidence: curves, corners and sparse
fragments are legitimate. Only repeated, significant local contradictions
throughout a sufficiently observed lifetime restrict inferred coverage.
"""
import time
import numpy as np
from .geometry import components, footprint


def assess(instance, template, observations, stride):
    prior = instance.motion
    rows = [] if prior is None else prior.evidence
    result = dict(status='insufficient', action='retain', measured_sources=len(rows),
                  reason='insufficient_independent_motion_observations')
    if len(rows) < 24:
        return result
    times = np.array([p.frame for p in rows], float)
    result.update(first_frame=int(times[0]), last_frame=int(times[-1]))
    if times[-1]-times[0] < 3*stride:
        return result
    values = np.array([components(p.matrix, template.anchor) for p in rows])
    # Do not interpret a sparse jump between flashes as frame-to-frame jitter.
    dense = (np.diff(times)[:-1] <= stride) & (np.diff(times)[1:] <= stride)
    indices = np.flatnonzero(dense)+1
    if len(indices) < 18:
        return result
    weight = (times[indices]-times[indices-1])/(times[indices+1]-times[indices-1])
    values[:, 2] = np.unwrap(values[:, 2])
    expected = ((1-weight[:, None])*values[indices-1]
                + weight[:, None]*values[indices+1])
    noise = np.array([max(.7, p.uncertainty) for p in rows])
    tolerance = np.maximum(.01*footprint(template.box), 3*(noise[indices]
        + (1-weight)*noise[indices-1] + weight*noise[indices+1]))
    error = np.abs(values[indices]-expected)
    error[:, 2] = np.abs(np.angle(np.exp(1j*(values[indices, 2]-expected[:, 2]))))
    error[:, 2:] *= footprint(template.box)*np.exp(values[indices, 3, None])
    error = np.column_stack((np.linalg.norm(error[:, :2], axis=1), error[:, 2:]))
    jitter = error > tolerance[:, None]
    supported = np.zeros((len(rows), 3), bool)
    for models, start, end in prior.segments:
        ids = np.flatnonzero((times >= start) & (times <= end))
        pixels = footprint(template.box)*np.exp(values[ids, 3])
        gate = np.maximum(3*noise[ids], .01*pixels)
        for j, valid in enumerate((all(m is not None for m in models[:2]),
                                   models[2] is not None, models[3] is not None)):
            if not valid:
                continue
            indices_for_component = ((0,1), (2,), (3,))[j]
            prediction = np.column_stack([models[k].design(times[ids])@models[k].coefficient
                                          for k in indices_for_component])
            delta = values[np.ix_(ids, indices_for_component)]-prediction
            if j == 1:
                delta = np.angle(np.exp(1j*delta))
            residual = np.linalg.norm(delta, axis=1)*(pixels if j else 1.)
            supported[ids, j] |= residual <= gate
    corroborated = np.array([len({observations[ref].source_id for ref in p.references
        if ref in observations and observations[ref].source_id != p.source_id}) >= 2
        for p in rows])
    # All three temporal thirds must contradict local continuity. A noisy
    # corner or a short bad interval cannot condemn the entire occurrence.
    thirds = np.minimum(2, ((times[indices]-times[0])/(times[-1]-times[0])*3).astype(int))
    rates = [jitter[thirds == part].mean(axis=0).tolist()
             if np.count_nonzero(thirds == part) >= 6 else None for part in range(3)]
    coverage = supported.mean(axis=0)
    incoherent = any(all(rate is not None and rate[j] >= .6 for rate in rates)
                     and coverage[j] < .2 for j in range(3))
    independently_supported = corroborated.mean() >= .25
    suspect = incoherent and not independently_supported
    result.update(status='suspected_scene' if suspect else
                  'image_supported_irregular' if incoherent else 'retained',
                  action='observed_only' if suspect else 'retain',
                  reason='persistent_local_motion_contradictions' if suspect else
                         'independent_images_support_irregular_motion' if incoherent else
                         'no_persistent_motion_contradiction',
                  local_tests=len(indices), components=['position', 'rotation', 'scale'],
                  jitter_fraction=jitter.mean(axis=0).tolist(),
                  jitter_fraction_by_third=rates,
                  coherent_fraction=coverage.tolist(),
                  corroborated_fraction=float(corroborated.mean()))
    return result


def prepare_trajectory_quality(engine):
    tick = time.perf_counter()
    for instance in engine.instances.values():
        instance.trajectory_quality = {}
    for instance in engine.active_instances():
        instance.trajectory_quality = assess(instance,
            engine.bank.templates[instance.template_id], engine.bank.observations,
            engine.config.global_stride)
    engine.timings['trajectory_quality_seconds'] = time.perf_counter()-tick


def observed_only(instance, frame=None):
    return (getattr(instance, 'trajectory_quality', {}).get('action') == 'observed_only'
            or frame is not None and any(row['first_frame']<=frame<=row['last_frame']
                for row in getattr(instance,'segmentation_quality',{}).get('intervals',())))
