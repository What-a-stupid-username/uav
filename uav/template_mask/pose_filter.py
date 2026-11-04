"""Measured-only motion models for final masks, boxes and review trajectories."""
import numpy as np
import time
from .geometry import components, discrepancy, footprint, from_components
from .motion import robust_fit


def output_matrix(instance, frame, matrix, anchor):
    """Apply one final motion estimate without rewriting original evidence."""
    constraints = getattr(instance, 'output_constraints', {}).get(frame)
    if constraints is None:
        return matrix
    value = components(matrix, anchor)
    supported = np.isfinite(constraints)
    value[supported] = constraints[supported]
    return from_components(value, anchor)


def _candidates(prior, template):
    """Reuse fitted models; reject static search approximations with drift.

    There is no model search here. The only extra fit is a two-column linear
    diagnostic for position components that the search prior calls static.
    Its role is to veto coordinate locking, never to invent a new motion mode.
    """
    times = np.array([p.frame for p in prior.evidence], float)
    values = np.array([components(p.matrix,template.anchor) for p in prior.evidence])
    sigma = np.array([p.uncertainty for p in prior.evidence])
    result = []
    for original,start,end in sorted(prior.segments,key=lambda s:s[2]-s[1],reverse=True):
        ids = np.flatnonzero((times>=start)&(times<=end))
        if len(ids)<8 or end<=start:
            continue
        models = list(original)
        noise = max(.7,float(np.median(sigma[ids])))
        for j in (0,1):
            if models[j] is None or models[j].mode!='static':
                continue
            time = (times[ids]-start)/(end-start)
            design = np.column_stack((np.ones(len(ids)),time))
            coefficient = robust_fit(design,values[ids,j])
            blocks = [float(np.median(values[part,j])) for part in np.array_split(ids,4)]
            if abs(coefficient[1])>2*noise or np.ptp(blocks)>2*noise:
                models[j] = None
        result.append(dict(start=start,end=end,models=models,
            measured_sources=len(ids),sigma=noise))
    return result


def prepare_output_filter(engine):
    """Filter supported motion, preserving turns and independently real changes.

    Predictions never supply fitting samples. Final estimates are stored apart
    from raw matches and cannot teach tracking, identity or template geometry.
    Unsupported components and corroborated departures keep the measured pose.
    """
    tick = time.perf_counter()
    reports = []
    for instance in engine.instances.values():
        instance.output_constraints = {}
        instance.boundary_output_frames = set()
        instance.boundary_output_previous = {}
    for instance in engine.active_instances():
        prior = instance.motion
        if prior is None or len(prior.evidence)<8:
            continue
        template = engine.bank.templates[instance.template_id]
        candidates = _candidates(prior,template)
        frames = np.array(sorted(instance.poses),int)
        poses = [instance.poses[f] for f in frames]
        values = np.array([components(p.matrix,template.anchor) for p in poses])
        output = np.full_like(values,np.nan)
        scales = np.array([1.,1.,footprint(template.box),footprint(template.box)])
        sigmas = np.array([p.uncertainty for p in poses])
        present = np.array([p.status!='absent' for p in poses])
        observations = engine.bank.observations
        corroborated = np.array([p.status=='measured' and len({observations[ref].source_id
            for ref in p.references if ref in observations
            and observations[ref].source_id!=p.source_id})>=2 for p in poses])
        selected = []
        # Long intervals win. Position coordinates always share their temporal
        # support. Each model evaluates all eligible frames in one batch.
        for indices in ((0,1),(2,),(3,)):
            unassigned = present.copy()
            for candidate in candidates:
                models = [candidate['models'][j] for j in indices]
                if any(m is None for m in models):
                    continue
                ids = np.flatnonzero(unassigned & (frames>=candidate['start'])
                                     & (frames<=candidate['end']))
                if not len(ids):
                    continue
                prediction = np.column_stack([m.design(frames[ids])@m.coefficient for m in models])
                actual = values[np.ix_(ids,indices)].copy()
                if indices==(2,):
                    actual += 2*np.pi*np.rint((prediction-actual)/(2*np.pi))
                error = np.max(np.abs(prediction-actual)*scales[list(indices)],axis=1)
                gate = 3*(sigmas[ids]+candidate['sigma'])
                keep = error<=np.where(corroborated[ids],gate,2*gate)
                ids,prediction = ids[keep],prediction[keep]
                if not len(ids):
                    continue
                output[np.ix_(ids,indices)] = prediction
                unassigned[ids] = False
                selected.append(dict(start=candidate['start'],end=candidate['end'],
                    components=[('x','y','rotation','scale')[j] for j in indices],
                    modes=[m.mode for m in models],measured_sources=candidate['measured_sources'],
                    output_frames=len(ids)))
        instance.output_constraints = {int(f):row.copy() for f,row in zip(frames,output)
                                       if np.isfinite(row).any()}
        if instance.output_constraints:
            maximum = max(discrepancy(output_matrix(instance,f,instance.poses[f].matrix,template.anchor),
                instance.poses[f].matrix,template.box) for f in instance.output_constraints)
            reports.append(dict(instance=instance.id,template=template.id,mode='validated_motion_models',
                intervals=selected,frames=len(instance.output_constraints),
                max_corner_correction_pixels=maximum))
    engine.output_pose_filter = reports
    if hasattr(engine,'timings'):
        engine.timings['output_pose_filter_seconds'] = time.perf_counter()-tick
