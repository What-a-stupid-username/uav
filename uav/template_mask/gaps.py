"""Two independently validated trajectory sides support conservative gaps.

Each side must predict the entire gap with small error and agree with the
other side. Joining endpoints with an invented bend does not pass this test.
Predictions retain unknown presence and never teach templates or motion fits.
"""
import numpy as np
from .geometry import bounds, components, discrepancy, footprint, from_components
from .motion import MotionPrior
from .records import Pose
from .presence import SAM_ABSENCE_REASONS


def path_continuation(left,right,template,sample_times=None):
    """Measured endpoint geometry supports variable-speed straight legs.

    This certifies association only: it never creates interpolated poses or
    evidence. A large gap jump relative to both local measured speeds fails.
    """
    if min(len(left),len(right))<8:return None
    a,b=left[-8:],right[:8]
    when=lambda p: (sample_times or {}).get(p.frame,p.frame)
    gap=when(b[0])-when(a[-1])
    if gap<=0:return None
    values=[np.array([components(p.matrix,template.anchor) for p in rows]) for rows in (a,b)]
    noise=max(.7,max(p.uncertainty for p in (*a,*b)))
    size=footprint(template.box)
    axes=[];speeds=[]
    for rows,v in zip((a,b),values):
        xy=v[:,:2];center=xy.mean(0)
        _,_,basis=np.linalg.svd(xy-center,full_matrices=False)
        axis=basis[0]
        if np.dot(xy[-1]-xy[0],axis)<0:axis=-axis
        if np.linalg.norm(xy[-1]-xy[0])<6*noise:return None
        if np.max(np.abs((xy-center)@basis[1]))>3*noise:return None
        if np.min(np.diff(xy@axis)) < -3*noise:return None
        dt=np.diff([when(p) for p in rows])
        if np.any(dt<=0):return None
        speeds.extend((np.linalg.norm(np.diff(xy,axis=0),axis=1)/dt).tolist())
        axes.append(axis)
    displacement=values[1][0,:2]-values[0][-1,:2]
    if np.dot(*axes)<.95:return None
    for axis in axes:
        if abs(np.cross(axis,displacement))>6*noise:return None
        if np.dot(axis,displacement)<-3*noise:return None
    if np.linalg.norm(displacement)>4*max(speeds)*gap+6*noise:return None
    # Position continuity cannot excuse an unrelated rotation or scale jump.
    for j in (2,3):
        x,y=values[0][:,j],values[1][:,j]
        if j==2:
            x=np.unwrap(x);y=np.unwrap(y);y+=2*np.pi*round((x[-1]-y[0])/(2*np.pi))
        rate=max(np.max(np.abs(np.diff(x))/np.diff([when(p) for p in a])),
                 np.max(np.abs(np.diff(y))/np.diff([when(p) for p in b])))
        if abs(y[0]-x[-1])*size>4*rate*size*gap+6*noise:return None
    return dict(kind='measured_straight_path',left_sources=[p.source_id for p in a],
                right_sources=[p.source_id for p in b],gap_time=float(gap))


def measured(instance):
    return sorted((p for p in instance.poses.values() if p.status == 'measured'
                   and p.presence and p.source_id is not None), key=lambda p: p.frame)


def boundary_prior(rows, template, side, *, fit_cache=None):
    """An independently validated straight leg is strong even on a bent path.

    Inspect the observations nearest this boundary, never the opposite end of
    the recording. Prefer the longest validated straight prefix/suffix; turns
    elsewhere must not weaken it. Failing that, retain the normal curve prior.
    """
    for length in (64, 32, 16, 8):
        if len(rows) < length:
            continue
        subset = rows[-length:] if side == 'left' else rows[:length]
        prior = MotionPrior(subset, template.box, template.anchor,fit_cache=fit_cache,whole_only=True)
        whole = prior.whole
        if (whole is not None and all(m is not None for m in whole[0])
                and all(m.mode in ('static', 'linear') for m in whole[0][:2])):
            return MotionPrior(subset,template.box,template.anchor,fit_cache=fit_cache)
    return MotionPrior(rows, template.box, template.anchor, fit_cache=fit_cache)


def trajectory_bridge(left, right, template, frames, *, fit_cache=None, sample_times=None):
    if sample_times is not None:
        # Physical motion follows original timestamps, not the unevenly sampled
        # output indices. Preserve output IDs as prediction keys only.
        from dataclasses import replace
        frames = tuple(frames)
        if all(p.frame in sample_times for p in (*left,*right)) and all(f in sample_times for f in frames):
            result = trajectory_bridge([replace(p,frame=sample_times[p.frame]) for p in left],
                [replace(p,frame=sample_times[p.frame]) for p in right],template,
                [sample_times[f] for f in frames],fit_cache=fit_cache)
            if result is None:return None
            return {f:result[0][sample_times[f]] for f in frames},dict(result[1],time_basis='source_frames')
    a = boundary_prior(left, template, 'left', fit_cache=fit_cache)
    b = boundary_prior(right, template, 'right', fit_cache=fit_cache)
    if len(a.evidence) < 8 or len(b.evidence) < 8:
        return None
    predictions = {}
    for frame in frames:
        pa, pb = a.predict(frame), b.predict(frame)
        if pa is None or pb is None or not pa[2] or not pb[2]:
            return None
        size = min(footprint(bounds(pa[0], template.box)), footprint(bounds(pb[0], template.box)))
        if max(pa[1], pb[1]) > .08*size:
            return None
        error = discrepancy(pa[0], pb[0], template.box)
        if error > 3*(pa[1]+pb[1]):
            return None
        # Select the closer independent side; uncertainty covers disagreement.
        chosen = pa if abs(frame-left[-1].frame) <= abs(frame-right[0].frame) else pb
        predictions[frame] = (chosen[0], max(pa[1], pb[1])+error)
    return predictions, dict(left=a.report(), right=b.report())


def review_measurements(engine):
    """Quarantine isolated single-reference conflicts with two-sided motion.

    Raw matches remain in the audit trail. A supported real turn, a weak corner,
    sparse history or disagreement between the two sides never gets smoothed.
    Decisions use original measurements only and are applied simultaneously.
    """
    e = engine
    changes = []
    for instance in e.active_instances():
        rows = measured(instance)
        template = e.bank.templates[instance.template_id]
        for index, pose in enumerate(rows):
            if index < 8 or len(rows)-index-1 < 8:
                continue
            sources = {e.bank.observations[ref].source_id for ref in pose.references
                if ref in e.bank.observations and e.bank.observations[ref].source_id != pose.source_id}
            if len(sources) >= 2 or e.associated_observations(instance, pose.frame):
                continue
            # Cheap triage only: normal dense samples need no pair of model
            # fits. An interpolation discrepancy never rejects a measurement.
            before, after = rows[index-1], rows[index+1]
            weight = (pose.frame-before.frame)/(after.frame-before.frame)
            endpoints = np.array([components(p.matrix, template.anchor) for p in (before,after)])
            endpoints[:,2] = np.unwrap(endpoints[:,2])
            interpolated = from_components((1-weight)*endpoints[0]+weight*endpoints[1], template.anchor)
            if discrepancy(interpolated, pose.matrix, template.box) <= 3*(
                    pose.uncertainty+before.uncertainty+after.uncertainty):
                continue
            # Restrict both fits to the neighbourhood of this measurement;
            # unrelated earlier/later motion legs cannot certify its position.
            left, right = rows[max(0,index-32):index], rows[index+1:index+33]
            bridge = trajectory_bridge(left, right, template, [pose.frame],
                                       fit_cache=getattr(e,'motion_fits',None),
                                       sample_times=getattr(e,'source_ids',None))
            if bridge is None:
                continue
            matrix, sigma = bridge[0][pose.frame]
            error = discrepancy(matrix, pose.matrix, template.box)
            if error <= 3*(sigma+pose.uncertainty):
                continue
            changes.append((instance, pose, matrix, sigma, error, bridge[1]))
    for instance, original, matrix, sigma, error, evidence in changes:
        e.pose_reviews.append(dict(instance=instance.id, template=instance.template_id,
            frame=original.frame, source=original.source_id, reason='isolated_single_reference_conflict',
            original_matrix=original.matrix.tolist(), original_uncertainty=original.uncertainty,
            original_references=original.references, replacement_matrix=matrix.tolist(),
            replacement_uncertainty=sigma, disagreement=error, evidence=evidence))
        instance.poses[original.frame] = Pose(original.frame, matrix.copy(), sigma, 'predicted',
            presence=False, reason='two_sided_measurement_review')
    return {instance.id for instance, *_ in changes}


class GapResolver:
    def __init__(self, engine):
        self.engine = engine
        self.events = []
        self.output_predictions = {}

    def _check(self, left, right, template, owners):
        start, end = left[-1].frame, right[0].frame
        if end <= start+1:
            return None
        for instance in owners:
            if any(start < p.frame < end and p.status == 'absent'
                   and p.reason not in SAM_ABSENCE_REASONS for p in instance.poses.values()):
                return None
        return trajectory_bridge(left, right, template, range(start, end+1),
                                 fit_cache=self.engine.motion_fits,
                                 sample_times=getattr(self.engine,'source_ids',None))

    def _fill(self, instance, start, end, bridge, image_at):
        e = self.engine
        predictions, evidence = bridge
        matched, predicted = [], []
        for frame in range(start+1, end):
            pose = instance.poses.get(frame)
            if pose is not None and pose.status == 'measured':
                continue
            # An independently established explicit disappearance would block a
            # bridge. The old SAM-miss heuristic alone is not such evidence.
            if pose is not None and pose.status == 'absent' and pose.reason not in SAM_ABSENCE_REASONS:
                continue
            matrix, sigma = predictions[frame]
            # Keep the validated two-sided output estimate through finalization.
            # A fragment's own one-sided prior must not replace this with a hold.
            self.output_predictions[instance.id,frame] = (matrix.copy(),sigma,True)
            # One targeted retry using two-sided geometry. Normal tracking can
            # now accept image fits close to this independently supported pose.
            if image_at is not None:
                e._track(instance, frame, e.source_ids[frame], image_at(frame), backward=True,
                         bridge_prediction=(matrix, sigma, True))
            pose = instance.poses.get(frame)
            if pose is not None and pose.status == 'measured':
                matched.append(frame)
            else:
                instance.poses[frame] = Pose(frame, matrix.copy(), sigma, 'predicted',
                    presence=False, reason='two_sided_trajectory_presence_unconfirmed')
                predicted.append(frame)
            e._crossings(frame)
        e.stats['trajectory_gap_matches'] += len(matched)
        e.stats['trajectory_gap_predictions'] += len(predicted)
        self.events.append(dict(instance=instance.id, template=instance.template_id,
            first_frame=start+1, last_frame=end-1, measured_frames=matched,
            predicted_frames=predicted, evidence=evidence))

    def resolve(self, image_at):
        e = self.engine
        # Rebuild from this geometry snapshot, never retain a revoked bridge.
        self.output_predictions = {}
        # First repair gaps already belonging to one instance.
        for instance in e.active_instances():
            rows = measured(instance)
            for index, (a, b) in enumerate(zip(rows, rows[1:])):
                if b.frame <= a.frame+1:
                    continue
                template = e.bank.templates[instance.template_id]
                bridge = self._check(rows[:index+1], rows[index+1:], template, (instance,))
                if bridge is not None:
                    self._fill(instance, a.frame, b.frame, bridge, image_at)
        # Two detections separated by a weak absence heuristic may have become
        # separate instances. Preserve both records and record a canonical ID;
        # fill only the gap, never duplicate the later measured trajectory.
        options = []
        instances = e.active_instances()
        for a in instances:
            left = measured(a)
            if not left:
                continue
            members = e.bank.members(a.template_id)
            for b in instances:
                if a.id == b.id or b.template_id not in members:
                    continue
                right = measured(b)
                if not right or left[-1].frame >= right[0].frame:
                    continue
                if any(c.id not in (a.id, b.id) and c.template_id in members
                       and any(left[-1].frame <= p.frame <= right[0].frame for p in measured(c))
                       for c in instances):
                    continue
                # No contemporaneous occurrence of this identity may compete
                # for the same gap. Uniqueness is checked below in both directions.
                template = e.bank.templates[a.template_id]
                inverse = np.linalg.inv(members[b.template_id])
                registered = [Pose(p.frame,p.matrix@inverse,p.uncertainty,p.status,
                    p.source_id,p.references,p.presence) for p in right]
                bridge = self._check(left, registered, template, (a, b))
                if bridge is not None:
                    options.append((a, b, left[-1].frame, right[0].frame, bridge))
        for a, b, start, end, bridge in options:
            if sum(x.id == a.id for x, *_ in options) != 1:
                continue
            if sum(y.id == b.id for _, y, *_ in options) != 1:
                continue
            # Retain provenance and permit later review/revocation of the link.
            b.continuation_of = a.continuation_of if a.continuation_of is not None else a.id
            # Earlier extrapolations on either side are retained as provenance,
            # but only one segment renders each timestamp after association.
            a.render_end = end-1
            b.render_start = end
            self._fill(a, start, end, bridge, image_at)
            self.events[-1]['continues_with_instance'] = b.id
