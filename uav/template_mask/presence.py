"""Directional boundary checks and continuous, evidence-free interval coverage."""
from dataclasses import replace
import numpy as np
from .geometry import area, bounds, clip_box, discrepancy, footprint, intersection, support_bounds
from .motion import MotionPrior
from .records import Pose


SAM_ABSENCE_REASONS = frozenset((
    'independent_local_absence_checks', 'global_detection_absent',
    'global_and_local_absence_checks'))
PRESENT_CHECKS = frozenset(('image_match', 'sam_candidate'))


def at_position(boxes, expected):
    return any(intersection(box, expected)/max(1., min(area(box), area(expected))) >= .4
               for box in boxes)


class PresenceResolver:
    def __init__(self, engine):
        self.engine = engine
        self.global_samples = {}
        self.checks = {}
        self.events = []
        self.dirty_frames = set()
        self.appearance_checked = set()
        self.completed = set()
        self.probes = {}
        self.sam_probes = set()
        self.coverage = {}
        self.edge_margins = {}
        self.probe_diagnostics = []

    def _observations(self, instance, frame):
        """Presence evidence belongs to an occurrence, not a spatial box."""
        e = self.engine
        observations = []
        for oid in e.frame_observations.get(frame, ()):
            obs = e.bank.observations[oid]
            if obs.ambiguous or obs.crossing_instances or obs.association_reason == 'stage2_refinement_rejected':
                continue
            assigned = obs.instance_id == instance.id and (obs.ownership_confirmed or obs.reliable)
            boundary = (obs.instance_id is None and obs.boundary_instance_id == instance.id
                        and obs.association_reason == 'boundary_sam_candidate')
            if assigned or boundary:
                observations.append(obs)
        return observations

    def _edge_prediction(self, instance, frame):
        """Continue a verified edge approach using measured motion uncertainty.

        Edge feature loss is not a new motion measurement. Fit native source
        times, retain original SAM pixels separately, and never teach templates
        with these predictions.
        """
        e = self.engine
        template = e.bank.templates[instance.template_id]
        h, w = e.shape
        for (iid, extra), detected in self.edge_margins.items():
            if (iid != instance.id or abs(frame-detected) > e.config.global_stride
                    or not self.covers(instance, frame)
                    or self.checks.get(iid, {}).get(extra) != 'not_detected'):
                continue
            exiting = extra > detected
            rows = sorted((p for p in instance.poses.values() if p.status == 'measured'
                and p.presence and p.source_id is not None
                and p.source_id != e.source_ids[frame]
                and (p.frame <= detected if exiting else p.frame >= detected)), key=lambda p:p.frame)
            rows = rows[-32:] if exiting else rows[:32]
            if len({p.source_id for p in rows}) < 8:
                continue
            prior = MotionPrior([replace(p, frame=p.source_id) for p in rows], template.box, template.anchor,
                                fit_cache=e.motion_fits)
            at_edge = prior.predict(e.source_ids[detected])
            nearby = prior.predict(e.source_ids[detected]+(1 if exiting else -1))
            prediction = prior.predict(e.source_ids[frame])
            if any(p is None or not p[2] for p in (at_edge, nearby, prediction)):
                continue
            if max(p[1] for p in (at_edge, nearby, prediction)) > .08*footprint(template.box):
                continue
            edge_box = bounds(at_edge[0], template.box)
            next_box = bounds(nearby[0], template.box)
            for oid in e.frame_observations[detected]:
                obs = e.bank.observations[oid]
                if (obs.ambiguous or obs.crossing_instances
                        or instance.id not in (obs.instance_id, obs.boundary_instance_id)):
                    continue
                data = e.cache.get(obs.probability_key)
                support = support_bounds((data['probability'] > e.config.pixel_threshold) & data['valid'])
                if support is None:
                    continue
                x, y = obs.origin
                box = (x+support[0], y+support[1], x+support[2], y+support[3])
                toward_edge = ((box[0] <= 0 and next_box[2] < edge_box[2]) or
                    (box[2] >= w and next_box[0] > edge_box[0]) or
                    (box[1] <= 0 and next_box[3] < edge_box[3]) or
                    (box[3] >= h and next_box[1] > edge_box[1]))
                if toward_edge and at_position([box], edge_box):
                    return prediction
        return None

    def _outside_edge_margin(self, instance, frame):
        """A checked-negative extra frame may lie beyond a measured frame edge.

        Transport the actual edge observation, not an expanded shared template
        or an uncertain search pose. Source timestamps preserve native timing
        when sampling (e.g. 30 -> 24 fps) alternates one/two-frame steps.
        """
        e = self.engine
        detected = self.edge_margins.get((instance.id, frame))
        if detected is None or self.checks.get(instance.id, {}).get(frame) != 'not_detected':
            return None
        current = instance.poses.get(frame)
        if current is not None and current.presence:
            return None
        before = frame > detected
        rows = sorted((p for p in instance.poses.values() if p.status == 'measured'
            and p.presence and p.source_id is not None
            and (p.frame <= detected if before else p.frame >= detected)), key=lambda p:p.frame)
        rows = rows[-32:] if before else rows[:32]
        if len({p.source_id for p in rows}) < 8:
            return None
        template = e.bank.templates[instance.template_id]
        prior = MotionPrior([replace(p, frame=p.source_id) for p in rows], template.box, template.anchor,
                            fit_cache=e.motion_fits)
        predictions = [prior.predict(e.source_ids[f]) for f in (detected, frame)]
        if any(p is None or not p[2] for p in predictions):
            return None
        a, b = predictions
        margin = 3*(a[1]+b[1])
        if not np.isfinite(margin):
            return None
        transform = b[0] @ np.linalg.inv(a[0])
        h, w = e.shape
        eligible = []
        for oid in e.frame_observations[detected]:
            obs = e.bank.observations[oid]
            if (obs.ambiguous or obs.crossing_instances or
                    instance.id not in (obs.instance_id, obs.boundary_instance_id)):
                continue
            data = e.cache.get(obs.probability_key)
            support = support_bounds((data['probability'] > e.config.pixel_threshold) & data['valid'])
            if support is None:
                continue
            x, y = obs.origin
            box = (x+support[0], y+support[1], x+support[2], y+support[3])
            projected = bounds(transform, box)
            outside = ((box[0] <= 0 and projected[2]+margin <= 0) or
                       (box[2] >= w and projected[0]-margin >= w) or
                       (box[1] <= 0 and projected[3]+margin <= 0) or
                       (box[3] >= h and projected[1]-margin >= h))
            eligible.append(outside)
        if eligible and all(eligible):
            return Pose(frame, b[0].copy(), b[1], 'absent', presence=False,
                        reason='boundary_margin_outside_frame')
        return None

    def covers(self, instance, frame):
        return frame in self.coverage.get(instance.id, set())

    def output_margin(self, instance, frame):
        """The extra lifetime frame is an output allowance, not an observation."""
        if (instance.id, frame) not in self.edge_margins:
            return False
        pose = instance.poses.get(frame)
        # Independent positive evidence, if already obtained, takes precedence.
        return not (pose is not None and pose.presence and
            (pose.status == 'measured' or pose.reason == 'boundary_sam_candidate'))

    def project(self, instance, frame, *, for_output=False):
        """Defer geometry, then use later measured motion without inventing votes."""
        if not self.covers(instance, frame) or frame in instance.ambiguous_frames:
            return
        pose = instance.poses.get(frame)
        localized_sam = pose is not None and pose.reason == 'boundary_sam_candidate' and pose.presence
        if pose is not None and pose.status == 'measured' and pose.presence:
            return
        if (pose is not None and pose.status == 'absent'
                and pose.reason not in SAM_ABSENCE_REASONS | {'boundary_margin_outside_frame'}):
            return
        outside = self._outside_edge_margin(instance, frame)
        if outside is not None:
            instance.poses[frame] = outside
            self.dirty_frames.add(frame)
            return
        edge_prediction = self._edge_prediction(instance, frame)
        if edge_prediction is not None:
            matrix, sigma, _ = edge_prediction
            instance.poses[frame] = Pose(frame, matrix.copy(), sigma, 'predicted',
                presence=False, reason='boundary_edge_motion_prediction')
            self.dirty_frames.add(frame)
            return
        if localized_sam:
            # Preserve localized original pixels until motion supports a pose.
            prediction = (self.engine.coverage_prediction if for_output else self.engine._prediction)(instance, frame)
            if prediction is None or not prediction[2]:
                return
        prediction = (self.engine.coverage_prediction if for_output else self.engine._prediction)(instance, frame)
        if prediction is None:
            return
        matrix, sigma, strong = prediction
        instance.poses[frame] = Pose(frame, matrix.copy(), sigma,
            'predicted' if strong else 'uncertain',
            presence=pose is not None and pose.presence and pose.reason == 'boundary_sam_presence',
            reason='boundary_sam_presence' if pose is not None and pose.presence
                   and pose.reason == 'boundary_sam_presence' else
                   'boundary_interval_motion_prediction' if strong else 'boundary_interval_pose_pending')
        self.dirty_frames.add(frame)

    def _presence_only(self, instance, frame, expected, boxes):
        """A SAM recheck certifies existence only, never pixels or geometry."""
        e = self.engine
        relevant = [box for box in boxes if at_position([box],expected)]
        # A foreign/ambiguous scheduled observation cannot lend its presence
        # to this occurrence merely because their boxes overlap.
        foreign = [o.box for oid in e.frame_observations.get(frame,())
            if (o := e.bank.observations[oid]).ambiguous or o.crossing_instances
            or o.instance_id not in (None,instance.id)
            or o.boundary_instance_id not in (None,instance.id)]
        present = any(not at_position(foreign,box) for box in relevant)
        pose = instance.poses[frame]
        if present:
            pose.status,pose.presence,pose.reason = 'uncertain',True,'boundary_sam_presence'
        else:
            pose.status,pose.presence,pose.reason = 'absent',False,'global_and_local_absence_checks'
        self.probes[(instance.id,e.source_ids[frame])] = present
        self.dirty_frames.add(frame)
        return present

    def refresh(self, *, for_output=False):
        for iid, frames in self.coverage.items():
            if self.engine._retired(self.engine.instances[iid]):
                continue
            for frame in sorted(frames):
                self.project(self.engine.instances[iid], frame, for_output=for_output)

    def _accept_interval(self, instance, start, end, image_at, already_checked=()):
        e = self.engine
        self.coverage.setdefault(instance.id, set()).update(range(start, end+1))
        instance.birth = min(instance.birth, start)
        # Local misses inside an accepted interval no longer end the occurrence.
        if instance.ended_at is not None and instance.ended_at < end:
            instance.ended_at = None
        for frame in range(start, end+1):
            if self.output_margin(instance, frame):
                self.project(instance, frame)
                continue
            pose = instance.poses.get(frame)
            if (pose is not None and pose.status=='absent'
                    and pose.reason not in SAM_ABSENCE_REASONS | {'boundary_margin_outside_frame'}):
                continue
            if pose is not None and pose.status == 'measured' and pose.presence:
                continue
            if pose is not None and pose.reason == 'boundary_sam_candidate' and pose.presence:
                self.project(instance, frame)
                continue
            if frame not in already_checked:
                e._track(instance, frame, e.source_ids[frame], image_at(frame),
                         backward=True, image_only=True, boundary=True)
            self.project(instance, frame)

    def _absent(self, instance, frame):
        pose = instance.poses.get(frame)
        if (pose is not None and pose.status=='absent'
                and pose.reason not in SAM_ABSENCE_REASONS):
            return
        if pose is None:
            prediction = self.engine._prediction(instance, frame)
            if prediction is not None:
                pose = Pose(frame, prediction[0], prediction[1], 'absent')
                instance.poses[frame] = pose
        if pose is not None and pose.status != 'measured':
            pose.status, pose.presence, pose.reason = 'absent', False, 'global_and_local_absence_checks'
        self.dirty_frames.add(frame)

    def _probe(self, instance, frame, image_at):
        """Image measurement first; deferred SAM certifies existence only."""
        e = self.engine
        if e._retired(instance):
            return None
        if self.output_margin(instance, frame):
            # Reusing coverage must never turn it into a new boundary detection.
            return (False if self.checks.get(instance.id, {}).get(frame) == 'not_detected'
                    else self.probes.get((instance.id, e.source_ids[frame])))
        pose = instance.poses.get(frame)
        if pose is not None and pose.status == 'measured' and pose.presence:
            self.probe_diagnostics.append(dict(instance=instance.id,frame=frame,
                source=e.source_ids[frame],result='sift_present',sam=False,reused=True))
            return True
        key = (instance.id, e.source_ids[frame])
        if key in self.probes:
            state = self.probes[key]
            if state is False and pose is not None:
                pose.status, pose.presence, pose.reason = 'absent', False, 'global_and_local_absence_checks'
                self.dirty_frames.add(frame)
                return False
            if state:
                # Existence is independent of the current geometry. SAM
                # rechecks do not have or need a bank observation to reuse.
                return True
            # Tracking or identity review may have replaced a positive probe's
            # pose. Reuse cached image/SAM work, not a boolean with lost geometry.
            self.probes.pop(key)
        image = image_at(frame)
        diagnostic = dict(instance=instance.id,frame=frame,source=e.source_ids[frame])
        e._track(instance, frame, e.source_ids[frame], image, backward=True,
                 image_only=True, boundary=True,diagnostics=diagnostic)
        pose = instance.poses.get(frame)
        if pose is None:
            return None
        if pose.status == 'measured' and pose.presence:
            self.dirty_frames.add(frame)
            self.probes[key] = True
            self.probe_diagnostics.append(dict(diagnostic,result='sift_present',sam=False))
            return True
        expected = bounds(pose.matrix, e.bank.templates[instance.template_id].box)
        roi = clip_box(expected, e.shape, padding=.25*footprint(expected))
        if area(roi) <= 0:
            return None
        if hasattr(e.detector, 'detect_presence'):
            candidates = e.detector.detect_presence(image, e.source_ids[frame])
        else:
            candidates = (e.coarse_detect(frame, e.source_ids[frame], image))
        self.sam_probes.add(key)
        boxes = candidates if hasattr(e.detector,'detect_presence') else [c.box for c in candidates]
        present = self._presence_only(instance,frame,expected,boxes)
        self.probe_diagnostics.append(dict(diagnostic,result='sam_present' if present else 'sam_absent',
            sam=True,mode='full',expected=tuple(expected)))
        return present

    def _boundary(self, instance, lo, hi, left, right, image_at, stage):
        """Deferred mode searches outward from the known positive endpoint."""
        key = (instance.id, lo, hi, left, right)
        if key in self.completed:
            return
        original_probes = set(self.sam_probes)
        probes, unresolved = [], []
        search = range(hi-1,lo,-1) if right else range(lo+1,hi)
        detected = hi if right else lo  # The positive global endpoint is fallback.
        for frame in search:
            state = self._probe(instance, frame, image_at)
            probes.append(frame)
            if state is True:
                detected = frame
            if state is False:
                break
            if state is None:
                unresolved.append(frame)
        start, end = (max(lo, detected-1), hi) if right else (lo, min(hi, detected+1))
        extra = start if right else end
        if extra != detected:
            self.edge_margins[(instance.id, extra)] = detected
        for frame in range(lo, hi+1):
            if not start <= frame <= end:
                self._absent(instance, frame)
        self._accept_interval(instance, start, end, image_at, already_checked=probes)
        self.completed.add(key)
        self.events.append(dict(instance=instance.id, template=instance.template_id,
            stage=stage, kind='appearance' if right else 'disappearance',
            first_frame=lo, last_frame=hi,
            method='from_present_end_first_absence',
            probe_frames=probes, local_sam_sources=sorted(source for iid, source in
                self.sam_probes-original_probes if iid == instance.id),
            detected_frame=detected, first_covered_frame=start, last_covered_frame=end,
            transition_frame=start if right else end+1, unresolved_frames=unresolved))

    def online(self, frame, image_at, new_instances):
        """Resolve both lifetime edges as soon as this global sample arrives."""
        e = self.engine
        previous = max((f for f in self.global_samples if f < frame), default=None)
        if previous is None:
            self.appearance_checked.update(new_instances)
            return
        for instance in e.active_instances():
            if instance.id not in self.appearance_checked:
                # A new remote occurrence has its own lifetime even when it
                # shares all strokes/features with an established template.
                prediction = e._prediction(instance, previous)
                if prediction is not None:
                    instance.poses.setdefault(previous, Pose(previous, prediction[0], prediction[1], 'uncertain'))
                    self._check(instance, previous)
                    positive = [f for f, p in instance.poses.items() if previous < f <= frame
                                and p.presence and p.status == 'measured']
                    left = self.checks.get(instance.id, {}).get(previous) in PRESENT_CHECKS
                    if not left and at_position(self.global_samples[previous][1],
                            bounds(instance.poses[previous].matrix, e.bank.templates[instance.template_id].box)):
                        # A global box at a proposed past position is not a pose.
                        # Confirm the endpoint before retaining its geometry.
                        left = self._probe(instance, previous, image_at)
                    if positive and left is False:
                        self._boundary(instance, previous, min(positive), False, True, image_at, 'first_pass')
                    elif positive and left is True:
                        # The occurrence predates discovery; only geometry
                        # remains to be matched inside the accepted interval.
                        self._accept_interval(instance, previous, min(positive), image_at)
                    elif left is None:
                        # An unlocalized search proposal is not pre-birth evidence.
                        instance.poses.pop(previous, None)
                        continue
                self.appearance_checked.add(instance.id)
            checks = self.checks.get(instance.id, {})
            if checks.get(frame) == 'not_detected' and checks.get(previous) in PRESENT_CHECKS:
                self._boundary(instance, previous, frame, True, False, image_at, 'first_pass')
            elif (checks.get(frame) in PRESENT_CHECKS
                    and checks.get(previous) in PRESENT_CHECKS
                    and instance.id not in new_instances):
                # Association has already continued the same occurrence. No
                # additional SAM lifetime checks are needed in the middle.
                self._accept_interval(instance, previous, frame, image_at)

    def check(self, frame, source_id, candidates):
        boxes = [tuple(c.box) for c in candidates]
        self.global_samples[frame] = (source_id, boxes)
        for instance in self.engine.active_instances():
            self._check(instance, frame)

    def _check(self, instance, frame):
        # Reverse tracking supplies poses before the original discovery frame.
        # The preceding global sample must check those poses as well, otherwise
        # a newly flashed occurrence is painted before it actually appeared.
        if instance.ended_at is not None and frame > instance.ended_at+2:
            return
        pose = instance.poses.get(frame)
        if pose is None:
            return
        if (pose.status=='absent' and pose.reason not in SAM_ABSENCE_REASONS
                and not self._observations(instance,frame)):
            self.checks.setdefault(instance.id,{})[frame]='not_detected'
            return
        if pose.status == 'measured' and pose.presence:
            result = 'image_match'
        elif self._observations(instance, frame):
            result = 'sam_candidate'
        else:
            # Global association has already run. Unowned/foreign candidates
            # remain separate hypotheses; they cannot restart this lifetime.
            result = 'not_detected'
            pose.status, pose.presence, pose.reason = 'absent', False, 'global_detection_absent'
        self.checks.setdefault(instance.id, {})[frame] = result

    def missing(self, instance, frame):
        if self.covers(instance, frame):
            return False
        checks = self.checks.get(instance.id, {})
        previous = max((f for f in checks if f <= frame), default=None)
        if previous is None or checks[previous] != 'not_detected':
            return False
        # A real image match wins over a SAM miss; prediction never does.
        return not any(previous <= f <= frame and p.presence
                       and (p.status == 'measured' or p.reason == 'boundary_sam_candidate')
                       for f, p in instance.poses.items())

    def resolve(self, image_at):
        e = self.engine
        # New templates/instances found backwards use the same saved global
        # detections. No additional global SAM pass or new object identities.
        instances = tuple(e.active_instances())
        for frame in self.global_samples:
            for instance in instances:
                self._check(instance, frame)
        for instance in instances:
            measured = [f for f, p in instance.poses.items()
                        if p.status == 'measured' and p.presence]
            if measured:
                instance.birth = min(instance.birth, min(measured))
            checks = self.checks.get(instance.id, {})
            ordered = sorted(checks.items())
            for (previous, left_result), (frame, result) in zip(ordered, ordered[1:]):
                left, right = left_result in PRESENT_CHECKS, result in PRESENT_CHECKS
                if left == right:
                    continue
                self._boundary(instance, previous, frame, left, right, image_at, 'second_pass')
            for frame, pose in instance.poses.items():
                if (not self.covers(instance, frame) and self.missing(instance, frame) and pose.status != 'measured'
                        and pose.reason != 'boundary_sam_candidate'):
                    pose.status, pose.presence, pose.reason = 'absent', False, 'global_detection_absent'
        self.refresh()

    def report(self):
        return dict(boundary_strategy=('present_end_first_absence_one_frame_margin'),
            probe_diagnostics=self.probe_diagnostics,
            sam_rechecks_presence_only=True,
            global_frames=sorted(self.global_samples),
            checks=[dict(instance=i, frame=f, result=result)
                    for i, rows in self.checks.items() for f, result in sorted(rows.items())],
            inferred_coverage=[dict(instance=i, frames=sorted(frames))
                               for i, frames in self.coverage.items()],
            outside_frame_margins=[dict(instance=i, frame=f, detected_frame=d)
                for (i, f), d in self.edge_margins.items()
                if (p := self.engine.instances[i].poses.get(f)) is not None
                and p.reason == 'boundary_margin_outside_frame'],
            output_only_margins=[dict(instance=i, frame=f, detected_frame=d,
                trajectory_evidence=False, template_evidence=False)
                for (i, f), d in self.edge_margins.items()
                if self.output_margin(self.engine.instances[i], f)],
            boundary_rechecks=self.events)
