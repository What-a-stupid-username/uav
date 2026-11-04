"""Online templates and instances, followed by bounded global evidence reconciliation."""
from collections import defaultdict
import math
import time
import cv2
import numpy as np
from .features import SiftMatcher, Match
from .geometry import (area, bounds, clip_box, discrepancy, footprint, intersection,
                       support_bounds, translation, warp, components, from_components, visible_support)
from .motion import MotionPrior, MotionFitCache, evidence_signature
from .rendering import MaskRenderer
from .records import Instance, Observation, Pose
from .templates import TemplateBank
from .pending import PendingResolver
from .gaps import GapResolver
from .reconciliation import TemplateReconciler
from .presence import PresenceResolver
from .interactions import InteractionResolver, relative_motion
from .schedule import is_discovery_frame

class TemplateEngine:
    def __init__(self, config, cache, detector, warnings, shape, *, debug=False):
        self.config, self.cache, self.detector = config, cache, detector
        self.debug = debug
        self.warnings, self.shape = warnings, tuple(shape)
        self.bank = TemplateBank(cache, config)
        self.matcher = SiftMatcher(cache, config)
        self.instances = {}
        self.frame_observations = defaultdict(list)
        self.source_ids = {}
        self.motion_revision = 0
        self.motion_fits = MotionFitCache()
        self.stats = dict(discoveries=0, template_calls=0,
                          ambiguous_frames=0, backward_recoveries=0, feedback_rounds=0,
                          pending_resolved=0,
                          trajectory_associations=0, trajectory_gap_matches=0,
                          trajectory_gap_predictions=0, provisional_templates=0,
                          template_reconciliations=0)
        self.timings = defaultdict(float)
        self.render_metrics = {}
        self.first_pass_marks = {}
        self.first_pass_observations = {}
        self.stage = 'first_pass'
        self.refinement_batches = []
        self.discovery_groups = {}
        self.fragment_associations = []
        self.composite_discoveries = []
        self.pose_reviews = []
        self.output_pose_filter = []
        self.edge_occurrences = []
        self.interactions = InteractionResolver(self)
        self.sam_stages = {}
        self._initial_sam_calls = getattr(detector, 'calls', 0)
        self._initial_sam_tasks = dict(getattr(detector, 'task_calls', {}))
        self.pending = PendingResolver(self)
        self.gaps = GapResolver(self)
        self.reconciler = TemplateReconciler(self)
        self.presence = PresenceResolver(self)
        self.renderer = MaskRenderer(self)
        self.discovery_buffer = None
        self.discovery_prefetch = None
        self.frame_count = None

    def coarse_detect(self, frame, source_id, image):
        """Aggregate SAM fragments before any new template/instance is created."""
        # Detector and engine share the current run's in-memory evidence.
        candidates = self.detector.detect(image, source_id, None, self.config.global_resolution)
        from .discovery import aggregate_candidates
        grouped = aggregate_candidates(candidates, self.shape, self.config.pixel_threshold)
        self.discovery_groups[source_id] = dict(frame=frame, source=source_id,
            raw_candidates=len(candidates), objects=len(grouped),
            groups=[dict(candidate=c.candidate_id, sources=list(c.source_candidates), box=c.box)
                    for c in grouped])
        # Preserve originals separately from the grouped observation. They never
        # receive persistent object IDs or cast additional fusion votes.
        if self.debug:
            for candidate in candidates:
                self.cache.put(self.cache.key('raw_discovery', source_id, candidate.candidate_id),
                    probability=candidate.probability,
                    valid=np.ones(candidate.probability.shape, bool) if candidate.valid is None else candidate.valid,
                    origin=np.asarray(candidate.origin), box=np.asarray(candidate.box), score=np.array(candidate.score))
        return grouped

    def reconcile_template(self, tid):
        self.reconciler.reconcile(tid)

    def observe(self, frame, source_id, image, candidate, resolution, tag):
        oid = self.cache.key('obs', source_id, tag, candidate.candidate_id)
        if oid in self.bank.observations:
            # Upsampling is another output timestamp, not another original observation.
            if oid not in self.frame_observations[frame]:
                self.frame_observations[frame].append(oid)
            return self.bank.observations[oid]
        p = np.asarray(candidate.probability, np.float32)
        valid = np.ones(p.shape, bool) if candidate.valid is None else candidate.valid
        if p.ndim != 2 or valid.shape != p.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('invalid candidate probability/validity')
        pkey = self.cache.key('probability', oid)
        self.cache.put(pkey, probability=p, valid=valid)
        # SAM needs broad scene context; registration needs this object only.
        # Keep out-of-box probabilities for coverage, but never use their
        # background keypoints to register a template reference.
        features = self.matcher.extract(image, source_id, candidate.box,
                                        feature_box=candidate.box)
        fkey = self.matcher.select(features, oid, np.where(valid,p,0.), candidate.origin,
                                   self.config.pixel_threshold, valid=valid)
        obs = Observation(oid, frame, source_id, candidate.candidate_id, tuple(candidate.box),
                          candidate.origin, pkey, fkey, candidate.score, resolution)
        obs.mask_role = ('discovery' if tag == 'global' else
                         'refinement' if isinstance(tag, tuple) and tag[0] == 'refinement' else 'local')
        obs.source_candidates = candidate.source_candidates or (candidate.candidate_id,)
        self.bank.observations[oid] = obs
        self.frame_observations[frame].append(oid)
        return obs

    def _shape_agreement(self, template, fit, obs):
        """Shared text alone cannot establish identity between two complete marks."""
        evidence = self._shape_evidence(template, fit, obs)
        return self._accept_shape_evidence(fit, evidence)

    @staticmethod
    def _accept_shape_evidence(fit, evidence):
        if evidence is None:
            return False
        forward, backward, ratio = evidence
        if .65 <= ratio <= 1/.65:
            return min(forward, backward) >= .8 and fit.coverage >= .5
        if ratio < .65:
            return forward >= .85
        return backward >= .85

    def _shape_evidence(self, template, fit, obs, *, identity=True):
        """Compare all visible template strokes, including outside the crop.

        Clipping the template to a small candidate used to turn containment
        into apparent equivalence. Out-of-frame strokes remain unknown.
        """
        data = self.cache.get(obs.probability_key)
        target = (data['probability'] > self.config.pixel_threshold) & data['valid']
        try:
            fusion = self.bank.fuse(template.id, identity=identity)
        except ValueError:
            return None
        local = translation(-obs.origin[0], -obs.origin[1])
        expected = warp(fusion['mask'],
            local @ fit.matrix @ translation(*fusion['origin']), target.shape, nearest=True) > 0
        if not target.any() or not expected.any():
            return None
        radius = max(1, min(5, int(math.ceil(fit.uncertainty*2))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
        e = cv2.dilate(expected.astype(np.uint8), kernel) > 0
        t = cv2.dilate(target.astype(np.uint8), kernel) > 0
        forward = float((target & e).sum()/target.sum())
        # Sum the projected support inside the frame, not only inside this
        # observation crop. Work in the template's own small coordinate grid.
        matrix = fit.matrix @ translation(*fusion['origin'])
        visible = visible_support(fusion['mask'],matrix,self.shape)
        expected_area = max(float(expected.sum()), float(visible)*abs(np.linalg.det(matrix[:2,:2])))
        backward = min(1.,float((expected & t).sum()/max(1.,expected_area)))
        projected = bounds(fit.matrix, template.box)
        support = support_bounds(target)
        target_box = (obs.origin[0]+support[0], obs.origin[1]+support[1],
                      obs.origin[0]+support[2], obs.origin[1]+support[3])
        ratio = area(target_box)/max(1., area(projected))
        return forward, backward, ratio

    def _new_instance(self, template, obs):
        iid = len(self.instances)
        instance = Instance(iid, template.id, obs.frame)
        self.instances[iid] = instance
        self._assign(instance, obs)
        return instance

    def _assign(self, instance, obs):
        if obs.instance_id is not None and obs.instance_id != instance.id:
            raise ValueError('an observation cannot belong to two instances')
        obs.ambiguous = False
        obs.ownership_confirmed = True
        obs.instance_id, obs.template_id = instance.id, instance.template_id
        instance.birth = min(instance.birth, obs.frame)
        if obs.id not in instance.observations:
            instance.observations.append(obs.id)
        sam_registered = obs.association_reason == 'continuous_sam_registration'
        instance.poses[obs.frame] = Pose(obs.frame, obs.pose.copy(), 1., 'measured',
            obs.source_id, () if sam_registered else (obs.id,), True,
            reason='continuous_sam_registration' if sam_registered else None)
        self.bank.add(obs)
        self._motion(instance)
        self.pending.resolved(obs)

    def _matches(self, template, target_key, max_locations=1, min_inliers=None):
        members = self.bank.members(template.id)
        return self.matcher.template_matches(template,self.bank.observations,target_key,
            max_locations,min_inliers,
            references=self.bank.matching_references(template.id,members=members),
            member_transforms=members)

    def _prepare_matches(self, templates, target_key):
        """Share GPU submission/readback for one observation's alternatives."""
        if self.matcher.gpu_neighbours is None:
            return
        references = []
        for template in templates:
            references.extend(self.bank.matching_references(template.id))
        self.matcher.prepare_references(references, self.bank.observations, target_key)

    def _retired(self, instance):
        if getattr(self.bank.templates[instance.template_id],'composite',False):
            return True
        if instance.alias_of is None or instance.alias_templates is None:
            return False
        a, b = instance.alias_templates
        return instance.alias_of in self.instances and b in self.bank.members(a)

    def active_instances(self):
        """Archive duplicate owners without deleting reversible evidence."""
        return [i for i in self.instances.values() if not self._retired(i)]

    def _motion(self, instance):
        template = self.bank.templates[instance.template_id]
        signature = evidence_signature(instance.poses.values(),template.box,template.anchor)
        if instance.motion is not None and getattr(instance.motion,'_signature',None)==signature:
            # Preserve the existing pending-association retry schedule. Reusing
            # numeric fits must not suppress a caller's explicit refresh event.
            self.motion_revision += 1
            return
        instance.motion = MotionPrior(instance.poses.values(), template.box, template.anchor,
                                      fit_cache=self.motion_fits)
        instance.motion._signature = signature
        self.motion_revision += 1

    def _prediction(self, instance, frame):
        if instance.motion is None:
            self._motion(instance)
        predicted = instance.motion.predict(frame)
        if predicted is not None and predicted[2]:
            # A single noisy nearest fit cannot revoke a validated regime.
            # MotionPrior requires a sustained independently measured change.
            return predicted
        measured = [p for p in instance.poses.values() if p.status == 'measured' and p.presence]
        nearest = min(measured, key=lambda p: abs(frame-p.frame), default=None)
        if nearest is None:
            return predicted
        template = self.bank.templates[instance.template_id]
        size = footprint(template.box)
        # Local measured velocity is a SEARCH hint for curves, turns and variable
        # speed; its uncertainty grows and it never supplies existence evidence.
        rows = sorted(measured, key=lambda p: abs(frame-p.frame))[:3]
        rows.sort(key=lambda p: p.frame)
        value = components(nearest.matrix, template.anchor)
        dt = abs(frame-nearest.frame)
        sigma = nearest.uncertainty + dt*.03*size
        if len(rows) >= 2:
            values = np.array([components(p.matrix, template.anchor) for p in rows])
            values[:, 2] = np.unwrap(values[:, 2])
            delta = rows[-1].frame-rows[-2].frame
            velocity = (values[-1]-values[-2])/max(1, delta)
            value = values[-1]+velocity*(frame-rows[-1].frame)
            if len(rows) == 3:
                previous = (values[1]-values[0])/max(1, rows[1].frame-rows[0].frame)
                acceleration = np.abs(velocity-previous)
                sigma += float(max(*acceleration[:2], *(acceleration[2:]*size)))*dt*dt/2
        return instance.motion.constrain(frame, from_components(value, template.anchor), sigma)

    def coverage_prediction(self, instance, frame):
        """Validated motion or a nearby measurement, never a contradicted model.

        The discrepancy of a failed regime is not localization uncertainty.
        Weak output uses measured geometry and local measured displacement;
        neither this estimate nor its uncertainty feeds tracking or templates.
        """
        bridge = getattr(self.gaps,'output_predictions',{}).get((instance.id,frame))
        if bridge is not None:
            return bridge[0].copy(),bridge[1],bridge[2]
        if instance.motion is None:
            self._motion(instance)
        prediction = instance.motion.predict(frame)
        if prediction is None or prediction[2]:
            return prediction
        rows = instance.motion.evidence
        # Interior misses have evidence on both sides. A nearest-frame hold
        # creates a visible pause and catch-up even on a well supported path.
        # Validate each side independently before using their geometry; this
        # remains output-only and supplies neither presence nor fit samples.
        left = [p for p in rows if p.frame < frame]
        right = [p for p in rows if p.frame > frame]
        if len(left) >= 8 and len(right) >= 8:
            from .gaps import trajectory_bridge
            bridge = trajectory_bridge(left[-32:], right[:32],
                self.bank.templates[instance.template_id], [frame],
                fit_cache=self.motion_fits, sample_times=self.source_ids)
            if bridge is not None:
                matrix, sigma = bridge[0][frame]
                return matrix.copy(), sigma, True
        nearest = min(rows, key=lambda p: (abs(p.frame-frame), p.frame))
        neighbours = sorted(rows, key=lambda p: (abs(p.frame-frame), p.frame))[:3]
        speeds = [discrepancy(a.matrix,b.matrix,self.bank.templates[instance.template_id].box)
                  /abs(a.frame-b.frame) for a,b in zip(neighbours,neighbours[1:])
                  if a.frame != b.frame]
        sigma = nearest.uncertainty + abs(frame-nearest.frame)*(max(speeds) if speeds else 0.)
        return nearest.matrix.copy(), sigma, False

    def _near_instances(self, tid, matrix, frame, uncertainty):
        members = self.bank.members(tid)
        near = []
        for instance in self.instances.values():
            if instance.template_id not in members:
                continue
            if instance.alias_of is not None and instance.alias_templates is not None:
                a, b = instance.alias_templates
                if b in self.bank.members(a):
                    continue
            if instance.ended_at is not None and frame > instance.ended_at:
                continue
            prediction = self._prediction(instance, frame)
            if prediction is None:
                continue
            earlier = [p for f, p in instance.poses.items() if f < frame]
            if earlier and max(earlier, key=lambda p: p.frame).status == 'absent':
                continue
            predicted, sigma, strong = prediction
            template = self.bank.templates[instance.template_id]
            local_matrix = matrix @ members[instance.template_id]
            # A remote reappearance is a new instance even if its template matches.
            gate = min(.6*footprint(bounds(local_matrix, template.box)), 3*(sigma+uncertainty))
            distance = discrepancy(predicted, local_matrix, template.box)
            if distance <= max(3*uncertainty, gate):
                near.append((distance/max(1., gate), instance))
        return sorted(near, key=lambda pair: pair[0])

    def _continuity(self, instance, obs, fit, template_id=None):
        if instance.motion is None:
            self._motion(instance)
        matrix = fit.matrix
        if template_id is not None:
            matrix = matrix @ self.bank.members(template_id)[instance.template_id]
        return instance.motion.continuity(obs.frame, matrix, fit.uncertainty, obs.source_id,
                                          fit_cache=self.motion_fits)

    def _fragment_registrations(self, instance, obs):
        """Independent image registration and validated motion alternatives."""
        pose = instance.poses.get(obs.frame)
        measured = pose is not None and pose.status == 'measured' and pose.presence
        if measured:
            sources = {self.bank.observations[r].source_id for r in pose.references
                       if r in self.bank.observations}
            if sources-{obs.source_id}:
                yield Match(pose.matrix, pose.uncertainty, 0, 0., pose.references), None, True
        elif (self.presence.missing(instance, obs.frame)
              or pose is not None and pose.status == 'absent'):
            return
        if instance.motion is None:
            self._motion(instance)
        prior = instance.motion.excluding_source(obs.source_id, fit_cache=self.motion_fits)
        # The same alternatives apply to missing AND noisy current matches.
        # Prefer nearby validated support; deduplicate equivalent registrations.
        for matrix, sigma, evidence in prior.continuity_predictions(
                obs.frame, box=self.bank.templates[instance.template_id].box):
            yield Match(matrix, sigma, 0, 0.), evidence, False

    def _registered_fragment(self, obs, *, preferred_templates=None):
        """Resolve occurrence ownership separately from template/pose evidence."""
        data = self.cache.get(obs.probability_key)
        target = (data['probability'] > self.config.pixel_threshold) & data['valid']
        if not target.any():
            return None
        compatible, checks = [], []
        for instance in self.instances.values():
            if (not self._renders_at(instance, obs.frame)
                    or self.presence.output_margin(instance, obs.frame)
                    or instance.ended_at is not None and obs.frame > instance.ended_at
                    or obs.frame in instance.ambiguous_frames):
                continue
            template = self.bank.templates[instance.template_id]
            if template.mixed or (preferred_templates is not None
                    and template.id not in preferred_templates):
                continue
            def agreement(fit):
                comparison = self._shape_evidence(template, fit, obs)
                return comparison[:2] if comparison is not None else (0., 0.)
            best_check = None
            for fit, evidence, measured in self._fragment_registrations(instance, obs):
                expected = bounds(fit.matrix, template.box)
                if intersection(obs.box, expected) <= 0:
                    continue
                if measured:
                    evidence = self._continuity(instance, obs, fit)
                forward, backward = agreement(fit)
                contains = intersection(obs.box, expected)/max(1., area(expected)) >= .8
                # At an independently measured pose, explaining the candidate
                # is sufficient for partial ownership. Missing template strokes
                # are not negative identity evidence, regardless of SAM box size.
                shape = forward >= .95
                supported = evidence is not None and (forward >= .9 or
                    backward >= .9 and forward >= .75 and contains)
                image_fit = None
                if evidence is not None and (not measured or not shape and not supported):
                    for candidate in self._matches(template, obs.feature_key, max_locations=4):
                        if (candidate.inliers < self.config.min_inliers
                                or discrepancy(candidate.matrix, fit.matrix, template.box)
                                   > 3*(candidate.uncertainty+fit.uncertainty)):
                            continue
                        corroboration = self._continuity(instance, obs, candidate)
                        f, b = agreement(candidate)
                        if corroboration is not None and f >= .7 and b >= .5:
                            image_fit, evidence, forward, backward = candidate, corroboration, f, b
                            supported = True
                            break
                # Additional unexplained strokes need their own hypothesis.
                # A measured location identifies the visible old fragment; it
                # does not prove every new pixel belongs to that fragment.
                accepted = (bool(shape or supported) if measured else bool(supported)) and forward >= .7
                check = dict(instance=instance.id, template=template.id, forward=forward,
                    backward=backward, measured_pose=measured, predicted_registration=not measured,
                    validated_motion=evidence is not None, matched_original=image_fit is not None,
                    accepted=accepted)
                if best_check is None or forward > best_check['forward']:
                    best_check = check
                if not accepted:
                    continue
                checks.append(check)
                compatible.append((instance, fit, evidence, image_fit, measured,
                    measured and forward >= .9, forward, backward))
                break
            else:
                if best_check is not None:
                    checks.append(best_check)
        # A complete match wins over a containing hypothesis. The latter may
        # have more references only because it accumulated multiple objects.
        complete = [row for row in compatible if min(row[-2:]) >= .8]
        if complete:
            preferred = set().union(*(set(self.bank.members(row[0].template_id)) for row in complete))
            compatible = [row for row in compatible if row[0].template_id in preferred]
        if obs.mask_role == 'discovery' and checks:
            self.fragment_associations.append(dict(observation=obs.id, frame=obs.frame,
                checks=checks, outcome='unique' if len(compatible)==1 else
                    'competing' if compatible else 'no_compatible_visible_strokes'))
        if len(compatible) != 1:
            return None
        instance, fit, evidence, image_fit, measured, share_mask, forward, _ = compatible[0]
        old_pose = instance.poses.get(obs.frame)
        has_measurement = old_pose is not None and old_pose.status == 'measured' and old_pose.presence
        registered = image_fit if image_fit is not None and not measured else fit
        obs.pose, obs.template_id = registered.matrix.copy(), instance.template_id
        obs.reliable = bool(share_mask or image_fit is not None and forward >= .9)
        obs.ambiguous = False
        obs.association_reason = ('validated_trajectory_continuity' if obs.reliable and image_fit is not None and not measured
            else 'registered_fragment_and_validated_trajectory' if obs.reliable and evidence is not None
            else 'partial_or_complete_at_measured_pose' if obs.reliable else
            'local_strokes_at_validated_trajectory')
        obs.trajectory_evidence = (dict(evidence, coordinate_registration='original_image' if
            measured or image_fit is not None else 'predicted', shared_template_update=obs.reliable)
            if evidence is not None else None)
        obs.instance_id, obs.ownership_confirmed = instance.id, True
        if obs.id not in instance.observations:
            instance.observations.append(obs.id)
        if image_fit is not None and not has_measurement:
            instance.poses[obs.frame] = Pose(obs.frame, image_fit.matrix.copy(), image_fit.uncertainty,
                'measured', obs.source_id, image_fit.references, True)
        elif not has_measurement:
            instance.poses[obs.frame] = Pose(obs.frame, fit.matrix.copy(), fit.uncertainty,
                'predicted', presence=False, reason='local_strokes_at_validated_trajectory')
        if obs.reliable:
            self.bank.add(obs)
        self.pending.resolved(obs)
        self.stats['trajectory_associations'] += 1
        return instance

    def associate(self, obs, *, allow_new=True, independent_new=False):
        if obs.instance_id is not None:
            return self.instances[obs.instance_id]
        if self._mixed_discovery(obs):
            return None
        candidates = []
        registrations = []
        complete_templates = set()
        queried = set()
        self._prepare_matches([t for t in self.bank.templates.values()
            if not t.mixed and not t.composite], obs.feature_key)
        for template in list(self.bank.templates.values()):
            if template.id in queried:
                continue
            queried.update(self.bank.members(template.id))
            if template.mixed:
                continue
            if getattr(template,'composite',False):
                continue
            fits = self._matches(template, obs.feature_key)
            if not fits:
                continue
            fit = fits[0]
            projected = bounds(fit.matrix, template.box)
            # Reject a shared word that explains little of BOTH complete identities.
            smaller = min(area(projected), area(obs.box))
            containment = intersection(projected, obs.box)/max(1., smaller)
            if fit.coverage < .08 or containment < .6:
                continue
            evidence = self._shape_evidence(template,fit,obs)
            registrations.append((template,fit,evidence))
            if not self._accept_shape_evidence(fit,evidence):
                continue
            # A partial image match can continue a localized occurrence, but
            # cannot give a new remote occurrence the missing template strokes.
            complete = evidence is not None and min(evidence[:2]) >= .8
            if not complete and not self._near_instances(template.id,fit.matrix,obs.frame,fit.uncertainty):
                continue
            # A larger candidate must retain its own identity until the added
            # pieces have measured fixed-layout evidence.
            if evidence is not None and evidence[0] < .7:
                continue
            candidates.append((template, fit))
            if complete:
                complete_templates.update(self.bank.members(template.id))
        # Compare image identities before a containing measured track can
        # claim the observation. Absence of a full match keeps fragment repair.
        if not complete_templates:
            from .composite import partition_discovery
            if partition_discovery(self,obs,registrations):
                return None
        if not complete_templates:
            # Only investigate a mixed whole when no established complete
            # template already explains this candidate.
            recovered = self._recover_mixed(obs)
            if recovered is not None:
                return recovered
        # Prefer current independent SAM geometry over assigning this original
        # to a predicted fragment pose. Otherwise the observation never becomes
        # the next segmentation reference and the continuation expires.
        if not registrations:
            from .sam_continuity import continue_observation
            continued = continue_observation(self,obs)
            if continued is not None:
                return continued
        registered = self._registered_fragment(obs,
            preferred_templates=complete_templates or None)
        if registered is not None:
            return registered
        if complete_templates:
            candidates = [(t,f) for t,f in candidates if t.id in complete_templates]
        if len(candidates) > 1:
            # A whole original observation may connect fragments, never just their proximity.
            for index, (a, ma) in enumerate(candidates):
                for b, mb in candidates[index+1:]:
                    ba, bb = bounds(ma.matrix, a.box), bounds(mb.matrix, b.box)
                    overlap = intersection(ba, bb)/max(1., min(area(ba), area(bb)))
                    kind = 'bridge' if overlap < .1 else 'overlap'
                    if kind == 'overlap' and min(ma.coverage, mb.coverage) < .65:
                        continue
                    self.bank.propose_link(a.id, b.id, np.linalg.inv(ma.matrix)@mb.matrix,
                        obs.source_id, kind, tolerance=4*max(ma.uncertainty, mb.uncertainty),
                        activate=False)
        # A link can become active above on this very observation. Refresh
        # aliases before counting trajectories, using prior measured agreement
        # only; otherwise the same occurrence competes with its own fragment.
        self.reconciler.refresh_aliases()
        candidates.sort(key=lambda item: (-len(item[1].references), -item[1].coverage, -item[1].inliers))
        # Independent, validated continuity can distinguish image-compatible
        # candidates. A search velocity/polyline cannot supply this evidence.
        supported = []
        for template, fit in candidates:
            for _, instance in self._near_instances(template.id, fit.matrix, obs.frame, fit.uncertainty):
                evidence = self._continuity(instance, obs, fit, template.id)
                if evidence is not None:
                    supported.append((template, fit, instance, evidence))
        supported = list({i.id: (t, f, i, evidence) for t, f, i, evidence in supported}.values())
        if len({i.id for _, _, i, _ in supported}) > 1:
            from .association import registered_occurrence_owner
            owner=registered_occurrence_owner(self,supported)
            if owner is None:
                obs.ambiguous = True
                return self.pending.defer(obs, 'competing_validated_trajectories')
            supported=[owner]
        if len(supported) == 1:
            template, fit, instance, evidence = supported[0]
            # A whole connecting disjoint fragments is not an alternative
            # occurrence: never let one trajectory absorb the mixed whole.
            separate_parts = any(intersection(bounds(fit.matrix, template.box), bounds(m.matrix, t.box))
                /max(1., min(area(bounds(fit.matrix, template.box)), area(bounds(m.matrix, t.box)))) < .6
                for t, m in candidates if t.id not in self.bank.members(template.id))
            if not separate_parts:
                transform = self.bank.members(template.id)[instance.template_id]
                obs.pose, obs.template_id, obs.reliable = fit.matrix @ transform, instance.template_id, True
                obs.association_reason, obs.trajectory_evidence = 'validated_trajectory_continuity', evidence
                self.stats['trajectory_associations'] += 1
                self._assign(instance, obs)
                return instance
        for template, fit in candidates:
            nearby = self._near_instances(template.id, fit.matrix, obs.frame, fit.uncertainty)
            # Prefer continuity; two independent views are required for a remote instance.
            if not nearby and len(fit.references) < 2:
                continue
            if len(nearby) > 1 and abs(nearby[0][0]-nearby[1][0]) < .3:
                obs.ambiguous = True
                for _, instance in nearby:
                    instance.ambiguous_frames.add(obs.frame)
                return self.pending.defer(obs, 'competing_instances')
            if len(candidates) > 1:
                unrelated = [t for t, m in candidates if t.id not in self.bank.members(template.id)]
                if unrelated:
                    # Keep pixels locally, but do not teach a contested whole to any fragment.
                    obs.ambiguous = True
                    return self.pending.defer(obs, 'competing_templates')
            obs.association_reason = 'original_image_match'
            if nearby:
                instance = nearby[0][1]
                transform = self.bank.members(template.id)[instance.template_id]
                obs.pose, obs.template_id, obs.reliable = fit.matrix @ transform, instance.template_id, True
                self._assign(instance, obs)
            else:
                obs.pose, obs.template_id, obs.reliable = fit.matrix, template.id, True
                instance = self._new_instance(template, obs)
            return instance
        if candidates:
            return self.pending.defer(obs, 'insufficient_independent_references')
        # Unexplained overlapping pixels get their own provisional template.
        # Proximity cannot establish identity or a coordinate transform.
        for instance in self.instances.values():
            if instance.ended_at is not None and obs.frame > instance.ended_at:
                continue
            prediction = self._prediction(instance, obs.frame)
            if prediction is None:
                continue
            template = self.bank.templates[instance.template_id]
            expected = bounds(prediction[0], template.box)
            if not independent_new and intersection(expected, obs.box)/max(1., min(area(expected), area(obs.box))) >= .7:
                obs.ambiguous = True
                return self.pending.defer(obs, 'overlap_without_identity')
        if not allow_new:
            return self.pending.defer(obs, 'no_existing_template')
        return self.pending.defer(obs, 'new_template')

    def _mixed_discovery(self, obs):
        """Recognize a shared detection using independently established tracks.

        The same SAM component can contain two existing watermarks. Detect this
        before fragment association or template links can teach either one the
        other one's strokes. A search prediction only locates support; separate
        original-image histories propose two possible owners. Relative past
        and future motion distinguishes crossings from rigid fragments.
        """
        if obs.crossing_instances:
            return True
        if obs.mask_role != 'discovery':
            return False
        data = self.cache.get(obs.probability_key)
        target = (data['probability'] > self.config.pixel_threshold) & data['valid']
        if not target.any():
            return False
        candidates = []
        local = translation(-obs.origin[0], -obs.origin[1])
        for instance in self.active_instances():
            if not self._renders_at(instance, obs.frame):
                continue
            pose = instance.poses.get(obs.frame)
            if pose is None or pose.status == 'absent':
                continue
            template = self.bank.templates[instance.template_id]
            if template.mixed:
                continue
            expected = bounds(pose.matrix, template.box)
            # A rotated/expanded template canvas contains empty space. Its box
            # only rejects disjoint searches; visible strokes below establish
            # whether this discovery contains the tracked object.
            if intersection(expected, obs.box) <= 0:
                continue
            # This only HOLDS a possible mixed observation, not certifies a
            # crossing. A brief difficult match must not feed both objects'
            # pixels to the larger template merely because sigma grew.
            recent = [p for f, p in instance.poses.items() if f < obs.frame and
                      p.status == 'measured' and p.presence]
            nearest = max(recent, key=lambda p: p.frame, default=None)
            if pose.status != 'measured' and (nearest is None or
                    obs.frame-nearest.frame > self.config.global_stride):
                continue
            if pose.uncertainty > .35*footprint(expected):
                continue
            fused = self.bank.fuse(template.id, identity=True)
            support = warp(fused['mean'] > self.config.pixel_threshold,
                local @ pose.matrix @ translation(*fused['origin']), target.shape, nearest=True) > 0
            if not support.any():
                continue
            radius = max(1, min(5, int(math.ceil(2*pose.uncertainty))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
            detected = cv2.dilate(target.astype(np.uint8), kernel) > 0
            if (support & detected).sum()/support.sum() < .5:
                continue
            candidates.append(instance)
        pairs = []
        for offset, a in enumerate(candidates):
            for b in candidates[offset+1:]:
                if self._separately_observed(a, b, obs.frame):
                    pairs.append((a, b))
        if not pairs:
            return False
        self.interactions.hold(obs, pairs)
        return True

    def _recover_mixed(self, obs):
        """Investigate a mixed whole using independent, complete original views.

        Repeated partial features cannot clear a mixed identity. Two new whole
        originals must preserve all visible strokes and their relative layout.
        The original contested observation is never promoted to shared evidence.
        """
        candidates = []
        for template in self.bank.templates.values():
            if not template.mixed:
                continue
            seed = self.bank.observations[template.observations[0]]
            if seed.source_id == obs.source_id:
                continue
            for fit in self._matches(template, obs.feature_key):
                if fit.coverage < .65 or not self._shape_agreement(template, fit, obs):
                    continue
                # Canvas bounds include retained probability outside the box
                # head. They can grow without the actual object becoming larger.
                expected = bounds(fit.matrix @ np.linalg.inv(seed.pose), seed.box)
                if not .8 <= area(obs.box)/max(1., area(expected)) <= 1.25:
                    continue
                nearby = self._near_instances(template.id, fit.matrix, obs.frame, fit.uncertainty)
                if len(nearby) == 1:
                    candidates.append((template, fit, nearby[0][1]))
        if len(candidates) != 1:
            return None
        template, fit, instance = candidates[0]
        obs.pose, obs.template_id, obs.reliable = fit.matrix.copy(), template.id, True
        obs.association_reason = 'mixed_hypothesis_complete_original_recheck'
        self._assign(instance, obs)
        if obs.id not in template.identity_confirmations:
            template.identity_confirmations.append(obs.id)
        sources = {self.bank.observations[o].source_id for o in template.identity_confirmations}
        if len(sources) >= 2:
            confirmations = set(template.identity_confirmations)
            for oid in template.observations:
                old = self.bank.observations[oid]
                if oid not in confirmations:
                    old.ambiguous = True
                    pose = instance.poses.get(old.frame)
                    if pose is not None:
                        pose.status, pose.presence = 'uncertain', False
                        pose.reason = 'original_mixed_observation'
            template.mixed, template.provisional = False, False
            template.version += 1
            self.bank.revision += 1
            self._motion(instance)
            self.reconciler.events.append(dict(event='mixed_hypothesis_resolved',
                template=template.id, sources=sorted(sources), frame=obs.frame))
        return instance

    def _provisional(self, obs, reason):
        """Store a hypothesis immediately; uncertainty of identity is not a miss."""
        obs.ambiguous = False
        template = self.bank.create(obs)
        template.investigation_reason = reason
        # Mixed pixels stay in their own hypothesis and never teach another
        # template, even when later registrations identify one constituent.
        template.mixed = reason in ('competing_instances', 'competing_templates',
            'competing_validated_trajectories', 'crossing_identity_ambiguous')
        instance = self._new_instance(template, obs)
        obs.association_reason = reason
        self.stats['provisional_templates'] += 1
        return instance


    def _observed_candidate(self, obs):
        data = self.cache.get(obs.probability_key)
        return warp((data['probability'] > self.config.pixel_threshold)&data['valid'],
                    translation(*obs.origin), self.shape, nearest=True) > 0

    def _track(self, instance, frame, source_id, image, backward=False, bridge_prediction=None,
               *, image_only=False, boundary=False, diagnostics=None):
        if self._retired(instance):
            return
        if self.presence.output_margin(instance, frame):
            # The additional lifetime frame borrows geometry only. Neither a
            # retry nor a gap bridge may promote this allowance to evidence.
            self.presence.project(instance, frame)
            return
        if ((instance.render_start is not None and frame < instance.render_start)
                or (instance.render_end is not None and frame > instance.render_end)):
            return
        checked = instance.poses.get(frame)
        interval = self.presence.covers(instance, frame)
        if (not boundary and bridge_prediction is None and checked is not None
                and not interval
                and checked.status == 'absent'
                and checked.reason == 'global_and_local_absence_checks'):
            # Stage-one boundary results also constrain the backward pass.
            # A new independent two-sided bridge can still reopen a missed gap.
            return
        # Once a scheduled global check loses an occurrence, wait for a later
        # global observation to reopen association. Only a separately validated
        # two-sided bridge can retry this gap; no expanding live extrapolation.
        if not boundary and bridge_prediction is None and self.presence.missing(instance, frame):
            prior = max((p for f, p in instance.poses.items() if f <= frame),
                        key=lambda p:p.frame, default=None)
            if prior is not None:
                instance.poses[frame] = Pose(frame, prior.matrix.copy(), prior.uncertainty,
                    'absent', presence=False, reason='global_detection_absent')
            return
        template = self.bank.templates[instance.template_id]
        prediction = bridge_prediction if bridge_prediction is not None else self._prediction(instance, frame)
        if prediction is None:
            if diagnostics is not None:
                diagnostics['sift_failure'] = 'no_search_pose'
            return
        matrix, sigma, strong = prediction
        predicted_box = bounds(matrix,template.box)
        predicted_size = footprint(predicted_box)
        roi = clip_box(predicted_box,self.shape,
                       padding=max(.25*predicted_size,3*sigma))
        if area(roi) <= 0:
            if diagnostics is not None:
                diagnostics['sift_failure'] = 'outside_frame'
            instance.poses[frame] = Pose(frame, matrix, sigma, 'uncertain', reason='outside_frame_presence_unknown')
            return
        key = self.matcher.extract(image, source_id, roi)
        minimum = max(4, self.config.min_inliers-2) if bridge_prediction is not None else None
        fits = self._matches(template, key, max_locations=4, min_inliers=minimum)
        if diagnostics is not None:
            diagnostics.update(search_box=roi,target_features=len(self.cache.get(key)['xy']),
                geometric_candidates=len(fits),sift_failure='no_geometric_fit' if not fits else 'pose_gate_rejected')
        if minimum is not None:
            # Weaker feature count is allowed ONLY inside a separately validated
            # two-sided bridge, with distributed inliers and independent views.
            fits = [f for f in fits if f.inliers >= self.config.min_inliers or
                    (f.coverage >= .35 and len({self.bank.observations[r].source_id
                     for r in f.references if self.bank.observations[r].source_id != source_id}) >= 2)]
        if fits:
            from .edge_identity import independent_sources
            # Prediction locates the search; it is not image evidence. A tiny
            # single-reference alternative must not outrank a distributed fit
            # corroborated by independent originals just because it is closer.
            def evidence_order(fit):
                sources=len(independent_sources(self,fit,source_id))
                corroborated=(sources>=2 and fit.inliers>=self.config.min_inliers
                              and fit.coverage>=.35)
                return (not corroborated,discrepancy(fit.matrix,matrix,template.box))
            fits.sort(key=evidence_order)
            size = predicted_size
            accepted = None
            for fit in fits:
                distance = discrepancy(fit.matrix, matrix, template.box)
                localized = distance <= max(4*fit.uncertainty, min(.65*size, 4*(sigma+fit.uncertainty)))
                conflict = distance > 4*(sigma+fit.uncertainty)
                independent = {self.bank.observations[r].source_id for r in fit.references
                    if r in self.bank.observations and self.bank.observations[r].source_id != source_id
                    and not self.bank.observations[r].ambiguous and self.bank.observations[r].reliable}
                # Prediction chooses a search region; it must not veto real
                # turns corroborated by distinct original frames. Keep the
                # size-scaled spatial gate, distributed features and unique
                # occurrence assignment so a remote flash cannot take over.
                alternatives = [other for other in self.active_instances()
                    if other.id != instance.id and other.template_id in self.bank.members(template.id)
                    and other.ended_at is None and other.birth <= frame
                    and (p := other.poses.get(frame)) is not None and p.status != 'absent'
                    and discrepancy(p.matrix, fit.matrix @ self.bank.members(template.id)[other.template_id],
                                    self.bank.templates[other.template_id].box) <= 4*(p.uncertainty+fit.uncertainty)]
                corroborated_turn = (len(independent) >= 2
                    and fit.inliers >= self.config.min_inliers and fit.coverage >= .35
                    and distance <= .65*size and not alternatives)
                if (localized and (not conflict or len(independent) >= 2)) or corroborated_turn:
                    accepted = fit
                    break
            if accepted is not None:
                fit = accepted
                result = Pose(frame, fit.matrix, fit.uncertainty, 'measured',source_id,fit.references,True)
                if diagnostics is not None:
                    diagnostics.update(sift_failure=None,inliers=fit.inliers,
                        references=len(fit.references),uncertainty=float(fit.uncertainty))
                old = instance.poses.get(frame)
                instance.poses[frame] = result
                if backward and (old is None or old.status != 'measured'):
                    self.stats['backward_recoveries'] += 1
                if instance.motion is None or sum(p.status == 'measured' and p.presence
                        for p in instance.poses.values()) >= len(instance.motion.evidence)+8:
                    self._motion(instance)
                return
        instance.poses[frame] = Pose(frame, matrix, sigma, 'uncertain', reason='no_original_image_match')
        if interval:
            # Existence was accepted at the lifetime boundary. Retry geometry
            # without further SAM presence checks or negative evidence.
            if (checked is not None and checked.presence
                    and checked.reason in ('boundary_sam_candidate', 'boundary_sam_presence')):
                instance.poses[frame] = checked
            self.presence.project(instance, frame)
            return
        return
        # Scheduled global detection checks presence independently of extra
        # identity investigation. Avoid another SAM call before that detection.
        # Keep unresolved existence explicit; motion alone never creates presence evidence.

    def _independent_instances(self, a, b, frame):
        if self._retired(a) or self._retired(b):
            return False
        if a.template_id == b.template_id:
            return True
        return relative_motion(self, a, b, frame)['state'] == 'independent_motion'

    def _separately_observed(self, a, b, frame):
        """Separate support proposes two owners; it does not prove two objects."""
        if self._retired(a) or self._retired(b):
            return False
        if a.template_id == b.template_id:
            return True
        # Different SAM candidates/templates may be a whole and its fragments.
        # Establish two occurrences from separately visible image measurements,
        # not from candidate numbering, overlapping boxes or shared work queues.
        sources = set()
        for f in set(a.poses) & set(b.poses):
            if f >= frame:
                continue
            pa, pb = a.poses[f], b.poses[f]
            if (pa.status != 'measured' or pb.status != 'measured'
                    or not pa.presence or not pb.presence
                    or pa.source_id is None or pa.source_id != pb.source_id):
                continue
            ba = bounds(pa.matrix, self.bank.templates[a.template_id].box)
            bb = bounds(pb.matrix, self.bank.templates[b.template_id].box)
            if intersection(ba, bb)/max(1., min(area(ba), area(bb))) < .2:
                sources.add(pa.source_id)
        return len(sources) >= 2

    def _crossings(self, frame):
        current = [i for i in self.instances.values() if frame in i.poses
                   and i.poses[frame].status != 'absent' and self._renders_at(i, frame)]
        boxes = {i.id:bounds(i.poses[frame].matrix,self.bank.templates[i.template_id].box)
                 for i in current}
        box_areas = {iid:area(box) for iid,box in boxes.items()}
        supports = {}
        def support(instance):
            if instance.id not in supports:
                fused = self.bank.fuse(instance.template_id, identity=True)
                supports[instance.id] = warp(fused['mean'] > self.config.pixel_threshold,
                    instance.poses[frame].matrix @ translation(*fused['origin']),
                    self.shape, nearest=True) > 0
            return supports[instance.id]
        for offset, a in enumerate(current):
            for b in current[offset+1:]:
                pa, pb = a.poses[frame], b.poses[frame]
                ba,bb = boxes[a.id],boxes[b.id]
                if intersection(ba,bb)/max(1.,min(box_areas[a.id],box_areas[b.id])) > .6:
                    # Relative-history analysis cannot affect non-overlapping
                    # objects here. Avoid recomputing it for every pair/frame.
                    if not self._independent_instances(a, b, frame):
                        continue
                    sa, sb = support(a), support(b)
                    if (sa & sb).sum()/max(1, min(sa.sum(), sb.sum())) < .15:
                        continue
                    same_pose = (a.template_id == b.template_id and
                        discrepancy(pa.matrix, pb.matrix, self.bank.templates[a.template_id].box)
                        <= 4*(pa.uncertainty+pb.uncertainty))
                    if not same_pose:
                        # Shared mask pixels do not invalidate independent
                        # image registration. Only uncertain current poses stay
                        # uncertain; mixed discovery is held before bank.add.
                        # Never erase measured history needed by the future
                        # relative-motion decision just because support overlaps.
                        self.stats['ambiguous_frames'] += 1
                        continue
                    a.ambiguous_frames.add(frame); b.ambiguous_frames.add(frame)
                    for instance in (a, b):
                        pose = instance.poses[frame]
                        pose.status, pose.reason = 'uncertain', 'crossing_identity_ambiguous'
                        pose.presence = False
                    for oid in self.frame_observations[frame]:
                        obs = self.bank.observations[oid]
                        if obs.instance_id in (a.id, b.id) and not obs.ambiguous:
                            obs.ambiguous = True
                            self.bank.revision += 1
                    self.stats['ambiguous_frames'] += 1

    def _clear_resolved_crossing(self, frame):
        """A resolved separation may use new observations, never recycle mixed pixels."""
        for instance in self.instances.values():
            pose = instance.poses.get(frame)
            if pose is not None and pose.status == 'measured':
                instance.ambiguous_frames.discard(frame)

    def _review_frame_rigidity(self, frame, source_id):
        rejected=self.matcher.contradicted_references(self.bank.observations,source_id)
        for oid in rejected:
            observation=self.bank.observations[oid]
            observation.ambiguous,observation.reliable=True,False
            observation.association_reason='independent_motion_inside_original'
            member=self.bank.templates[observation.template_id]
            member.composite=not any(self.bank.observations[o].reliable
                and not self.bank.observations[o].ambiguous
                and self.bank.observations[o].mask_role=='discovery' for o in member.observations)
            member.version+=1
            self.bank.revision+=1
            if member.composite:
                for link in list(self.bank.links.values()):
                    if link.active and member.id in (link.a,link.b):
                        self.bank.revoke(link.a,link.b,'independent_motion_inside_original')
            self.reconciler.events.append(dict(event='nonrigid_original',
                template=member.id,observation=oid,frame=frame,source=source_id))

    def first_frame(self, frame, source_id, image, *, image_at=None):
        tick = time.monotonic()
        discover = is_discovery_frame(frame,self.config.global_stride,self.frame_count)
        self.source_ids[frame] = source_id
        previous = next((f for f in reversed(self.source_ids) if f < frame and self.source_ids[f] == source_id), None)
        if previous is not None:
            for instance in self.instances.values():
                prior = instance.poses.get(previous)
                if prior is not None:
                    instance.poses[frame] = Pose(frame, prior.matrix.copy(), prior.uncertainty, prior.status,
                        prior.source_id, prior.references, prior.presence, prior.reason)
            self.frame_observations[frame] = list(self.frame_observations[previous])
            if not discover:
                self._mark_first_frame(frame)
                return
        if discover:self.matcher.begin_frame_evidence(source_id)
        # Track from original image without requiring a current SAM mask.
        for instance in self.active_instances():
            if instance.ended_at is None or frame <= instance.ended_at+2:
                self._track(instance, frame, source_id, image)
        if discover:
            previous_instances = set(self.instances)
            self.stats['discoveries'] += 1
            if self.discovery_buffer is not None:
                self.detector.accept_global(self.discovery_buffer.take(frame))
            candidates = self.coarse_detect(frame, source_id, image)
            observations = [self.observe(frame,source_id,image,candidate,
                self.config.global_resolution,'global') for candidate in candidates]
            for obs in observations:
                self.associate(obs)
            self._review_frame_rigidity(frame, source_id)
            self.presence.check(frame, source_id, candidates)
            if image_at is not None:
                self.presence.online(frame, image_at, set(self.instances)-previous_instances)
        self.pending.retry()
        self._clear_resolved_crossing(frame)
        self._crossings(frame)
        if discover:
            self.interactions.resolve()
        for instance in self.active_instances():
            if frame in instance.poses and instance.poses[frame].status == 'measured':
                # Fit only when enough new original evidence has accumulated.
                if discover:
                    self._motion(instance)
        dirty = self.presence.dirty_frames | {frame}
        for changed in sorted(dirty):
            self._mark_first_frame(changed)
        self.presence.dirty_frames.clear()
        self.timings['first_pass_seconds'] += time.monotonic()-tick

    def _mark_first_frame(self, frame):
        """Save lightweight debug state, never render a first-pass mask."""
        if self.debug:
            from .review import capture_first_frame
            capture_first_frame(self, frame)

    def second_pass(self, image_at):
        from .stage2 import run_stage2
        run_stage2(self,image_at)

    def _unresolved_pose_jobs(self, affected):
        """Only unresolved image registrations, independent of video length."""
        jobs = set()
        for instance in self.active_instances():
            earliest = (instance.birth if instance.id in self.presence.appearance_checked
                        else max(0,instance.birth-self.config.global_stride))
            frames = set(instance.poses) | self.presence.coverage.get(instance.id,set())
            # Partitioned children inherit a real original-image registration,
            # but not the rejected whole's motion. Recheck its former interval
            # with the child template instead of leaving only isolated keyframes.
            for record in self.composite_discoveries:
                if instance.id not in record['instances']:
                    continue
                parent = self.bank.observations[record['observation']]
                old = self.instances.get(parent.instance_id)
                if old is not None:
                    frames.update(old.poses)
            for frame in frames:
                if (frame not in self.source_ids or frame < earliest
                        or instance.ended_at is not None and frame > instance.ended_at+2
                        or self.presence.output_margin(instance,frame)):
                    continue
                pose = instance.poses.get(frame)
                if pose is not None and pose.status in ('measured','absent'):
                    continue
                if pose is not None and pose.status == 'predicted' and instance.template_id not in affected:
                    continue
                if self.associated_observations(instance,frame):
                    continue
                if (not self.presence.covers(instance,frame) and self.presence.missing(instance,frame)):
                    continue
                jobs.add((frame,instance.id))
        return sorted(jobs)

    def _finalize_poses(self):
        """Output geometry is decided after the final bounded identity feedback."""
        from .gaps import review_measurements
        review_measurements(self)
        for instance in self.active_instances():
            self._motion(instance)
        self.presence.refresh(for_output=True)
        for instance in self.active_instances():
            for frame, pose in list(instance.poses.items()):
                if (pose.status in ('measured', 'absent') or pose.reason == 'boundary_sam_candidate'
                        or self.presence.output_margin(instance, frame)
                        or self.associated_observations(instance, frame)):
                    continue
                prediction = self.coverage_prediction(instance, frame)
                if prediction is not None:
                    instance.poses[frame] = Pose(frame, prediction[0].copy(), prediction[1],
                        'predicted' if prediction[2] else 'uncertain',
                        presence=pose.presence and pose.reason == 'boundary_sam_presence', reason=pose.reason)
        self.reconciler.refresh_aliases()
        from .pose_filter import prepare_output_filter
        prepare_output_filter(self)
        from .trajectory_quality import prepare_trajectory_quality
        prepare_trajectory_quality(self)
        from .segmentation_support import prepare_segmentation_support
        prepare_segmentation_support(self)
        for instance in self.active_instances():
            if instance.trajectory_quality.get('action') == 'observed_only':
                rows = instance.motion.evidence
                for frame in (rows[0].frame, rows[-1].frame):
                    self.warnings.add(instance.id,instance.template_id,frame,
                        'suspected_scene_incoherent_trajectory','restrict_to_observed_pixels')

    def output_matrix(self, instance, frame, matrix):
        from .pose_filter import output_matrix
        return output_matrix(instance, frame, matrix,
                             self.bank.templates[instance.template_id].anchor)

    def _verify_links(self):
        for link in list(self.bank.links.values()):
            if not link.active:
                continue
            left = [i for i in self.instances.values() if i.template_id == link.a]
            right = [i for i in self.instances.values() if i.template_id == link.b]
            for a in left:
                for b in right:
                    common = sorted(set(a.poses)&set(b.poses))
                    agreeing = False
                    for frame in common:
                        pa, pb = a.poses[frame], b.poses[frame]
                        if pa.status != 'measured' or pb.status != 'measured':
                            continue
                        error = discrepancy(pa.matrix@link.b_to_a, pb.matrix, self.bank.templates[link.b].box)
                        tolerance = 4*(pa.uncertainty+pb.uncertainty)
                        if error <= tolerance:
                            agreeing = True
                        elif agreeing and link.kind == 'bridge':
                            self.bank.revoke(link.a, link.b, 'independent_motion_after_bridge')
                            break

    def _observed_mask(self, frame, instance=None, reliable_only=False):
        mask = np.zeros(self.shape, bool)
        for oid in self.frame_observations[frame]:
            obs = self.bank.observations[oid]
            if instance is not None and obs.instance_id != instance.id:
                continue
            mixed = obs.template_id is not None and self.bank.templates[obs.template_id].mixed
            if reliable_only and (not obs.reliable or obs.ambiguous or mixed):
                continue
            data = self.cache.get(obs.probability_key)
            mask |= warp((data['probability'] > self.config.pixel_threshold)&data['valid'],
                         translation(*obs.origin), self.shape, nearest=True) > 0
        return mask

    def _renders_at(self, instance, frame):
        if self.stage != 'first_pass' and getattr(self.bank.templates[instance.template_id],'composite',False):
            return False
        if instance.alias_of is not None and instance.alias_templates is not None:
            a, b = instance.alias_templates
            target = self.instances.get(instance.alias_of)
            if b in self.bank.members(a) and target is not None:
                canonical = target.poses.get(frame)
                own = instance.poses.get(frame)
                # Archived owners contribute only unique original measurements,
                # never stale uncertain/predicted coverage after disappearance.
                if (own is None or own.status != 'measured' or not own.presence
                        or canonical is not None and canonical.status == 'measured'):
                    return False
        return ((instance.render_start is None or frame >= instance.render_start)
                and (instance.render_end is None or frame <= instance.render_end))

    def associated_observations(self, instance, frame):
        """Actual detections assigned to this occurrence, not template votes."""
        return [o for oid in self.frame_observations.get(frame, ())
            if (o := self.bank.observations[oid]).instance_id == instance.id
            and self.usable_output_observation(o)]

    def usable_output_observation(self, observation):
        """One eligibility rule shared by planning and final rendering."""
        o = observation
        return ((o.ownership_confirmed or o.reliable) and not o.ambiguous
            and not o.crossing_instances and o.pose is not None
            and np.isfinite(o.pose).all() and o.template_id in self.bank.templates
            and not self.bank.templates[o.template_id].mixed
            and o.association_reason != 'stage2_refinement_rejected')

    def render(self, frame, final=True, *, diagnostic=False):
        return self.renderer.render(frame, final, diagnostic=diagnostic)

    def report(self):
        return dict(engine='template', config=self.config.as_dict(),
            stage2_phases=getattr(self,'stage2_phases',{}),
            edge_identity_checks=getattr(self,'edge_identity_checks',[]),
            edge_occurrences=self.edge_occurrences,
            validated_edge_occurrences=getattr(self,'validated_edge_occurrences',()),
            output_plan=self.output_plan.report() if getattr(self,'output_plan',None) is not None else None,
            stats=dict(self.stats, templates=len(self.bank.templates), instances=len(self.instances),
                active_templates=len({min(self.bank.members(t)) for t in self.bank.templates
                    if not getattr(self.bank.templates[t],'composite',False)}),
                active_instances=len(self.active_instances()),
                motion_fit_cache_hits=self.motion_fits.hits,
                motion_fit_cache_misses=self.motion_fits.misses,
                motion_fit_cache_entries=len(self.motion_fits.fits),
                archived_instances=len(self.instances)-len(self.active_instances()),
                suspected_scene_instances=sum(getattr(i,'trajectory_quality',{}).get('status') == 'suspected_scene'
                                              for i in self.active_instances()),
                final_conservative_frames=sum(bool(m.get('uncertainty')) for m in self.render_metrics.values()),
                observations=len(self.bank.observations), sift_extractions=self.matcher.extractions,
                sift_matches=self.matcher.matches, sift_reference_cache_hits=self.matcher.reference_cache_hits,
                identity_switches=None, object_fragmentation=None,
                identity_metrics_note='requires human annotations; not inferred from ID counts'),
            timing=dict(self.timings), cache=self.cache.stats(), matching=self.matcher.matching_report(),
            discovery_prefetch=self.discovery_prefetch,
            warnings=self.warnings.rows,
            sam_stages=self.sam_stages, refinement_batches=self.refinement_batches,
            fragment_associations=self.fragment_associations, pose_reviews=self.pose_reviews,
            measurement_review=getattr(self,'measurement_review_stats',{}),
            measurement_review_checks=getattr(self,'measurement_review_checks',[]),
            composite_discoveries=getattr(self,'composite_discoveries',[]),
            output_pose_filter=self.output_pose_filter,
            discovery_groups=list(self.discovery_groups.values()),
            crossing_observations=list(self.interactions.records.values()),
            pending=self.pending.report(), trajectory_gaps=self.gaps.events,
            presence=self.presence.report(),
            reconciliation=self.reconciler.events, global_identity=self.reconciler.global_reviews,
            template_identities=[dict(a=l.a, b=l.b, active=l.active, reason=l.reason,
                evidence=l.evidence) for l in self.bank.identity_links.values()],
            first_pass=dict(debug=self.debug, mask_generated=False, frames=[dict(frame=f, instances=marks)
                for f, marks in sorted(self.first_pass_marks.items())]),
            rendering=dict(area_checks=False, final_dilation_min=self.config.dilation, final_dilation_max=max(12, self.config.dilation),
                uncertainty_padding_max=12, weak_pose_fallback='nearest_measurement',
                pose_filter_padding_max=4,pose_filter_padding='min_4px_or_half_base_radius_nonaccumulating',
                probability_fusion='rolling_reference_mean', global_mask_pixels=True,
                coarse_fallback_dilation_multiplier=2,
                refinement_crop=dict(scale=1.7, minimum_short_side_fraction=0.35, resolution=1008),
                probability_reference_limit=self.config.mask_reference_limit,
                local_crop='square_max_1p7_target_or_35_percent_short_side',
                local_input_resolutions=[1008],
                neighbour_attenuation='gaussian_outside_global_box_half_size_sigma',
                object_bounds='valid_mask_support',
                first_pass_dilated=False, frames=[self.render_metrics[f] for f in sorted(self.render_metrics)]),
            links=[dict(a=l.a, b=l.b, kind=l.kind, active=l.active, revoked=l.revoked,
                        reason=l.reason, sources=sorted(l.evidence),
                        trajectory_support=getattr(l,'trajectory_support',{}),
                        transform=l.b_to_a.tolist()) for l in self.bank.links.values()],
            templates=[dict(id=t.id, box=t.box, anchor=t.anchor, members=sorted(self.bank.members(t.id)),
                       identity_members=sorted(self.bank.identity_members(t.id)),
                       provisional=t.provisional, mixed=t.mixed, composite=getattr(t,'composite',False),
                       investigation_reason=t.investigation_reason,
                       identity_confirmations=t.identity_confirmations,
                       references=[dict(observation=r.observation_id, source=r.source_id, quality=r.quality) for r in t.references],
                       refinement_sources=sorted(t.refinement_sources),
                       mask_sources=self.bank.fuse(t.id)['sources'].tolist(),
                       mask_refined=bool(self.bank.fuse(t.id)['refined']),
                       mask_view_events=t.mask_view_events,
                       ) for t in self.bank.templates.values()],
            instances=[dict(id=i.id, template=i.template_id, birth=i.birth, ended_at=i.ended_at,
                continuation_of=i.continuation_of, alias_of=i.alias_of,
                alias_templates=i.alias_templates, render_start=i.render_start, render_end=i.render_end,
                motion=i.motion.report() if i.motion else None,
                trajectory_quality=getattr(i,'trajectory_quality',{}),
                segmentation_quality=getattr(i,'segmentation_quality',{}),
                poses=[dict(frame=p.frame, source_id=p.source_id, matrix=p.matrix.tolist(), uncertainty=p.uncertainty,
                       status=p.status, presence=p.presence, reason=p.reason, references=p.references)
                       | (dict(output_matrix=self.output_matrix(i,p.frame,p.matrix).tolist())
                          if p.frame in getattr(i,'output_constraints',{}) else {})
                       for p in sorted(i.poses.values(), key=lambda p: p.frame)]) for i in self.instances.values()])
