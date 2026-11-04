"""Final output: observed pixels, predicted coverage and optional edge margins.

This module consumes evidence; it never changes ownership, templates or poses.
Every ordinary contribution goes through the same output dilation.
"""
from dataclasses import dataclass
import math
import numpy as np
from .geometry import dilate, discrepancy, support_bounds, translation, warp, visible_support
from .policy import SkipVideo, uncertain_mask, uncertainty_radius
from .records import Instance
from .pose_filter import output_matrix
from .output_geometry import selected_observation, selected_matrix
from .trajectory_quality import observed_only

MAX_POSE_FILTER_PADDING = 4  # Additional source pixels, never accumulated.


@dataclass
class Contribution:
    owner: Instance
    raw: np.ndarray
    mask: np.ndarray
    conservative: bool = False
    optional: bool = False
    method: str | None = None
    explained: bool = False


class MaskRenderer:
    def __init__(self, engine):
        self.engine = engine

    def fusion(self, instance, final):
        e = self.engine
        if not final:
            return e.bank.fuse(instance.template_id, coarse=True), True
        return e.bank.output_fusion(instance.template_id)

    def dilate(self, mask, metrics=None, *, multiplier=1, pose_correction=0.):
        e = self.engine
        box = support_bounds(mask)
        if box is None:
            return mask
        w,h = box[2]-box[0],box[3]-box[1]
        radius = max(e.config.dilation, min(max(12, e.config.dilation), math.ceil(.02*math.sqrt(w*h))))
        radius *= multiplier
        # Filtering removes image-fit jitter, but its edge displacement still
        # merits a small coverage allowance. Cap it at half the normal margin;
        # a noisy match must never create an unbounded growing halo.
        extra = min(MAX_POSE_FILTER_PADDING,
                    math.ceil(max(0.,pose_correction-1e-6)), math.ceil(radius*.5))
        radius += extra
        if metrics is not None:
            metrics['dilation_radii'].append(radius)
            if extra:
                metrics.setdefault('pose_filter_padding',[]).append(dict(
                    correction_pixels=float(pose_correction),extra_radius=extra))
        return dilate(mask, radius, box=box) if radius else mask

    def finish(self, mask, owner, frame, final, metrics, diagnostic, *, coarse=False,
               reason=None, sigma=0., observation=None, pose_correction=0.):
        """Apply uncertainty only when requested, then normal output dilation."""
        e = self.engine
        if not final:
            return Contribution(owner, mask, mask)
        if reason is not None:
            mask = uncertain_mask(mask, sigma, owner, frame, e.config, e.warnings,
                                  reason, diagnostic=diagnostic)
        expanded = self.dilate(mask, metrics, multiplier=2 if coarse else 1,
                               pose_correction=pose_correction)
        if coarse:
            metrics.setdefault('coarse_fallback', []).append(dict(instance=owner.id, template=owner.template_id))
            e.warnings.add(owner.id, owner.template_id, frame, 'coarse_mask_fallback', 'coarse_mask_double_dilation')
        conservative = reason is not None and expanded.any() and e.config.uncertain_mask_policy != 'ignore'
        if conservative:
            row = dict(instance=owner.id, template=owner.template_id, reason=reason,
                sigma=float(sigma), extra_radius=uncertainty_radius(sigma) if math.isfinite(sigma) else None,
                requested_radius=int(math.ceil(3*sigma)) if math.isfinite(sigma) else None)
            if observation is not None:
                row['observation'] = observation
            metrics['uncertainty'].append(row)
        return Contribution(owner, mask, expanded, conservative,
                            explained=reason is None and not coarse)

    def unassigned(self, frame, metrics, diagnostic, *, explained=None):
        e = self.engine
        for oid in e.frame_observations[frame]:
            obs = e.bank.observations[oid]
            if obs.association_reason == 'stage2_refinement_rejected':
                continue
            mixed = obs.template_id is not None and e.bank.templates[obs.template_id].mixed
            composite = obs.template_id is not None and getattr(e.bank.templates[obs.template_id],'composite',False)
            if (obs.reliable or obs.ownership_confirmed) and not obs.ambiguous and not mixed and not composite:
                continue
            mask = e._observed_candidate(obs)
            for child_id in getattr(obs,'partition_children',()):
                mask &= ~e._observed_candidate(e.bank.observations[child_id])
            if explained is not None:
                mask &= ~explained
            if not mask.any():
                continue
            owner = e.instances.get(obs.instance_id,
                e.instances.get(obs.boundary_instance_id, Instance(-1, -1, frame)))
            reason = ('composite_template_original_pixels_only' if composite else
                      'mixed_hypothesis_identity_unresolved' if mixed else
                      'ambiguous_candidate' if obs.ambiguous else
                      obs.association_reason or 'unassigned_candidate')
            yield self.finish(mask, owner, frame, True, metrics, diagnostic,
                coarse=obs.mask_role == 'discovery',
                reason=reason, sigma=1., observation=obs.id)

    def occurrence(self, instance, frame, final, metrics, diagnostic):
        e = self.engine
        if final and observed_only(instance,frame):
            # A suspected scene track cannot propagate a shared template through
            # SIFT-only matches. Preserve actual assigned SAM pixels for review.
            observations = [e.bank.observations[oid] for oid in e.frame_observations.get(frame, ())
                if e.bank.observations[oid].instance_id == instance.id
                and not e.bank.observations[oid].ambiguous]
            metrics.setdefault('trajectory_restrictions', []).append(dict(
                instance=instance.id, template=instance.template_id,
                reason=('continuous_sift_sam_disagreement' if
                    getattr(instance,'segmentation_quality',{}).get('action')=='restrict_intervals'
                    else 'suspected_scene_observed_only'), observations=[o.id for o in observations]))
            if not observations:
                return None
            mask = np.zeros(e.shape, bool)
            for obs in observations:
                mask |= e._observed_candidate(obs)
            return self.finish(mask, instance, frame, final, metrics, diagnostic,
                coarse=any((o.mask_role == 'discovery' for o in observations)))
        if e.presence.output_margin(instance, frame):
            mask, method = self.margin(instance, frame, final, metrics)
            return Contribution(instance, mask, mask, conservative=True, optional=True, method=method)
        associated = e.associated_observations(instance, frame)
        observed = selected_observation(e, instance, frame, associated)
        pose = instance.poses.get(frame)
        if observed is None and (pose is None or pose.status == 'absent' or pose.reason == 'boundary_sam_candidate'):
            return None
        try:
            fusion, coarse = self.fusion(instance, final)
        except ValueError as error:
            if not final:
                return None
            e.warnings.add(instance.id, instance.template_id, frame, str(error), 'skip_video')
            raise SkipVideo(str(error)) from error
        matrix = observed.pose if observed is not None else pose.matrix
        correction = 0.
        if final:
            template = e.bank.templates[instance.template_id]
            filtered = selected_matrix(e,instance,frame,observed)
            correction = discrepancy(matrix,filtered,template.box)
            matrix = filtered
        mask = warp(fusion['mask'],
                    matrix @ translation(*fusion['origin']), e.shape, nearest=True) > 0
        if (final and not mask.any() and frame in getattr(instance,'boundary_output_frames',())
                and bool(fusion['has_support']) and visible_support(fusion['mask'],
                    matrix@translation(*fusion['origin']),e.shape)==0):
            # Validated boundary transport located this template outside the
            # image. Empty visible support is not an unlocalizable occurrence.
            return Contribution(instance,mask,mask)
        reason, sigma = None, 0.
        if observed is not None:
            if not mask.any() or not len(fusion['sources']):
                mask = np.zeros(e.shape, bool)
                for obs in associated:
                    mask |= e._observed_candidate(obs)
                coarse = any((o.mask_role == 'discovery' for o in associated))
            if final:
                metrics.setdefault('observed_presence', []).append(dict(instance=instance.id,
                    template=instance.template_id, observations=[o.id for o in associated],
                    registration='predicted' if (observed.trajectory_evidence or {}).get(
                        'coordinate_registration') == 'predicted' else 'original_image'))
        elif not len(fusion['sources']):
            reason, sigma = 'no_usable_local_probability', float('inf')
        elif pose.status != 'measured' or e.bank.templates[instance.template_id].mixed:
            reason = ('mixed_hypothesis_identity_unresolved' if e.bank.templates[instance.template_id].mixed
                      else pose.reason or 'presence_unknown')
            sigma = pose.uncertainty
        return self.finish(mask, instance, frame, final, metrics, diagnostic,
                           coarse=coarse, reason=reason, sigma=sigma,pose_correction=correction)

    def margin(self, instance, frame, final, metrics):
        """Optional padding: prediction, nearest real observation, then empty."""
        e = self.engine
        pose = instance.poses.get(frame)
        prediction = e._prediction(instance, frame) if pose is None else None
        matrix = pose.matrix if pose is not None else prediction[0] if prediction is not None else None
        correction = 0.
        if final and matrix is not None:
            template = e.bank.templates[instance.template_id]
            filtered = output_matrix(instance,frame,matrix,template.anchor)
            correction = discrepancy(matrix,filtered,template.box)
            matrix = filtered
        if matrix is not None and np.isfinite(matrix).all() and abs(np.linalg.det(matrix[:2, :2])) > 1e-9:
            try:
                fusion, coarse = self.fusion(instance, final)
                support = fusion['mask']
                if bool(fusion['has_support']) and len(fusion['sources']):
                    return self.margin_projection(support, matrix @ translation(*fusion['origin']),
                                                  final, metrics, coarse,correction), 'prediction'
            except ValueError:
                pass
        observations = [o for o in e.bank.observations.values()
            if instance.id in (o.instance_id, o.boundary_instance_id)
            and not o.ambiguous and not o.crossing_instances
            and o.association_reason != 'stage2_refinement_rejected']
        for obs in sorted(observations, key=lambda o: (abs(o.frame-frame), o.frame)):
            data = e.cache.get(obs.probability_key)
            support = (data['probability'] > e.config.pixel_threshold) & data['valid']
            if support.any():
                return self.margin_projection(support, translation(*obs.origin), final, metrics,
                    obs.mask_role == 'discovery'), 'nearest_observation'
        return np.zeros(e.shape, bool), 'empty'

    def margin_projection(self, support, matrix, final, metrics, coarse,pose_correction=0.):
        e = self.engine
        normal = max(12,e.config.dilation)*(2 if coarse else 1)
        padding = normal+min(MAX_POSE_FILTER_PADDING,math.ceil(normal*.5)) if final else 0
        h, w = e.shape
        mask = warp(support, translation(padding, padding) @ matrix,
                    (h+2*padding, w+2*padding), nearest=True) > 0
        if final:
            mask = self.dilate(mask, metrics, multiplier=2 if coarse else 1,
                               pose_correction=pose_correction)
        return mask[padding:padding+h, padding:padding+w]

    def render(self, frame, final=True, *, diagnostic=False):
        e = self.engine
        canvas = np.zeros(e.shape, bool) if final else e._observed_mask(frame)
        raw, conservative = canvas.copy(), np.zeros(e.shape, bool)
        metrics = dict(frame=frame, diagnostic=diagnostic, dilation_radii=[], uncertainty=[],
                       pixels_before_dilation=int(raw.sum()), pixels_after_dilation=0)
        explained = np.zeros(e.shape, bool) if final else None
        def contributions():
            plan = getattr(e,'output_plan',None) if final else None
            owners = (e.instances[i] for i in plan.frame_owners.get(frame,())) if plan is not None else e.instances.values()
            for instance in owners:
                if plan is not None or e._renders_at(instance, frame):
                    layer = self.occurrence(instance, frame, final, metrics, diagnostic)
                    if layer is not None:
                        if final and layer.explained and not layer.optional:
                            explained[:] |= layer.mask
                        yield layer
            if final:
                # Discovery may contain both already-owned strokes and new
                # pixels. Expand only the unexplained residual, never the
                # entire duplicate coarse candidate on each discovery frame.
                yield from self.unassigned(frame, metrics, diagnostic, explained=explained)
        for layer in contributions():
            raw |= layer.raw
            canvas |= layer.mask
            if layer.conservative:
                conservative |= layer.mask
            if final and layer.optional:
                metrics.setdefault('output_margins', []).append(dict(instance=layer.owner.id,
                    template=layer.owner.template_id, method=layer.method, pixels=int(layer.mask.sum())))
        metrics['pixels_after_dilation'] = int(canvas.sum())
        if final:
            metrics.update(pixels_before_dilation=int(raw.sum()), conservative_pixels=int(conservative.sum()))
            if e.debug:
                e.cache.put(e.cache.key('conservative_mask', frame), packed=np.packbits(conservative))
            e.render_metrics[frame] = metrics
        return canvas
