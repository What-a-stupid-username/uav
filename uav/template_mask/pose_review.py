"""Sparse original-image review of short, weakly registered motion intervals.

Motion proposes a search, never a replacement measurement. Decisions use an
immutable snapshot and require multiple independent image references.
"""
import numpy as np
from .edge_identity import independent_sources
from .edge_trajectory import boundary_prediction
from .gaps import measured
from .geometry import bounds, clip_box, discrepancy, footprint
from .records import Pose, Reference


def review_matches(engine, template, target_key, source_id):
    """Keep normal references; add nearby originals only for a failed review.

    The retained eight views optimize template coverage, not every difficult
    temporal boundary. Supplementary image pairs reuse cached features without
    modifying the template's references or adding observations.
    """
    e = engine
    fits = e._matches(template,target_key,max_locations=4)
    if any(len(independent_sources(e,f,source_id))>=2 for f in fits):
        return fits
    members = e.bank.members(template.id)
    by_source = {}
    for tid in members:
        for oid in e.bank.templates[tid].observations:
            obs = e.bank.observations[oid]
            if (obs.source_id==source_id or not obs.reliable or obs.ambiguous
                    or obs.pose is None or obs.crossing_instances):
                continue
            current = by_source.get(obs.source_id)
            if current is None or (current.mask_role=='discovery' and obs.mask_role!='discovery'):
                by_source[obs.source_id] = obs
    nearby = sorted(by_source.values(),key=lambda o:(abs(o.source_id-source_id),o.source_id))[:4]
    references = list(e.bank.matching_references(template.id,members=members))
    original_count = len(references)
    retained = {r.source_id for r in references}
    references.extend(Reference(o.id,o.source_id,o.feature_key,0.) for o in nearby
                      if o.source_id not in retained)
    if len(references)==original_count:
        return fits
    return e.matcher.template_matches(template,e.bank.observations,target_key,4,
        references=references,member_transforms=members)


def review_weak_intervals(engine, image_at):
    e = engine
    changes = []
    stats = dict(weak_intervals=0,motion_conflicts=0,image_queries=0,accepted=0)
    checks = []
    for instance in e.active_instances():
        template = e.bank.templates[instance.template_id]
        rows = measured(instance)
        strong = [len(independent_sources(e, p, p.source_id)) >= 2 for p in rows]
        cursor = 0
        while cursor < len(rows):
            if strong[cursor]:
                cursor += 1
                continue
            start = cursor
            while cursor < len(rows) and not strong[cursor]:
                cursor += 1
            weak = rows[start:cursor]
            if weak[-1].frame-weak[0].frame >= e.config.global_stride:
                continue
            stats['weak_intervals'] += 1
            sides = []
            for side, indices in (('left', range(start-1, max(-1,start-33), -1)),
                                  ('right', range(cursor, min(len(rows),cursor+32)))):
                samples = []
                for index in indices:
                    if not strong[index]:
                        break
                    samples.append(rows[index])
                samples.sort(key=lambda p:p.frame)
                if len({p.source_id for p in samples}) >= 4:
                    sides.append((side,samples))
            if not sides:
                continue
            for original in weak:
                if (original.frame in instance.ambiguous_frames
                        or e.associated_observations(instance,original.frame)
                        or e.presence.output_margin(instance,original.frame)):
                    continue
                predictions = []
                for side, samples in sides:
                    nearest = samples[-1] if side == 'left' else samples[0]
                    if abs(original.frame-nearest.frame) > e.config.global_stride:
                        continue
                    result = boundary_prediction(e,samples,template,side,original.source_id)
                    if result is not None and result[2]:
                        predictions.append((result,samples))
                if not predictions:
                    continue
                (matrix,sigma,_),samples = min(predictions,key=lambda p:p[0][1])
                if any(discrepancy(matrix,p[0],template.box)>4*(sigma+p[1])
                       for p,_ in predictions):
                    continue
                if discrepancy(matrix,original.matrix,template.box)<=4*(sigma+original.uncertainty):
                    continue
                stats['motion_conflicts'] += 1
                predicted_box = bounds(matrix,template.box)
                original_box = bounds(original.matrix,template.box)
                search_box = (*np.minimum(predicted_box[:2],original_box[:2]),
                              *np.maximum(predicted_box[2:],original_box[2:]))
                roi = clip_box(search_box,e.shape,
                               padding=max(3*sigma,.25*footprint(predicted_box)))
                if roi[2]<=roi[0] or roi[3]<=roi[1]:
                    continue
                stats['image_queries'] += 1
                key = e.matcher.extract(image_at(original.frame),original.source_id,roi)
                candidates = review_matches(e,template,key,original.source_id)
                check = dict(instance=instance.id,frame=original.frame,
                    candidates=[dict(inliers=f.inliers,coverage=f.coverage,
                        reference_sources=len(independent_sources(e,f,original.source_id)),
                        prediction_error=discrepancy(f.matrix,matrix,template.box),
                        uncertainty=float(f.uncertainty)) for f in candidates],accepted=False)
                checks.append(check)
                fits = [fit for fit in candidates
                        if fit.inliers>=e.config.min_inliers and fit.coverage>=.35
                        and len(independent_sources(e,fit,original.source_id))>=2]
                # A real turn or repeated motif supported by independent images
                # vetoes replacement, even when it disagrees with the path.
                if not fits or any(discrepancy(f.matrix,matrix,template.box)
                                   >4*(sigma+f.uncertainty) for f in fits):
                    continue
                fit = min(fits,key=lambda f:discrepancy(f.matrix,matrix,template.box))
                if any(discrepancy(fit.matrix,f.matrix,template.box)
                       >4*(fit.uncertainty+f.uncertainty) for f in fits):
                    continue
                # Never steal a separately measured occurrence of this template.
                members = e.bank.members(template.id)
                if any(other.id!=instance.id and other.template_id in members
                       and (p:=other.poses.get(original.frame)) is not None
                       and p.status=='measured' and p.presence
                       and discrepancy(fit.matrix@members[other.template_id],p.matrix,
                           e.bank.templates[other.template_id].box)<=4*(fit.uncertainty+p.uncertainty)
                       for other in e.instances.values()):
                    continue
                changes.append((instance,original,fit,samples))
                check['accepted'] = True
    for instance,original,fit,samples in changes:
        e.pose_reviews.append(dict(instance=instance.id,template=instance.template_id,
            frame=original.frame,source=original.source_id,reason='independent_image_interval_review',
            original_matrix=original.matrix.tolist(),original_references=original.references,
            replacement_matrix=fit.matrix.tolist(),replacement_references=fit.references,
            evidence_sources=[p.source_id for p in samples]))
        instance.poses[original.frame] = Pose(original.frame,fit.matrix.copy(),fit.uncertainty,
            'measured',original.source_id,fit.references,True)
    stats['accepted'] = len(changes)
    e.measurement_review_stats = stats
    e.measurement_review_checks = checks
    return {instance.template_id for instance, *_ in changes}
