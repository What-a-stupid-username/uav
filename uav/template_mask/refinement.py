"""One stage-two refinement batch per registered shared template."""
import time
import json
from dataclasses import replace
import numpy as np
import cv2
from .geometry import area, bounds, discrepancy, intersection, translation, warp, points
from .ownership import OwnershipEvidence


def exclude_independent_foreground(engine, original, candidate, *, refined=False, evidence=None):
    """A target registration does not assign every SAM pixel to that target.

    Independently moving, image-localized marks contribute unknown regions,
    never negative votes, to this target's refinement. The batch review uses
    complete templates from one snapshot, independent of commit order.
    """
    e = engine
    owner = e.instances.get(original.instance_id)
    if owner is None:
        return candidate
    members = e.bank.members(original.template_id)
    evidence = evidence or OwnershipEvidence(e)
    excluded = np.zeros(candidate.probability.shape, bool)
    for other in evidence.independent(members):
        pose = other.poses.get(original.frame)
        if pose is None or pose.status != 'measured' or not pose.presence:
            continue
        fusion = evidence.fusion(other.template_id, refined=refined)
        h,w = fusion['mask'].shape
        projected = pose.matrix @ translation(*fusion['origin'])
        if intersection(candidate.box, bounds(projected,(0,0,w,h))) <= 0:
            continue
        support = warp(fusion['mask'], translation(-candidate.origin[0],-candidate.origin[1])
            @ pose.matrix @ translation(*fusion['origin']), excluded.shape, nearest=True) > 0
        radius = max(1, min(3, int(np.ceil(pose.uncertainty))))
        excluded |= cv2.dilate(support.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)) > 0
    if not excluded.any():
        return candidate
    valid = ~excluded if candidate.valid is None else candidate.valid & ~excluded
    return replace(candidate, valid=valid)


def review_refinement_ownership(engine):
    """Evaluate complete fine templates before changing any view validity.

    Partial discoveries can localize a larger shared template. Revisit the
    stored fine views once all groups have finished, so neither a fragment
    mask nor template processing order determines the exclusion region.
    """
    from .records import Candidate
    e = engine
    updates = []
    evidence = OwnershipEvidence(e)
    for obs in list(e.bank.observations.values()):
        if obs.mask_role != 'refinement' or not obs.reliable or obs.ambiguous:
            continue
        raw_key = getattr(obs,'original_probability_key',None) or obs.probability_key
        data = e.cache.get(raw_key)
        candidate = Candidate(data['probability'], obs.origin, obs.box, obs.score,
                              obs.candidate_id, data['valid'])
        cleaned = exclude_independent_foreground(e, obs, candidate, refined=True, evidence=evidence)
        current = e.cache.get(obs.probability_key)
        if not np.array_equal(current['valid'],cleaned.valid):
            updates.append((obs, cleaned))
    for obs, candidate in updates:
        # Retain the original probability/validity record for revocation and
        # debugging. Derived ownership is a separate immutable cache entry.
        raw_key = getattr(obs,'original_probability_key',None) or obs.probability_key
        obs.original_probability_key = raw_key
        key = e.cache.key('refinement_ownership',raw_key,e.bank.revision)
        e.cache.put(key, probability=candidate.probability, valid=candidate.valid)
        obs.probability_key = key
        if obs.feature_key is not None:
            original_features = getattr(obs,'original_feature_key',None) or obs.feature_key
            obs.original_feature_key = original_features
            features = e.cache.get(original_features)
            xy = np.rint(features['xy']-obs.origin).astype(int)
            h,w = candidate.valid.shape
            inside = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            ids = np.flatnonzero(inside)
            ids = ids[candidate.valid[xy[ids,1],xy[ids,0]]]
            feature_key = e.cache.key('ownership_features',original_features,e.bank.revision)
            e.cache.put(feature_key,**{name:value[ids] for name,value in features.items()})
            obs.feature_key = feature_key
            template = e.bank.templates[obs.template_id]
            template.references = [replace(ref,feature_key=feature_key)
                if ref.observation_id==obs.id else ref for ref in template.references]
    if updates:
        e.bank.revision += 1
    return len(updates)


def bounded_candidate(candidate,transform,low,high):
    """Ownership validity is separate from cached, unmodified SAM probabilities."""
    h,w = candidate.probability.shape
    # Similarity projection is separable along each image axis. Broadcasting
    # avoids allocating two full coordinate grids and an H*W by 2 point array
    # for every SAM candidate. Accumulate in the original transform dtype.
    x = np.arange(w, dtype=transform.dtype)+candidate.origin[0]
    y = np.arange(h, dtype=transform.dtype)+candidate.origin[1]
    inside = np.ones((h,w),bool)
    for axis in range(2):
        projected = (x[None,:]*transform[axis,0]+y[:,None]*transform[axis,1]
                     +transform[axis,2])
        inside &= (projected>=low[axis]) & (projected<high[axis])
    l,t,r,b = candidate.box
    probability_support = inside & ((x[None,:]>=l-12)&(x[None,:]<r+12)
               &(y[:,None]>=t-12)&(y[:,None]<b+12))
    valid = inside if candidate.valid is None else inside & candidate.valid
    # SAM evaluated the entire context crop. Its background outside this
    # candidate's proposal is a real zero vote, not missing image evidence.
    # Keep genuinely unknown ownership and out-of-canvas regions invalid.
    return replace(candidate,probability=np.where(valid & probability_support,candidate.probability,0.),valid=valid)


def select_registered_candidates(engine, original, registered):
    """Do not union a good target with a nested multi-object alternative.

    Evaluate an immutable batch before adding any view to the template. A
    smaller candidate must explain the original target at least as well as
    the containing candidate. Complementary fragments cannot eliminate each
    other merely because they are small or have fewer detected pixels.
    """
    if len(registered) < 2:
        return registered, []
    e = engine
    data = e.cache.get(original.probability_key)
    target = (data['probability'] > e.config.pixel_threshold) & data['valid']
    prepared = []
    for obs, fit in registered:
        pixels = e.cache.get(obs.probability_key)
        support = (pixels['probability'] > e.config.pixel_threshold) & pixels['valid']
        matrix = translation(-original.origin[0],-original.origin[1]) @ original.pose @ np.linalg.inv(fit.matrix) @ translation(*obs.origin)
        in_target = warp(support,matrix,target.shape,nearest=True) > 0
        recall = float((in_target & target).sum()/max(1,target.sum()))
        prepared.append((obs,fit,support,int(support.sum()),recall))
    retained, rejected = [], []
    for large, fit, support, count, recall in prepared:
        replacement = None
        for small, other_fit, small_support, small_count, small_recall in prepared:
            if small.id == large.id or small_count == 0 or small_count >= .8*count:
                continue
            if small_recall < .8 or small_recall+.05 < recall:
                continue
            transform = translation(-large.origin[0],-large.origin[1]) @ fit.matrix @ np.linalg.inv(other_fit.matrix) @ translation(*small.origin)
            projected = warp(small_support,transform,support.shape,nearest=True)>0
            contained = float((projected & support).sum()/max(1,projected.sum()))
            if contained >= .9:
                replacement = small.id
                break
        if replacement is None:
            retained.append((large,fit))
        else:
            rejected.append(dict(observation=large.id,reason='containing_candidate_has_no_extra_target_support',
                                 preferred_observation=replacement))
    return retained, rejected


def select_originals(engine, members, *, evidence=None):
    """Stratify original observations over time; retain the best in each bin."""
    # A short duplicate record may not span enough time to prove separation.
    # All registered occurrences provide the group's independent-motion
    # evidence; a current measured pose still localizes the actual occluder.
    evidence = evidence or OwnershipEvidence(engine)
    independent = evidence.independent(members)
    by_source = {}
    for tid in members:
        for oid in engine.bank.templates[tid].observations:
            obs = engine.bank.observations[oid]
            if not obs.reliable or obs.ambiguous or obs.mask_role != 'discovery':
                continue
            # As in legacy Stage2, overlapping independent objects are poor
            # template references. Retain fragments of the same rigid object.
            if any((pose := other.poses.get(obs.frame)) is not None
                   and pose.status=='measured' and pose.presence
                   and intersection(obs.box,bounds(pose.matrix,
                       engine.bank.templates[other.template_id].box))>0
                   for other in independent):
                continue
            features = len(engine.cache.get(obs.feature_key)['xy'])
            quality = (features, area(obs.box), obs.score)
            if obs.source_id not in by_source or quality > by_source[obs.source_id][0]:
                by_source[obs.source_id] = (quality, obs)
    ordered = sorted(by_source.values(), key=lambda row: row[1].source_id)
    bins = np.array_split(np.arange(len(ordered)), min(len(ordered), engine.config.mask_reference_limit)) if ordered else []
    return [max((ordered[i] for i in ids), key=lambda row: row[0])[1] for ids in bins]


def refine_templates(engine, image_at, *, probe=False):
    """Probe until one usable original, or fill a registered group's budget.

    Previously accepted originals count toward the final shared budget. Failed
    tasks remain recorded and are never called again in the completion phase.
    """
    e = engine
    tick = time.monotonic()
    processed = set()
    evidence = OwnershipEvidence(e)
    referenced = {i.template_id for i in e.active_instances()}
    for template in list(e.bank.templates.values()):
        if template.id in processed:
            continue
        if getattr(template,'composite',False):
            continue
        members = e.bank.members(template.id)
        processed.update(members)
        if not referenced.intersection(members):
            continue
        selected = select_originals(e, members, evidence=evidence)
        if probe:
            selected.sort(key=lambda o: (len(e.cache.get(o.feature_key)['xy']),
                                         area(o.box), o.score), reverse=True)
        # Legacy Stage2 fixes its fusion canvas from the registered original
        # boxes plus 12 pixels, before seeing any refinement probabilities.
        # A local SAM context crop must never redefine an object's geometry.
        extents = [bounds(members[o.template_id] @ np.linalg.inv(o.pose),o.box) for o in selected]
        low = np.floor(np.min([b[:2] for b in extents],axis=0))-12 if extents else None
        high = np.ceil(np.max([b[2:] for b in extents],axis=0))+12 if extents else None
        batch = dict(template=template.id, members=sorted(members),
                     phase='probe' if probe else 'completion',
                     selected_sources=[o.source_id for o in selected], views=[],
                     crop_scale=1.7, minimum_short_side_fraction=.35, resolution=1008)
        e.refinement_batches.append(batch)
        successful = set(e.bank.fuse(template.id)['sources'].tolist())
        attempted_sources = e.bank.refinement_sources(template.id)
        batch['reused_sources'] = sorted(successful)
        for obs in selected:
            if len(successful) >= (1 if probe else e.config.mask_reference_limit):
                break
            owner = e.bank.templates[obs.template_id]
            if obs.source_id in attempted_sources:
                continue
            owner.refinement_sources.add(obs.source_id)
            attempted_sources.add(obs.source_id)
            e.stats['template_calls'] += 1
            image = image_at(obs.frame)
            results = e.detector.detect(image, obs.source_id, obs.box, 1008, refinement=True)
            accepted = []
            registered = []
            for candidate in results:
                # Legacy UAV requires the candidate itself to belong mainly
                # to the target. A containing multi-object mask must not pass
                # merely because it contains the smaller target box.
                if intersection(candidate.box, obs.box)/max(1., area(candidate.box)) < .5:
                    continue
                transform = members[owner.id] @ np.linalg.inv(obs.pose)
                candidate = bounded_candidate(candidate,transform,low,high)
                fine = e.observe(obs.frame, obs.source_id, image, candidate, 1008,
                                 ('refinement', owner.id, obs.id))
                if fine.instance_id is not None:
                    accepted.append(fine.id)
                    continue
                # Geometry comes from raw original pixels, with features confined
                # to each object's box. A coarse mask cannot veto new fine strokes.
                fits = e._matches(owner, fine.feature_key)
                fit = next((f for f in fits if discrepancy(f.matrix, obs.pose, owner.box)
                            <= 4*f.uncertainty), None)
                if fit is None:
                    continue
                registered.append((fine,fit))
            registered, rejected_candidates = select_registered_candidates(e,obs,registered)
            for fine,fit in registered:
                # Refinement segments this known original, not a new trajectory
                # measurement. Preserve its registered pose like legacy Stage2.
                fine.pose, fine.template_id, fine.reliable = obs.pose.copy(), owner.id, True
                fine.instance_id = obs.instance_id
                fine.association_reason = 'stage2_registered_refinement'
                instance = e.instances[obs.instance_id]
                if fine.id not in instance.observations:
                    instance.observations.append(fine.id)
                e.bank.add(fine)
                accepted.append(fine.id)
            # Failed/unrelated candidates are diagnostic only; they do not create
            # new identities or become unassigned output pixels after stage two.
            for candidate in results:
                oid = e.cache.key('obs', obs.source_id, ('refinement', owner.id, obs.id), candidate.candidate_id)
                if oid in e.bank.observations and oid not in accepted:
                    rejected = e.bank.observations[oid]
                    rejected.association_reason = 'stage2_refinement_rejected'
            batch['views'].append(dict(source=obs.source_id, frame=obs.frame, accepted=accepted,
                                       rejected_candidates=rejected_candidates,
                                       reason=None if accepted else 'no_registered_refinement'))
            if accepted:
                fusion = e.bank.fuse(template.id)
                if bool(fusion['has_support']):
                    successful = set(fusion['sources'].tolist())
        fusion = e.bank.fuse(template.id)
        batch['accepted_sources'] = fusion['sources'].tolist()
        batch['coarse_fallback'] = (not bool(batch['accepted_sources']) or
                                    not bool(fusion['has_support']))
        print(json.dumps(dict(event='template_refinement_batch', **batch)), flush=True)
    review_refinement_ownership(e)
    e.timings['template_refinement_seconds'] += time.monotonic()-tick
