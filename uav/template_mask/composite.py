"""Partition a discovery explained by separately registered complete marks.

The original SAM observation remains immutable. Children share its source ID,
retain only pixels uniquely explained by their original-image registration,
and never assert that the two templates form a new common shape.
"""
from dataclasses import replace
import cv2
import numpy as np
from .geometry import discrepancy, translation, warp, support_bounds
from .records import Pose
from .features import Match


def partition_discovery(engine, obs, registrations, *, require_independent=False, refined=False,
                        evaluate_only=False, ownership=None):
    e = engine
    if require_independent and ownership is None:
        from .ownership import OwnershipEvidence
        ownership = OwnershipEvidence(e)
    if obs.mask_role != 'discovery' or getattr(obs,'partition_children', ()):
        return False
    data = e.cache.get(obs.probability_key)
    target = (data['probability'] > e.config.pixel_threshold) & data['valid']
    count = int(target.sum())
    if not count:
        return False
    proposals = []
    for template, fit, evidence in registrations:
        localized = evidence is not None and len(evidence)>3 and evidence[3]
        if (evidence is None or evidence[0] >= .8
                or evidence[1] < (.1 if require_independent and localized else .85)):
            continue
        sources = {e.bank.observations[r].source_id for r in fit.references
                   if r in e.bank.observations and e.bank.observations[r].reliable
                   and not e.bank.observations[r].ambiguous
                   and e.bank.observations[r].source_id != obs.source_id}
        if len(sources) < 2:
            continue
        fusion = e.bank.fuse(template.id,identity=not refined)
        support = warp(fusion['mask'],translation(-obs.origin[0],-obs.origin[1])
            @ fit.matrix @ translation(*fusion['origin']),target.shape,nearest=True)>0
        radius = max(1,min(3,int(np.ceil(fit.uncertainty))))
        near = cv2.dilate(support.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))>0
        explained = near & target
        # Independent motion can disprove a rigid whole even when one mark is
        # small or SAM includes substantial unexplained background. Online
        # ownership partitioning still requires the stricter whole explanation.
        minimum = max(32,.03*count) if require_independent else .15*count
        if explained.sum() < minimum:
            continue
        if any(t.id in e.bank.members(template.id) and
               discrepancy(m.matrix,fit.matrix,template.box) <= 4*(m.uncertainty+fit.uncertainty)
               for t,m,_ in proposals):
            continue
        proposals.append((template,fit,explained))
    pairs = []
    for k,left in enumerate(proposals):
        for right in proposals[k+1:]:
            a,b = left[2],right[2]
            if (a & b).sum() > .1*min(a.sum(),b.sum()):
                continue
            coverage = float((a | b).sum()/count)
            if coverage >= .85 or require_independent:
                if require_independent:
                    members_a, members_b = e.bank.members(left[0].id), e.bank.members(right[0].id)
                    owners_a = [i for i in ownership.owners if i.template_id in members_a]
                    owners_b = [i for i in ownership.owners if i.template_id in members_b]
                    # The classification uses all measured originals. The query
                    # frame only annotates past/future counts, not this state.
                    if not any(ia.id != ib.id and ownership.separate(ia,ib)
                               for ia in owners_a for ib in owners_b):
                        continue
                pairs.append((coverage,left,right))
    if not pairs:
        return False
    pairs.sort(key=lambda row:row[0],reverse=True)
    # Distinct explanations using different templates remain unresolved.
    best = pairs[0]
    roots = {min(e.bank.members(t.id)) for t,_,_ in best[1:]}
    if not require_independent and any(score >= best[0]-.05 and {min(e.bank.members(t.id)) for t,_,_ in (a,b)} != roots
           for score,a,b in pairs[1:]):
        return False
    if evaluate_only:
        return best
    return apply_partition(e,obs,best)


def apply_partition(e,obs,best):
    """Commit one already validated explanation, preserving original pixels."""
    data = e.cache.get(obs.probability_key)
    target = (data['probability'] > e.config.pixel_threshold) & data['valid']
    children,owners = [],[]
    prepared=[]
    for index,(template,fit,pixels) in enumerate(best[1:]):
        other = best[2 if index==0 else 1][2]
        own = pixels & ~other
        box = support_bounds(own)
        if box is None:
            return False
        prepared.append((index,template,fit,own,box))
    # Validate every child before mutating any cache, observation or instance.
    # A failed second child cannot leave a committed first child behind.
    for index,template,fit,own,box in prepared:
        x,y=obs.origin
        probability = np.where(own,data['probability'],0.)
        # Every other foreground area is unknown to this child, not zero evidence.
        valid = data['valid'] & (~target | own)
        child_id = e.cache.key('partition',obs.id,index)
        pkey,fkey = e.cache.key('probability',child_id),e.cache.key('observation_features',child_id)
        e.cache.put(pkey,probability=probability,valid=valid)
        features = e.cache.get(obs.feature_key)
        xy = np.rint(features['xy']-obs.origin).astype(int)
        inside = ((xy[:,0]>=0)&(xy[:,0]<own.shape[1])&(xy[:,1]>=0)&(xy[:,1]<own.shape[0]))
        ids=np.flatnonzero(inside)
        ids=ids[own[xy[ids,1],xy[ids,0]]]
        e.cache.put(fkey,**{name:value[ids] for name,value in features.items()})
        child = replace(obs,id=child_id,candidate_id=obs.candidate_id+f'/part:{index}',
            box=(x+box[0],y+box[1],x+box[2],y+box[3]),probability_key=pkey,feature_key=fkey,
            instance_id=None,template_id=template.id,pose=fit.matrix.copy(),reliable=True,
            ambiguous=False,ownership_confirmed=True,association_reason='partitioned_original_image_match',
            crossing_instances=(),partition_children=())
        e.bank.observations[child.id]=child
        e.frame_observations[obs.frame].append(child.id)
        nearby=e._near_instances(template.id,fit.matrix,obs.frame,fit.uncertainty)
        if len(nearby)==1:
            owner=nearby[0][1]
            child.pose=fit.matrix @ e.bank.members(template.id)[owner.template_id]
            child.template_id=owner.template_id
            old=owner.poses.get(obs.frame)
            e._assign(owner,child)
            if old is not None and old.status=='measured':
                owner.poses[obs.frame]=old
        else:
            owner=e._new_instance(template,child)
        # Geometry is certified by original references, not by the SAM union.
        if owner.poses[obs.frame].references == (child.id,):
            owner.poses[obs.frame]=Pose(obs.frame,child.pose.copy(),fit.uncertainty,
                'measured',obs.source_id,fit.references,True)
        owners.append(owner.id);children.append(child.id)
    obs.ambiguous,obs.reliable=True,False
    obs.association_reason='composite_discovery_partitioned'
    obs.partition_children=tuple(children)
    e.composite_discoveries.append(dict(observation=obs.id,frame=obs.frame,source=obs.source_id,
        coverage=best[0],children=children,instances=owners,
        action='separate_original_image_registrations_no_shared_template'))
    return True


def review_composite_templates(engine, *, refined=False):
    """Revisit retained discovery evidence once with complete motion histories.

    No timeline decode, new segmentation, or inference-generated evidence is
    needed. A single composite detection is not enough to reject a template:
    the registered constituent identities must also have measured independent
    relative motion. Originals remain available as local output coverage.
    """
    e = engine
    roots = sorted({min(e.bank.members(t)) for t in e.bank.templates})
    # This review applies mutations only after all proposals are evaluated.
    # Reuse group/owner metadata throughout that immutable snapshot.
    groups = {tid: e.bank.members(tid) for tid in roots}
    reference_sources = {tid: {e.bank.observations[r.observation_id].source_id
                              for r in e.bank.matching_references(tid)} for tid in roots}
    active = tuple(e.active_instances())
    from .ownership import OwnershipEvidence
    ownership = OwnershipEvidence(e)
    owners = {tid: [(owner, np.linalg.inv(groups[tid][owner.template_id]))
                    for owner in active if owner.template_id in groups[tid]] for tid in roots}
    # Partitioning requires two distinct independently moving constituents.
    # Establish that necessary condition before matching every original to
    # every template. Relative motion uses the same immutable snapshot as the
    # final partition check; shape/registration alone cannot bypass it.
    eligible = [tid for tid in roots if len(reference_sources[tid]) >= 2
                and not e.bank.templates[tid].mixed
                and not getattr(e.bank.templates[tid], 'composite', False)]
    independent_pairs = [(a, b) for index, a in enumerate(eligible)
                         for b in eligible[index+1:]
                         if any(ownership.separate(ia, ib) or ownership.separate(ib, ia)
                                for ia, _ in owners[a] for ib, _ in owners[b])]
    if not independent_pairs:
        return set()
    # Strong, repeatedly observed hypotheses explain weaker composite ones.
    # An accumulated large area or extra duplicate owners supplies no priority.
    roots.sort(key=lambda t: -len(reference_sources[t]))
    changed = set()
    proposals = []
    for tid in reversed(roots):
        members = set(groups[tid])
        constituents = {other for pair in independent_pairs if tid not in pair for other in pair}
        if not constituents:
            continue
        original_ids = [oid for t in members for oid in e.bank.templates[t].observations
                        if e.bank.observations[oid].mask_role == 'discovery'
                        and not e.bank.observations[oid].ambiguous]
        for oid in original_ids:
            obs = e.bank.observations[oid]
            if getattr(obs,'partition_children',()):
                continue
            registrations = []
            for other in roots:
                template = e.bank.templates[other]
                if other not in constituents or other in members or template.mixed or getattr(template,'composite',False):
                    continue
                # Match independent originals only; weak single-view hypotheses
                # cannot dismantle an established template.
                if len(reference_sources[other]) < 2:
                    continue
                for fit in e._matches(template,obs.feature_key,max_locations=4):
                    if fit.coverage < .08:
                        continue
                    registrations.append((template,fit,e._shape_evidence(template,fit,obs,identity=not refined)))
                # A union crop may give its SIFT budget to just one component.
                # Independently measured current poses still localize the other
                # component; accept only their visible pixels, never the union.
                for owner, inverse_transform in owners[other]:
                    pose=owner.poses.get(obs.frame)
                    if pose is None or pose.status!='measured' or not pose.presence:continue
                    refs=tuple(r for r in pose.references if r in e.bank.observations
                               and e.bank.observations[r].source_id!=obs.source_id
                               and e.bank.observations[r].reliable and not e.bank.observations[r].ambiguous)
                    if len({e.bank.observations[r].source_id for r in refs})<2:continue
                    fit=Match(pose.matrix@inverse_transform,
                              pose.uncertainty,0,0.,refs)
                    shape=e._shape_evidence(template,fit,obs,identity=not refined)
                    if shape is not None:registrations.append((template,fit,(*shape,True)))
            proposal = partition_discovery(e,obs,registrations,require_independent=True,
                                           refined=refined,evaluate_only=True,ownership=ownership)
            if proposal:
                proposals.append((members,obs,registrations,proposal))
                break
    # Every explanation above used the same bank. Apply only explanations whose
    # constituent templates are not themselves rejected by this snapshot.
    rejected = set().union(*(members for members,_,_,_ in proposals)) if proposals else set()
    for members,obs,registrations,proposal in proposals:
        if any(t.id in rejected for t,_,_ in proposal[1:]):
            continue
        if not apply_partition(e,obs,proposal):
            continue
        for t in members:
            e.bank.templates[t].composite = True
            changed.add(t)
        e.composite_discoveries[-1]['action'] = 'reject_composite_template_using_independent_motion'
        e.composite_discoveries[-1]['rejected_templates'] = sorted(members)
    if changed:
        e.bank.revision += 1
    return changed
