"""Verify frame-clipped identities with independent neighbouring originals.

Only missing strokes outside the image are excused. SAM is not called, and
the existing instance lifetime is retained. Probe matches do not teach masks.
"""
import numpy as np
from .geometry import bounds, clip_box, discrepancy, translation, visible_support, footprint
from .edge_trajectory import BoundaryEvidence, match_visible_boundary, corroborate


def independent_sources(engine, fit, source_id):
    """A query image cannot corroborate its own reference registration."""
    observations=engine.bank.observations
    return {obs.source_id for ref in fit.references
            if (obs:=observations.get(ref)) is not None and obs.source_id!=source_id
            and getattr(obs,'reliable',True) and not getattr(obs,'ambiguous',False)}


def edge_originals(engine, originals):
    """Bound work by edge evidence, not by the template's total lifetime."""
    h,w=engine.shape
    sources={}
    for obs in originals:
        margin=max(2.,.05*min(obs.box[2]-obs.box[0],obs.box[3]-obs.box[1]))
        if min(obs.box[0],obs.box[1],w-obs.box[2],h-obs.box[3])>margin:
            continue
        sources.setdefault(obs.source_id,obs)
    ordered=sorted(sources.values(),key=lambda o:o.frame)
    ids=np.unique(np.linspace(0,len(ordered)-1,min(8,len(ordered))).astype(int)) if ordered else ()
    return [ordered[i] for i in ids]


def resolve_edges(engine,image_at):
    e=engine
    proposals=[]
    checks=[]
    registrations_cache={}
    trajectory_evidence=BoundaryEvidence(e)
    roots=sorted({min(e.bank.members(t)) for t in e.bank.templates
                  if not e.bank.templates[t].composite and not e.bank.templates[t].mixed})
    for tid in roots:
        members=e.bank.members(tid)
        originals=[e.bank.observations[oid] for t in members
                   for oid in e.bank.templates[t].observations
                   if e.bank.observations[oid].mask_role=='discovery'
                   and e.bank.observations[oid].reliable and not e.bank.observations[oid].ambiguous]
        # A truncated mark can persist for many discovery frames while slowly
        # entering. More observations do not turn a fragment into a whole.
        originals=edge_originals(e,originals)
        views_by_source={}
        for observation in e.bank.observations.values():
            if (observation.template_id in members and observation.mask_role=='refinement'
                    and observation.reliable and not observation.ambiguous):
                views_by_source.setdefault(observation.source_id,[]).append(observation)
        matches=[]
        for obs in originals:
            if obs.mask_role!='discovery':continue
            for other in roots:
                if other==tid:continue
                template=e.bank.templates[other]
                views=[obs]+views_by_source.get(obs.source_id,[])
                fits=[fit for view in views for fit in e._matches(template,view.feature_key,max_locations=4)]
                fits.extend(match_visible_boundary(e,template,obs,evidence=trajectory_evidence))
                if len(originals)<=2:
                    # Fine references may register in the reverse direction
                    # even when a clipped coarse observation has few features.
                    # Reuse global registration, including source provenance.
                    from .global_identity import registrations
                    from .features import Match
                    pair=(other,tid)
                    if pair not in registrations_cache:
                        registrations_cache[pair]=registrations(e,other,tid)
                    seed_pose=obs.pose@np.linalg.inv(members[obs.template_id])
                    for row in registrations_cache[pair]:
                        refs=tuple(oid for oid in (row['observation'],*row['references'])
                            if (o:=e.bank.observations[oid]).source_id!=obs.source_id
                            and o.reliable and not o.ambiguous
                            and o.template_id in e.bank.members(other))
                        fits.append(Match(seed_pose@np.linalg.inv(row['matrix']),row['sigma'],
                            row['inliers'],0.,refs))
                for fit in fits:
                    sources=independent_sources(e,fit,obs.source_id)
                    motion_localized=fit.inliers==0 and not fit.references
                    if not motion_localized and (not sources or fit.inliers<e.config.min_inliers):continue
                    evidence=[(view,e._shape_evidence(template,fit,view,identity=False)) for view in views]
                    evidence=[(view,shape) for view,shape in evidence if shape is not None]
                    if not evidence:continue
                    view,shape=max(evidence,key=lambda row:min(row[1][:2]))
                    check=dict(template=tid,target=other,frame=obs.frame,inliers=fit.inliers,
                               reference_sources=len(sources),shape=shape,verified_sources=[])
                    checks.append(check)
                    if shape is None or shape[0]<.9:continue
                    trajectory=corroborate(e,template,view,fit,evidence=trajectory_evidence)
                    if shape[1]<.9 and trajectory is None:continue
                    fusion=e.bank.output_fusion(other)[0]
                    visible=visible_support(fusion['mask'],fit.matrix@translation(*fusion['origin']),e.shape)
                    total=np.count_nonzero(fusion['mask'])
                    # Clipping is an image-boundary fact, not a minimum 20%
                    # missing-area rule. Slightly clipped entries need the same
                    # independently verified coordinate association.
                    if visible>=total or not visible:continue
                    if len(sources)<2 and trajectory is None:continue
                    matches.append((other,view,fit,check,trajectory))
        groups={m[0] for m in matches}
        if len(groups)!=1:continue
        # Repeated strokes can give incompatible coordinates in ONE template.
        # Do not let reference count silently resolve that ambiguity.
        relations_by_match = []
        for target, view, candidate, _, _ in matches:
            seed = view.pose @ np.linalg.inv(members[view.template_id])
            relations_by_match.append(np.linalg.inv(candidate.matrix) @ seed)
        consistent = all(discrepancy(a, b, e.bank.templates[tid].box)
            <= 4*(matches[i][2].uncertainty + matches[j][2].uncertainty)
            for i, a in enumerate(relations_by_match)
            for j, b in enumerate(relations_by_match[:i]))
        if not consistent:
            for row in matches:
                row[3]['rejection'] = 'ambiguous_template_coordinates'
            continue
        other,obs,fit,check,trajectory=max(matches,key=lambda r:(len(r[2].references),r[2].inliers))
        template=e.bank.templates[other]
        # The seed and one further original must independently establish the
        # same coordinate relation. A fully clipped neighbour is unknown.
        verified=[obs.source_id]
        # Originals may belong to a non-root member. Register the group's
        # fixed coordinates, not that observation's private fragment origin.
        seed_pose=obs.pose@np.linalg.inv(members[obs.template_id])
        relations=[np.linalg.inv(fit.matrix)@seed_pose]
        if trajectory is not None:
            # The independent measured path corroborates this coordinate
            # relation; it does not create another fragment observation.
            check['trajectory']=trajectory
            if fit.inliers==0:
                check['registration']='visible_mask_at_measured_boundary_trajectory'
        for frame in (obs.frame-1,obs.frame+1):
            if frame not in e.source_ids or e.source_ids[frame]==obs.source_id:continue
            roi=clip_box(bounds(fit.matrix,template.box),e.shape,padding=.15*footprint(obs.box))
            if roi[2]<=roi[0] or roi[3]<=roi[1]:continue
            key=e.matcher.extract(image_at(frame),e.source_ids[frame],roi)
            options=e._matches(template,key,max_locations=4)
            good=[m for m in options if m.inliers>=e.config.min_inliers and
                  len(independent_sources(e,m,e.source_ids[frame]))>=2]
            check.setdefault('probes',[]).append(dict(frame=frame,candidates=len(options),qualified=len(good)))
            if not good:continue
            partial=e._matches(e.bank.templates[tid],key,max_locations=4)
            expected=np.linalg.inv(fit.matrix)@seed_pose
            coordinates=[np.linalg.inv(whole.matrix)@m.matrix for whole in good for m in partial
                         if discrepancy(whole.matrix@expected,m.matrix,e.bank.templates[tid].box)
                         <=4*(m.uncertainty+whole.uncertainty)]
            if len(coordinates)!=1:continue
            verified.append(e.source_ids[frame])
            relations.append(coordinates[0])
        check['verified_sources']=verified
        if len(set(verified))<2 and trajectory is None:continue
        proposals.append((other,tid,fit.uncertainty,list(zip(verified,relations)),obs.id,trajectory))
    e.edge_identity_checks=checks
    changed=set()
    for other,tid,sigma,sources,oid,trajectory in proposals:
        for source,observed in sources:
            link=e.bank.propose_link(other,tid,observed,source,kind='verified_frame_clipped_identity',
                tolerance=4*sigma,trajectory_support=trajectory)
        if link.active:
            changed.update(e.bank.members(other))
            if trajectory is not None:
                fragment=e.bank.observations[oid].instance_id
                record=dict(fragment=fragment,instance=trajectory['instance'],
                    observation=oid,frame=e.bank.observations[oid].frame,
                    template=other,fragment_template=tid,evidence=trajectory,
                    registration=fit.matrix.tolist(),uncertainty=float(fit.uncertainty))
                if not any(r['fragment']==fragment and r['instance']==record['instance']
                           for r in e.edge_occurrences):
                    e.edge_occurrences.append(record)
            e.reconciler.events.append(dict(event='verified_frame_clipped_identity',a=other,b=tid,
                sources=[s for s,_ in sources],observation=oid))
    return changed
