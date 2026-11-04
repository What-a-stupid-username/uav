"""One snapshot of post-refinement identity evidence, before any link is applied.

Original-image registration establishes coordinates. Measured trajectories
can reunite incomplete SAM fragments independently of their detected shape.
Shape agreement is a separate route for sharing templates across occurrences.
"""
import cv2
import numpy as np
from .geometry import bounds, discrepancy, translation, warp
from .gaps import measured, trajectory_bridge, path_continuation
from .presence import SAM_ABSENCE_REASONS
from .records import Pose
from types import SimpleNamespace


def occurrence_groups(engine,owners,members):
    """Group duplicate records only through consistent simultaneous originals.

    A remote recurrence never joins through shape or endpoint proximity. Each
    overlapping pair in a component must agree in registered coordinates.
    """
    cache=getattr(engine,'_association_occurrences',None)
    owners=sorted(owners,key=lambda i:i.id)
    key=(tuple(i.id for i in owners),tuple((tid,np.asarray(matrix).tobytes())
                                          for tid,matrix in sorted(members.items())))
    if cache is not None and key in cache:return cache[key]
    groups = [[i] for i in owners]
    measured_frames={i.id:{p.frame for p in measured(i)} for i in owners}
    compatible = {}
    for n,a in enumerate(owners):
        for b in owners[n+1:]:
            relation = np.linalg.inv(members[a.template_id])@members[b.template_id]
            compatible[a.id,b.id] = trajectory_agreement(engine,a,b,relation)['accepted']
    for (a,b),accepted in compatible.items():
        if not accepted:continue
        left=next(g for g in groups if any(i.id==a for i in g))
        right=next(g for g in groups if any(i.id==b for i in g))
        if left is right:continue
        conflict=False
        for x in left:
            for y in right:
                common=measured_frames[x.id]&measured_frames[y.id]
                if len(common)>=4 and not compatible.get(tuple(sorted((x.id,y.id))),False):
                    conflict=True
        if not conflict:left.extend(right);groups.remove(right)
    result=[]
    for group in groups:
        root=min(group,key=lambda i:i.id)
        poses={}
        for owner in group:
            transform=np.linalg.inv(members[owner.template_id])@members[root.template_id]
            for p in measured(owner):
                old=poses.get(p.frame)
                if old is None or p.uncertainty<old.uncertainty:
                    poses[p.frame]=Pose(p.frame,p.matrix@transform,p.uncertainty,p.status,
                        p.source_id,p.references,p.presence,p.reason)
        for owner in group:
            for p in owner.poses.values():
                if p.status=='absent' and p.reason not in SAM_ABSENCE_REASONS:
                    poses.setdefault(p.frame,p)
        result.append((SimpleNamespace(id=root.id,template_id=root.template_id,poses=poses,
            ambiguous_frames=set()),tuple(i.id for i in group)))
    if cache is not None:cache[key]=result
    return result


def trajectory_agreement(engine, a, b, b_to_a, sigma=0., *, expanding=False):
    rows = []
    seen = set()
    box = engine.bank.templates[b.template_id].box
    for frame in sorted(set(a.poses) & set(b.poses)):
        pa, pb = a.poses[frame], b.poses[frame]
        if (pa.status != 'measured' or pb.status != 'measured' or not pa.presence or not pb.presence
                or pa.source_id is None or pa.source_id != pb.source_id or pa.source_id in seen
                or frame in a.ambiguous_frames or frame in b.ambiguous_frames):
            continue
        seen.add(pa.source_id)
        error = discrepancy(pa.matrix @ b_to_a, pb.matrix, box)
        tolerance = 4*(pa.uncertainty+pb.uncertainty+sigma)
        rows.append((frame, pa.source_id, error/max(1e-6, tolerance)))
    result = dict(accepted=False, sources=[s for _,s,_ in rows],
                  frames=[f for f,_,_ in rows], normalized_errors=[v for _,_,v in rows])
    if len(rows) < 4 or rows[-1][0]-rows[0][0] < engine.config.global_stride:
        return result
    good = np.array([v <= 1 for _,_,v in rows])
    # Allow isolated registration noise, not a sustained independent path or
    # an intersection that only agrees briefly in the middle of two tracks.
    halves = np.array_split(good, 2)
    longest = current = 0
    for ok in good:
        current = 0 if ok else current+1
        longest = max(longest, current)
    result['accepted'] = bool(good.mean() >= .8 and all(h.mean() >= .75 for h in halves) and longest < 3)
    if expanding:
        # A short shared leg is useful for tracking but cannot establish that
        # two sets of strokes form one template. Demand distributed measured
        # support across three discovery intervals, checking all common poses.
        bins = {(f-rows[0][0])//engine.config.global_stride for f,_,v in rows if v <= 1}
        enough = len(rows) >= 8 and len(bins) >= 3 and rows[-1][0]-rows[0][0] >= 2*engine.config.global_stride
        if not enough:
            result.update(accepted=False, reason='brief_shared_motion_is_not_template_identity')
    return result


def occurrence_agreement(engine, a, b, b_to_a, sigma=0., *, expanding=False):
    """Overlap or an independently validated gap in registered coordinates."""
    result = dict(trajectory_agreement(engine,a,b,b_to_a,sigma,expanding=expanding), kind='overlap')
    left,right = measured(a),measured(b)
    if not left or not right:
        return result
    if right[-1].frame < left[0].frame:
        return occurrence_agreement(engine,b,a,np.linalg.inv(b_to_a),sigma,expanding=expanding)
    if left[-1].frame >= right[0].frame:
        return result
    start,end = left[-1].frame,right[0].frame
    if any(start < p.frame < end and p.status == 'absent'
           and p.reason not in SAM_ABSENCE_REASONS
           for owner in (a,b) for p in owner.poses.values()):
        return dict(result,kind='continuation',reason='confirmed_disappearance')
    # pb maps B to image; pb @ inverse(B-to-A) maps A to image.
    inverse = np.linalg.inv(b_to_a)
    converted = [Pose(p.frame,p.matrix@inverse,p.uncertainty+sigma,p.status,
        p.source_id,p.references,p.presence) for p in right]
    bridge = trajectory_bridge(left,converted,engine.bank.templates[a.template_id],range(start,end+1),
                               fit_cache=engine.motion_fits,sample_times=getattr(engine,'source_ids',None))
    path = path_continuation(left,converted,engine.bank.templates[a.template_id],
                             getattr(engine,'source_ids',None)) if bridge is None else None
    return dict(result,accepted=bridge is not None or path is not None,kind='continuation',
        left_instance=a.id,right_instance=b.id,first_frame=start,last_frame=end,
        evidence=path if bridge is None else bridge[1])


def shape_agreement(engine, a, b, b_to_a, sigma):
    fa, fb = engine.bank.output_fusion(a)[0], engine.bank.output_fusion(b)[0]
    ah, aw = fa['mean'].shape; bh, bw = fb['mean'].shape
    ba = (*fa['origin'], *(fa['origin']+[aw,ah]))
    bb = bounds(b_to_a, (*fb['origin'], *(fb['origin']+[bw,bh])))
    low = np.floor(np.minimum(ba[:2],bb[:2])).astype(int)-3
    high = np.ceil(np.maximum(ba[2:],bb[2:])).astype(int)+3
    shape = tuple((high-low)[::-1])
    if min(shape) <= 0 or np.prod(shape) > engine.config.max_canvas_pixels:
        return dict(kind='unresolved', reason='comparison_canvas_unavailable')
    ma = warp(fa['mask'],
              translation(*(-low)) @ translation(*fa['origin']), shape, nearest=True) > 0
    mb = warp(fb['mask'],
              translation(*(-low)) @ b_to_a @ translation(*fb['origin']), shape, nearest=True) > 0
    radius = max(1,min(3,int(np.ceil(sigma))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
    da = cv2.dilate(ma.astype(np.uint8),kernel)>0
    db = cv2.dilate(mb.astype(np.uint8),kernel)>0
    ab = float((ma & db).sum()/max(1,ma.sum()))
    ba = float((mb & da).sum()/max(1,mb.sum()))
    kind = ('equivalent' if min(ab,ba) >= .85 else
            'a_contains_b' if ba >= .8 else 'b_contains_a' if ab >= .8 else 'unresolved')
    return dict(kind=kind, a_explained=ab, b_explained=ba,
                a_pixels=int(ma.sum()), b_pixels=int(mb.sum()))


def registration_views(engine, source):
    """Prepare a source group once within an immutable association review."""
    e = engine
    cache = getattr(e, '_association_views', None)
    if cache is not None and source in cache:
        return cache[source]
    members = e.bank.members(source)
    views = e.bank.selected_views(source)
    observations = [(o,m) for jobs in views.values() for o,m in jobs]
    if not observations:
        observations = [(e.bank.observations[r.observation_id],
                         members[e.bank.observations[r.observation_id].template_id])
                        for r in e.bank.matching_references(source)]
    result = [(obs, np.linalg.inv(transform)) for obs, transform in observations
              if not obs.crossing_instances and obs.pose is not None
              and obs.reliable and not obs.ambiguous]
    if cache is not None:
        cache[source] = result
    return result


def registrations(engine, a, b):
    """Query both directions, counting distinct originals in matched references."""
    e = engine
    proposals = []
    for target, source, reverse in ((a,b,False),(b,a,True)):
        for obs, inverse_transform in registration_views(e, source):
            for fit in e._matches(e.bank.templates[target], obs.feature_key, max_locations=4):
                references = [e.bank.observations[r] for r in fit.references
                    if r in e.bank.observations and e.bank.observations[r].source_id != obs.source_id
                    and e.bank.observations[r].reliable and not e.bank.observations[r].ambiguous]
                if not references or fit.inliers < e.config.min_inliers or fit.coverage < .08:
                    continue
                matrix = np.linalg.inv(fit.matrix) @ obs.pose @ inverse_transform
                if reverse:
                    matrix = np.linalg.inv(matrix)
                proposals.append(dict(matrix=matrix, sigma=fit.uncertainty,
                    inliers=fit.inliers,references=tuple(r.id for r in references),
                    sources={obs.source_id, *(r.source_id for r in references)}, observation=obs.id))
    return proposals


def evaluate_pair(engine, a, b):
    e = engine
    if any(getattr(e.bank.templates[t],'composite',False) for t in (a,b)):
        return dict(a=a,b=b,accepted=False,reason='composite_template_is_not_shareable')
    proposals = registrations(e,a,b)
    row = dict(a=a,b=b,accepted=False,reason='no_independent_registration')
    if not proposals:
        return row
    # A qualified geometric alternative must have more than one original.
    # Compare clusters before choosing; no mutation depends on pair order.
    clusters = []
    for proposal in proposals:
        group = next((g for g in clusters if discrepancy(g[0]['matrix'],proposal['matrix'],e.bank.templates[b].box)
            <= 4*(g[0]['sigma']+proposal['sigma'])),None)
        if group is None:
            clusters.append([proposal])
        else:
            group.append(proposal)
    alternatives = []
    for group in clusters:
        sources = sorted(set().union(*(p['sources'] for p in group)))
        if len(sources) < 2:
            continue
        best = min(group,key=lambda p:sum(discrepancy(p['matrix'],q['matrix'],e.bank.templates[b].box) for q in group))
        matrix,sigma = best['matrix'],best['sigma']
        shape = shape_agreement(e,a,b,matrix,sigma)
        candidate = dict(transform=matrix.tolist(),sigma=float(sigma),shape=shape,sources=sources,
            observations=sorted({p['observation'] for p in group}),accepted=False,
            reason='different_or_unexplained_strokes')
        candidate.update(occurrence_evidence(e,a,b,matrix,sigma,shape))
        # Several retained references voting for ONE query still describe one
        # visible fragment. Non-equivalent shapes need registrations obtained
        # from at least two distinct query originals before sharing pixels.
        query_sources = {e.bank.observations[p['observation']].source_id for p in group}
        candidate['registration_query_sources'] = sorted(query_sources)
        if candidate['accepted'] and (shape['kind'] != 'equivalent'
                or candidate['reason']=='registered_equivalent_occurrence') and len(query_sources) < 2:
            candidate.update(accepted=False,reason='fragment_needs_two_registered_query_originals')
        alternatives.append(candidate)
    valid = [c for c in alternatives if c['accepted']]
    # Repeated motifs can fit several rotations geometrically. A unique
    # original-measurement trajectory selects coordinates more strongly than
    # shape equivalence at an unrelated pose. Two trajectory explanations
    # remain ambiguous.
    located = [c for c in valid if c['reason']=='registered_occurrence_trajectory']
    if len(located)==1:
        valid=located
    if len(valid) == 1:
        return dict(row,**valid[0],alternatives=alternatives)
    return dict(row,reason='competing_global_explanations' if valid else
                'no_supported_global_explanation',alternatives=alternatives)


def occurrence_evidence(e,a,b,matrix,sigma,shape,*,groups=None):
    members_a,members_b = (e.bank.members(a),e.bank.members(b)) if groups is None else groups
    owners_a = [i for i in e.instances.values() if i.template_id in members_a]
    owners_b = [i for i in e.instances.values() if i.template_id in members_b]
    groups_a=occurrence_groups(e,owners_a,members_a)
    groups_b=occurrence_groups(e,owners_b,members_b)
    alignments = []
    for ia,ids_a in groups_a:
        left = np.linalg.inv(members_a[ia.template_id]) @ matrix
        for ib,ids_b in groups_b:
            relation = left @ members_b[ib.template_id]
            result = occurrence_agreement(e,ia,ib,relation,sigma,
                                          expanding=shape['kind'] != 'equivalent')
            # Keep original owner IDs for revocation and audit; one decision is
            # based on the combined real measurements of a proven occurrence.
            for aid in ids_a:
                for bid in ids_b:
                    local=np.linalg.inv(members_a[e.instances[aid].template_id])@matrix@members_b[e.instances[bid].template_id]
                    alignments.append(dict(a=aid,b=bid,transform=local.tolist(),
                        occurrence_a=ia.id,occurrence_b=ib.id,**result))
    # First ask whether an entire hypothesis is a uniquely registered piece of
    # an existing occurrence. SAM shape is not involved in this decision. One
    # side may already contain many independent instances of the shared mark.
    supported = []
    # Containment is directional: every occurrence of the SMALLER hypothesis
    # needs a corresponding complete occurrence. One complete occurrence that
    # matches one of many partial ones cannot lend its extra strokes to all.
    sides = (('b',owners_b),) if shape['kind']=='a_contains_b' else (
            (('a',owners_a),) if shape['kind']=='b_contains_a' else
            (('a',owners_a),('b',owners_b)))
    for side, owners in sides:
        other='b' if side=='a' else 'a'
        if owners and all(len({r['occurrence_'+other] for r in alignments
                               if r['accepted'] and r[side]==i.id})==1 for i in owners):
            supported.append([(side,i.id) for i in owners])
    result = dict(trajectories=alignments,required_occurrences=[],accepted=False)
    if supported:
        required = min(supported,key=len)
        return dict(result,required_occurrences=required,accepted=True,
                    reason='registered_occurrence_trajectory')
    if shape['kind']=='equivalent':
        # Equivalent complete originals do not need a long shared leg to
        # resolve a previously mixed hypothesis. Four simultaneous originals
        # validate occurrence coordinates; distributed image queries are still
        # checked separately by evaluate_pair below.
        def located(row):
            errors=np.asarray(row['normalized_errors'])
            return row['kind']=='overlap' and len(errors)>=4 and np.all(errors<=1)
        for side,owners in sides:
            other='b' if side=='a' else 'a'
            other_groups=groups_b if other=='b' else groups_a
            grouped={owner.id:owner for owner,_ in other_groups}
            def supported_owner(owner):
                matches=[r for r in alignments if located(r) and r[side]==owner.id]
                ids={r['occurrence_'+other] for r in matches}
                if not ids:return False
                # The query can bridge two disjoint fragments of the same
                # path. Multiple *simultaneous* incompatible locations remain
                # competing identities, but adjacent records are not a veto.
                for aid in ids:
                    for bid in ids:
                        if aid>=bid:continue
                        left,right=grouped[aid],grouped[bid]
                        common=set(left.poses)&set(right.poses)
                        if len(common)>=4:return False
                return True
            if owners and all(supported_owner(i) for i in owners):
                return dict(result,required_occurrences=[(side,i.id) for i in owners],
                            accepted=True,reason='registered_equivalent_occurrence')
    # Separate occurrences can share equivalent complete shapes, but a shared
    # word or partial line at another location is insufficient.
    if (shape['kind']=='equivalent' and not e.bank.templates[a].mixed
            and not e.bank.templates[b].mixed):
        return dict(result,accepted=True,reason='registered_equivalent_templates')
    return dict(result,reason='fragment_identity_needs_unique_shared_trajectory')
