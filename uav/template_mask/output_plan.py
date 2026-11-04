"""Final canonical ownership table; provenance records remain untouched."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OutputPlan:
    instances: tuple
    template_groups: tuple
    frame_owners: dict
    occurrences: tuple

    def report(self):
        return dict(instances=list(self.instances),
                    template_groups=[list(g) for g in self.template_groups],
                    frame_owners={str(f):list(ids) for f,ids in sorted(self.frame_owners.items())},
                    occurrences=[dict(id=min(g),members=list(g)) for g in self.occurrences],
                    occurrence_count=len(self.occurrences))


def build_output_plan(engine):
    e = engine
    from .global_identity import occurrence_agreement
    from .gaps import measured
    observed_frames = {i:set() for i in e.instances}
    for obs in e.bank.observations.values():
        if obs.instance_id in observed_frames and e.usable_output_observation(obs):
            observed_frames[obs.instance_id].add(obs.frame)
    owners = tuple(i.id for i in e.instances.values()
        if not e.bank.templates[i.template_id].composite
        and (observed_frames[i.id] or any(p.status!='absent' for p in i.poses.values())))
    groups = sorted({tuple(sorted(e.bank.members(e.instances[i].template_id))) for i in owners})
    transforms = {g[0]:e.bank.members(g[0]) for g in groups}
    boundary_pairs={tuple(sorted((r['fragment'],r['instance'])))
                    for r in getattr(e,'validated_edge_occurrences',())}
    occurrences = []
    for group in groups:
        members = transforms[group[0]]
        candidates = [e.instances[i] for i in owners if e.instances[i].template_id in members]
        # Cache pair decisions. A merged occurrence cannot acquire a conflicting
        # simultaneous path through transitive aliases.
        compatible = {}
        measurements = {i.id:measured(i) for i in candidates}
        measured_frames = {i:set(p.frame for p in rows) for i,rows in measurements.items()}
        for n,a in enumerate(candidates):
            for b in candidates[n+1:]:
                continuation = a.continuation_of==b.id or b.continuation_of==a.id
                boundary = tuple(sorted((a.id,b.id))) in boundary_pairs
                if not continuation and not boundary and len(measured_frames[a.id]&measured_frames[b.id])<4:
                    compatible[a.id,b.id] = False
                    continue
                relation = np.linalg.inv(members[a.template_id])@members[b.template_id]
                evidence = occurrence_agreement(e,a,b,relation)
                # A template may recur at another time. Only an explicit gap
                # resolution can turn two non-overlapping lifetimes into one
                # output occurrence; shared shape alone never does that.
                compatible[a.id,b.id] = evidence['accepted'] and (
                    evidence['kind']=='overlap' or continuation)
                if boundary:
                    # Explicit image+trajectory boundary verification can join
                    # a single clipped observation to its complete occurrence.
                    # Newly measured contrary layout still vetoes the relation.
                    common=measured_frames[a.id]&measured_frames[b.id]
                    from .geometry import discrepancy
                    compatible[a.id,b.id] = all(discrepancy(a.poses[f].matrix@relation,
                        b.poses[f].matrix,e.bank.templates[b.template_id].box)
                        <=4*(a.poses[f].uncertainty+b.poses[f].uncertainty) for f in common)
        components = [[i.id] for i in candidates]
        for a,b in sorted((a,b) for (a,b),ok in compatible.items() if ok):
            left=next(c for c in components if a in c);right=next(c for c in components if b in c)
            if left is right:continue
            # Non-overlapping fragments may be linked through a complete path.
            # Simultaneous contradictory measurements forbid that union.
            conflict=False
            for x in left:
                for y in right:
                    common=measured_frames[x]&measured_frames[y]
                    if len(common)>=4 and not compatible.get(tuple(sorted((x,y))),False):conflict=True
            if not conflict:
                left.extend(right);components.remove(right)
        occurrences.extend(tuple(sorted(c)) for c in components)
    frame_owners = {}
    for occurrence in occurrences:
        frames=sorted({f for i in occurrence for f in e.instances[i].poses}
                      |{f for i in occurrence for f in e.presence.coverage.get(i,set())}
                      |{f for i in occurrence for f in observed_frames[i]})
        for frame in frames:
            # The validated occurrence, rather than its archived original
            # owner, controls duplicate suppression. Otherwise a missed match
            # in every alias leaves a hole after the old owner disappears.
            def eligible(i):
                return ((i.render_start is None or frame>=i.render_start)
                        and (i.render_end is None or frame<=i.render_end))
            choices=[e.instances[i] for i in occurrence if eligible(e.instances[i])
                     and (e.presence.output_margin(e.instances[i],frame)
                          or frame in observed_frames[i]
                          or frame in e.instances[i].poses and e.instances[i].poses[frame].status!='absent')]
            if not choices:continue
            def quality(i):
                pose=i.poses.get(frame)
                observed=frame in observed_frames[i.id]
                if pose is None:return (not observed,not observed,float('inf'),0,i.id)
                return (not observed and pose.status!='measured',not observed and not pose.presence,pose.uncertainty,
                        -len(pose.references),i.id)
            frame_owners.setdefault(frame,[]).append(min(choices,key=quality).id)
    return OutputPlan(owners,tuple(groups),{f:tuple(ids) for f,ids in frame_owners.items()},
                      tuple(occurrences))
