"""Provisional template identity lifecycle and retained-evidence reconciliation.

Cases retain unresolved identities. The bank owns original observations, fixed coordinates,
progressively updated references and masks. A case never supplies identity.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PendingCase:
    id: int
    template_id: int
    instance_id: int
    observations: list[str] = field(default_factory=list)
    checked_sources: set[int] = field(default_factory=set)
    closed: bool = False

class PendingResolver:
    def __init__(self, engine):
        self.engine = engine
        self.cases = {}
        self.membership = {}
        self.events = []
        self.retried = {}

    def defer(self, obs, reason):
        e = self.engine
        obs.association_reason = reason
        if obs.instance_id is not None:
            return e.instances[obs.instance_id]
        # Exact duplicate outputs of the same original are not a new identity.
        # No cross-frame association is inferred from proximity/SAM numbering.
        for oid in e.frame_observations[obs.frame]:
            other = e.bank.observations[oid]
            if other.id == obs.id or other.instance_id is None or other.source_id != obs.source_id:
                continue
            if other.box != obs.box or other.origin != obs.origin:
                continue
            a, b = e.cache.get(other.probability_key), e.cache.get(obs.probability_key)
            if not (np.array_equal(a['probability'], b['probability']) and np.array_equal(a['valid'], b['valid'])):
                continue
            owner = e.instances[other.instance_id]
            obs.pose, obs.template_id, obs.reliable = other.pose.copy(), owner.template_id, True
            e._assign(owner, obs)
            return owner
        owner = e._provisional(obs, reason)
        case = PendingCase(len(self.cases), owner.template_id, owner.id, [obs.id], {obs.source_id})
        self.cases[case.id] = case
        self.membership[obs.id] = case.id
        self.events.append(dict(event='provisional_created', case=case.id, observation=obs.id,
            frame=obs.frame, template=owner.template_id, instance=owner.id, reason=reason))
        return owner

    def resolved(self, obs):
        # Evidence accumulation in its own template is not identity resolution.
        if obs.template_id is None:
            return
        case = next((c for c in self.cases.values() if c.template_id == obs.template_id and not c.closed), None)
        if case is not None:
            if obs.id not in case.observations:
                case.observations.append(obs.id)
            case.checked_sources.add(obs.source_id)
            self.membership[obs.id] = case.id

    def retry(self):
        e = self.engine
        signature = (e.bank.revision, e.motion_revision)
        for case in list(self.cases.values()):
            if self.retried.get(case.id) == signature:
                continue
            self.retried[case.id] = signature
            e.reconcile_template(case.template_id)
            members = e.bank.members(case.template_id)
            if len(members) > 1 and not case.closed:
                case.closed = True
                for oid in case.observations:
                    self.membership.pop(oid, None)
                e.stats['pending_resolved'] += len(case.observations)
                self.events.append(dict(event='template_linked', case=case.id,
                    template=case.template_id, members=sorted(members)))
        for case in self.cases.values():
            if case.closed and len(e.bank.members(case.template_id)) == 1:
                case.closed = False
                for oid in case.observations:
                    self.membership[oid] = case.id
                self.events.append(dict(event='template_reopened', case=case.id, template=case.template_id))
        e.reconciler.refresh_aliases()

    def report(self):
        e = self.engine
        unresolved = [o for o in self.membership if e.bank.templates[e.bank.observations[o].template_id].provisional]
        return dict(unresolved=len(unresolved), identity_investigations=len(self.membership), events=self.events,
            cases=[dict(id=c.id, template=c.template_id, instance=c.instance_id, closed=c.closed,
                status='linked' if c.closed else ('provisional' if e.bank.templates[c.template_id].provisional else 'established_independent'),
                checked_sources=sorted(c.checked_sources), observations=c.observations,
                unresolved=[] if c.closed or not e.bank.templates[c.template_id].provisional else c.observations) for c in self.cases.values()],
            observations=[dict(id=o.id, frame=o.frame, source=o.source_id, instance=o.instance_id,
                template=o.template_id, reason=o.association_reason, trajectory=o.trajectory_evidence,
                ownership_confirmed=o.ownership_confirmed, template_update_eligible=o.reliable)
                for o in e.bank.observations.values() if o.association_reason])
