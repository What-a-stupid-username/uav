"""Progressive, reversible registration of already existing hypotheses."""
import numpy as np
from .geometry import discrepancy, discrepancies, translation
from .gaps import measured, trajectory_bridge
from .records import Pose
from .presence import SAM_ABSENCE_REASONS


class TemplateReconciler:
    def __init__(self, engine):
        self.engine = engine
        self.events = []
        self.signatures = {}
        self.global_reviews = []
        self.resolved_mixed = set()
        self.global_relations = []

    def query_references(self, template):
        """Bound live identity queries; retain full matching and Stage 2 evidence."""
        refs = template.references
        if self.engine.stage != 'first_pass':
            return refs
        observations = self.engine.bank.observations
        eligible = [r for r in refs if observations[r.observation_id].reliable
                    and not observations[r.observation_id].ambiguous]
        if len(eligible) <= 2:
            return eligible
        ordered = sorted(eligible, key=lambda r:r.source_id)
        middle = (len(ordered)+1)//2
        selected = {max(group,key=lambda r:r.quality).observation_id
                    for group in (ordered[:middle],ordered[middle:])}
        return [r for r in refs if r.observation_id in selected]

    def global_review(self):
        """Evaluate one immutable evidence snapshot, then apply reversible links."""
        from .association import evaluate_associations
        e = self.engine
        proposals = evaluate_associations(e)
        changed = set()
        for row in proposals:
            if not row['accepted']:
                continue
            a,b = row['a'],row['b']
            matrix = np.asarray(row['transform'])
            connected = e.bank.members(a)
            if b in connected:
                # The immutable review may accept every pair in a component.
                # Keep all decisions for audit, but maintain only the edges
                # needed to register coordinates. Redundant cycles otherwise
                # multiply graph traversal and occurrence revalidation work.
                if discrepancy(connected[b],matrix,e.bank.templates[b].box)>4*row['sigma']:
                    row.update(accepted=False,reason='inconsistent_coordinate_cycle')
                else:
                    row['coordinate_application']='already_connected'
                continue
            previous = e.bank.links.get((a,b))
            was_active = previous is not None and previous.active
            link = None
            for source in row['sources']:
                link = e.bank.propose_link(a,b,matrix,source,kind='global_refined_identity',
                                          tolerance=4*row['sigma'])
            if link is None or not link.active:
                row.update(accepted=False,reason='coordinate_link_rejected')
                continue
            if not was_active:
                changed.update(e.bank.members(a))
            self.global_relations = [r for r in self.global_relations
                                     if (r['a'],r['b'])!=(a,b)]
            self.global_relations.append(row)
            for tid in e.bank.members(a):
                template = e.bank.templates[tid]
                if template.mixed:
                    self.resolved_mixed.add(tid)
                    template.mixed = False
                    template.version += 1
                    e.bank.revision += 1
            if not was_active:
                e.stats['template_reconciliations'] += 1
                self.events.append(dict(event='global_refined_templates',a=a,b=b,
                    sources=row['sources'],shape=row['shape'],trajectories=row['trajectories']))
        self.global_reviews.extend(proposals)
        self.refresh_aliases()
        return changed

    def reconcile(self, tid):
        e = self.engine
        template = e.bank.templates[tid]
        if template.mixed or getattr(template,'composite',False):
            return
        signature = e.bank.revision
        if self.signatures.get(tid) == signature:
            return
        self.signatures[tid] = signature
        own_members = e.bank.members(tid)
        # Use diverse retained originals; repeated SAM crops from one source do
        # not validate coordinate layout or cast additional identity votes.
        refs = self.query_references(template)
        for reference in refs:
            obs = e.bank.observations[reference.observation_id]
            if obs.ambiguous or not obs.reliable:
                continue
            candidates = []
            queried = set(own_members)
            e._prepare_matches([other for other in e.bank.templates.values()
                if other.id not in own_members and not other.mixed and not other.composite],
                obs.feature_key)
            for other in e.bank.templates.values():
                if other.id in queried or other.mixed or getattr(other,'composite',False):
                    continue
                queried.update(e.bank.members(other.id))
                for fit in e._matches(other, obs.feature_key):
                    independent = [r for r in fit.references
                        if e.bank.observations[r].source_id != obs.source_id]
                    if not independent or fit.coverage < .08:
                        continue
                    shape = e._shape_evidence(other, fit, obs)
                    if not e._accept_shape_evidence(fit, shape):
                        continue
                    candidates.append((other, fit, shape))
                    break
            # Preserve alternatives rather than collapsing two complete marks
            # because they happen to contain the same fragment.
            groups = {min(e.bank.members(t.id)) for t, _, _ in candidates}
            if len(groups) != 1:
                continue
            other, fit, shape = candidates[0]
            equivalent = shape is not None and min(shape[:2]) >= .85 and fit.coverage >= .5
            transform = np.linalg.inv(fit.matrix) @ obs.pose @ np.linalg.inv(own_members[obs.template_id])
            if equivalent:
                # A reference itself may show only the shared line. Compare
                # both complete accumulated hypotheses before activating it.
                from .global_identity import shape_agreement
                equivalent = shape_agreement(e,other.id,tid,transform,
                                             fit.uncertainty)['kind'] == 'equivalent'
            key = tuple(sorted((tid, other.id)))
            was_active = key in e.bank.links and e.bank.links[key].active
            link = e.bank.propose_link(other.id, tid, transform, obs.source_id,
                kind='registered_hypothesis', tolerance=4*fit.uncertainty,
                activate=equivalent)
            if link is not None and link.active and not was_active:
                e.stats['template_reconciliations'] += 1
                self.events.append(dict(event='registered_templates', a=other.id, b=tid,
                    sources=sorted(link.evidence), b_to_a=transform.tolist()))
        self.refresh_aliases()

    def refresh_aliases(self):
        e = self.engine
        from .global_identity import occurrence_agreement, trajectory_agreement, occurrence_evidence
        # Pair scans revisit the same roots thousands of times. Coordinate
        # relations only depend on the bank; a revocation invalidates this
        # snapshot immediately. Never cache instance motion or alias decisions.
        member_cache = {}
        member_revision = e.bank.revision
        def members_at(tid):
            nonlocal member_revision
            if member_revision != e.bank.revision:
                member_cache.clear()
                member_revision = e.bank.revision
            if tid not in member_cache:
                member_cache[tid] = e.bank.members(tid)
            return member_cache[tid]
        for relation in self.global_relations:
            link = e.bank.links.get((relation['a'],relation['b']))
            if link is None or not link.active:
                continue
            # Equivalent complete shapes can describe independent instances.
            # Fragment/mixed links additionally assert the same occurrence;
            # sustained contrary original motion revokes that assertion.
            if relation.get('required_occurrences'):
                # Validate with the same occurrence evidence used to accept the
                # relation. Reverting to individual fragments here used to
                # immediately revoke a sound link supported by their union.
                groups = relation.get('occurrence_groups')
                if groups is not None:
                    groups=tuple({t:np.asarray(m) for t,m in g.items()} for g in groups)
                evidence = occurrence_evidence(e,relation['a'],relation['b'],
                    np.asarray(relation['transform']),relation['sigma'],relation['shape'],groups=groups)
                if not evidence['accepted']:
                    # Losing sufficient support is not contrary evidence. A
                    # continuation may become too sparse to fit after outlier
                    # review. Revoke only a sustained measured layout conflict
                    # with every alternative for a required occurrence.
                    def contradicted(row):
                        errors=np.asarray(row['normalized_errors'])
                        run=0
                        for bad in errors>1:
                            run=run+1 if bad else 0
                            if run>=3:return True
                        return False
                    conflicts=[]
                    for side,iid in relation['required_occurrences']:
                        choices=[r for r in evidence['trajectories'] if r[side]==iid]
                        conflicts.append(bool(choices) and all(contradicted(r) for r in choices))
                    if any(conflicts):
                        e.bank.revoke(link.a,link.b,'global_occurrence_layout_contradiction')
        for tid in self.resolved_mixed:
            active = any(l.active and l.kind == 'global_refined_identity'
                         and (l.a in members_at(tid) and l.b in members_at(tid))
                         for l in e.bank.links.values())
            if not active and not e.bank.templates[tid].mixed:
                e.bank.templates[tid].mixed = True
                e.bank.templates[tid].version += 1
                e.bank.revision += 1
        for source in sorted(e.instances.values(), key=lambda i: i.id):
            # Never copy measurements into another owner. Revocation must not
            # leave derived poses that can keep certifying the old association.
            source.alias_of = None
            source.alias_templates = None
            template = e.bank.templates[source.template_id]
            if template.investigation_reason is None or template.mixed:
                continue
            candidates = []
            for target in e.instances.values():
                if target.id >= source.id or target.template_id == source.template_id:
                    continue
                members = members_at(target.template_id)
                transform = members.get(source.template_id)
                if transform is None:
                    continue
                pairs = []
                for frame in sorted(set(source.poses)&set(target.poses)):
                    a, b = source.poses[frame], target.poses[frame]
                    if a.status != 'measured' or b.status != 'measured' or not a.presence or not b.presence:
                        continue
                    pairs.append((a,b))
                rows = []
                if pairs:
                    # Compare the same historical measurements in one NumPy
                    # batch instead of rebuilding four corners per timestamp.
                    errors = discrepancies(
                        np.asarray([b.matrix for a,b in pairs]) @ transform,
                        np.asarray([a.matrix for a,b in pairs]), template.box)
                    rows = [(a.source_id, error <= 4*(a.uncertainty+b.uncertainty))
                            for (a,b),error in zip(pairs,errors)]
                global_link = any(l.active and l.kind == 'global_refined_identity'
                                  and l.a in members and l.b in members for l in e.bank.links.values())
                if global_link:
                    sigma = max((r['sigma'] for r in self.global_relations
                        if r['a'] in members and r['b'] in members),default=0.)
                    agrees = trajectory_agreement(e,target,source,transform,sigma)['accepted']
                else:
                    agrees = len({s for s, ok in rows if ok}) >= 2 and all(ok for _, ok in rows)
                if agrees:
                    candidates.append((target, transform))
            # Several agreeing targets may already be aliases of one occurrence.
            # Count canonical occurrences, not each preserved provenance record.
            canonical = {}
            for target, transform in candidates:
                root = target
                while root.alias_of is not None:
                    root = e.instances[root.alias_of]
                root_transform = members_at(root.template_id).get(source.template_id)
                if root_transform is not None:
                    canonical[root.id] = (root, root_transform)
            if len(canonical) != 1:
                continue
            target, transform = next(iter(canonical.values()))
            source.alias_of = target.id
            source.alias_templates = (target.template_id, source.template_id)

    def trajectory_identities(self):
        """Strong continuation can support template identity without fake pixels.

        Compare observable screen anchors, fitting each side independently. This
        records a reversible identity link. Pixel fusion still requires a known
        coordinate transform; a template ID is not a coordinate measurement.
        """
        e = self.engine
        # Re-evaluate old associations as measurements change. Contrary motion
        # or confirmed disappearance revokes identity without deleting data.
        for link in e.bank.identity_links.values():
            if not link.active:
                continue
            a = e.instances[link.evidence['left_instance']]
            b = e.instances[link.evidence['right_instance']]
            if self._identity_bridge(a, b) is None:
                e.bank.revoke_identity(link.a, link.b, 'trajectory_identity_contradiction')
                b.continuation_of = None
        options = []
        for a in e.active_instances():
            left = measured(a)
            ta = e.bank.templates[a.template_id]
            if len(left) < 8 or ta.mixed:
                continue
            for b in e.active_instances():
                tb = e.bank.templates[b.template_id]
                right = measured(b)
                if (a.id == b.id or a.template_id == b.template_id or len(right) < 8
                        or tb.mixed or left[-1].frame >= right[0].frame):
                    continue
                if not self._trajectory_compatible(a, b):
                    continue
                if any(c.id not in (a.id, b.id)
                       and c.template_id in e.bank.identity_members(a.template_id) | e.bank.identity_members(b.template_id)
                       and any(left[-1].frame <= p.frame <= right[0].frame for p in measured(c))
                       for c in e.active_instances()):
                    continue
                bridge = self._identity_bridge(a,b)
                if bridge is not None:
                    options.append((a, b, bridge[1]))
        for a, b, evidence in options:
            if sum(x.id == a.id for x, _, _ in options) != 1 or sum(y.id == b.id for _, y, _ in options) != 1:
                continue
            evidence = dict(evidence, kind='two_sided_trajectory',
                            left_instance=a.id, right_instance=b.id,
                            first_frame=measured(a)[-1].frame, last_frame=measured(b)[0].frame,
                            coordinate_registration='unresolved')
            link = e.bank.associate_identity(a.template_id, b.template_id, evidence)
            if link.active:
                b.continuation_of = a.continuation_of if a.continuation_of is not None else a.id

    def _trajectory_compatible(self, a, b):
        """Presence constraints, independent of incomplete SAM shapes."""
        e = self.engine
        left, right = measured(a), measured(b)
        if not left or not right or left[-1].frame >= right[0].frame:
            return False
        start, end = left[-1].frame, right[0].frame
        if any(start < p.frame < end and p.status == 'absent'
                   and p.reason not in SAM_ABSENCE_REASONS
               for owner in (a, b) for p in owner.poses.values()):
            return False
        ta, tb = e.bank.templates[a.template_id], e.bank.templates[b.template_id]
        if ta.mixed or tb.mixed:
            return False
        return True

    def _identity_bridge(self,a,b):
        if not self._trajectory_compatible(a,b):
            return None
        e = self.engine
        ta,tb = e.bank.templates[a.template_id],e.bank.templates[b.template_id]
        left,right = measured(a),measured(b)
        # Screen-anchor continuity supplies identity evidence only. No layout
        # transform or shared pixels are invented from bounding-box centers.
        lrows = [Pose(p.frame,p.matrix@translation(*ta.anchor),p.uncertainty,
            p.status,p.source_id,p.references,p.presence) for p in left]
        rrows = [Pose(p.frame,p.matrix@translation(*tb.anchor),p.uncertainty,
            p.status,p.source_id,p.references,p.presence) for p in right]
        from dataclasses import replace
        anchor_template = replace(ta,anchor=(0.,0.),
            box=tuple(np.asarray(ta.box)-np.tile(ta.anchor,2)))
        return trajectory_bridge(lrows,rrows,anchor_template,
            range(left[-1].frame,right[0].frame+1),fit_cache=e.motion_fits,
            sample_times=e.source_ids)
