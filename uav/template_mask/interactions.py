"""Reversible distinction between rigid fragments and independently moving marks.

SAM grouping proposes a shared observation. It cannot decide how many physical
objects generated it. Only original-image measured relative poses are evidence;
future measurements revise the same record instead of creating a mixed object.
"""
import numpy as np
from .geometry import discrepancy


def relative_motion(engine, a, b, frame):
    """Test one constant similarity transform on past AND future measurements."""
    rows = []
    sources = set()
    for f in sorted(set(a.poses) & set(b.poses)):
        pa, pb = a.poses[f], b.poses[f]
        if (pa.status != 'measured' or pb.status != 'measured' or
                not pa.presence or not pb.presence or pa.source_id is None or
                pa.source_id != pb.source_id or pa.source_id in sources or
                f in a.ambiguous_frames or f in b.ambiguous_frames):
            continue
        # Template-to-image poses must be backed by uncontested originals.
        if any(engine.bank.observations[r].ambiguous
               for p in (pa, pb) for r in p.references if r in engine.bank.observations):
            continue
        sources.add(pa.source_id)
        rows.append((f, pa, pb))
    result = dict(state='unresolved', sources=sorted(sources),
                  past_sources=sorted({pa.source_id for f, pa, _ in rows if f < frame}),
                  future_sources=sorted({pa.source_id for f, pa, _ in rows if f > frame}),
                  layout=None, incompatible_sources=[])
    if len(rows) < 4 or rows[-1][0]-rows[0][0] < engine.config.global_stride:
        return result
    box = engine.bank.templates[b.template_id].box
    # Choose a robust measured layout, then validate ALL originals against it.
    # Sampling only chooses the candidate medoid, never drops contrary evidence.
    choices = np.unique(np.linspace(0, len(rows)-1, min(17, len(rows))).astype(int))
    # Batch exactly the same four-corner discrepancies. Keep every original;
    # only the medoid candidates are sampled, as before.
    left = np.stack([pa.matrix for _, pa, _ in rows])
    right = np.stack([pb.matrix for _, _, pb in rows])
    layouts = np.linalg.inv(left) @ right
    sigma = np.array([pa.uncertainty+pb.uncertainty for _, pa, pb in rows])
    l, t, r, bottom = box
    corners = np.array([[l,t,1.], [r,t,1.], [l,bottom,1.], [r,bottom,1.]])
    observed = np.einsum('nij,kj->nki', right[:, :2], corners)
    def errors(layout):
        projected = np.einsum('nij,kj->nki', (left @ layout)[:, :2], corners)
        return np.max(np.linalg.norm(projected-observed, axis=2), axis=1)
    normalization = np.maximum(1e-6, 4*sigma)
    # Evaluate medoid candidates in one batch, preserving candidate order and
    # the exact matrix/corner operations. Avoid one NumPy dispatch per layout.
    projected = np.einsum('cnij,kj->cnki',
        (left[None] @ layouts[choices,None])[:,:,:2], corners)
    candidate_errors = np.max(np.linalg.norm(projected-observed[None],axis=3),axis=2)
    scores = np.median(candidate_errors/normalization[None],axis=1)
    layout = layouts[choices[int(np.argmin(scores))]]
    residual = errors(layout)/normalization
    result['layout'] = layout.tolist()
    result['incompatible_sources'] = [pa.source_id for (_, pa, _), r in zip(rows, residual) if r > 1]
    result['max_normalized_residual'] = float(residual.max())
    if np.all(residual <= 1):
        result['state'] = 'rigid_layout'
        return result
    # Require two corroborating originals on each side of a layout change.
    # One bad SIFT fit, box separation or a polyline prediction cannot establish
    # independent objects. A fixed-offset pair moving on any curve stays rigid.
    # Every split compares against the same first two originals. Compute those
    # distances once, then count contrary originals in each suffix.
    distant_counts = [np.cumsum((errors(reference) > 8*sigma)[::-1])[::-1]
                      for reference in layouts[:2]]
    first, second = rows[:2]
    early_agreement = discrepancy(second[1].matrix @ layouts[0], second[2].matrix, box) <= \
        8*(first[1].uncertainty+first[2].uncertainty+second[1].uncertainty+second[2].uncertainty)
    for split in range(2, len(rows)-1):
        for counts in distant_counts:
            if counts[split] < 2:
                break
        else:
            # Both early layouts must agree with each other; otherwise these
            # might merely be two unrelated fitting outliers.
            if early_agreement:
                result['state'] = 'independent_motion'
                return result
    return result


class InteractionResolver:
    def __init__(self, engine):
        self.engine = engine
        self.records = {}

    def hold(self, obs, pairs):
        e = self.engine
        owners = sorted({i.id for pair in pairs for i in pair})
        obs.crossing_instances = tuple(owners)
        obs.ambiguous, obs.reliable = True, False
        obs.association_reason = 'mixed_discovery_pending_trajectory'
        self.records[obs.id] = dict(frame=obs.frame, source=obs.source_id, observation=obs.id,
            instances=owners, pairs=[dict(a=a.id, b=b.id) for a, b in pairs],
            source_candidates=list(obs.source_candidates), state='unresolved',
            action='retain_local_pixels_without_template_update')
        self.resolve(obs.id)

    def resolve(self, observation_id=None):
        e = self.engine
        records = [self.records[observation_id]] if observation_id else list(self.records.values())
        for record in records:
            obs = e.bank.observations[record['observation']]
            states = []
            for pair in record['pairs']:
                a, b = e.instances[pair['a']], e.instances[pair['b']]
                evidence = relative_motion(e, a, b, record['frame'])
                pair['evidence'] = evidence
                states.append(evidence['state'])
                if evidence['state'] == 'rigid_layout':
                    self._bridge(obs, a, b, evidence)
                elif evidence['state'] == 'independent_motion':
                    self._revoke_contradiction(a, b)
            two_sided = all(len(p['evidence']['past_sources']) >= 2 and
                            len(p['evidence']['future_sources']) >= 2 for p in record['pairs'])
            record['state'] = ('crossing' if two_sided and all(s == 'independent_motion' for s in states) else
                               'rigid_fragments' if two_sided and all(s == 'rigid_layout' for s in states) else 'unresolved')
            # A rigid trajectory is supporting evidence; pixel fusion still
            # needs a whole original registering to both fragments, twice.
            record['action'] = ('retain_tracks_and_local_pixels' if record['state'] == 'crossing' else
                'validate_whole_observation_registration' if record['state'] == 'rigid_fragments' else
                'retain_both_hypotheses_and_local_pixels')

    def _bridge(self, obs, a, b, evidence):
        e = self.engine
        # Keep both measured paths alive through the first pass. Retiring a
        # fragment now would remove future evidence that can reject a brief
        # spell of coordinated motion.
        if e.stage == 'first_pass':
            return
        if len(evidence['past_sources']) < 2 or len(evidence['future_sources']) < 2:
            return
        if a.template_id == b.template_id:
            return
        ta, tb = e.bank.templates[a.template_id], e.bank.templates[b.template_id]
        if tb.id in e.bank.members(ta.id):
            return
        left = e._matches(ta, obs.feature_key)
        right = e._matches(tb, obs.feature_key)
        relative = np.asarray(evidence['layout'])
        from .global_identity import trajectory_agreement
        if not trajectory_agreement(e,a,b,relative,expanding=True)['accepted']:
            return
        for fa in left:
            if fa.coverage < .35 or not e._shape_agreement(ta, fa, obs):
                continue
            for fb in right:
                if fb.coverage < .35 or not e._shape_agreement(tb, fb, obs):
                    continue
                tolerance = 4*(fa.uncertainty+fb.uncertainty)
                if discrepancy(fa.matrix @ relative, fb.matrix, tb.box) > tolerance:
                    continue
                # A whole must also align to the stored trajectories at this
                # timestamp; text matches at other positions do not bridge it.
                pa, pb = a.poses.get(obs.frame), b.poses.get(obs.frame)
                if pa is None or pb is None:
                    continue
                if (discrepancy(fa.matrix, pa.matrix, ta.box) > 4*(fa.uncertainty+pa.uncertainty) or
                        discrepancy(fb.matrix, pb.matrix, tb.box) > 4*(fb.uncertainty+pb.uncertainty)):
                    continue
                e.bank.propose_link(ta.id, tb.id, np.linalg.inv(fa.matrix) @ fb.matrix,
                    obs.source_id, kind='trajectory_validated_bridge', tolerance=tolerance)
                return

    def _revoke_contradiction(self, a, b):
        e = self.engine
        # Revoke every coordinate path that asserts a fixed layout between this
        # pair. Preserve original transforms and measurements for reconsideration.
        members = e.bank.members(a.template_id)
        if b.template_id not in members or a.template_id == b.template_id:
            return
        for link in list(e.bank.links.values()):
            if link.active and link.a in members and link.b in members:
                e.bank.revoke(link.a, link.b, 'measured_relative_motion_contradiction')
