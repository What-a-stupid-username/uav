"""Non-destructive template evidence and reversible coordinate associations."""
from collections import defaultdict, deque
import numpy as np
from .geometry import bounds, discrepancy, support_bounds, translation, warp
from .records import Template, Reference, Link, IdentityLink


class TemplateBank:
    def __init__(self, cache, config):
        self.cache, self.config = cache, config
        self.templates = {}
        self.observations = {}
        self.links = {}
        self.identity_links = {}
        self.revision = 0

    def create(self, observation):
        tid = len(self.templates)
        l, t, r, b = observation.box
        data = self.cache.get(observation.probability_key)
        support = support_bounds((data['probability'] > self.config.pixel_threshold) & data['valid'])
        if support is not None:
            x, y = observation.origin
            box = (x+support[0]-l, y+support[1]-t,
                   x+support[2]-l, y+support[3]-t)
        else:
            box = (0., 0., r-l, b-t)
        template = Template(tid, box, ((r-l)/2, (b-t)/2))
        self.templates[tid] = template
        observation.template_id = tid
        observation.pose = translation(l, t)
        observation.reliable = True
        self.add(observation)
        return template

    def add(self, observation):
        self.observations[observation.id] = observation
        if not observation.reliable or observation.ambiguous:
            return False
        template = self.templates[observation.template_id]
        if observation.id in template.observations:
            return False
        template.observations.append(observation.id)
        data = self.cache.get(observation.probability_key)
        # Storage includes SAM box margins and subthreshold interpolation
        # fringe, neither of which is object geometry. Only actual mask support
        # expands fixed template coordinates; keep full probabilities as evidence.
        support = support_bounds((data['probability'] > self.config.pixel_threshold) & data['valid'])
        x, y = observation.origin
        local_observations = [self.observations[o] for o in template.observations
                              if self.observations[o].mask_role != 'discovery']
        coarse_extent = (observation.mask_role == 'discovery' and (not local_observations))
        if (observation.mask_role != 'discovery' or coarse_extent) and support is not None:
            support_box = (x+support[0], y+support[1], x+support[2], y+support[3])
            extent = bounds(np.linalg.inv(observation.pose), support_box)
            # Replace a discovery-only extent once local pixels arrive, while
            # preserving the original coordinate system, pose and fixed anchor.
            old = template.box
            template.box = (tuple(extent) if observation.mask_role != 'discovery' and len(local_observations) == 1 else
                tuple(np.r_[np.minimum(old[:2], extent[:2]), np.maximum(old[2:], extent[2:])]))
        quality = float((data['probability'] > self.config.pixel_threshold).sum())
        feature_count = len(self.cache.get(observation.feature_key)['xy'])
        quality *= np.log2(2 + feature_count)
        ref = Reference(observation.id, observation.source_id, observation.feature_key, quality)
        previous = next((r for r in template.references if r.source_id == ref.source_id), None)
        if previous is None or previous.quality < ref.quality:
            if previous is not None:
                template.references.remove(previous)
            template.references.append(ref)
        # Preserve temporal diversity and replace the least useful interior view.
        if len(template.references) > self.config.max_references:
            refs = sorted(template.references, key=lambda r: r.source_id)
            choices = refs[1:-1] or refs
            remove = min(choices, key=lambda r: r.quality * min(abs(r.source_id-q.source_id)
                         for q in refs if q is not r))
            template.references.remove(remove)
        template.version += 1
        if observation.mask_role != 'discovery':
            template.mask_views.setdefault(observation.source_id, []).append(observation.id)
            self._retain_views(template)
        sources = {self.observations[o].source_id for o in template.observations
                   if self.observations[o].reliable and not self.observations[o].ambiguous}
        if len(sources) >= 2 and not template.mixed:
            template.provisional = False
        self.revision += 1
        return True

    def propose_link(self, a, b, b_to_a, source_id, kind='overlap', tolerance=2., *, activate=True,
                     trajectory_support=None):
        if a == b:
            return None
        if a > b:
            a, b = b, a
            b_to_a = np.linalg.inv(b_to_a)
        key = (a, b)
        link = self.links.get(key)
        if link is None:
            link = Link(a, b, b_to_a.copy(), kind)
            self.links[key] = link
        if link.revoked:
            return link
        if discrepancy(link.b_to_a, b_to_a, self.templates[b].box) > tolerance:
            self.revoke(a, b, 'inconsistent_relative_layout')
            return link
        link.evidence.setdefault(source_id, b_to_a.copy())
        # A measured path corroborates identity, not another coordinate vote.
        if trajectory_support is not None:
            independent = set(trajectory_support.get('measured_sources', ())) - set(link.evidence)
            if len(independent) >= 4:
                if not hasattr(link, 'trajectory_support'):
                    link.trajectory_support = {}
                link.trajectory_support[source_id] = dict(trajectory_support)
        supported = len(link.evidence) >= 2 or bool(getattr(link, 'trajectory_support', {}))
        if activate and supported and not link.active:
            connected = self.members(a)
            if b in connected and discrepancy(connected[b], b_to_a, self.templates[b].box) > tolerance:
                self.revoke(a, b, 'inconsistent_coordinate_cycle')
            else:
                link.kind = kind
                link.active = True
                self.revision += 1
        return link

    def revoke(self, a, b, reason):
        link = self.links[tuple(sorted((a, b)))]
        link.active = False
        link.revoked = True
        link.reason = reason
        self.revision += 1

    def members(self, root):
        result = {root: np.eye(3)}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for link in self.links.values():
                if not link.active or current not in (link.a, link.b):
                    continue
                other = link.b if current == link.a else link.a
                transform = link.b_to_a if current == link.a else np.linalg.inv(link.b_to_a)
                if other not in result:
                    result[other] = result[current] @ transform
                    queue.append(other)
        return result

    def associate_identity(self, a, b, evidence):
        """Identity evidence without pretending template coordinates are known."""
        key = tuple(sorted((a, b)))
        existing = self.identity_links.get(key)
        if existing is not None:
            return existing
        link = IdentityLink(*key, evidence)
        self.identity_links[key] = link
        self.revision += 1
        return link

    def identity_members(self, root):
        found, queue = {root}, deque([root])
        while queue:
            current = queue.popleft()
            adjacent = set(self.members(current))
            for link in self.identity_links.values():
                if link.active and current in (link.a, link.b):
                    adjacent.update((link.a, link.b))
            for other in adjacent-found:
                found.add(other); queue.append(other)
        return found

    def revoke_identity(self, a, b, reason):
        link = self.identity_links[tuple(sorted((a, b)))]
        if link.active:
            link.active, link.reason = False, reason
            self.revision += 1

    def refinement_sources(self, tid):
        return set().union(*(self.templates[t].refinement_sources for t in self.members(tid)))

    def matching_references(self, tid, *, members=None):
        """One bounded active reference set for a coordinate-registered group."""
        by_source = {}
        for member in (self.members(tid) if members is None else members):
            for ref in self.templates[member].references:
                obs = self.observations[ref.observation_id]
                if not obs.reliable or obs.ambiguous:
                    continue
                previous = by_source.get(ref.source_id)
                if previous is None or ref.quality > previous.quality:
                    by_source[ref.source_id] = ref
        refs = sorted(by_source.values(), key=lambda r: r.source_id)
        while len(refs) > self.config.max_references:
            choices = refs[1:-1] or refs
            remove = min(choices, key=lambda r: r.quality * min(abs(r.source_id-q.source_id)
                         for q in refs if q is not r))
            refs.remove(remove)
        return refs

    def _eligible_views(self, members, instance_id=None):
        by_source = defaultdict(list)
        for tid, transform in members.items():
            for ids in self.templates[tid].mask_views.values():
                for oid in ids:
                    obs = self.observations[oid]
                    if (obs.reliable and not obs.ambiguous and obs.mask_role != 'discovery'
                            and (instance_id is None or obs.instance_id == instance_id)):
                        by_source[obs.source_id].append((obs, transform))
        # Match the legacy stage2 rule: usable refinement replaces coarse masks.
        refined = {s: [(o, m) for o, m in jobs if o.mask_role == 'refinement']
                   for s, jobs in by_source.items()}
        refined = {s: jobs for s, jobs in refined.items() if jobs}
        return refined or by_source

    def _view_quality(self, jobs):
        confidence, features = [], []
        for obs, _ in jobs:
            data = self.cache.get(obs.probability_key)
            p = data['probability']
            support = (p > self.config.pixel_threshold) & data['valid']
            if support.any():
                confidence.append(float(p[support].mean()))
                features.append(len(self.cache.get(obs.feature_key)['xy']))
        # Do not reward a larger false-positive area or multiply object scores.
        return max(confidence, default=0.) + .02*np.log2(2+max(features, default=0))

    def _select_views(self, views):
        selected = dict(views)
        quality = ({s:self._view_quality(jobs) for s,jobs in selected.items()}
                   if len(selected)>self.config.mask_reference_limit else {})
        while len(selected) > self.config.mask_reference_limit:
            # Near-duplicate backgrounds are less useful than an equally clear
            # temporally distinct original. Oldest loses exact ties.
            span = max(1, max(selected)-min(selected))
            # On a sorted time axis the nearest remaining source is always
            # adjacent. Preserve exact utilities and oldest-source tie breaks
            # without scanning every other source for every candidate.
            ordered = sorted(selected)
            distances = {source: min(
                source-ordered[n-1] if n else float('inf'),
                ordered[n+1]-source if n+1<len(ordered) else float('inf'))
                for n,source in enumerate(ordered)}
            utility = {s: quality[s] + .1*distances[s]/span for s in selected}
            del selected[min(selected, key=lambda s: (utility[s], s))]
        return selected

    def _retain_views(self, template):
        before = set(template.mask_views)
        selected = self._select_views(self._eligible_views({template.id: np.eye(3)}))
        template.mask_views = {s: [o.id for o, _ in jobs] for s, jobs in selected.items()}
        if before != set(selected):
            template.mask_view_events.append(dict(removed=sorted(before-set(selected)),
                retained=sorted(selected), reason='rolling_probability_reference_selection'))

    def fuse(self, tid, instance_id=None, *, identity=False, coarse=False):
        """Legacy-style aligned probability mean over at most eight originals.

        Candidate duplicates, resolutions and upsampled frames share one vote.
        Fusion is rebuilt from immutable observations after link revocation.
        """
        coarse = coarse or identity
        # Explicit coarse fallback and deferred identity use identical discovery
        # observations, canvas and arithmetic. Share their cached result.
        key = ('fusion-v3', tid, self.revision, instance_id, coarse)
        aliases = getattr(self, '_fusion_keys', None)
        if aliases is None:
            self._fusion_keys = aliases = {}
        if key in aliases:
            return self.cache.get(aliases[key])
        if key in self.cache:
            return self.cache.get(key)
        members = self.members(tid)
        # An unrelated template update increments the bank revision too. A
        # second, dependency-complete key preserves the exact fusion across
        # those updates without relaxing invalidation for this group.
        dependencies = []
        for member, transform in members.items():
            template = self.templates[member]
            observations = tuple((oid, obs.probability_key, obs.feature_key,
                obs.reliable, obs.ambiguous, obs.mask_role, obs.instance_id,
                obs.source_id, tuple(obs.origin), obs.pose.dtype.str, obs.pose.tobytes())
                for oid in template.observations for obs in (self.observations[oid],))
            dependencies.append((member, tuple(template.box), transform.dtype.str,
                transform.tobytes(), observations,
                tuple((source,tuple(ids)) for source,ids in template.mask_views.items())))
        shared_key = ('fusion-dependencies-v1',tid,tuple(dependencies),instance_id,
                      identity and not coarse,coarse,self.config.pixel_threshold,
                      self.config.mask_reference_limit,self.config.max_canvas_pixels)
        if shared_key in aliases:
            aliases[key] = aliases[shared_key]
            return self.cache.get(aliases[key])
        by_source = {} if coarse else self._select_views(self._eligible_views(members,instance_id))
        use_discovery = coarse or (identity and not by_source)
        if use_discovery:
            by_source = {}
            # Global coarse evidence bootstraps identity. In deferred mode it
            # is also the explicit fallback if the fine batch has no usable view.
            for t, transform in members.items():
                for oid in self.templates[t].observations:
                    obs = self.observations[oid]
                    if (obs.mask_role == 'discovery' and obs.reliable and not obs.ambiguous
                            and (instance_id is None or obs.instance_id == instance_id)):
                        by_source.setdefault(obs.source_id, []).append((obs, transform))
            by_source = self._select_views(by_source)
        boxes = [bounds(transform, self.templates[t].box) for t, transform in members.items()]
        if use_discovery:
            # Fine observations may shrink the displayed object extent. The
            # fallback canvas must still retain the original coarse strokes,
            # including support beyond SAM's proposal box.
            for jobs in by_source.values():
                for obs, transform in jobs:
                    data = self.cache.get(obs.probability_key)
                    support = support_bounds((data['probability'] > self.config.pixel_threshold) & data['valid'])
                    if support is not None:
                        x, y = obs.origin
                        support = (x+support[0], y+support[1], x+support[2], y+support[3])
                        boxes.append(bounds(transform @ np.linalg.inv(obs.pose), support))
        low = np.floor(np.min([b[:2] for b in boxes], axis=0)).astype(int)-2
        high = np.ceil(np.max([b[2:] for b in boxes], axis=0)).astype(int)+2
        shape = tuple((high-low)[::-1])
        if np.prod(shape) > self.config.max_canvas_pixels or min(shape) <= 0:
            raise ValueError('template_canvas_limit')
        total = np.zeros(shape, np.float32)
        counts = np.zeros(shape, np.uint16)
        for jobs in by_source.values():
            probability = np.zeros(shape, np.float32)
            valid = np.zeros(shape, bool)
            for obs, transform in jobs:
                data = self.cache.get(obs.probability_key)
                matrix = translation(*(-low)) @ transform @ np.linalg.inv(obs.pose) @ translation(*obs.origin)
                p = warp(np.where(data['valid'], data['probability'], 0.), matrix, shape)
                v = warp(data['valid'], matrix, shape, nearest=True) > 0
                probability = np.maximum(probability, np.where(v, p, 0.))
                valid |= v
            total += probability*valid
            counts += valid
        mean = total/np.maximum(1, counts)
        mask = (mean > self.config.pixel_threshold).astype(np.float32)
        self.cache.put(key, mean=mean, counts=counts, origin=low,
                       mask=mask, has_support=np.array(bool(mask.any())),
                       sources=np.array(sorted(by_source), dtype=np.int64),
                       refined=np.array(any(o.mask_role == 'refinement'
                           for jobs in by_source.values() for o, _ in jobs)))
        aliases[shared_key] = key
        return self.cache.get(key)

    def selected_views(self, tid, instance_id=None):
        return self._select_views(self._eligible_views(self.members(tid), instance_id))

    def output_fusion(self, tid):
        """Choose usable fine strokes, otherwise the explicit coarse fallback."""
        fine = self.fuse(tid)
        fallback = (not len(fine['sources']) or not bool(fine['has_support']))
        return (self.fuse(tid, coarse=True) if fallback else fine), fallback
