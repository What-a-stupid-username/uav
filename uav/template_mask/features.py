"""Original-image SIFT with cached ROI extraction and multi-location matching."""
from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import clip_box, discrepancy
from .native_fit import _module


@dataclass
class Match:
    matrix: np.ndarray
    uncertainty: float
    inliers: int
    coverage: float
    references: tuple[str, ...] = ()
    reference_indices: tuple[int, ...] = ()


def ratio_matches(reference, target):
    """OpenCV exact L2 neighbours as arrays, in original query order."""
    if len(reference) < 2 or not len(target):
        return np.empty(0,dtype=int), np.empty(0,dtype=int)
    distances, indices = cv2.batchDistance(np.asarray(target,np.float32),
        np.asarray(reference,np.float32), cv2.CV_32F, normType=cv2.NORM_L2, K=2)
    # DMatch.distance was a Python float: preserve its float64 ratio arithmetic.
    keep = (indices[:, 1] >= 0) & (distances[:, 0].astype(np.float64)
                                   < .76*distances[:, 1].astype(np.float64))
    return indices[keep, 0], np.flatnonzero(keep)


class SiftMatcher:
    def __init__(self, cache, config):
        self.cache, self.config = cache, config
        if config.sift_backend=='cuda':
            from uav.cuda_sift import CudaSift
            self.sift=CudaSift(threshold=.5,scale_up=True,nfeatures=config.max_features,geometry=True,priority=-1)
        else:
            self.sift = cv2.SIFT_create(nfeatures=config.max_features, contrastThreshold=.015)
        self.extractions = self.matches = 0
        self._reference_extents = {}
        self.reference_cache_hits = 0
        from .rigidity import RigidityEvidence
        self.rigidity = RigidityEvidence()
        self.frame_pairs = None
        self.feature_sources = {}
        self.evidence_source = None
        self.gpu_neighbours = None
        if config.resolved_match_backend == 'cuda':
            from .neighbours import CudaNeighbours
            self.gpu_neighbours = CudaNeighbours()

    def close(self):
        try:
            if self.config.sift_backend=='cuda':
                self.sift.close()
        finally:
            if self.gpu_neighbours is not None:
                self.gpu_neighbours.close()

    def _match_key(self, *parts):
        # Match entries are internal run-local keys, never observation IDs.
        # Hash structured values directly instead of repeatedly JSON-encoding
        # nested feature IDs and every reference's transform.
        return ('match-backend', 'cuda' if self.gpu_neighbours is not None else 'cpu', *parts)

    def matching_report(self):
        return (self.gpu_neighbours.report() if self.gpu_neighbours is not None
                else dict(backend='cpu'))

    def begin_frame_evidence(self, source_id=None):
        self.frame_pairs = {}
        self.evidence_source = source_id

    def contradicted_references(self, observations, source_id):
        """Aggregate this keyframe's computed pairs without another search."""
        pairs,self.frame_pairs = self.frame_pairs or {},None
        originals = {o.feature_key:o for o in observations.values()
                     if o.reliable and not o.ambiguous and o.source_id!=source_id}
        rejected=[]
        for reference, jobs in pairs.items():
            obs=originals.get(reference)
            if obs is None:continue
            fits=[fit for rows in jobs.values() for fit in rows]
            if self.rigidity.observe(reference,source_id,fits,obs.box):
                rejected.append(obs.id)
        return rejected

    def _record_pair(self, reference, target, fits):
        if (self.frame_pairs is not None and (self.evidence_source is None
                or self.feature_sources.get(target)==self.evidence_source)):
            self.frame_pairs.setdefault(reference,{})[target]=[
                Match(f.matrix.copy(),f.uncertainty,f.inliers,f.coverage,
                      reference_indices=f.reference_indices) for f in fits]
        return fits

    def extract(self, image, source_id, box=None, *, feature_box=None):
        h, w = image.shape[:2]
        box = clip_box(box or (0, 0, w, h), (h, w))
        key = (self.cache.key('sift-object-v1', int(source_id), box, tuple(feature_box), self.config.max_features)
               if feature_box is not None else
               self.cache.key('sift', int(source_id), box, self.config.max_features))
        if self.config.sift_backend!='cpu':
            key=self.cache.key('sift-backend',self.config.sift_backend,'contrast-0.5',key)
        if key not in self.cache:
            l, t, r, b = box
            gray = image[t:b, l:r]
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            mask = None
            if feature_box is not None:
                # Limit keypoint centres BEFORE nfeatures ranking. Background
                # must not consume the object's feature budget. Coordinates
                # stay native; SAM's large context crop is unrelated to SIFT.
                a, y, z, v = feature_box
                xx, yy = np.arange(l, r), np.arange(t, b)
                mask = np.uint8((yy[:, None] >= y) & (yy[:, None] < v)
                                & (xx[None, :] >= a) & (xx[None, :] < z))*255
            if gray.size and self.config.sift_backend == 'cuda':
                xy,size,desc = self.sift.detect_arrays(gray,mask)
            else:
                kp,desc = self.sift.detectAndCompute(gray, mask) if gray.size else ([],None)
                xy = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
                size = np.array([k.size for k in kp], np.float32)
            xy = xy + [l,t]
            if desc is None:
                desc = np.empty((0, 128), np.float32)
            # RootSIFT retains shape under changes of contrast/opacity.
            desc = np.sqrt(desc / np.maximum(desc.sum(1, keepdims=True), 1e-9))
            self.cache.put(key, xy=xy, size=size, desc=desc)
            self.extractions += 1
        self.feature_sources[key] = source_id
        return key

    def select(self, key, observation_id, probability, origin, threshold, *, valid=None):
        data = self.cache.get(key)
        xy = np.rint(data['xy'] - origin).astype(int)
        h, w = probability.shape
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        # A small support margin retains stroke edges without selecting the ROI background.
        support = cv2.dilate((probability > threshold).astype(np.uint8), np.ones((5, 5), np.uint8))
        ids = np.flatnonzero(inside)
        keep = ids[support[xy[ids, 1], xy[ids, 0]] > 0]
        if valid is not None:
            keep = keep[valid[xy[keep,1],xy[keep,0]]]
        result = self.cache.key('observation_features', observation_id)
        self.cache.put(result, **{name: value[keep] for name, value in data.items()})
        if key in self.feature_sources:
            self.feature_sources[result] = self.feature_sources[key]
        return result

    def fit(self, reference, target, max_locations=1, min_inliers=None, *, neighbours=None):
        """Repeated RANSAC removes target inliers, leaving other instances available."""
        self.matches += 1
        minimum = self.config.min_inliers if min_inliers is None else max(4, min_inliers)
        if min(len(reference['xy']), len(target['xy'])) < minimum:
            return []
        # Each query's neighbours depend only on the fixed reference. Removing
        # a fitted occurrence removes query rows, never changes their neighbours.
        if neighbours is None:
            search = self.gpu_neighbours if self.gpu_neighbours is not None else ratio_matches
            neighbours = search(reference['desc'],target['desc'])
        ri,ti = neighbours
        remaining = np.ones(len(target['xy']),bool)
        extent = None
        results = []
        for _ in range(max_locations):
            active = remaining[ti]
            current_ri,current_ti = ri[active],ti[active]
            if len(current_ti) < minimum:
                break
            src, dst = reference['xy'][current_ri], target['xy'][current_ti]
            localization = _module().localization(target['size'][current_ti])
            matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=3*localization, maxIters=2500, confidence=.995, refineIters=10)
            if matrix is None:
                break
            keep = mask.ravel().astype(bool)
            inlier_count = int(np.count_nonzero(keep))
            # Reject low support before allocating/sorting reference indices.
            if inlier_count < minimum or inlier_count/len(keep) < .35:
                break
            reference_indices = np.flatnonzero(np.bincount(
                current_ri[keep],minlength=len(reference['xy'])))
            if len(reference_indices) < minimum:
                break
            full = np.eye(3); full[:2] = matrix
            scale = np.linalg.norm(full[:2, 0])
            if not np.isfinite(full).all() or not .1 < scale < 10:
                break
            inlier_src, inlier_dst = src[keep], dst[keep]
            distributed = _module().distributed(
                np.ascontiguousarray(inlier_src,dtype=np.float64),localization)
            if distributed < 0:
                centered = inlier_src - inlier_src.mean(0)
                distributed = np.linalg.svd(centered, compute_uv=False)[-1] >= localization
            if not distributed:
                break
            if extent is None:
                # Reference geometry is immutable in the evidence cache. Its
                # extent is shared across target frames and RANSAC options.
                xy = reference['xy']
                cached = self._reference_extents.get(id(xy)) if not xy.flags.writeable else None
                if cached is not None and cached[0] is xy:
                    extent = cached[1]
                else:
                    extent = np.ptp(xy,axis=0)
                    if not xy.flags.writeable:
                        self._reference_extents[id(xy)] = xy,extent
            residual,coverage = _module().match_statistics(full,
                np.ascontiguousarray(inlier_src,dtype=np.float64),
                np.ascontiguousarray(inlier_dst,dtype=np.float64),*extent)
            uncertainty = max(localization,residual)
            results.append(Match(full, uncertainty, inlier_count, coverage,
                                 reference_indices=tuple(reference_indices.tolist())))
            # Exclude the fitted occurrence, not all matches of its descriptors.
            remaining[current_ti[keep]] = False
        return results

    def reference_matches(self, reference_key, target_key, max_locations=1, min_inliers=None, *, neighbours=None):
        """Immutable original feature pairs survive template/reference updates.

        Cache the image-to-image fit, before applying observation poses or group
        transforms. A changed template still reprojects and regroups all fits.
        """
        minimum = self.config.min_inliers if min_inliers is None else max(4, min_inliers)
        key = self._match_key('reference-matches-v2', reference_key, target_key,
                             max_locations, minimum)
        if key not in self.cache and max_locations in (1,4):
            alternative = self._match_key('reference-matches-v2',reference_key,target_key,
                                          4 if max_locations==1 else 1,minimum)
            if alternative in self.cache:
                previous = self.cache.get(alternative)
                # RANSAC visits locations in order. A failed first location
                # ends both queries; a four-location result supplies its prefix.
                if max_locations==1 or not previous:
                    self.cache.put(key, **{name:value[:max_locations]
                        for name,value in previous.items()})
        if key in self.cache:
            self.reference_cache_hits += 1
            data = self.cache.get(key)
            if not data:
                return self._record_pair(reference_key,target_key,[])
            return self._record_pair(reference_key,target_key,[Match(matrix.copy(), float(sigma), int(count), float(coverage),
                          reference_indices=tuple(ids[ids>=0].tolist()))
                    for matrix, sigma, count, coverage, ids in zip(data['matrices'], data['sigma'],
                        data['inliers'], data['coverage'], data['reference_indices'])])
        if neighbours is None:
            reference, target = self.cache.get(reference_key), self.cache.get(target_key)
            if min(len(reference['xy']),len(target['xy'])) >= minimum:
                neighbour_key = self._match_key('ratio-neighbours-v1',reference_key,target_key)
                if neighbour_key not in self.cache:
                    search = self.gpu_neighbours if self.gpu_neighbours is not None else ratio_matches
                    ri,ti = search(reference['desc'],target['desc'])
                    self.cache.put(neighbour_key,ri=ri,ti=ti)
                data = self.cache.get(neighbour_key)
                neighbours = data['ri'],data['ti']
        fits = self.fit(self.cache.get(reference_key),self.cache.get(target_key),
                        max_locations,min_inliers,neighbours=neighbours)
        if not fits:
            self.cache.put(key)
            return self._record_pair(reference_key,target_key,[])
        width=max((len(m.reference_indices) for m in fits),default=0)
        self.cache.put(key, matrices=np.array([m.matrix for m in fits]).reshape(-1, 3, 3),
            sigma=np.array([m.uncertainty for m in fits]),
            inliers=np.array([m.inliers for m in fits]), coverage=np.array([m.coverage for m in fits]),
            reference_indices=np.array([list(m.reference_indices)+[-1]*(width-len(m.reference_indices))
                for m in fits],dtype=np.int32).reshape(len(fits),width))
        return self._record_pair(reference_key,target_key,fits)

    def prepare_references(self, references, observations, target_key, max_locations=1, min_inliers=None):
        """Precompute independent searches across templates in one readback.

        Only immutable descriptor neighbours are cached here. Registration,
        rigidity evidence and ownership still run in the caller's order.
        """
        if self.gpu_neighbours is not None:
            self._prepare_neighbours(references, observations, target_key, max_locations, min_inliers)

    def _prepare_neighbours(self, references, observations, target_key, max_locations, min_inliers):
        """Batch only uncached independent GPU searches; do not reorder RANSAC."""
        if self.gpu_neighbours is None:
            return {}
        minimum = self.config.min_inliers if min_inliers is None else max(4,min_inliers)
        target = self.cache.get(target_key)
        if len(target['xy']) < minimum:
            return {}
        missing = {}
        prepared = {}
        for ref in references:
            obs = observations[ref.observation_id]
            if obs.ambiguous or not obs.reliable:
                continue
            key = self._match_key('reference-matches-v2',ref.feature_key,target_key,max_locations,minimum)
            if key not in self.cache and ref.feature_key not in missing:
                neighbour_key = self._match_key('ratio-neighbours-v1',ref.feature_key,target_key)
                if neighbour_key in self.cache:
                    data = self.cache.get(neighbour_key)
                    prepared[ref.feature_key] = data['ri'],data['ti']
                    continue
                data = self.cache.get(ref.feature_key)
                if len(data['xy']) >= minimum:
                    missing[ref.feature_key] = data['desc']
        if not missing:
            return prepared
        pairs = self.gpu_neighbours.many(list(missing.values()),target['desc'])
        for reference_key,(ri,ti) in zip(missing,pairs):
            key = self._match_key('ratio-neighbours-v1',reference_key,target_key)
            self.cache.put(key,ri=ri,ti=ti)
            prepared[reference_key] = ri,ti
        return prepared

    def template_matches(self, template, observations, target_key, max_locations=1, min_inliers=None,
                         *, references=None, member_transforms=None):
        references = template.references if references is None else references
        member_transforms = member_transforms or {template.id: np.eye(3)}
        signature = tuple((r.observation_id, r.feature_key, observations[r.observation_id].source_id,
                           observations[r.observation_id].ambiguous,
                           observations[r.observation_id].reliable,
                           observations[r.observation_id].pose.tobytes(),
                           member_transforms[observations[r.observation_id].template_id].tobytes())
                          for r in references)
        # A new observation need not change the retained matching references.
        # All geometry/reference inputs are already present in this key; the
        # broader template version would invalidate identical registrations.
        key = self._match_key('group-matches-v4', template.id, tuple(template.box),
                              signature, target_key, max_locations, min_inliers)
        if key in self.cache:
            data = self.cache.get(key)
            if not data:
                return []
            return [Match(matrix.copy(), float(sigma), int(count), float(coverage),
                    tuple(str(r) for r in refs if r)) for matrix, sigma, count, coverage, refs in
                    zip(data['matrices'], data['sigma'], data['inliers'], data['coverage'], data['references'])]
        neighbours = self._prepare_neighbours(references,observations,target_key,max_locations,min_inliers)
        fits = []
        inverses = {}
        for ref in references:
            obs = observations[ref.observation_id]
            if obs.ambiguous or not obs.reliable:
                continue
            for fit in self.reference_matches(ref.feature_key,target_key,max_locations,min_inliers,
                                              neighbours=neighbours.get(ref.feature_key)):
                if obs.template_id not in inverses:
                    inverses[obs.template_id] = np.linalg.inv(member_transforms[obs.template_id])
                fit.matrix = fit.matrix @ obs.pose @ inverses[obs.template_id]
                fit.references = (ref.observation_id,)
                fits.append(fit)
        # Independent original frames, not repeated candidates, provide corroboration.
        groups = []
        for fit in sorted(fits, key=lambda m: (-m.inliers, m.uncertainty)):
            group = next((g for g in groups if discrepancy(g[0].matrix, fit.matrix, template.box)
                          <= max(g[0].uncertainty, fit.uncertainty)*4), None)
            if group is None:
                groups.append([fit])
            else:
                group.append(fit)
        out = []
        for group in groups:
            best = group[0]
            refs = {}
            for fit in group:
                oid = fit.references[0]
                refs.setdefault(observations[oid].source_id, oid)
            out.append(Match(best.matrix, best.uncertainty, best.inliers, best.coverage, tuple(refs.values())))
        out.sort(key=lambda m: (-len(m.references), -m.inliers, m.uncertainty))
        if not out:
            self.cache.put(key)
            return []
        width = max((len(m.references) for m in out), default=0)
        self.cache.put(key, matrices=np.array([m.matrix for m in out]).reshape(-1, 3, 3),
            sigma=np.array([m.uncertainty for m in out]), inliers=np.array([m.inliers for m in out]),
            coverage=np.array([m.coverage for m in out]),
            references=np.array([list(m.references)+['']*(width-len(m.references)) for m in out], dtype=str).reshape(len(out), width))
        return out
