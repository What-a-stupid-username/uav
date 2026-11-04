"""Independent measured-only component priors, validated on time blocks."""
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
from .geometry import components, from_components, footprint, discrepancy
from .numerics import median, quantile
from .native_fit import fit_and_score, fit_batch


class MotionFitCache:
    """Run-local reuse of pure fits, including failed model candidates."""
    def __init__(self):
        self.fits = {}
        self.systems = {}
        self.designs = {}
        self.sampling = {}
        self.windows = {}
        self.hits = self.misses = 0

    def prepare_window(self, times, values, sigmas, size):
        """Pure window preparation shared by overlapping prior rebuilds."""
        key = times.tobytes(), values.tobytes(), sigmas.tobytes(), size
        if key not in self.windows:
            self.windows[key] = prepare_window(times,values,sigmas,size)
        return self.windows[key]

    def fit(self,times,values,tolerance,component,*,straight_only=False):
        times,values = np.asarray(times,float),np.asarray(values,float)
        key = (times.tobytes(),values.tobytes(),float(tolerance),component,straight_only)
        if key in self.fits:
            self.hits += 1
        else:
            self.misses += 1
            self.fits[key] = fit_pose_component(times,values,tolerance,component,
                straight_only=straight_only,systems=self.systems,designs=self.designs,
                sampling=self.sampling)
        result = self.fits[key]
        # Regime logic uses model identity to distinguish successive fits.
        # Share computed numbers, never mutable models or their identities.
        return None if result is None else ComponentModel(result.mode,result.coefficient.copy(),
            result.residual,result.origin,result.span,result.cycles,result.degree,result.knots)


def prepare_window(times, values, sigmas, size):
    times, values = times.copy(), values.copy()
    if np.ptp(values[:,2]) > np.pi:
        values[:,2] = np.unwrap(values[:,2])
    sigma = max(.7,float(median(sigmas)))
    times.flags.writeable = values.flags.writeable = False
    return times,values,(2*sigma,2*sigma,2*sigma/size,2*sigma/size)


def evidence_signature(poses,box,anchor):
    """Exact inputs of a measured-only prior, excluding output predictions."""
    return (tuple(box),tuple(anchor),tuple((p.frame,p.source_id,float(p.uncertainty),
        np.asarray(p.matrix).tobytes()) for p in poses
        if p.status=='measured' and p.presence and p.source_id is not None))


def design_matrix(model,times,cache=None):
    if cache is None:
        return model.design(times)
    key=(model.mode,model.origin,model.span,model.cycles,model.degree,model.knots,
         times.dtype.str,times.shape,times.tobytes())
    if key not in cache:
        cache[key]=model.design(times)
    return cache[key]


@dataclass(frozen=True)
class _Holdout:
    """Immutable partition shared by every fit at the same sample times."""
    held: np.ndarray
    keep: np.ndarray
    key: tuple

    @classmethod
    def prepare(cls, held, count):
        held = np.array(held,dtype=int,copy=True)
        keep = np.ones(count,bool)
        keep[held] = False
        held.flags.writeable = keep.flags.writeable = False
        return cls(held,keep,tuple(held))


@dataclass
class _FitDiagnostics:
    """Tolerance-independent results of one exact numeric system."""
    coefficient: np.ndarray
    residual_score: float | None = None
    held_errors: dict = field(default_factory=dict)

    def score(self, design, values):
        if self.residual_score is None:
            self.residual_score = quantile(np.abs(values-design@self.coefficient), .8)
        return self.residual_score

    def held_error(self, design, values, held, systems, *, check_rank=False):
        partition = held if isinstance(held,_Holdout) else _Holdout.prepare(held,len(values))
        key = (check_rank,partition.key)
        if key not in self.held_errors:
            held,keep = partition.held,partition.keep
            training = design[keep]
            if check_rank and np.linalg.matrix_rank(training) < design.shape[1]:
                error = None
            else:
                # Validation only reads coefficients. The public robust_fit
                # API still returns an owned copy to callers that may mutate it.
                coefficient = _fit_system(training,values[keep],systems).coefficient
                error = float(quantile(np.abs(values[held]-design[held]@coefficient), .8))
            self.held_errors[key] = error
        return self.held_errors[key]


class _SamplingLayout:
    """Shared preparation for components fitted at identical original times."""
    def __init__(self, times):
        self.times = times.copy()
        self.origin = float(times[0])
        self.span = float(np.ptp(times))

    @cached_property
    def dense(self):
        bins = np.minimum(7, ((self.times-self.origin)/self.span*8).astype(int))
        return len(np.unique(bins)) >= 7 and np.max(self.steps) <= .2*self.span

    @cached_property
    def steps(self):
        return np.diff(self.times)

    @cached_property
    def blocks(self):
        return np.array_split(np.arange(len(self.times)), 4)

    @cached_property
    def validation_blocks(self):
        return tuple(_Holdout.prepare(held,len(self.times)) for held in self.blocks)

    @cached_property
    def spline_validation_blocks(self):
        return tuple(_Holdout.prepare(held,len(self.times)) for held in self.spline_blocks)

    @cached_property
    def periodic_frequencies(self):
        return np.linspace(2.5,min(12.,len(self.times)/8),100)

    @cached_property
    def periodic_designs(self):
        # This layout owns an immutable copy of its original timestamps. All
        # components/tolerances use exactly the same frequency grid and basis.
        result = []
        for cycles in self.periodic_frequencies:
            model = ComponentModel('periodic',np.empty(0),0.,self.origin,self.span,cycles,0)
            design = model.design(self.times)
            prefix = (design.shape,design.dtype.str,design.tobytes())
            result.append((cycles,design,prefix))
        return result

    @cached_property
    def spline_blocks(self):
        count = len(self.times)
        edge = max(4, count//32)
        return [np.arange(edge), *np.array_split(np.arange(edge, count-edge),
            min(14, (count-2*edge)//4)), np.arange(count-edge, count)]


def _sampling_layout(times, cache):
    if cache is None:
        return _SamplingLayout(times)
    key = (times.dtype.str, times.shape, times.tobytes())
    if key not in cache:
        cache[key] = _SamplingLayout(times)
    return cache[key]


def _system_key(design,values):
    return (design.shape,design.dtype.str,design.tobytes(),values.dtype.str,values.tobytes())


def _fit_system(design, values, systems):
    key=None
    if systems is not None:
        key=_system_key(design,values)
        if key in systems:
            return systems[key]
    coefficient,score = fit_and_score(design, values)
    result = _FitDiagnostics(coefficient,score)
    if systems is not None:
        systems[key] = result
    return result


def robust_fit(design, values, *, systems=None):
    # Callers own their coefficients; cached diagnostics remain independent.
    return _fit_system(design, values, systems).coefficient.copy()


@dataclass
class ComponentModel:
    mode: str
    coefficient: np.ndarray
    residual: float
    origin: float
    span: float
    cycles: float = 0.
    degree: int = 3
    knots: tuple = ()

    def design(self, times):
        t = (np.asarray(times)-self.origin)/self.span
        if self.mode == 'static':
            return np.ones((t.size, 1))
        if self.mode == 'linear':
            return np.column_stack((np.ones(t.size), t))
        if self.mode == 'accelerating':
            return np.column_stack((np.ones(t.size), t, t*t))
        if self.mode == 'smooth':
            return np.polynomial.legendre.legvander(2*t-1, self.degree)
        if self.mode == 'smooth_spline':
            from scipy.interpolate import BSpline
            return BSpline.design_matrix(t, self.knots, 3, extrapolate=True).toarray()
        return np.column_stack((np.ones(t.size), np.sin(2*np.pi*self.cycles*t), np.cos(2*np.pi*self.cycles*t)))

    def predict_many(self, times):
        """Evaluate one model over original times without per-frame arrays."""
        if self.mode == 'static':
            return np.full(np.asarray(times).size,self.coefficient[0],dtype=float)
        return self.design(times) @ self.coefficient

    def predict(self, frame):
        if self.mode == 'static':
            return float(self.coefficient[0])
        return float((self.design([frame])@self.coefficient)[0])


def _model_trials(count, straight_only, modes, *, layout=None):
    """Generate expensive periodic candidates only after simpler fits fail."""
    for mode, degree, minimum in (('static', 0, 8), ('linear', 1, 8),
            ('accelerating', 2, 16), ('smooth', 3, 24),
            ('smooth', 4, 24), ('smooth', 5, 24)):
        if (count >= minimum and (not straight_only or mode in ('static', 'linear'))
                and (modes is None or mode in modes)):
            yield mode, 0., degree
    if (not straight_only and count >= 25
            and (modes is None or 'periodic' in modes)):
        frequencies = (np.linspace(2.5,min(12.,count/8),100) if layout is None
                       else layout.periodic_frequencies)
        for cycles in frequencies:
            yield 'periodic', cycles, 0


def fit_component(times, values, tolerance, *, straight_only=False, modes=None,
                  systems=None, designs=None, sampling=None):
    times, values = np.asarray(times, float), np.asarray(values, float)
    if len(times) < 8:
        return None
    layout = _sampling_layout(times, sampling)
    origin, span = layout.origin, layout.span
    if span <= 0:
        return None
    periodic = None
    blocks = dense = None
    periodic_trials = None
    for mode, cycles, degree in _model_trials(len(times),straight_only,modes,layout=layout):
        if mode in ('smooth', 'accelerating', 'periodic'):
            # Two dense endpoint clusters cannot certify a curve through a long
            # unobserved middle. Location fits never establish existence there.
            if dense is None:
                dense = layout.dense
            if not dense:
                continue
        # Periodic candidates already own their prepared basis. Construct a
        # model only if its residual and every holdout pass validation.
        if mode == 'periodic' and periodic_trials is None:
            # Every periodic frequency is inspected before choosing a winner.
            # Fit those independent full systems as one batch; retain the
            # original ordered/lazy holdout tests and all acceptance rules.
            matrices=[];keys=[]
            periodic_trials={}
            value_key = (values.dtype.str,values.tobytes())
            for frequency,matrix,prefix in layout.periodic_designs:
                key = prefix+value_key
                if systems is not None and key in systems:
                    periodic_trials[frequency] = (systems[key],matrix)
                else:
                    matrices.append(matrix);keys.append((frequency,key))
            if matrices:
                coefficients,scores=fit_batch(matrices,values)
                for (frequency,key),matrix,coefficient,score in zip(keys,matrices,coefficients,scores):
                    trial=_FitDiagnostics(coefficient,float(score))
                    periodic_trials[frequency] = (trial,matrix)
                    if systems is not None:
                        systems[key]=trial
        if mode == 'periodic':
            trial,design = periodic_trials[cycles]
        else:
            model = ComponentModel(mode,np.empty(0),0.,origin,span,cycles,degree)
            design = design_matrix(model,times,designs)
            trial = _fit_system(design,values,systems)
        residual_score = trial.score(design, values)
        if residual_score > tolerance:
            continue
        if blocks is None:
            blocks = layout.validation_blocks
        validated = True
        for held in blocks:
            if trial.held_error(design, values, held, systems) > 2*tolerance:
                validated = False
                break
        if not validated:
            continue
        residual = max(tolerance*.25,float(residual_score))
        if mode != 'periodic':
            model.coefficient = trial.coefficient.copy()
            model.residual = residual
            return model
        # Strict improvement preserves min()'s first-candidate tie behaviour.
        if periodic is None or residual < periodic.residual:
            periodic = ComponentModel(mode,trial.coefficient.copy(),residual,
                                      origin,span,cycles,degree)
    if periodic is not None:
        return periodic
    return None if straight_only or modes is not None else fit_spline(
        times, values, tolerance, systems=systems, designs=designs, sampling=sampling)


def fit_pose_component(times, values, tolerance, component, *, straight_only=False,
                       systems=None, designs=None, sampling=None):
    """Shape priors distinguish coherent rotation/scale from fit jitter.

    Estimate localization jitter from second differences, after accounting for
    unequal original timestamps. Stable velocity and slow cycles have little
    second difference; alternating SIFT angle/scale errors do not. This changes
    model evidence only, never the independent image measurements themselves.
    """
    if component < 2 or len(times) < 8:
        return fit_component(times, values, tolerance, straight_only=straight_only,
            systems=systems, designs=designs, sampling=sampling)
    times, values = np.asarray(times, float), np.asarray(values, float)
    layout = _sampling_layout(times, sampling)
    dt = layout.steps
    velocity = np.diff(values)/dt
    acceleration = np.diff(velocity)*np.minimum(dt[:-1], dt[1:])
    noise = float(median(np.abs(acceleration-median(acceleration))))/1.652
    shape_tolerance = max(tolerance, 2.5*noise)
    static = fit_component(times, values, shape_tolerance, modes={'static'},
        systems=systems, designs=designs, sampling=sampling)
    if static is not None:
        return static
    if component == 3 and not straight_only:
        # A slow breathing scale is preferable to a spline that later diverges.
        periodic = fit_component(times, values, shape_tolerance, modes={'periodic'},
            systems=systems, designs=designs, sampling=sampling)
        if periodic is not None:
            return periodic
    model = fit_component(times, values, tolerance, straight_only=straight_only,
        systems=systems, designs=designs, sampling=sampling)
    if model is None:
        return None
    if component == 2:
        predicted = model.design(np.linspace(times[0], times[-1], max(32, len(times)))) @ model.coefficient
        changes = np.diff(predicted)
        direction = np.sign(predicted[-1]-predicted[0])
        # Back-and-forth angular fits need their own independently supported
        # intervals; do not explain random wobble with one oscillatory model.
        if not direction or np.any(direction*changes < -tolerance/len(changes)):
            return None
        blocks = [float(median(values[ids])) for ids in layout.blocks]
        if sum(direction*np.diff(blocks) > shape_tolerance) < 2:
            return None
    return model


def fit_spline(times, values, tolerance, *, systems=None, designs=None, sampling=None):
    """Validate a low-complexity C2 curve using contiguous local holdouts.

    A curve need not extrapolate an entire missing quarter to be useful locally.
    Endpoint holdouts still test extrapolation; interior holdouts test gaps.
    Sparse endpoint clusters and arbitrary corners must not gain strong support
    merely by interpolating every measured point.
    """
    if len(times) < 32:
        return None
    layout = _sampling_layout(times, sampling)
    origin, span = layout.origin, layout.span
    if not layout.dense:
        return None
    # Endpoint tests cover a short extrapolation. A validated interior curve
    # does not promise extrapolation of an entire quarter of the recording.
    blocks = layout.spline_validation_blocks
    for count in (1, 2, 3, 5, 7):
        if len(times) < 6*(count+4):
            continue
        knots = tuple(np.r_[np.zeros(4), np.linspace(0., 1., count+2)[1:-1], np.ones(4)])
        model = ComponentModel('smooth_spline', np.empty(0), 0., origin, span,
                               degree=3, knots=knots)
        design = design_matrix(model,times,designs)
        trial = _fit_system(design, values, systems)
        residual_score = trial.score(design, values)
        if residual_score > tolerance:
            continue
        valid, held_errors = True, []
        for held in blocks:
            error = trial.held_error(design, values, held, systems, check_rank=True)
            if error is None:
                valid = False
                break
            held_errors.append(error)
            if error > 2*tolerance:
                valid = False
                break
        if valid:
            model.coefficient = trial.coefficient.copy()
            model.residual = max(tolerance*.25,float(residual_score),max(held_errors))
            return model
    return None


class MotionPrior:
    def __getstate__(self):
        # Derived query caches are cheap to reconstruct and are not evidence.
        return {k:v for k,v in vars(self).items()
                if k not in ('_excluded_priors', '_times', '_typical_step', '_pending_deviations',
                             '_predictions', '_continuity_predictions')}

    @cached_property
    def _times(self):
        return np.array([p.frame for p in self.evidence], float)

    @cached_property
    def _typical_step(self):
        return max(1., float(median(np.diff(self._times)))) if len(self.evidence) > 1 else 1.

    @cached_property
    def _pending_deviations(self):
        """Contrary-source residuals are fixed for this immutable fitted prior."""
        size = footprint(self.box)
        return tuple(tuple(float(quantile([
            abs(components(self.evidence[k].matrix, self.anchor)[j]
                - model.predict(self._times[k])) * (1. if j < 2 else size)
            for k in ids], .8)) for model, ids, _ in changes)
            for j, changes in enumerate(self.unconfirmed_changes))

    def __init__(self, poses, box, anchor, *, fit_cache=None, whole_only=False):
        measured = {}
        for pose in sorted(poses, key=lambda p: p.frame):
            if pose.status == 'measured' and pose.presence and pose.source_id is not None:
                measured.setdefault(pose.source_id, pose)
        self.evidence = sorted(measured.values(), key=lambda p: p.frame)
        self.anchor, self.box = anchor, box
        self.models = [None]*4
        self.start = self.end = None
        self.segments = []
        self.whole = None
        self.regimes = [[] for _ in range(4)]
        self.pending_changes = [[] for _ in range(4)]
        self.unconfirmed_changes = [[] for _ in range(4)]
        # Reuse only pure numeric fits. Every prior still owns its evidence,
        # support intervals, regime validation and source exclusion decisions.
        fit = fit_pose_component if fit_cache is None else fit_cache.fit
        if len(self.evidence) < 8:
            return
        size = footprint(box)
        times_all = self._times
        values_all = np.array([components(p.matrix,anchor) for p in self.evidence])
        sigmas_all = np.array([p.uncertainty for p in self.evidence])
        indices = {id(p):k for k,p in enumerate(self.evidence)}
        prepared = {}
        def fit_rows(rows, *, straight_only=False):
            begin,end=indices[id(rows[0])],indices[id(rows[-1])]+1
            # Prefix/tail/local scans may revisit the same window with a
            # different model restriction. Its measured preparation is shared.
            key = begin,end
            if key not in prepared:
                prepare = prepare_window if fit_cache is None else fit_cache.prepare_window
                prepared[key] = prepare(times_all[begin:end],values_all[begin:end],
                                        sigmas_all[begin:end],size)
            times,values,tolerances = prepared[key]
            models = [fit(times, values[:, j], tolerances[j], j,
                                   straight_only=straight_only) for j in range(4)]
            return models, times[0], times[-1]

        for length in dict.fromkeys((len(self.evidence), 64, 32, 16, 8)):
            if length > len(self.evidence):
                continue
            models, start, end = fit_rows(self.evidence[-length:])
            segment = (models, start, end)
            self.segments.append(segment)
            if self.whole is None:
                self.whole = segment
            if sum(m is not None for m in models) > sum(m is not None for m in self.models):
                self.models, self.start, self.end = segment
            if whole_only:
                return
            if all(m is not None for m in models):
                break
        # A failing whole-record fit must not discard independently validated
        # earlier curve segments. Overlapping windows retain their own support
        # intervals; they are never joined into an invented corner across a gap.
        # Even a passing broad spline must not conceal a simpler local straight
        # leg. In that case only test cheap static/linear local candidates.
        if len(self.evidence) > 8:
            straight_only = all(m is not None for m in self.whole[0])
            ranges = {(s[1], s[2]) for s in self.segments}
            # Short early legs need the same opportunity as the short tail.
            # They remain measured-only models, supported only in their range.
            for length in (64, 32, 16, 8):
                if length >= len(self.evidence):
                    continue
                rows = self.evidence[:length]
                if (rows[0].frame, rows[-1].frame) in ranges:
                    continue
                segment = fit_rows(rows, straight_only=straight_only)
                if all(m is not None for m in segment[0]):
                    self.segments.append(segment)
                    ranges.add((segment[1], segment[2]))
            for length in (64, 32, 16, 8):
                if length >= len(self.evidence):
                    continue
                for offset in range(0, len(self.evidence)-length+1, length//2):
                    rows = self.evidence[offset:offset+length]
                    if (rows[0].frame, rows[-1].frame) in ranges:
                        continue
                    segment = fit_rows(rows, straight_only=straight_only or length < 32)
                    if all(m is not None for m in segment[0]):
                        self.segments.append(segment)
                        ranges.add((segment[1], segment[2]))
        self._establish_regimes(fit,times_all,values_all,sigmas_all)

    def _establish_regimes(self,fit=fit_pose_component,times=None,values=None,sigmas=None):
        """A new component regime needs eight consecutive contrary originals.

        Validated windows propose models, but do not revoke an established one
        merely because they are nearer or contain fewer fitted components.
        A contradiction run must itself fit and pass time-block holdouts. Raw
        image measurements are retained throughout, including genuine turns.
        """
        if times is None:
            times=np.array([p.frame for p in self.evidence],float)
            values=np.array([components(p.matrix,self.anchor) for p in self.evidence])
            sigmas=np.array([p.uncertainty for p in self.evidence])
        values=values.copy()
        values[:, 2] = np.unwrap(values[:, 2])
        size = footprint(self.box)
        scales = np.array([1., 1., size, size])
        sigmas = np.maximum(.7,sigmas)
        for j in range(4):
            # Prefer the broadest validated model starting at the first source.
            # A full curve remains a curve, rather than a succession of lines.
            initial = [s for s in self.segments if s[0][j] is not None
                       and s[1] == times[0]]
            if initial:
                models, start, end = max(initial, key=lambda s: s[2]-s[1])
                model = models[j]
                confirmed = int(np.searchsorted(times, end, side='right'))-1
            else:
                model, confirmed = None, None
                for stop in range(8, len(times)+1, 8):
                    trial = fit(times[:stop], values[:stop, j],
                                          2*float(median(sigmas[:stop]))/scales[j], j)
                    if trial is not None:
                        model, confirmed = trial, stop-1
                        break
                if model is None:
                    continue
            regimes = self.regimes[j]
            regimes.append(dict(model=model, start=float(times[0]),
                confirmed_at=float(times[confirmed]), support_end=float(times[confirmed]),
                support_sources=int(confirmed+1), contradicted_sources=0))
            contrary = []
            predicted = model.predict_many(times)
            for k, time in enumerate(times):
                error = abs(values[k, j]-predicted[k])*scales[j]
                tolerance = 3*(sigmas[k]+model.residual*scales[j])
                if error <= tolerance:
                    if contrary:
                        self.unconfirmed_changes[j].append((model, contrary, float(time)))
                    contrary = []
                    regimes[-1]['support_end'] = float(time)
                    continue
                contrary.append(k)
                if len(contrary) < 8 or (len(contrary) > 32 and len(contrary) % 8):
                    continue
                # No interpolation across two sparse endpoint clusters. Adjacent
                # measurements must cover the proposed interval sufficiently.
                ids = np.array(contrary)
                local_times = times[ids]
                if np.max(np.diff(local_times)) > .25*np.ptp(local_times):
                    continue
                replacement = fit(local_times, values[ids, j],
                    2*float(median(sigmas[ids]))/scales[j], j)
                if replacement is None:
                    continue
                regimes.append(dict(model=replacement, start=float(local_times[0]),
                    confirmed_at=float(time), support_end=float(time),
                    support_sources=len(ids), contradicted_sources=len(ids)))
                model, contrary = replacement, []
                predicted = model.predict_many(times)
            self.pending_changes[j] = contrary
            if contrary:
                self.unconfirmed_changes[j].append((model, contrary, None))

    def _nearest_evidence(self, frame):
        """Sorted original times preserve the earlier observation on ties."""
        right = int(np.searchsorted(self._times,frame))
        if right == 0:
            return self.evidence[0]
        if right == len(self.evidence):
            return self.evidence[-1]
        left = right-1
        return self.evidence[left if frame-self._times[left] <= self._times[right]-frame else right]

    def _component_prediction(self, frame, fallback=None):
        nearest = self._nearest_evidence(frame)
        value = components(nearest.matrix if fallback is None else fallback[0], self.anchor)
        size = footprint(self.box)
        times = self._times
        step = self._typical_step
        errors, known, pending = [], [], []
        for j, regimes in enumerate(self.regimes):
            available = [r for r in regimes if r['start'] <= frame]
            regime = available[-1] if available else regimes[0] if regimes else None
            if regime is None:
                errors.append(nearest.uncertainty + size*.03*abs(frame-nearest.frame)
                              if fallback is None else fallback[1])
                known.append(False)
                pending.append(False)
                continue
            model = regime['model']
            value[j] = model.predict(frame)
            beyond = max(0., regime['start']-frame, frame-regime['support_end'])/model.span
            power = model.degree if model.mode in ('smooth', 'smooth_spline') else 2 if model.mode == 'accelerating' else 1
            error = model.residual*(1+3*beyond)**power*(1+abs(frame-nearest.frame)/step*.1)
            scale = 1. if j < 2 else size
            error *= scale
            # Until a coherent replacement exists, retain the old geometry but
            # expose actual contrary measurements in its uncertainty. Never
            # extrapolate their two-point velocity/acceleration as a new mode.
            is_pending = False
            for change_index, (old_model, ids, resolved_at) in enumerate(self.unconfirmed_changes[j]):
                if (old_model is model and frame >= times[ids[0]]
                        and (resolved_at is None or frame < resolved_at)):
                    is_pending = True
                    error = max(error, self._pending_deviations[j][change_index])
            errors.append(error)
            known.append(True)
            pending.append(is_pending)
        return from_components(value, self.anchor), max(errors), all(known) and not any(pending)

    def constrain(self, frame, matrix, uncertainty):
        """Keep independently established components of a weak search pose."""
        if not self.evidence:
            return matrix, uncertainty, False
        return self._component_prediction(frame, (matrix, uncertainty))

    def _window(self, frame):
        eligible = [s for s in self.segments if s[1] <= frame <= s[2]]
        if eligible:
            return max(eligible, key=lambda s: (sum(m is not None for m in s[0]), s[2]-s[1]))
        if self.segments:
            # Extrapolate from the nearest supported end, never the tail just
            # because it happened to be the most recent validated segment.
            return min(self.segments, key=lambda s: (max(s[1]-frame, frame-s[2], 0),
                -sum(m is not None for m in s[0])))
        return self.models, self.start, self.end

    def predict(self, frame, *, segment=None):
        if not self.evidence:
            return None
        # A fitted prior owns its models for its entire lifetime. Include the
        # selected segment: local line support and whole curves are distinct.
        key = (frame, None if segment is None else
               (tuple(id(model) for model in segment[0]), segment[1], segment[2]))
        if not hasattr(self, '_predictions'):
            self._predictions = {}
        if key not in self._predictions:
            self._predictions[key] = (self._component_prediction(frame) if segment is None
                                      else self._segment_prediction(frame, segment))
        matrix, uncertainty, strong = self._predictions[key]
        return matrix.copy(), uncertainty, strong

    def _segment_prediction(self, frame, segment):
        nearest = self._nearest_evidence(frame)
        value = components(nearest.matrix, self.anchor)
        dt = abs(frame-nearest.frame)
        size = footprint(self.box)
        models, start, end = segment
        strong = all(m is not None for m in models)
        errors = []
        density_factor = 1+abs(frame-nearest.frame)/self._typical_step*.1
        for j, model in enumerate(models):
            if model is None:
                errors.append(nearest.uncertainty + size*.03*dt)
                continue
            value[j] = model.predict(frame)
            beyond = max(0., start-frame, frame-end)/model.span
            growth = (1+3*beyond)**(model.degree if model.mode in ('smooth', 'smooth_spline') else 2 if model.mode == 'accelerating' else 1)
            error = model.residual*growth*density_factor
            errors.append(error if j < 2 else error*size)
        return from_components(value, self.anchor), max(errors), strong

    def report(self):
        def window(models, start, end):
            return dict(start=float(start), end=float(end),
                        measured_sources=int(sum(start <= p.frame <= end for p in self.evidence)),
                        components=[m.mode if m else 'undetermined' for m in models],
                        strong=all(m is not None for m in models))
        whole = window(*self.whole) if self.whole else None
        return dict(measured_sources=len(self.evidence), start=self.start, end=self.end,
                    components=[m.mode if m else 'undetermined' for m in self.models],
                    cycles=[m.cycles if m and m.mode == 'periodic' else None for m in self.models],
                    scope='whole_trajectory' if whole and self.start == whole['start'] and self.end == whole['end'] else 'local_window',
                    whole_trajectory=whole, validated_windows=[window(*s) for s in self.segments],
                    component_regimes=[[dict(mode=r['model'].mode,
                        **{k:v for k,v in r.items() if k != 'model'}) for r in rows]
                        for rows in self.regimes],
                    pending_change_sources=[[self.evidence[k].source_id for k in ids]
                        for ids in self.pending_changes], minimum_change_sources=8)

    def excluding_source(self, source_id, *, fit_cache=None):
        """One independent prior per excluded original, shared by all queries.

        A MotionPrior is replaced when its measured inputs change; the derived
        prior therefore has the same lifetime and cannot certify its own source.
        Older debug snapshots acquire this runtime cache lazily.
        """
        if source_id is None or not any(p.source_id == source_id for p in self.evidence):
            return self
        if not hasattr(self, '_excluded_priors'):
            self._excluded_priors = {}
        if source_id not in self._excluded_priors:
            self._excluded_priors[source_id] = MotionPrior(
                [p for p in self.evidence if p.source_id != source_id],
                self.box, self.anchor, fit_cache=fit_cache)
        return self._excluded_priors[source_id]

    def continuity_predictions(self, frame, *, box=None):
        """Validated alternatives shared by candidates querying this fitted prior.

        These are query results, never new measurements. Source exclusion is
        performed by the caller before querying this immutable prior.
        """
        if not hasattr(self, '_continuity_predictions'):
            self._continuity_predictions = {}
        box = self.box if box is None else box
        key = frame, tuple(box)
        if key not in self._continuity_predictions:
            segments = sorted((s for s in self.segments if all(m is not None for m in s[0])),
                key=lambda s: (max(s[1]-frame, frame-s[2], 0), -(s[2]-s[1])))
            accepted = []
            seen = []
            for segment in [None, *segments]:
                prediction = self.predict(frame, segment=segment)
                if (prediction is None or not prediction[2] or not np.isfinite(prediction[1])
                        or prediction[1] > .08*footprint(box)):
                    continue
                matrix, sigma, _ = prediction
                if any(discrepancy(matrix, other, box) <= sigma for other in seen):
                    continue
                seen.append(matrix)
                evidence = self.continuity(frame, matrix, sigma, segment=segment)
                if evidence is not None:
                    accepted.append((matrix, sigma, evidence))
            self._continuity_predictions[key] = accepted
        return [(matrix.copy(), sigma, dict(evidence, components=list(evidence['components'])))
                for matrix, sigma, evidence in self._continuity_predictions[key]]

    def continuity(self, frame, matrix, uncertainty, source_id=None, *, segment=None, fit_cache=None):
        """Independent trajectory support for an IMAGE-supported candidate pose.

        Exclude this source before fitting: a current match cannot certify its
        own continuity. An arbitrary polyline/search extrapolation has no model
        here. Large prediction uncertainty never becomes evidence of identity.
        """
        from .geometry import bounds, discrepancy
        prior = self.excluding_source(source_id, fit_cache=fit_cache)
        # Validate identity against the local straight leg independently, even
        # when a broader curve is used for precise search positions. A straight
        # approximation must not flatten an accurately measured smooth curve.
        local_lines = [s for s in prior.segments if s[1] <= frame <= s[2]
            and all(m is not None for m in s[0])
            and all(m.mode in ('static', 'linear') for m in s[0][:2])]
        local_lines.sort(key=lambda s: s[2]-s[1], reverse=True)
        candidates = ([segment] if segment is not None else
            [*local_lines, None, prior._window(frame)])
        for candidate in candidates:
            prediction = prior.predict(frame, segment=candidate)
            if prediction is None or not prediction[2]:
                continue
            predicted, sigma, _ = prediction
            size = footprint(bounds(predicted, self.box))
            if not np.isfinite(sigma) or sigma > .08*size:
                continue
            error = discrepancy(predicted, matrix, self.box)
            tolerance = 3*(sigma+uncertainty)
            if error > tolerance:
                continue
            if candidate is None:
                selected = [[r for r in rows if r['start'] <= frame][-1]
                    if any(r['start'] <= frame for r in rows) else rows[0]
                    for rows in prior.regimes]
                support_start = max(r['start'] for r in selected)
                support_end = min(r['support_end'] for r in selected)
                modes = [r['model'].mode for r in selected]
            else:
                support_start, support_end = candidate[1:]
                modes = [m.mode if m else 'undetermined' for m in candidate[0]]
            return dict(normalized_error=error/max(1e-9, tolerance),
                        uncertainty=float(sigma), measured_sources=len(prior.evidence),
                        support_start=float(support_start), support_end=float(support_end),
                        components=modes)
        return None
