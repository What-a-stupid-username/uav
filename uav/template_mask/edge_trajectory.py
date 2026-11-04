"""Measured motion can corroborate a clipped original's registration.

The fragment need not establish a motion model of its own. The complete
occurrence must predict its location independently, and visible strokes must
agree. Predictions never become image measurements or template pixel votes.
"""
from dataclasses import replace
import numpy as np
from .gaps import measured, boundary_prior
from .geometry import discrepancy, footprint, bounds, translation, visible_support
from .geometry import components, from_components
from .motion import robust_fit
from .geometry import warp
import cv2


class BoundaryEvidence:
    """Reuse pure trajectory queries within one immutable edge review."""
    def __init__(self, engine):
        self.engine = engine
        self.candidates = {}

    def query(self, template, observation):
        key = (template.id, observation.frame, observation.source_id)
        if key not in self.candidates:
            self.candidates[key] = tuple(trajectory_candidates(self.engine,template,observation))
        return self.candidates[key]


def trajectory_candidates(engine, template, observation):
    """Yield independently validated complete-side poses in template coordinates."""
    e=engine
    members=e.bank.members(template.id)
    for owner in e.active_instances():
        if owner.template_id not in members:
            continue
        rows=[p for p in measured(owner) if p.source_id!=observation.source_id
              and p.frame not in owner.ambiguous_frames]
        for side,samples in (('left',[p for p in rows if p.frame<observation.frame][-32:]),
                             ('right',[p for p in rows if p.frame>observation.frame][:32])):
            if len({p.source_id for p in samples})<4:
                continue
            nearest=samples[-1] if side=='left' else samples[0]
            if abs(nearest.frame-observation.frame)>e.config.global_stride:
                continue
            transform=np.linalg.inv(members[owner.template_id])
            registered=[replace(p,matrix=p.matrix@transform) for p in samples]
            prediction=boundary_prediction(e,registered,template,side,observation.source_id)
            if prediction is None or not prediction[2]:
                continue
            matrix,sigma,_=prediction
            if sigma>.08*footprint(bounds(matrix,template.box)):
                continue
            yield matrix,sigma,dict(instance=owner.id,source=nearest.source_id,
                measured_sources=[p.source_id for p in samples],side=side)


def match_visible_boundary(engine, template, observation, *, evidence=None):
    """Local mask registration around measured motion, ignoring offscreen strokes.

    Translation is searched only within the prior's uncertainty. Rotation and
    scale come from real complete-side measurements. A near-perfect observed
    foreground match is required; a temporal guess alone never creates a link.
    """
    from .features import Match
    e=engine
    data=e.cache.get(observation.probability_key)
    target=(data['probability']>e.config.pixel_threshold)&data['valid']
    if target.sum()<32:
        return []
    fusion=e.bank.output_fusion(template.id)[0]
    total=np.count_nonzero(fusion['mask'])
    matches=[]
    candidates = (evidence.query(template,observation) if evidence is not None
                  else trajectory_candidates(e,template,observation))
    for prediction,sigma,trajectory in candidates:
        radius=int(np.ceil(3*sigma))
        # This is a local localization check, not an expanding mask fallback.
        if radius>min(24,.1*footprint(template.box)):
            continue
        origin=np.asarray(observation.origin)-radius
        shape=(target.shape[0]+2*radius,target.shape[1]+2*radius)
        visible=visible_support(fusion['mask'],prediction@translation(*fusion['origin']),e.shape)
        if not 0<visible<total:
            continue
        projected=warp(fusion['mask'],translation(*(-origin))@prediction
            @translation(*fusion['origin']),shape,nearest=True)>0
        projected=cv2.dilate(projected.astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)
        scores=cv2.matchTemplate(projected,target.astype(np.float32),cv2.TM_CCORR)/float(target.sum())
        best=float(scores.max())
        if best<.9:
            continue
        yy,xx=np.nonzero(scores>=best-.01)
        # Repeated text can support distant alignments. Keep that ambiguity.
        if np.ptp(xx)>max(4,radius) or np.ptp(yy)>max(4,radius):
            continue
        order=np.argmin((xx-radius)**2+(yy-radius)**2)
        dx,dy=radius-int(xx[order]),radius-int(yy[order])
        matrix=translation(dx,dy)@prediction
        shape_evidence=e._shape_evidence(template,Match(matrix,max(.7,sigma),0,0.,()),observation,identity=False)
        if shape_evidence is None or shape_evidence[0]<.9:
            continue
        matches.append(Match(matrix,max(.7,sigma),0,0.,()))
    return matches


def boundary_prediction(engine, samples, template, side, source):
    """Reuse full priors, or validate a short local straight segment.

    The short fit only corroborates an image registration at an adjacent
    boundary. Four independent images, two-sided time-block holdouts and a
    strict extrapolation range are required; it is never tracking evidence.
    """
    registered = [replace(p,frame=p.source_id) for p in samples]
    if len(registered)>=8:
        return boundary_prior(registered,template,side,fit_cache=engine.motion_fits).predict(source)
    if len(registered)<4:
        return None
    times=np.array([p.source_id for p in samples],float)
    span=np.ptp(times)
    if span<=0 or min(abs(times-source))>np.median(np.diff(times))*2:
        return None
    values=np.array([components(p.matrix,template.anchor) for p in samples])
    values[:,2]=np.unwrap(values[:,2])
    design=np.column_stack((np.ones(len(times)),(times-times[0])/span))
    scales=np.array([1.,1.,footprint(template.box),footprint(template.box)])
    noise=max(.7,max(p.uncertainty for p in samples))
    coefficients=np.column_stack([robust_fit(design,values[:,j]) for j in range(4)])
    errors=[np.max(np.abs(design@coefficients-values)*scales)]
    for held in np.array_split(np.arange(len(times)),2):
        keep=np.ones(len(times),bool);keep[held]=False
        trial=np.column_stack([robust_fit(design[keep],values[keep,j]) for j in range(4)])
        errors.append(np.max(np.abs(design[held]@trial-values[held])*scales))
    residual=float(max(errors))
    if residual>3*noise:
        return None
    value=np.array([1.,(source-times[0])/span])@coefficients
    sigma=(noise+residual)*(1+min(abs(times-source))/span)
    return from_components(value,template.anchor),sigma,True


def corroborate(engine, template, observation, fit, *, evidence=None):
    e = engine
    fusion = e.bank.output_fusion(template.id)[0]
    total = np.count_nonzero(fusion['mask'])
    if not total:
        return None
    visible = visible_support(fusion['mask'],fit.matrix@translation(*fusion['origin']),e.shape)
    if not 0 < visible < total:
        return None
    shape = e._shape_evidence(template,fit,observation,identity=False)
    # SAM may miss visible strokes too. Observed foreground must agree with
    # the complete template; completeness is supplied by independent motion.
    if shape is None or shape[0] < .9:
        return None
    accepted = []
    candidates = (evidence.query(template,observation) if evidence is not None
                  else trajectory_candidates(e,template,observation))
    for matrix,sigma,trajectory in candidates:
        error=discrepancy(matrix,fit.matrix,template.box)
        if error<=4*(sigma+fit.uncertainty):
            accepted.append(dict(trajectory,sigma=float(sigma),error=error))
    # Multiple independent occurrences at the same entry remain ambiguous.
    identities = {row['instance'] for row in accepted}
    return min(accepted,key=lambda row:row['error']) if len(identities)==1 else None


def validated_occurrences(engine):
    """Recheck saved boundary associations against the current evidence snapshot."""
    from .features import Match
    e = engine
    evidence = BoundaryEvidence(e)
    for record in getattr(e, 'edge_occurrences', ()):
        fragment = e.instances[record['fragment']]
        owner = e.instances[record['instance']]
        members = e.bank.members(record['template'])
        obs = e.bank.observations[record['observation']]
        if (fragment.template_id not in members or owner.template_id not in members
                or obs.template_id not in members or obs.pose is None
                or not obs.reliable or obs.ambiguous):
            continue
        # Derive current coordinates; a saved registration is not permanent
        # permission to join occurrences after geometry has changed.
        matrix = obs.pose @ np.linalg.inv(members[obs.template_id])
        fit = Match(matrix, record.get('uncertainty', .7), 0, 0., ())
        current = corroborate(e, e.bank.templates[record['template']], obs, fit, evidence=evidence)
        if current is not None and current['instance'] == owner.id:
            yield dict(record, evidence=current)


def prepare_boundary_output(engine):
    """Transport unresolved boundary output with its verified complete path.

    Keep the registered fragment at the seed image. Subsequent output follows
    the measured path's displacement instead of freezing the fragment at its
    clipped discovery box. Constraints never modify image measurements.
    """
    e=engine
    # The method is repeatable after evidence changes. A previous successful
    # transport must not survive if its template link was revoked.
    for instance in e.instances.values():
        for frame in getattr(instance,'boundary_output_frames',()):
            previous = getattr(instance, 'boundary_output_previous', {}).get(frame)
            if previous is None:
                instance.output_constraints.pop(frame,None)
            else:
                instance.output_constraints[frame] = previous
        instance.boundary_output_frames=set()
        instance.boundary_output_previous={}
    e.validated_edge_occurrences = tuple(validated_occurrences(e))
    for record in e.validated_edge_occurrences:
        fragment=e.instances[record['fragment']]
        owner=e.instances[record['instance']]
        members=e.bank.members(record['template'])
        if fragment.template_id not in members or owner.template_id not in members:
            continue
        obs=e.bank.observations[record['observation']]
        template=e.bank.templates[record['template']]
        samples=[p for p in measured(owner) if p.source_id in record['evidence']['measured_sources']]
        transform=np.linalg.inv(members[owner.template_id])
        samples=[replace(p,matrix=p.matrix@transform) for p in samples]
        side=record['evidence']['side']
        seed=boundary_prediction(e,samples,template,side,obs.source_id)
        if seed is None or not seed[2]:
            continue
        # obs.pose maps its private coordinates into the seed image.
        fragment_to_seed=obs.pose@np.linalg.inv(e.bank.members(fragment.template_id)[obs.template_id])
        for frame,pose in fragment.poses.items():
            if pose.status in ('measured','absent') or abs(frame-obs.frame)>e.config.global_stride:
                continue
            prediction=boundary_prediction(e,samples,template,side,e.source_ids[frame])
            if prediction is None or not prediction[2]:
                continue
            matrix=prediction[0]@np.linalg.inv(seed[0])@fragment_to_seed
            if frame not in fragment.boundary_output_frames:
                previous = fragment.output_constraints.get(frame)
                fragment.boundary_output_previous[frame] = None if previous is None else previous.copy()
            fragment.output_constraints[frame]=components(matrix,
                e.bank.templates[fragment.template_id].anchor)
            if not hasattr(fragment,'boundary_output_frames'):
                fragment.boundary_output_frames=set()
            fragment.boundary_output_frames.add(frame)
