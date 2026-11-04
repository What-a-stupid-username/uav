"""Continue unique nearby occurrences using independent SAM pixel evidence.

This is a translation registration between observed segmentations, not a
feature match, a remote template-identity merge or a predicted observation.
"""
import numpy as np
from .geometry import translation, warp, footprint, discrepancy


def register(engine, previous, current):
    e = engine
    supports=[]
    centers=[]
    for obs in (previous,current):
        data=e.cache.get(obs.probability_key)
        support=(data['probability']>e.config.pixel_threshold)&data['valid']
        yy,xx=np.nonzero(support)
        if len(xx)<32:return None
        supports.append(support)
        centers.append(np.array([xx.mean(),yy.mean()])+obs.origin)
    delta=centers[1]-centers[0]
    # Nearby observed geometry is a search bound, not identity evidence.
    size=min(footprint(previous.box),footprint(current.box))
    if np.linalg.norm(delta)>.6*size:return None
    a,b=supports
    forward=warp(a,translation(*(np.asarray(previous.origin)+delta-current.origin)),b.shape,nearest=True)>0
    backward=warp(b,translation(*(np.asarray(current.origin)-delta-previous.origin)),a.shape,nearest=True)>0
    recall=float((forward&b).sum()/max(1,b.sum()))
    precision=float((backward&a).sum()/max(1,a.sum()))
    if min(recall,precision)<.9:return None
    return translation(*delta)@previous.pose,dict(forward=recall,backward=precision,
        source=previous.source_id,observation=previous.id,registration='sam_translation')


def continue_observation(engine, observation):
    e=engine;obs=observation
    if obs.mask_role!='discovery' or obs.ambiguous or obs.crossing_instances:return None
    choices=[]
    for instance in e.active_instances():
        if instance.ended_at is not None and obs.frame>instance.ended_at:continue
        originals=[e.bank.observations[oid] for oid in instance.observations
            if e.bank.observations[oid].mask_role=='discovery'
            and e.bank.observations[oid].reliable and not e.bank.observations[oid].ambiguous
            and not e.bank.observations[oid].crossing_instances
            and e.bank.observations[oid].frame<obs.frame]
        if not originals:continue
        previous=max(originals,key=lambda o:o.frame)
        if (obs.source_id==previous.source_id or obs.frame-previous.frame>e.config.global_stride
                or previous.pose is None):continue
        # A checked disappearance or independent current owner cannot be
        # bridged by shape similarity. Flashes keep separate lifetimes.
        if any(previous.frame<f<obs.frame and p.status=='absent'
               for f,p in instance.poses.items()):continue
        if any(e.bank.observations[oid].instance_id==instance.id
               for oid in e.frame_observations[obs.frame] if oid!=obs.id):continue
        result=register(e,previous,obs)
        if result is None:continue
        measured=instance.poses.get(obs.frame)
        if (measured is not None and measured.status=='measured' and measured.presence
                and discrepancy(measured.matrix,result[0],e.bank.templates[instance.template_id].box)
                    >4*(measured.uncertainty+1.)):
            continue
        # Another candidate in this same discovery can explain the previous
        # occurrence equally well. Leave both unresolved rather than steal it.
        if any(other.id!=obs.id and other.mask_role=='discovery'
               and register(e,previous,other) is not None
               for oid in e.frame_observations[obs.frame]
               if (other:=e.bank.observations[oid]).instance_id is None):continue
        choices.append((instance,*result))
    if len(choices)!=1:return None
    instance,matrix,evidence=choices[0]
    obs.pose,obs.template_id,obs.reliable=matrix,instance.template_id,True
    obs.association_reason='continuous_sam_registration'
    obs.trajectory_evidence=evidence
    e._assign(instance,obs)
    return instance
