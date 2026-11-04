"""Continuous image tracking and repeated independent SAM disagreement.

Missing geometry, clipping and overlap are unknown. Image tracking proves
existence, never independently the semantic identity of a watermark.
"""
from .geometry import bounds, footprint, intersection


def assess(engine, instance):
    e = engine
    template = e.bank.templates[instance.template_id]
    h,w = e.shape
    intervals,run = [],[]
    previous = None
    seen = set()

    def localized(frame):
        pose = instance.poses.get(frame)
        if (pose is None or pose.status!='measured' or not pose.presence
                or pose.reason=='continuous_sam_registration'
                or pose.source_id is None or frame in instance.ambiguous_frames):
            return None
        box = bounds(pose.matrix,template.box)
        margin = max(2.,3*pose.uncertainty)
        if (box[0]<margin or box[1]<margin or box[2]>w-margin or box[3]>h-margin
                or pose.uncertainty>.03*footprint(box)):
            return None
        return box

    def finish():
        if len(run)>=3:
            intervals.append(dict(first_frame=run[0][0],last_frame=run[-1][0],
                negative_sources=[source for _,source in run],
                reason='continuous_sift_sam_disagreement'))
        run.clear()

    for frame,(source,boxes) in sorted(e.presence.global_samples.items()):
        if source in seen:
            continue
        seen.add(source)
        box = localized(frame)
        continuous = previous is None or all(localized(f) is not None
            for f in range(previous+1,frame+1))
        # Any overlap may be a partial detection or crossing: unknown.
        observed = any((o:=e.bank.observations[oid]).instance_id==instance.id
            and not o.ambiguous and (o.reliable or o.ownership_confirmed)
            for oid in e.frame_observations.get(frame,()))
        negative = (box is not None and not observed
                    and not any(intersection(box,other)>0 for other in boxes))
        if not continuous or not negative:
            finish()
        if negative:
            run.append((frame,source))
        previous = frame
    finish()
    return dict(action='restrict_intervals' if intervals else 'retain',intervals=intervals)


def prepare_segmentation_support(engine):
    for instance in engine.instances.values():
        instance.segmentation_quality = {}
    for instance in engine.active_instances():
        instance.segmentation_quality = assess(engine,instance)
        for interval in instance.segmentation_quality['intervals']:
            for frame in (interval['first_frame'],interval['last_frame']):
                engine.warnings.add(instance.id,instance.template_id,frame,
                    interval['reason'],'restrict_to_observed_pixels')
