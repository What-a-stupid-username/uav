"""Four-frame global-box coverage gate, before tracking or repair initialization."""
import random
import time


def check_coverage(source, threshold, seed, detect, *, single_threshold=None, capture_factory=None):
    import cv2
    if capture_factory is None:capture_factory=cv2.VideoCapture
    start=time.monotonic();cap=capture_factory(str(source))
    try:
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or count<=0 or fps<=0:
            raise ValueError('video must have positive FPS and frame count')
        # No repeated observations for clips shorter than four source frames.
        indices=sorted(random.Random(seed).sample(range(count),min(4,count)))
        samples=[]
        for index in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES,index):raise RuntimeError('preflight seek failed')
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'preflight failed to decode source frame {index}')
            measurement=detect(bgr)
            fraction=float(measurement['coverage'])
            single=float(measurement['single_coverage'])
            if not 0<=single<=fraction<=1:raise ValueError('invalid detected coverage')
            samples.append(dict(source_frame=index,seconds=index/fps,coverage=fraction,single_coverage=single))
        stats=coverage_statistics([s['coverage'] for s in samples])
        maximum_single=max(s['single_coverage'] for s in samples)
        reasons=[]
        if threshold is not None and stats['max_coverage']>threshold:reasons.append('watermark_coverage_exceeded')
        if single_threshold is not None and maximum_single>single_threshold:reasons.append('single_watermark_coverage_exceeded')
        return dict(samples=samples,**stats,threshold=threshold,
                    max_single_coverage=maximum_single,single_threshold=single_threshold,
                    skipped=bool(reasons),skip_reasons=reasons,seconds=time.monotonic()-start,
                    scope='entire_source',seed=seed,metric='box_union',aggregation='max',
                    object_threshold=.3,dilated=False)
    finally:cap.release()


def coverage_statistics(values):
    maximum=max(values,default=0.)
    retained=[v for v in values if v>0 and v>=maximum*.5]
    return dict(mean_coverage=sum(values)/len(values) if values else 0.,max_coverage=maximum,
                filtered_mean_coverage=sum(retained)/len(retained) if retained else 0.,
                retained_frames=len(retained),no_detection=not retained)


def box_union_coverage(boxes,width,height):
    """Exact rectangle union, clipped to frame; no pixel rasterization."""
    boxes=[(max(0.,min(width,float(l))),max(0.,min(height,float(t))),
            max(0.,min(width,float(r))),max(0.,min(height,float(b)))) for l,t,r,b in boxes]
    boxes=[b for b in boxes if b[2]>b[0] and b[3]>b[1]]
    xs=sorted({x for l,t,r,b in boxes for x in (l,r)})
    area=0.
    for l,r in zip(xs,xs[1:]):
        spans=sorted((t,b) for x,t,y,b in boxes if x<r and y>l)
        end=0.;height_sum=0.
        for t,b in spans:
            height_sum+=max(0.,b-max(t,end));end=max(end,b)
        area+=(r-l)*height_sum
    return area/(width*height)


def box_coverage_statistics(boxes,width,height):
    return dict(coverage=box_union_coverage(boxes,width,height),
                single_coverage=max((box_union_coverage([box],width,height) for box in boxes),default=0.))
