import cv2
import numpy as np
from uav.template_mask.review_trajectory import dashed_polyline

def draw_trajectory(panel, samples, frame, color, *, tail_frames=48, full_path_dashed=False):
    """Full offline path at low alpha, plus a time-faded recent tail.

    Samples contain (frame, x, y), so sparse observations fade by elapsed
    frames rather than their position in an observation list.
    """
    if len(samples)<2:return
    points=np.array([[x,y] for _,x,y in samples],np.float32)
    h,w=panel.shape[:2]
    l,t=np.maximum(0,np.floor(points.min(0)).astype(int)-3)
    r,b=np.minimum([w,h],np.ceil(points.max(0)).astype(int)+4)
    if r<=l or b<=t:return
    # One small coverage/alpha canvas; alpha compositing never darkens pixels
    # outside the strokes and avoids a full-frame copy for each segment.
    alpha=np.zeros((b-t,r-l),np.float32)
    shifted=np.rint(points-[l,t]).astype(np.int32)
    full=np.zeros_like(alpha,dtype=np.uint8)
    if full_path_dashed:
        dashed_polyline(full,points-[l,t])
    else:
        cv2.polylines(full,[shifted],False,255,2,cv2.LINE_AA)
    base=full.astype(np.float32)*(.4/255.)
    for i in range(1,len(samples)):
        f0,f1=samples[i-1][0],samples[i][0]
        if f1<=f0 or f0>=frame or f1<frame-tail_frames:continue
        # Subdivide sparse observations so the fade varies along a segment,
        # rather than assigning one opacity to a potentially long line.
        for end in range(max(f0+1,frame-tail_frames+1),min(f1,frame)+1):
            q0=(end-1-f0)/(f1-f0);q1=(end-f0)/(f1-f0)
            p0=np.rint(points[i-1]*(1-q0)+points[i]*q0-[l,t]).astype(np.int32)
            p1=np.rint(points[i-1]*(1-q1)+points[i]*q1-[l,t]).astype(np.int32)
            freshness=max(0.,1.-(frame-end+.5)/tail_frames)
            strength=.95*freshness**2
            stroke=np.zeros_like(full)
            cv2.line(stroke,tuple(p0),tuple(p1),255,2,cv2.LINE_AA)
            np.maximum(alpha,stroke.astype(np.float32)*(strength/255.),out=alpha)
    # Composite the broad tail over the full path. Its faint end is no longer
    # clipped to the baseline opacity by a max() against the full path.
    alpha=base+(1-base)*alpha
    roi=panel[t:b,l:r];a=alpha[:,:,None]
    roi[:]=np.rint(roi.astype(np.float32)*(1-a)+np.array(color,np.float32)*a).astype(np.uint8)
