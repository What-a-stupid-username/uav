"""Prepared dashed paths for debug rendering only."""
import cv2
import numpy as np


def dashed_polyline(canvas, points, *, dash=10., gap=8.):
    """Continuous arc-length dashes, independent of observation density."""
    phase=0.
    period=dash+gap
    for start,stop in zip(points[:-1],points[1:]):
        vector=stop-start
        length=float(np.linalg.norm(vector))
        if length<=1e-6:continue
        distance=0.
        while distance<length-1e-6:
            active=phase<dash
            step=min(length-distance,(dash if active else period)-phase)
            if active and step>1e-6:
                a=np.rint(start+vector*(distance/length)).astype(np.int32)
                b=np.rint(start+vector*((distance+step)/length)).astype(np.int32)
                cv2.line(canvas,tuple(a),tuple(b),255,2,cv2.LINE_AA)
            distance+=step
            phase=(phase+step)%period


class PreparedTrajectory:
    """Rasterize the full path once; blend only pixels touched by its dashes.

    Geometry and opacity stay fixed across frames. Prepared data is memory-only
    debug state and never participates in mask generation or tracking.
    """
    def __init__(self, shape, samples, color):
        self.color = np.asarray(color, np.float32)
        self.rows = self.cols = np.empty(0, np.int32)
        self.alpha = np.empty((0, 1), np.float32)
        if len(samples) < 2:
            return
        points = np.array([[x,y] for _,x,y in samples], np.float32)
        h,w = shape
        l,t = np.maximum(0,np.floor(points.min(0)).astype(int)-3)
        r,b = np.minimum([w,h],np.ceil(points.max(0)).astype(int)+4)
        if r <= l or b <= t:
            return
        full = np.zeros((b-t,r-l), np.uint8)
        dashed_polyline(full, points-[l,t])
        yy,xx = np.nonzero(full)
        self.rows,self.cols = yy+t,xx+l
        self.alpha = (full[yy,xx].astype(np.float32)*(.4/255.))[:,None]

    def draw(self, panel):
        if not len(self.alpha):
            return
        original = panel[self.rows,self.cols].astype(np.float32)
        panel[self.rows,self.cols] = np.rint(
            original*(1-self.alpha)+self.color*self.alpha).astype(np.uint8)
