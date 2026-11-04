"""Fixed-coordinate similarity geometry, independent of bounding-box growth."""
import math
from functools import lru_cache
import cv2
import numpy as np


def translation(x=0., y=0.):
    matrix = np.eye(3)
    matrix[:2, 2] = [x, y]
    return matrix


def points(matrix, xy):
    return np.asarray(xy) @ matrix[:2, :2].T + matrix[:2, 2]


@lru_cache(maxsize=4096, typed=True)
def _box_corners(l, t, r, b):
    """Immutable source corners shared by repeated pose comparisons."""
    corners = np.asarray([[l, t], [r, t], [l, b], [r, b]])
    corners.flags.writeable = False
    return corners


def bounds(matrix, box):
    if matrix.dtype == np.float64 and matrix.shape == (3,3) and matrix.flags.c_contiguous:
        from .native_fit import _module
        return np.asarray(_module().bounds(matrix,*box))
    xy = points(matrix, _box_corners(*box))
    result = np.empty(4,dtype=xy.dtype)
    result[:2],result[2:] = xy.min(0),xy.max(0)
    return result


def footprint(box):
    return max(1., math.hypot(box[2] - box[0], box[3] - box[1]))


def discrepancy(a, b, box):
    if (a.dtype == np.float64 and b.dtype == np.float64
            and a.shape == (3,3) and b.shape == (3,3)
            and a.flags.c_contiguous and b.flags.c_contiguous):
        from .native_fit import _module
        return _module().discrepancy(a,b,*box)
    xy = _box_corners(*box)
    return float(np.max(np.linalg.norm(points(a, xy) - points(b, xy), axis=1)))


def discrepancies(a, b, box):
    """The same corner error for a batch of paired affine poses."""
    a, b = np.asarray(a), np.asarray(b)
    xy = _box_corners(*box)
    left = xy @ np.swapaxes(a[..., :2, :2], -1, -2) + a[..., None, :2, 2]
    right = xy @ np.swapaxes(b[..., :2, :2], -1, -2) + b[..., None, :2, 2]
    return np.max(np.linalg.norm(left-right, axis=-1), axis=-1)


def clip_box(box, shape, padding=0.):
    h, w = shape
    l, t, r, b = box
    return (max(0, min(w, math.floor(l-padding))), max(0, min(h, math.floor(t-padding))),
            max(0, min(w, math.ceil(r+padding))), max(0, min(h, math.ceil(b+padding))))


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def area(box):
    return max(0., box[2]-box[0]) * max(0., box[3]-box[1])


def warp(patch, matrix, shape, nearest=False):
    return cv2.warpAffine(np.asarray(patch, np.float32), matrix[:2].astype(np.float32),
                          (shape[1], shape[0]), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def support_bounds(mask):
    """Half-open occupied bounds without expanding every positive coordinate."""
    x, y, w, h = cv2.boundingRect(np.asarray(mask, np.uint8))
    return (x, y, x+w, y+h) if w and h else None


def visible_support(mask, matrix, shape):
    """Count visible support exactly; interior canvases need no pixel projection."""
    h,w = mask.shape
    frame_h,frame_w = shape
    extent = bounds(matrix,(0,0,w-1,h-1))
    if (extent[0] >= 0 and extent[1] >= 0 and
            extent[2] < frame_w and extent[3] < frame_h):
        return int(np.count_nonzero(mask))
    if (mask.dtype in (np.float32,np.bool_,np.uint8) and mask.flags.c_contiguous
            and matrix.dtype==np.float64 and matrix.flags.c_contiguous):
        from .native_fit import _module
        return _module().visible_support(mask,matrix,h,w,frame_h,frame_w)
    yy,xx = np.nonzero(mask)
    projected = points(matrix,np.column_stack((xx,yy)))
    return int(((projected[:,0]>=0) & (projected[:,0]<frame_w) &
                (projected[:,1]>=0) & (projected[:,1]<frame_h)).sum())


def dilate(mask, radius, *, box=None):
    """Exact elliptical dilation on the occupied ROI plus its full halo."""
    result = np.zeros(mask.shape, bool)
    box = support_bounds(mask) if box is None else box
    if box is None:
        return result
    if not radius:
        return mask.copy()
    x, y, right, bottom = box
    l, t = max(0, x-radius), max(0, y-radius)
    r, b = min(mask.shape[1], right+radius), min(mask.shape[0], bottom+radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
    result[t:b, l:r] = cv2.dilate(mask[t:b, l:r].astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return result


def components(matrix, anchor=(0., 0.)):
    if matrix.dtype == np.float64 and matrix.shape == (3,3) and matrix.flags.c_contiguous:
        from .native_fit import _module
        return np.asarray(_module().components(matrix,*anchor))
    xy = points(matrix, [anchor])[0]
    return np.array([xy[0],xy[1],math.atan2(matrix[1, 0], matrix[0, 0]),
                     math.log(max(1e-9,np.linalg.norm(matrix[:2,0])))])


def from_components(value, anchor=(0., 0.)):
    x, y, angle, logscale = value
    # Scalar clipping needs no NumPy array/dispatch. Comparisons leave NaN
    # unchanged, matching np.clip, while infinities retain the same limits.
    logscale = float(logscale)
    scale = math.exp(-8. if logscale < -8. else 8. if logscale > 8. else logscale)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]])
    matrix[:2, 2] = [x, y] - matrix[:2, :2] @ np.asarray(anchor)
    return matrix
