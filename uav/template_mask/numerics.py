"""Small finite-vector reductions, preserving NumPy's interpolation order."""
import numpy as np
from functools import lru_cache
from scipy.linalg.lapack import dgelsd, dgelsd_lwork


@lru_cache(maxsize=512)
def _gelsd_workspace(rows, columns):
    """LAPACK workspace sizes depend on dimensions, never sample values."""
    work, integer_work, info = dgelsd_lwork(rows, columns, 1)
    if info != 0:
        raise ValueError(f'GELSD workspace query failed: {info}')
    return int(work), integer_work


def least_squares(design, values):
    """NumPy's GELSD/rcond convention with less overhead for small real fits.

    Inputs are not overwritten. Only the common float64 single-response,
    overdetermined system uses the direct wrapper; other forms retain NumPy.
    """
    if (design.dtype != np.float64 or values.dtype != np.float64
            or design.ndim != 2 or values.ndim != 1
            or design.shape[0] != len(values)
            or not 0 < design.shape[1] <= design.shape[0]):
        return np.linalg.lstsq(design, values, rcond=None)[0]
    rows, columns = design.shape
    if columns == 1:
        # Static models (including their IRLS-weighted systems) have one
        # column. Its scalar normal equation needs no SVD. Extreme norms keep
        # the scaled LAPACK path to avoid overflow/underflow.
        column = design[:, 0]
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            squared = column@column
            numerator = column@values
        if np.finfo(float).tiny < squared < np.inf and abs(numerator) < np.inf:
            return np.array([numerator/squared])
    coefficient, _, _, info = dgelsd(design, values[:, None],
        *_gelsd_workspace(rows, columns), cond=np.finfo(float).eps*max(design.shape))
    if info > 0:
        raise np.linalg.LinAlgError('SVD did not converge in Linear Least Squares')
    if info < 0:
        raise ValueError(f'illegal GELSD argument: {-info}')
    return coefficient[:columns, 0].copy()


def median(values):
    values=np.asarray(values)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        return np.median(values)
    middle=values.size//2
    indices=(middle-1,middle) if values.size%2==0 else (middle,)
    selected=np.partition(values,indices)
    if values.dtype==np.float64:
        return (selected[middle-1]+selected[middle])/2 if values.size%2==0 else selected[middle]
    return selected[indices[0]:middle+1].mean()


def quantile(values,q):
    values=np.asarray(values)
    if values.ndim != 1 or not values.size or not 0 <= q <= 1 or not np.isfinite(values).all():
        return np.quantile(values,q)
    index=(values.size-1)*q
    low=int(index);high=min(low+1,values.size-1)
    selected=np.partition(values,(low,high))
    a,b=selected[low],selected[high]
    fraction=index-low
    # NumPy's linear quantile interpolates from the nearer endpoint.
    return b-(b-a)*(1-fraction) if fraction>=.5 else a+(b-a)*fraction
