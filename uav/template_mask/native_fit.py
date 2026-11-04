"""Cached CPU IRLS kernel using the installed SciPy's LAPACK implementation."""
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
from scipy.linalg import cython_lapack
from .numerics import _gelsd_workspace


def _cached_library(source, directory):
    from filelock import FileLock
    # The extension uses Python's buffer protocol and receives SciPy's LAPACK
    # function pointer; no NumPy headers or machine-specific BLAS link is needed.
    options = ['-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC']
    fingerprint = hashlib.sha256(source.read_bytes()+repr((options,platform.machine(),
        platform.system(),sys.byteorder,sys.implementation.cache_tag)).encode()).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    library, stamp = directory/'libmotion_fit.so', directory/'source.sha256'
    with FileLock(str(directory/'build.lock')):
        if not library.exists() or not stamp.exists() or stamp.read_text()!=fingerprint:
            compiler = shutil.which('g++')
            if compiler is None:
                raise RuntimeError('Building the motion kernel requires g++; use the UAV image and mount tmp/ writable')
            temporary, mark = directory/'libmotion_fit.build.so', directory/'source.sha256.tmp'
            try:
                subprocess.run([compiler,*options,'-I'+sysconfig.get_path('include'),
                    str(source),'-o',str(temporary)],check=True)
                os.replace(temporary,library)
                mark.write_text(fingerprint)
                os.replace(mark,stamp)
            finally:
                temporary.unlink(missing_ok=True)
                mark.unlink(missing_ok=True)
    return library


@lru_cache(maxsize=1)
def _module():
    source = Path(__file__).with_name('robust_fit.cpp')
    directory = Path(__file__).resolve().parents[2]/'tmp/extensions/uav_motion_fit'
    spec=importlib.util.spec_from_file_location('_uav_motion_fit',_cached_library(source,directory))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _kernel():
    return _module().fit,cython_lapack.__pyx_capi__['dgelsd']


def fit_and_score(design,values):
    """Five IRLS rounds and residual score through a single buffer-protocol call."""
    design = np.ascontiguousarray(design,dtype=np.float64)
    values = np.ascontiguousarray(values,dtype=np.float64)
    if (design.ndim!=2 or values.ndim!=1 or design.shape[0]!=len(values)
            or not 0 < design.shape[1] <= design.shape[0]):
        raise ValueError('native robust fit requires an overdetermined real system')
    function,solver = _kernel()
    result = np.empty(design.shape[1]+1,np.float64)
    info = function(design,values,result,*design.shape,solver,*_gelsd_workspace(*design.shape))
    if info > 0:
        raise np.linalg.LinAlgError('SVD did not converge in robust fit')
    if info < 0:
        raise ValueError(f'illegal robust fit argument: {-info}')
    return result[:-1],float(result[-1])


def robust_coefficients(design,values):
    return fit_and_score(design,values)[0]


DEFAULT_WORKERS = 4


@lru_cache(maxsize=8)
def _pool(workers):
    # Construct once, submit chunks rather than individual tiny matrices.
    # ThreadPoolExecutor joins these workers at process exit. Native buffers
    # are thread-local; the template/evidence dictionaries stay on the caller.
    return ThreadPoolExecutor(max_workers=workers,thread_name_prefix='uav-motion')


def _fit_chunk(designs,values):
    count,rows,columns=designs.shape
    output=np.empty((count,columns+1),np.float64)
    function,solver=_kernel()
    info=function(designs,values,output,rows,columns,solver,
                  *_gelsd_workspace(rows,columns),count)
    if info:
        raise np.linalg.LinAlgError(f'robust batch fit failed: {info}')
    return output


def fit_batch(designs,values,*,workers=None):
    """Fit equal-size independent systems with deterministic output order."""
    designs=np.ascontiguousarray(designs,dtype=np.float64)
    values=np.ascontiguousarray(values,dtype=np.float64)
    if (designs.ndim!=3 or values.ndim!=1 or designs.shape[1]!=len(values)
            or not 0 < designs.shape[2] <= designs.shape[1]):
        raise ValueError('invalid robust batch dimensions')
    count,_,columns=designs.shape
    workers=DEFAULT_WORKERS if workers is None else workers
    if type(workers) is not int or workers<1:
        raise ValueError('workers must be a positive integer')
    if not count:
        return np.empty((0,columns)),np.empty(0)
    # Start native import/workspace queries on the calling thread, before any
    # worker can race a cold import. Data/results remain in memory only.
    _kernel();_gelsd_workspace(*designs.shape[1:])
    if workers==1 or count<16:
        result=_fit_chunk(designs,values)
    else:
        chunks=np.array_split(designs,min(workers,count))
        result=np.concatenate(list(_pool(workers).map(_fit_chunk,chunks,[values]*len(chunks))))
    return result[:,:-1],result[:,-1]
