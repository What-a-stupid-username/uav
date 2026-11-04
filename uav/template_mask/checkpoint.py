"""Explicit debug-only Stage1 snapshots; normal evidence remains in memory.

Store data, not live engines, model weights, OpenCV handles or bound methods.
Fresh runtime components execute the current Stage2 code after restoration.
"""
import builtins
from collections import defaultdict
import os
from pathlib import Path
import pickle
import time

import numpy as np
from numpy.core.numeric import _frombuffer
from numpy.core.multiarray import _reconstruct, scalar
from . import records
from .motion import ComponentModel, MotionPrior
from .pending import PendingCase


VERSION = 1
COMPONENTS = ('bank', 'pending', 'gaps', 'reconciler', 'presence', 'interactions')
RUNTIME = {'config', 'cache', 'detector', 'warnings', 'matcher', 'renderer', 'motion_fits',
           'discovery_buffer', *COMPONENTS}


class _DataReader(pickle.Unpickler):
    """Only the explicit numeric/record vocabulary of a local snapshot."""
    def find_class(self, module, name):
        allowed = {('collections', 'defaultdict'): defaultdict,
                   ('uav.template_mask.motion', 'ComponentModel'): ComponentModel,
                   ('uav.template_mask.motion', 'MotionPrior'): MotionPrior,
                   ('uav.template_mask.pending', 'PendingCase'): PendingCase}
        allowed.update({('builtins', n): getattr(builtins,n)
                        for n in ('dict','list','set','tuple','int','float','bool')})
        allowed.update({('uav.template_mask.records', n): getattr(records,n)
            for n in ('Observation','Template','Reference','Link','IdentityLink','Instance','Pose')})
        allowed.update({('numpy', n): getattr(np,n) for n in ('ndarray','dtype')})
        # Protocol 5 arrays use _frombuffer; scalar records use scalar. NumPy
        # renamed core to _core between releases, so accept either spelling.
        for prefix in ('numpy.core','numpy._core'):
            allowed[(prefix+'.numeric','_frombuffer')] = _frombuffer
            allowed[(prefix+'.multiarray','scalar')] = scalar
            allowed[(prefix+'.multiarray','_reconstruct')] = _reconstruct
        if (module,name) not in allowed:
            raise ValueError(f'unsupported checkpoint type: {module}.{name}')
        return allowed[(module,name)]


def signature(source, frames, config, engine):
    """File stat only: no media hash or content inspection for validation."""
    source = Path(source)
    stat = source.stat()
    options = config.as_dict()
    if not options['sam3_trt_engine']:
        options.pop('sam3_trt_engine')
    # These are output-only controls; changing them does not invalidate Stage1.
    for name in ('uncertain_mask_policy','dilation'):
        options.pop(name)
    # Normalize auto/explicit selection; older checkpoints used CPU matching.
    options.pop('match_backend')
    if config.resolved_match_backend != 'cpu':
        options['match_backend'] = config.resolved_match_backend
    # Existing CPU snapshots predate the explicit backend option.
    if options['sift_backend']=='cpu':
        options.pop('sift_backend')
    else:
        options['sift_contrast_threshold'] = .5
    # Earlier checkpoints used eager SAM and complex RoPE.
    for name in ('sam3_compile', 'sam3_real_rope', 'sam3_graph_grounding'):
        if not options[name]:
            options.pop(name)
    return dict(source=str(source.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
        start=frames.start,count=frames.count,source_fps=frames.source_fps,
        shape=engine.shape,debug=engine.debug,config=options,
        boundary_contract='presence-only-near-end-v1',
        discovery_contract='first-stride-last-v1',
        engine=type(engine).__module__+'.'+type(engine).__qualname__)


def _data(obj, excluded):
    return {k:v for k,v in vars(obj).items() if k not in excluded and not callable(v)}


def save(path, engine, expected):
    if engine.stage != 'first_pass':
        raise ValueError('Stage1 checkpoint must precede Stage2')
    tick = time.monotonic()
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data = dict(version=VERSION,signature=expected,
        engine=_data(engine,RUNTIME),
        components={n:_data(getattr(engine,n),{'engine','cache','config'}) for n in COMPONENTS},
        matcher={n:getattr(engine.matcher,n) for n in ('extractions','matches','reference_cache_hits')},
        detector={n:getattr(engine.detector,n) for n in ('calls','cache_hits','task_calls')},
        warnings=engine.warnings.rows,
        evidence=engine.cache.memory,
        cache_stats={n:getattr(engine.cache,n) for n in ('bytes','peak_bytes','hits','misses')})
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(data,stream,protocol=5)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path),reused=False,bytes=path.stat().st_size,
        seconds=time.monotonic()-tick,first_pass_seconds=engine.timings['first_pass_seconds'])


def load(path, engine, expected):
    tick = time.monotonic()
    path = Path(path)
    with path.open('rb') as stream:
        data = _DataReader(stream).load()
    if data['version'] != VERSION or data['signature'] != expected:
        raise ValueError('Stage1 checkpoint does not match source/interval/config/engine; refresh it explicitly')
    engine.__dict__.update(data['engine'])
    for name in COMPONENTS:
        getattr(engine,name).__dict__.update(data['components'][name])
    engine.matcher.__dict__.update(data['matcher'])
    engine.detector.__dict__.update(data['detector'])
    engine.warnings.rows[:] = data['warnings']
    engine.cache.memory = data['evidence']
    engine.cache.__dict__.update(data['cache_stats'])
    for arrays in engine.cache.memory.values():
        for array in arrays.values():
            array.flags.writeable = False
    original = engine.timings['first_pass_seconds']
    engine.timings['first_pass_seconds'] = 0.
    return dict(path=str(path),reused=True,bytes=path.stat().st_size,
        seconds=time.monotonic()-tick,first_pass_seconds=original)
