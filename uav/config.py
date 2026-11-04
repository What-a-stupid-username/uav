"""Explicit inference configuration shared by Python API and CLI; no env fallback."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import math
from typing import get_type_hints
from threading import RLock

@dataclass(frozen=True)
class UAVConfig:
    block_overlap: int = 3
    debug: bool = False
    segment_seconds: float = 15.0
    segment_overlap: float = 1.0
    mask_pixel_threshold: float = 0.23
    mask_dilation: int = 4
    mask_reference_limit: int = 8
    flow_completion_scale: float = 0.5
    fps: float = 24.0
    max_seconds: float | None = None
    max_watermark_coverage: float | None = 0.15
    max_single_watermark_coverage: float | None = 0.15
    output: str | None = None
    propainter_compile_ffn: bool = True
    propainter_trt_encoder: bool = True
    propainter_trt_raft: bool = True
    propainter_trt_propagation: bool = True
    propainter_compile_precise: bool = True
    propainter_compile_raft: bool = True
    propainter_debug_stages: bool = False
    propainter_feature_blend: bool = True
    propainter_full_reference_range: bool = False
    propainter_graph_propagation: bool = True
    propainter_graph_completion: bool = True
    propainter_measure_propagation: bool = False
    propainter_pack_qkv: bool = True
    propainter_raft_channels_last: bool = True
    propainter_reference_cache_mib: int = 128
    propainter_reference_host_cache_mib: int = 4096
    propainter_reuse_correlation: bool = False
    propainter_sdpa: bool = True
    propainter_sparse_context: bool = True
    propainter_stream_deform: bool = True
    propainter_transformer_only: bool = False
    raft_iters: int = 4
    raft_scale: float = 0.5
    repair_scale: float = 0.7
    repair_context_ratio: float = 64 / 720
    random_start: bool = False
    reference_count: int = 6
    reference_seconds: float = 3.0
    repair: bool = True
    sam3_compile: bool = True
    sam3_trt_engine: str = ''
    sam3_global_resolution: int = 1008
    sam3_global_stride: int = 8
    sam3_graph_grounding: bool = True
    sam3_real_rope: bool = True
    sift_backend: str = "cuda"
    match_backend: str = "auto"
    uncertain_mask_policy: str = "conservative"
    start_source_frame: int = 0
    seed: int = 20261019
    source: str | None = None

    def __post_init__(self):
        types=get_type_hints(type(self))
        for field in fields(self):
            value=getattr(self,field.name);kind=types[field.name]
            if kind is bool and type(value) is not bool: raise ValueError(f'{field.name} must be a bool')
            if kind is int and type(value) is not int: raise ValueError(f'{field.name} must be an integer')
        choices = {
            'sam3_global_resolution': (504,672,1008),
            'sam3_global_stride': (4,8,16),
            'block_overlap': (3,4,5), 'reference_count': (0,2,4,6),
        }
        for name, values in choices.items():
            if getattr(self,name) not in values: raise ValueError(f'{name} must be one of {values}')
        for name in ('raft_scale','flow_completion_scale','repair_scale'):
            value=getattr(self,name)
            if not math.isfinite(value) or not 0<value<=1: raise ValueError(f'{name} must be in (0, 1]')
        if not math.isfinite(self.repair_context_ratio) or not 0 <= self.repair_context_ratio <= .5:
            raise ValueError('repair_context_ratio must be in [0, 0.5]')
        for name in ('fps','max_seconds'):
            value=getattr(self,name)
            if value is not None and (not math.isfinite(value) or value<=0): raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.reference_seconds) or self.reference_seconds<0: raise ValueError('reference_seconds must be finite and nonnegative')
        for name in ('max_watermark_coverage', 'max_single_watermark_coverage'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{name} must be in [0, 1] or None')
        if not math.isfinite(self.mask_pixel_threshold) or not 0<self.mask_pixel_threshold<1:raise ValueError('mask_pixel_threshold must be in (0, 1)')
        if self.mask_dilation<0:raise ValueError('mask_dilation must be nonnegative')
        if not 1 <= self.mask_reference_limit <= 8:
            raise ValueError('mask_reference_limit must be between 1 and 8')
        for name in ('raft_iters',):
            if getattr(self,name)<1: raise ValueError(f'{name} must be positive')
        if self.propainter_reference_cache_mib<0: raise ValueError('reference cache must be nonnegative')
        if self.propainter_reference_host_cache_mib<0: raise ValueError('reference host cache must be nonnegative')
        if not math.isfinite(self.segment_seconds) or self.segment_seconds <= 0:
            raise ValueError('segment_seconds must be finite and positive')
        if not math.isfinite(self.segment_overlap) or not 0 <= self.segment_overlap < self.segment_seconds:
            raise ValueError('segment_overlap must be nonnegative and less than segment_seconds')
        if round(self.segment_seconds*self.fps) < 1 or round(self.segment_seconds*self.fps) <= round(self.segment_overlap*self.fps):
            raise ValueError('segment must contain more frames than overlap')
        if self.start_source_frame < 0: raise ValueError('start_source_frame must be nonnegative')
        if self.random_start and self.start_source_frame: raise ValueError('random_start conflicts with start_source_frame')
        if self.sift_backend not in ('cpu','cuda'): raise ValueError('unknown sift_backend')
        if not isinstance(self.sam3_trt_engine, str): raise ValueError('sam3_trt_engine must be a path string')
        if self.sam3_trt_engine and not self.sam3_real_rope:
            raise ValueError('TensorRT encoder requires sam3_real_rope')
        if self.match_backend not in ('auto','cpu','cuda'): raise ValueError('unknown match_backend')
        if self.uncertain_mask_policy not in ('conservative','skip','ignore'): raise ValueError('unknown uncertain_mask_policy')
        if self.random_start and self.max_seconds is None: raise ValueError('random_start requires max_seconds')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def high_quality(cls, **overrides):
        """Validated quality preset; explicit Python overrides take precedence."""
        return cls(**dict(dict(block_overlap=5,repair_scale=1.,raft_iters=8),**overrides))

# Model instances and CUDA Graph buffers are sequential, as is the pipeline.
# Bind a validated immutable configuration for one run; restore even on failure.
_active = UAVConfig()
_lock = RLock()

def get_config():
    return _active

@contextmanager
def use_config(config):
    if not isinstance(config,UAVConfig): raise TypeError('expected UAVConfig')
    global _active
    with _lock:
        previous=_active
        _active=config
        try: yield config
        finally: _active=previous
