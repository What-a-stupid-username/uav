"""Explicit shared-template options; no environment-variable fallback."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class TemplateConfig:
    fps: float = 24.
    global_stride: int = 8
    global_resolution: int = 1008
    sam_context_scale: float = 3.
    sam_context_min_side: int = 336
    pixel_threshold: float = .23
    object_threshold: float = .3
    max_references: int = 8
    mask_reference_limit: int = 8
    max_canvas_pixels: int = 4_000_000
    max_features: int = 3000
    sift_backend: str = 'cuda'
    match_backend: str = 'auto'
    sam3_compile: bool = True
    sam3_trt_engine: str = ''
    sam3_real_rope: bool = True
    sam3_graph_grounding: bool = True
    min_inliers: int = 6
    dilation: int = 4
    uncertain_mask_policy: str = 'conservative'

    def __post_init__(self):
        if not isinstance(self.sam3_trt_engine, str):
            raise ValueError('sam3_trt_engine must be a path string')
        if self.sam3_trt_engine and not self.sam3_real_rope:
            raise ValueError('TensorRT encoder requires sam3_real_rope')
        for name in ('sam3_compile', 'sam3_real_rope', 'sam3_graph_grounding'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a bool')
        if self.sift_backend not in ('cpu','cuda'):
            raise ValueError('unknown sift_backend')
        if self.match_backend not in ('auto', 'cpu', 'cuda'):
            raise ValueError('unknown match_backend')
        if self.sift_backend=='cuda' and self.max_features>8192:
            raise ValueError('CUDA SIFT supports at most 8192 features')
        if self.uncertain_mask_policy not in ('conservative', 'skip', 'ignore'):
            raise ValueError('unknown uncertain_mask_policy')
        for name in ('global_stride', 'max_references', 'mask_reference_limit',
                     'max_canvas_pixels', 'max_features', 'min_inliers'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.sam_context_min_side) is not int or self.sam_context_min_side < 1:
            raise ValueError('sam_context_min_side must be a positive integer')
        if (not math.isfinite(self.sam_context_scale) or self.sam_context_scale <= 1):
            raise ValueError('sam_context_scale must be finite and greater than 1')
        if self.max_references > 8:
            raise ValueError('at most 8 primary references are supported')
        if self.mask_reference_limit > 8:
            raise ValueError('at most 8 probability references are supported')
        if type(self.dilation) is not int or self.dilation < 0:
            raise ValueError('dilation must be nonnegative')
        for name in ('global_resolution',):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % 28:
                raise ValueError(f'{name} must be a positive multiple of 28')
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('fps must be positive and finite')
        for name in ('pixel_threshold', 'object_threshold'):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f'{name} must be in (0, 1)')

    @property
    def resolved_match_backend(self):
        return self.sift_backend if self.match_backend == 'auto' else self.match_backend

    def as_dict(self):
        return asdict(self)
