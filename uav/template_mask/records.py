"""Observations, templates and independently moving instances have separate IDs."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Candidate:
    probability: np.ndarray
    origin: tuple[int, int]
    box: tuple[float, float, float, float]
    score: float
    candidate_id: str
    valid: np.ndarray | None = None
    source_candidates: tuple[str, ...] = ()


@dataclass
class Observation:
    id: str
    frame: int
    source_id: int
    candidate_id: str
    box: tuple
    origin: tuple
    probability_key: str
    feature_key: str
    score: float
    resolution: int
    instance_id: int | None = None
    template_id: int | None = None
    pose: np.ndarray | None = None
    reliable: bool = False
    ambiguous: bool = False
    association_reason: str | None = None
    trajectory_evidence: dict | None = None
    boundary_instance_id: int | None = None
    # Resolution alone cannot distinguish full-image discovery from refinement.
    mask_role: str = 'local'
    source_candidates: tuple[str, ...] = ()
    # Possible owners of a shared detection (crossing OR rigid fragments).
    crossing_instances: tuple[int, ...] = ()
    # Occurrence ownership is independent of eligibility to teach the template.
    ownership_confirmed: bool = False
    partition_children: tuple[str, ...] = ()
    original_probability_key: str | None = None
    original_feature_key: str | None = None


@dataclass
class Reference:
    observation_id: str
    source_id: int
    feature_key: str
    quality: float


@dataclass
class Template:
    id: int
    box: tuple
    anchor: tuple = (0., 0.)
    observations: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    refinement_sources: set[int] = field(default_factory=set)
    version: int = 0
    provisional: bool = True
    mixed: bool = False
    investigation_reason: str | None = None
    mask_views: dict[int, list[str]] = field(default_factory=dict)
    mask_view_events: list[dict] = field(default_factory=list)
    identity_confirmations: list[str] = field(default_factory=list)
    composite: bool = False


@dataclass
class Pose:
    frame: int
    matrix: np.ndarray
    uncertainty: float
    status: str
    source_id: int | None = None
    references: tuple[str, ...] = ()
    presence: bool = False
    reason: str | None = None


@dataclass
class Instance:
    id: int
    template_id: int
    birth: int
    poses: dict[int, Pose] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    ambiguous_frames: set[int] = field(default_factory=set)
    motion: object = None
    ended_at: int | None = None
    continuation_of: int | None = None
    render_start: int | None = None
    render_end: int | None = None
    alias_of: int | None = None
    alias_templates: tuple[int, int] | None = None
    # Stage2 motion-filter output only; estimates never become measurements.
    output_constraints: dict[int, np.ndarray] = field(default_factory=dict)
    trajectory_quality: dict = field(default_factory=dict)


@dataclass
class Link:
    a: int
    b: int
    b_to_a: np.ndarray
    kind: str
    evidence: dict[int, np.ndarray] = field(default_factory=dict)
    trajectory_support: dict = field(default_factory=dict)
    active: bool = False
    revoked: bool = False
    reason: str | None = None


@dataclass
class IdentityLink:
    a: int
    b: int
    evidence: dict
    active: bool = True
    reason: str | None = None
