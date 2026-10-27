"""Portable analysis schema. No backend or GUI types."""
from dataclasses import dataclass, field

ANALYSIS_VERSION = "2.1.0"
EVENT_NAMES = ("takeoff", "apex", "tuck_start", "maximum_tuck", "opening", "landing")

@dataclass(frozen=True)
class AnalysisParameters:
    facing: str  # 'right' or 'left' in the displayed image
    takeoff: float  # source-video PTS seconds
    landing: float
    event_overrides: dict[str, float | None] = field(default_factory=dict)
    viewpoint: str = "assumed_side_view"
    min_confidence: float = .5
    min_torso_pixels: float = 8.0
    max_orientation_step_deg: float = 90.0
    max_gap_seconds: float = .15
    smoothing_window_seconds: float = .15
    smoothing_max_points: int = 5
    smoothing_order: int = 2
    gravity: float = 9.80665
    athlete_height_m: float | None = None
    athlete_height_pixels: float | None = None
    calibration_source: str | None = None
    calibration_timestamp: float | None = None
    calibration_disagreement_fraction: float = .25
    fit_residual_torso_fraction: float = .15
    tuck_start_fraction: float = .5
    opening_fraction: float = .7

@dataclass(frozen=True)
class Event:
    source_timestamp: float | None
    provenance: str
    note: str = ""

@dataclass(frozen=True)
class Measurement:
    value: float | None
    unit: str
    note: str = ""

@dataclass
class AnalysisResult:
    analysis_version: str
    source_timestamps: list[float]
    frame_indices: list[int]
    series: dict[str, list[float | None]]
    events: dict[str, Event]
    quality_flags: list[list[str]]
    units: dict[str, str]
    calibration: dict
    parameters: dict
    smoothing: dict
    facing_direction: str
    viewpoint: str
    summary: dict[str, Measurement]
    warnings: list[str]
    metadata: dict
