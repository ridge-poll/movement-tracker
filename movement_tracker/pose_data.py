"""Backend-independent data. Coordinates refer to the oriented image."""
from dataclasses import dataclass, field

LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)

# Standard MediaPipe pose topology, without importing MediaPipe into rendering.
CONNECTIONS = tuple((LANDMARK_NAMES[a], LANDMARK_NAMES[b]) for a, b in (
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
))

@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None
    presence: float | None = None

    @property
    def confidence(self) -> float | None:
        # A display convenience, not an independently calibrated probability.
        values = [v for v in (self.visibility, self.presence) if v is not None]
        return min(values) if values else None

@dataclass(frozen=True)
class PoseFrame:
    timestamp: float
    frame_index: int
    width: int
    height: int
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    source_timestamp: float | None = None
    timestamp_estimated: bool = False
    inference_seconds: float = 0.0
    manual_landmarks: tuple[str, ...] = ()

    @property
    def confidence(self) -> float | None:
        scores = [p.confidence for p in self.landmarks.values() if p.confidence is not None]
        return sum(scores) / len(scores) if scores else None
