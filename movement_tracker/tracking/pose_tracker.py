from typing import Protocol
import numpy as np
from movement_tracker.pose_data import PoseFrame

class PoseTracker(Protocol):
    def process_frame(self, rgb: np.ndarray, timestamp: float, frame_index: int) -> PoseFrame: ...
    def close(self) -> None: ...
