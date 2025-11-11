from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class CameraPose:  # camera-to-world
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

CameraDict = Dict[str, Dict]