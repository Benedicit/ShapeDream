from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple
from .types import CameraIntrinsics, CameraPose

def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> CameraIntrinsics:
    f = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
    return CameraIntrinsics(width, height, f, f, width * 0.5, height * 0.5)

def look_at(eye: np.ndarray,
            center: np.ndarray = np.zeros(3, dtype=np.float32),
            up: np.ndarray = np.array([0,0,1], dtype=np.float32)) -> CameraPose:
    # camera looks along -Z in local frame (SD/MVDream default)
    fwd = center - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, fwd); true_up /= (np.linalg.norm(true_up) + 1e-9)
    Rcw = np.stack([right, true_up, -fwd], axis=1).astype(np.float32)
    return CameraPose(R=Rcw, t=eye.astype(np.float32))

def make_camera_ring(num_az: int = 24, elevations_deg: Tuple[float, ...]=(10.0,25.0),
                     radius: float=1.6) -> List[CameraPose]:
    poses = []
    for el in elevations_deg:
        elr = math.radians(el)
        for a in range(num_az):
            az = 2.0 * math.pi * a / num_az
            eye = np.array([radius*math.cos(az)*math.cos(elr),
                            radius*math.sin(az)*math.cos(elr),
                            radius*math.sin(elr)], dtype=np.float32)
            poses.append(look_at(eye))
    return poses
