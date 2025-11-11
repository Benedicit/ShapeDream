from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0: return points.astype(np.float32)
    p = points.astype(np.float32)
    mins = p.min(axis=0, keepdims=True)
    coords = np.floor((p - mins) / voxel_size).astype(np.int64)
    key = coords[:,0]*73856093 ^ coords[:,1]*19349663 ^ coords[:,2]*83492791
    _, idx = np.unique(key, return_index=True)
    return p[np.sort(idx)]

def center_scale_unit_sphere(points: np.ndarray) -> Tuple[np.ndarray, Dict]:
    pc = points.astype(np.float32).copy()
    c = pc.mean(axis=0, keepdims=True)
    pc -= c
    s = np.linalg.norm(pc, axis=1).max() or 1.0
    pc /= s
    return pc, {"center": c.squeeze(0), "scale": float(s)}

def pca_canonicalize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pc = points - points.mean(axis=0, keepdims=True)
    cov = (pc.T @ pc) / max(1, pc.shape[0] - 1)
    U, _, _ = np.linalg.svd(cov)
    if np.linalg.det(U) < 0: U[:,2] *= -1  # right-handed
    return (pc @ U).astype(np.float32), U.astype(np.float32)
