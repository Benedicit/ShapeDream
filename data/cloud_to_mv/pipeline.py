from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from .preprocess import voxel_downsample, center_scale_unit_sphere, pca_canonicalize
from .features import FeatureExtractor

def preprocess_pointcloud(points: np.ndarray,
                          voxel_size: float=0.01,
                          do_pca: bool=True) -> Dict:
    pc = voxel_downsample(points, voxel_size)
    pc, meta_cs = center_scale_unit_sphere(pc)
    R = None
    if do_pca:
        pc, R = pca_canonicalize(pc)
    else:
        R = np.eye(3, dtype=np.float32)
    extractor = FeatureExtractor()
    feats = extractor.extract(pc)
    return {"points": pc.astype(np.float32),
            "features": feats,
            "meta": {**meta_cs, "R": R, "backend": extractor.backend}}