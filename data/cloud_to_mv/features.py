from __future__ import annotations
import numpy as np
from typing import Dict, Optional

class FeatureExtractor:
    """
    extract(points:(N,3), normals:optional) -> {"local": (N,C), "global": (G,)}
    Tries TP3D PointNet++ first, then yanx27, else a deterministic tiny MLP.
    """
    def __init__(self):
        self.backend = None
        self._try_tp3d() or self._try_yanx27() or self._fallback()

    def _try_tp3d(self) -> bool:
        try:
            import torch
            from torch_points3d.applications.pretrained_api import PretrainedRegistry
            self.model = PretrainedRegistry.from_pretrained("pointnet2_msg-s3dis-1")
            self.model.eval().requires_grad_(False)
            self.backend = "tp3d"
            return True
        except Exception:
            return False

    def _try_yanx27(self) -> bool:
        try:
            import torch, sys
            from pathlib import Path
            REPO = Path("Pointnet_Pointnet2_pytorch")
            if not REPO.exists(): return False
            sys.path.append(str(REPO / "models"))
            from pointnet2_part_seg_msg import get_model as get_pointnet2_partseg_msg
            ckpt = REPO / "log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth"
            if not ckpt.exists(): return False
            net = get_pointnet2_partseg_msg(num_part=50, normal_channel=True)
            state = torch.load(str(ckpt), map_location="cpu")
            net.load_state_dict(state["model_state_dict"], strict=False)
            net.eval().requires_grad_(False)
            self.model = net; self.backend = "yanx27"; return True
        except Exception:
            return False

    def _fallback(self) -> bool:
        class TinyMLP:
            def __call__(self, xyz: np.ndarray) -> Dict[str, np.ndarray]:
                rng = np.random.default_rng(0)
                Wl = rng.normal(size=(3,64)).astype(np.float32)
                Wg = rng.normal(size=(3,128)).astype(np.float32)
                Fl = xyz @ Wl
                g = (xyz.mean(axis=0, keepdims=True) @ Wg).squeeze(0)
                return {"local": Fl.astype(np.float32), "global": g.astype(np.float32)}
        self.model = TinyMLP(); self.backend = "fallback"; return True

    def extract(self, points: np.ndarray, normals: Optional[np.ndarray]=None) -> Dict[str, np.ndarray]:
        if self.backend == "tp3d":
            import torch
            from torch_geometric.data import Data
            data = Data(pos=torch.from_numpy(points).float())
            with torch.no_grad():
                self.model.forward(data)
                bb = getattr(self.model, "backbone", None)
                g = getattr(bb, "global_feat", None)
                g = g.detach().cpu().numpy().squeeze() if g is not None else np.zeros(1024, np.float32)
                F = getattr(bb, "fp_features", None)
                if F is not None and len(F) > 0:
                    F = F[-1].detach().cpu().numpy()  # (B,C,N)
                    F = F[0].transpose(1,0)          # (N,C)
                else:
                    F = np.zeros((points.shape[0], 128), np.float32)
            return {"local": F.astype(np.float32), "global": g.astype(np.float32)}
        if self.backend == "yanx27":
            import torch
            xyz = torch.from_numpy(points.T).unsqueeze(0).float()
            nrm = torch.from_numpy(normals.T).unsqueeze(0).float() if normals is not None else None
            with torch.no_grad():
                seg_logits, trans_feat = self.model(torch.cat([xyz,nrm],dim=1) if nrm is not None else xyz, None)
                F = getattr(self.model, "feat", None)
                F = F[0].permute(1,0).contiguous().cpu().numpy() if F is not None else np.zeros((points.shape[0],128), np.float32)
                g = getattr(self.model, "global_feat", None)
                g = g.squeeze(0).cpu().numpy() if g is not None else np.zeros(1024, np.float32)
            return {"local": F.astype(np.float32), "global": g.astype(np.float32)}
        return self.model(points)
