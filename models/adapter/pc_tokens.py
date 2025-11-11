from __future__ import annotations
import math, numpy as np
import torch
import torch.nn as nn

def sinusoidal_pose_embed(R: torch.Tensor, t: torch.Tensor, fov: torch.Tensor, dim: int=64):
    """
    R:(B,3,3), t:(B,3), fov:(B,) -> (B,dim) pose embedding
    """
    def rot_to_euler(R):
        sy = torch.sqrt(R[:,0,0]**2 + R[:,1,0]**2)
        near_zero = sy < 1e-6
        x = torch.atan2(R[:,2,1], R[:,2,2])
        y = torch.atan2(-R[:,2,0], sy)
        z = torch.where(~near_zero, torch.atan2(R[:,1,0], R[:,0,0]), torch.zeros_like(sy))
        return torch.stack([x,y,z], -1)
    eul = rot_to_euler(R)
    vec = torch.cat([eul, t, fov[:,None]], -1)  # (B,7)
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=vec.device))
    x = vec / 3.0
    sinus = torch.cat([torch.sin(x.unsqueeze(-1)*freqs),
                       torch.cos(x.unsqueeze(-1)*freqs)], dim=-1)  # (B,7,dim)
    emb = sinus.flatten(1)
    return nn.functional.normalize(emb, dim=-1)[..., :dim]

class TokenPool(nn.Module):
    def __init__(self, in_dim: int, num_tokens: int=16, token_dim: int=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.query = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.to_q = nn.Linear(token_dim, token_dim, bias=False)
        self.to_k = nn.Linear(in_dim, token_dim, bias=False)
        self.to_v = nn.Linear(in_dim, token_dim, bias=False)
        self.out = nn.Linear(token_dim, token_dim)

    def forward(self, feats: torch.Tensor):  # (B,N,C)
        B,N,C = feats.shape
        Q = self.to_q(self.query)[None].expand(B, -1, -1)      # (B,M,D)
        K = self.to_k(feats)                                   # (B,N,D)
        V = self.to_v(feats)                                   # (B,N,D)
        attn = torch.softmax(Q @ K.transpose(1,2) / (K.shape[-1]**0.5), dim=-1)
        T = attn @ V
        return self.out(T)                                     # (B,M,D)

class PointCloudTokenEncoder(nn.Module):
    def __init__(self, in_dim: int, num_tokens: int=16, token_dim: int=256, cam_dim: int=64):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.pool = TokenPool(256, num_tokens=num_tokens, token_dim=token_dim)
        self.cam_proj = nn.Linear(cam_dim, token_dim)

    def forward(self, per_point_feats: torch.Tensor,
                cam_R: torch.Tensor, cam_t: torch.Tensor, cam_fov: torch.Tensor):
        x = self.pre(per_point_feats)            # (B,N,256)
        tokens = self.pool(x)                    # (B,M,token_dim)
        cam_emb = sinusoidal_pose_embed(cam_R, cam_t, cam_fov, dim=64)  # (B,64)
        tokens = tokens + self.cam_proj(cam_emb)[:,None,:]
        return tokens
