from __future__ import annotations
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int=8, alpha: float=1.0, bias: bool=False):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / max(1, rank)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling

class KVInjector(nn.Module):
    """
    Produces K'/V' from adapter tokens and concatenates with base K,V using a learnable gate.
    """
    def __init__(self, token_dim: int, kv_dim: int, rank: int=8):
        super().__init__()
        self.to_k = LoRALinear(token_dim, kv_dim, rank=rank, alpha=1.0, bias=False)
        self.to_v = LoRALinear(token_dim, kv_dim, rank=rank, alpha=1.0, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))  # start closed

    def forward(self, base_K: torch.Tensor, base_V: torch.Tensor, tokens: torch.Tensor):
        """
        base_*(B,H,T,D), tokens(B,M,token_dim) -> concat along T
        """
        B,H,T,D = base_K.shape
        M = tokens.shape[1]
        Kp = self.to_k(tokens)  # (B,M,D)
        Vp = self.to_v(tokens)  # (B,M,D)
        Kp = Kp[:,None,:,:].expand(B,H,M,D)
        Vp = Vp[:,None,:,:].expand(B,H,M,D)
        w = torch.sigmoid(self.gate)
        K = torch.cat([base_K, w*Kp], dim=2)
        V = torch.cat([base_V, w*Vp], dim=2)
        return K, V
