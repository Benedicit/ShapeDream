from __future__ import annotations
import types
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .pc_tokens import PointCloudTokenEncoder
from .kv_injector import KVInjector

class PCAdapter:
    """
    Wrap a MVDream/SD U-Net with a point-cloud token adapter and KV injection.

    Two integration modes:
      A) direct KV hook: for CrossAttention.forward(q,k,v,...) signature
      B) encoder_hidden_states concat: for CrossAttention.forward(hidden_states, encoder_hidden_states, ...)
    """
    def __init__(self, unet: nn.Module, in_dim: int, token_dim: int=256, num_tokens: int=16,
                 rank: int=8, device: str='cuda', target_block_name_substr: str='attn2'):
        self.device = torch.device(device)
        self.unet = unet
        self.encoder = PointCloudTokenEncoder(in_dim, num_tokens=num_tokens, token_dim=token_dim).to(self.device)
        self.injectors: Dict[str, KVInjector] = {}
        self._tokens: Optional[torch.Tensor] = None
        self._kv_hooks = []
        self._pre_hooks = []
        # Try direct KV first; if not possible, fall back to encoder_hidden_states concat
        for name, mod in self.unet.named_modules():
            if target_block_name_substr in name and hasattr(mod, 'forward'):
                if self._supports_qkv(mod.forward):
                    inj = KVInjector(token_dim=token_dim, kv_dim=getattr(mod, 'heads', 8)*getattr(mod, 'dim_head', 64), rank=rank).to(self.device)
                    self.injectors[name] = inj
                    self._wrap_qkv_forward(name, mod, inj)
                else:
                    self._register_kv_pre_hook(name, mod)  # fallback
        if not (self._kv_hooks or self._pre_hooks):
            raise RuntimeError("No attention blocks found to hook. Adjust `target_block_name_substr` to match your repo.")

    def parameters(self):
        params = list(self.encoder.parameters())
        for inj in self.injectors.values():
            params += [p for p in inj.parameters() if p.requires_grad]
        return params

    def set_adapter_tokens(self, tokens: torch.Tensor):
        self._tokens = tokens  # (B,M,token_dim)

    def encode_from_pointcloud(self, per_point_feats: torch.Tensor,
                               cam_R: torch.Tensor, cam_t: torch.Tensor, cam_fov: torch.Tensor):
        return self.encoder(per_point_feats, cam_R, cam_t, cam_fov)

    # ---------- internal helpers ----------
    @staticmethod
    def _supports_qkv(forward_fn):
        # crude check for (q,k,v,...) signature
        return forward_fn.__code__.co_varnames[:4] == ('self','q','k','v')

    def _wrap_qkv_forward(self, name: str, module: nn.Module, injector: KVInjector):
        orig_forward = module.forward
        def wrapped_forward(q, k, v, *args, **kwargs):
            if self._tokens is not None:
                # expect k,v shaped (B,H,T,D)
                K, V = injector(k, v, self._tokens)
                return orig_forward(q, K, V, *args, **kwargs)
            return orig_forward(q, k, v, *args, **kwargs)
        module.forward = types.MethodType(wrapped_forward, module)
        self._kv_hooks.append((name, module))

    def _register_kv_pre_hook(self, name: str, module: nn.Module):
        """
        Fallback for CrossAttention.forward(hidden_states, encoder_hidden_states, ...).
        We concatenate adapter tokens (after a linear) to encoder_hidden_states along T.
        """
        proj = nn.Linear(self.encoder.cam_proj.out_features, getattr(module, 'context_dim', 768), bias=False).to(self.device)
        # freeze proj? No: keep trainable (belongs to adapter path)
        def pre_hook(mod, args):
            if self._tokens is None: return
            # args: (hidden_states, encoder_hidden_states, attention_mask, ...)
            if len(args) < 2 or args[1] is None: return
            hidden, enc = args[0], args[1]  # shapes: (B,T,d), (B,S,d)
            B = enc.shape[0]
            tok = self._tokens  # (B,M,Dtok)
            tok_lin = proj(tok) # (B,M,d)
            enc2 = torch.cat([enc, tok_lin], dim=1)
            new_args = (hidden, enc2) + args[2:]
            return new_args
        h = module.register_forward_pre_hook(pre_hook, with_kwargs=False)
        self._pre_hooks.append(h)
