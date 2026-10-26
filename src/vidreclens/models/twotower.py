import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, feats, n_items, d=128, use_id=True, id_drop=0.3, mod_drop=0.2, drop=0.1):
        super().__init__()
        self.names = list(feats)
        self.id_drop = id_drop
        self.mod_drop = mod_drop
        for k, v in feats.items():
            self.register_buffer(f"f_{k}", v, persistent=False)
        self.proj = nn.ModuleDict({k: nn.Linear(v.shape[1], d) for k, v in feats.items()})
        self.id = nn.Embedding(n_items, d) if use_id else None
        if self.id is not None:
            nn.init.normal_(self.id.weight, std=0.1)
        self.norm = nn.LayerNorm(d)
        self.item_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        self.user_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d))
        self.log_scale = nn.Parameter(torch.tensor(math.log(14.0)))

    def encode_items(self):
        h = 0
        for k in self.names:
            z = self.proj[k](getattr(self, f"f_{k}"))
            if self.training and self.mod_drop > 0:
                z = z * (torch.rand(z.size(0), 1, device=z.device) > self.mod_drop)
            h = h + z
        if self.id is not None:
            e = self.id.weight
            if self.training and self.id_drop > 0:
                e = e * (torch.rand(e.size(0), 1, device=e.device) > self.id_drop)
            h = h + e
        out = F.normalize(self.item_mlp(self.norm(h)), dim=-1)
        return torch.cat([out, out.new_zeros(1, out.size(1))])

    def encode_users(self, IR, hist, mask):
        m = mask.unsqueeze(-1).to(IR.dtype)
        x = (IR[hist] * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.user_mlp(x), dim=-1)
