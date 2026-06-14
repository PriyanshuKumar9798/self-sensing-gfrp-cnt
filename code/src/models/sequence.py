"""Phase 3 sequence models (PROJECT_PLAN §5 Phase 3 step 2-3).

A small causal 1-D TCN (3 dilated blocks, kernel 3, dilations 1/2/4, 32 channels,
residual; ~19k params) and a small LSTM (1 layer, hidden 64). Both expose a
shared backbone feeding four heads: DTF (Huber), load (MSE), deflection (MSE),
stage (cross-entropy). The TCN is causal by left-padding; the LSTM is causal by
construction. Both read out at the most recent timestep.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    """Dilated 1-D convolution with left-only padding (no future leakage)."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, 3, dilation)
        self.c2 = CausalConv1d(ch, ch, 3, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.relu(self.c1(x)))
        y = self.drop(F.relu(self.c2(y)))
        return F.relu(x + y)


class TCNBackbone(nn.Module):
    def __init__(self, f_in: int, ch: int = 32, dilations=(1, 2, 4)):
        super().__init__()
        self.inp = nn.Conv1d(f_in, ch, 1)
        self.blocks = nn.ModuleList(TCNBlock(ch, d) for d in dilations)
        self.out_dim = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        h = self.inp(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return h[:, :, -1]  # last (most recent) timestep -> (B, ch)


class LSTMBackbone(nn.Module):
    def __init__(self, f_in: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(f_in, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1]


class MultiTaskNet(nn.Module):
    """Shared backbone + four task heads."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        d = backbone.out_dim
        self.dtf = nn.Linear(d, 1)
        self.load = nn.Linear(d, 1)
        self.defl = nn.Linear(d, 1)
        self.stage = nn.Linear(d, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "dtf": self.dtf(h).squeeze(-1),
            "load": self.load(h).squeeze(-1),
            "defl": self.defl(h).squeeze(-1),
            "stage": self.stage(h),
        }


def make_model(kind: str, f_in: int) -> MultiTaskNet:
    if kind == "tcn":
        return MultiTaskNet(TCNBackbone(f_in))
    if kind == "lstm":
        return MultiTaskNet(LSTMBackbone(f_in))
    raise ValueError(f"unknown model kind: {kind}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
