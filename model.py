#!/usr/bin/env python3
"""Minimal model module for inference server.

The live draft server imports MLPScorer from here and loads weights from model_mlp.pt.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPScorer(nn.Module):
    # Simple MLP scorer that outputs one scalar score per candidate row.
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # Return per-row scores for a batch/list of candidate feature vectors.
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)
