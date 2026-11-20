"""
PyTorch LSTM Architecture for Longitudinal Vitals & Biomarker Streams.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VitalsLSTM(nn.Module):
    """
    2-layer LSTM encoder over (batch, seq_len, n_features) longitudinal vitals/biomarkers,
    producing a 30-day MACE risk logit.
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 48, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)
