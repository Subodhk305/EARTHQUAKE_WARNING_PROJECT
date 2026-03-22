"""
CNN-LSTM Hybrid Neural Network for Seismic Waveform Feature Extraction.

Architecture:
  Waveform input (batch, channels, time_steps)
      │
  ┌───▼──────────────────────────────────┐
  │  1D CNN  (Conv1d → BN → ReLU → Pool) │  × 3 blocks
  └───────────────────────────────────────┘
      │
  ┌───▼──────────────────────────────────┐
  │  Bidirectional LSTM  (2 layers)       │
  └───────────────────────────────────────┘
      │
  ┌───▼──────────────────────────────────┐
  │  Attention Pooling                    │
  └───────────────────────────────────────┘
      │
  Feature Vector → XGBoost classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """Conv1D → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, pool: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class AttentionPooling(nn.Module):
    """Soft attention over LSTM time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq, hidden)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq, 1)
        return (lstm_out * weights).sum(dim=1)            # (batch, hidden)


class CNNLSTMModel(nn.Module):
    """
    Full CNN-LSTM model returning a feature embedding.
    Final classification is handled by XGBoost using these embeddings
    concatenated with structured historical features.
    """

    def __init__(
        self,
        n_channels: int = 3,        # Z, N, E seismic channels
        cnn_channels: list[int] = [32, 64, 128],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        feature_dim: int = 128,     # output embedding dimension
    ):
        super().__init__()

        # ── CNN blocks ────────────────────────────────────────────────────────
        cnn_blocks = []
        in_ch = n_channels
        for out_ch in cnn_channels:
            cnn_blocks.append(ConvBlock1D(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = AttentionPooling(lstm_out_dim)

        # ── Projection head ───────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Auxiliary head (supervised pre-training) ──────────────────────────
        # Predicts magnitude class (5 classes) during standalone training.
        # Not used during XGBoost inference.
        self.classifier = nn.Linear(feature_dim, 5)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embedding only (used by XGBoost pipeline)."""
        # x: (batch, n_channels, time_steps)
        cnn_out = self.cnn(x)                  # (batch, 128, T')
        cnn_out = cnn_out.permute(0, 2, 1)     # (batch, T', 128) for LSTM
        lstm_out, _ = self.lstm(cnn_out)       # (batch, T', hidden*2)
        pooled = self.attention(lstm_out)      # (batch, hidden*2)
        return self.proj(pooled)               # (batch, feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features) — used during training."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits, features
