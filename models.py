"""
Models: StockLSTM (with optional Attention) and StockTransformer
Supports both regression and classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Self-attention over LSTM outputs to aggregate sequence info."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        scores = self.attn(x).squeeze(-1)          # (batch, seq_len)
        weights = F.softmax(scores, dim=1)         # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False,
        task: str = "regression",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        self.task = task

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        feat_dim = hidden_dim * self.num_directions

        if use_attention:
            self.attention = TemporalAttention(feat_dim)
            self.norm = nn.LayerNorm(feat_dim)
        else:
            self.norm = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(dropout)

        out_dim = 1 if task == "regression" else 1  # classification uses 1 logit
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, feat_dim)

        if self.use_attention:
            last, _ = self.attention(lstm_out)
        else:
            last = lstm_out[:, -1, :]

        last = self.norm(last)
        last = self.dropout(last)
        out = self.fc(last)         # (batch, 1)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class StockTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 5000,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        out_dim = 1 if task == "regression" else 1
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        out = self.fc(x)
        return out


def build_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    model_type = model_type.lower()
    task = kwargs.get("task", "regression")
    if model_type == "lstm":
        return StockLSTM(
            input_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
            use_attention=kwargs.get("use_attention", False),
            task=task,
        )
    elif model_type == "transformer":
        return StockTransformer(
            input_dim=input_dim,
            d_model=kwargs.get("hidden_dim", 128),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 256),
            dropout=kwargs.get("dropout", 0.2),
            task=task,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
