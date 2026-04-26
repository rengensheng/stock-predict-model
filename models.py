"""
Models: StockLSTM and StockTransformer for return prediction.
"""

import math
import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Multi-layer LSTM for time-series regression.
    Output: predicted future return (scalar).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)
        last = lstm_out[:, -1, :]   # 取最后时刻
        last = self.norm(last)
        last = self.dropout(last)
        out = self.fc(last)         # (batch, 1)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    """经典正弦位置编码。"""

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
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class StockTransformer(nn.Module):
    """
    Transformer Encoder for time-series regression.
    Uses mean pooling over the sequence and an MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 5000,
    ):
        super().__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # (batch, seq_len, d_model)
        # mean pooling
        x = x.mean(dim=1)               # (batch, d_model)
        x = self.norm(x)
        x = self.dropout(x)
        out = self.fc(x)                # (batch, 1)
        return out


def build_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "lstm":
        return StockLSTM(
            input_dim=input_dim,
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
        )
    elif model_type == "transformer":
        return StockTransformer(
            input_dim=input_dim,
            d_model=kwargs.get("hidden_dim", 128),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 256),
            dropout=kwargs.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
