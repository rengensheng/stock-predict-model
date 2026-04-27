"""
Models: StockLSTM (with optional Attention) and StockTransformer
Supports both regression and classification.
Enhanced with feature grouping, residual connections, and financial-specific architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGroupEncoder(nn.Module):
    """
    将特征按组分别编码，然后拼接。
    金融数据中不同指标（量价、均线、动量等）有不同的特性，分组编码可以更好的学习。
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_groups: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_groups = num_groups
        self.group_dim = hidden_dim // num_groups
        
        # 计算每组的输入维度，均匀分配余数
        base_input = input_dim // num_groups
        remainder = input_dim % num_groups
        
        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_input + (1 if i < remainder else 0), self.group_dim),
                nn.LayerNorm(self.group_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for i in range(num_groups)
        ])
        
        # 拼接后的实际维度（可能小于 hidden_dim，需投影）
        concat_dim = self.group_dim * num_groups
        self.cross_group = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 残差连接投影（确保输入输出维度一致）
        self.res_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        grouped_outputs = []
        start_idx = 0
        
        for i, encoder in enumerate(self.group_encoders):
            group_size = encoder[0].in_features
            end_idx = start_idx + group_size
            group_input = x[:, :, start_idx:end_idx]
            grouped_outputs.append(encoder(group_input))
            start_idx = end_idx
        
        # 拼接各组输出
        concatenated = torch.cat(grouped_outputs, dim=-1)
        
        # 组间交互
        output = self.cross_group(concatenated)
        
        # 残差连接
        output = output + self.res_proj(x)
        
        return output


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


class MultiHeadTemporalAttention(nn.Module):
    """多头注意力机制，捕捉不同子空间的时序依赖"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        residual = x
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        
        output = self.out_proj(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False,
        use_feature_grouping: bool = True,
        task: str = "regression",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        self.use_feature_grouping = use_feature_grouping
        self.task = task

        # 特征分组编码（可选）
        if use_feature_grouping:
            self.feature_encoder = FeatureGroupEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_groups=5,
                dropout=dropout
            )
            lstm_input_dim = hidden_dim
        else:
            self.feature_encoder = None
            lstm_input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        feat_dim = hidden_dim * self.num_directions

        if use_attention:
            self.attention = MultiHeadTemporalAttention(feat_dim, num_heads=4)
        
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

        # 更深的输出层
        out_dim = 1 if task == "regression" else 1  # classification uses 1 logit
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征分组编码
        if self.use_feature_grouping and self.feature_encoder is not None:
            x = self.feature_encoder(x)
        
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, feat_dim)

        if self.use_attention:
            last = self.attention(lstm_out)[:, -1, :]
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
        use_feature_grouping: bool = True,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        self.use_feature_grouping = use_feature_grouping
        
        # 特征分组编码（可选）
        if use_feature_grouping:
            self.feature_encoder = FeatureGroupEncoder(
                input_dim=input_dim,
                hidden_dim=d_model,
                num_groups=5,
                dropout=dropout
            )
            self.input_proj = nn.Linear(d_model, d_model)
        else:
            self.feature_encoder = None
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
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征分组编码
        if self.use_feature_grouping and self.feature_encoder is not None:
            x = self.feature_encoder(x)
        
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
            use_feature_grouping=kwargs.get("use_feature_grouping", True),
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
            use_feature_grouping=kwargs.get("use_feature_grouping", True),
            task=task,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
