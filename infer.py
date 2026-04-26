"""
Inference & Backtesting script.
Supports:
  - Loading best checkpoint
  - Running predictions on validation / test sets
  - Simple directional backtest (sign-based position)
  - Metrics: IC, RankIC, Directional Accuracy, Sharpe, Max Drawdown
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import build_model
from data_pipeline import StockDataPipeline
from train import evaluate_model
from utils import (
    setup_logger,
    directional_accuracy,
    ic_metric,
    rank_ic_metric,
    sharpe_ratio,
    max_drawdown,
    plot_backtest,
)

logger = setup_logger("infer")


def load_best_model(
    model_type: str,
    input_dim: int,
    checkpoint_path: str,
    device: torch.device,
    **model_kwargs,
) -> nn.Module:
    model = build_model(model_type, input_dim, **model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss={checkpoint.get('val_loss', '?'):.6f}"
    )
    return model


def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """返回展平后的预测值 numpy 数组。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()


def backtest_directional(
    df: pd.DataFrame,
    pred_col: str = "pred_return",
    actual_col: str = "future_return",
    close_col: str = "close",
    fee_rate: float = 0.0,
) -> pd.DataFrame:
    """
    基于预测方向构造简单策略回测。
    position = sign(pred_return)
    策略收益 = position * future_return（假设能拿到该段未来收益，用于评估预测能力）
    同时计算基于 close 的 Buy & Hold 累计收益作为基准。

    注意：future_return 在 horizon > 1 时存在时间重叠，此处结果主要用于评估模型区分度，
    而非精确的最终资金曲线。如需精确回测，需使用事件驱动框架按 horizon 调仓。
    """
    df = df.copy()
    df["position"] = np.sign(df[pred_col])
    df["strategy_return"] = df["position"] * df[actual_col] - fee_rate * np.abs(
        df["position"] - df["position"].shift(1).fillna(0)
    )
    df["strategy_cum"] = df["strategy_return"].cumsum()

    # Buy & Hold 用 close 计算真实累计收益（更严谨）
    first_close = df[close_col].iloc[0]
    df["buyhold_cum"] = df[close_col] / first_close - 1

    return df


def evaluate_and_backtest(
    model: nn.Module,
    pipeline: StockDataPipeline,
    split: str = "test",
    device: torch.device = torch.device("cpu"),
    result_dir: str = "./results",
) -> Dict[str, float]:
    """
    在指定数据集（val / test）上进行推理 + 回测 + 指标打印。
    """
    os.makedirs(result_dir, exist_ok=True)

    if split == "train":
        loader = pipeline.train_loader
        df_split = pipeline.train_df
        dates = pipeline.train_dates
    elif split == "val":
        loader = pipeline.val_loader
        df_split = pipeline.val_df
        dates = pipeline.val_dates
    else:
        loader = pipeline.test_loader
        df_split = pipeline.test_df
        dates = pipeline.test_dates

    # 1) 评估指标
    metrics = evaluate_model(model, loader, device)
    preds = metrics["preds"]
    targets = metrics["targets"]

    # 2) 构造回测 DataFrame
    # 序列起始位置 i 按 step 跳跃，末尾日期索引 = i + seq_len - 1
    date_indices = list(range(pipeline.seq_len - 1, len(df_split), pipeline.step))
    bt_df = df_split.iloc[date_indices].copy().reset_index(drop=True)
    bt_df = bt_df.iloc[: len(preds)].copy()  # 安全对齐
    bt_df["pred_return"] = preds
    bt_df["true_return"] = targets
    bt_df["date"] = dates.iloc[: len(preds)].values

    # 3) 回测
    bt_df = backtest_directional(bt_df)

    # 4) 计算高阶指标
    sr = sharpe_ratio(bt_df["strategy_return"].values)
    mdd = max_drawdown(bt_df["strategy_cum"].values + 1.0)  # 传净值序列
    buyhold_sr = sharpe_ratio(bt_df["true_return"].values)
    buyhold_mdd = max_drawdown(bt_df["buyhold_cum"].values + 1.0)

    logger.info(f"========== {split.upper()} SET RESULTS ==========")
    logger.info(f"MSE Loss        : {metrics['loss']:.6f}")
    logger.info(f"IC              : {metrics['ic']:.4f}")
    logger.info(f"Rank IC         : {metrics['rank_ic']:.4f}")
    logger.info(f"Directional Acc : {metrics['dir_acc']:.4f}")
    logger.info(f"--- Strategy (sign-based) ---")
    logger.info(f"Cumulative Ret  : {bt_df['strategy_cum'].iloc[-1]:.4f}")
    logger.info(f"Sharpe Ratio    : {sr:.4f}")
    logger.info(f"Max Drawdown    : {mdd:.4f}")
    logger.info(f"--- Buy & Hold ---")
    logger.info(f"Cumulative Ret  : {bt_df['buyhold_cum'].iloc[-1]:.4f}")
    logger.info(f"Sharpe Ratio    : {buyhold_sr:.4f}")
    logger.info(f"Max Drawdown    : {buyhold_mdd:.4f}")

    # 5) 保存结果
    csv_path = os.path.join(result_dir, f"backtest_{split}.csv")
    bt_df.to_csv(csv_path, index=False)
    logger.info(f"Backtest details saved to {csv_path}")

    fig_path = os.path.join(result_dir, f"backtest_{split}.png")
    plot_backtest(bt_df, save_path=fig_path)
    logger.info(f"Backtest plot saved to {fig_path}")

    return {
        "loss": metrics["loss"],
        "ic": metrics["ic"],
        "rank_ic": metrics["rank_ic"],
        "dir_acc": metrics["dir_acc"],
        "strategy_cum_ret": bt_df["strategy_cum"].iloc[-1],
        "strategy_sharpe": sr,
        "strategy_mdd": mdd,
        "buyhold_cum_ret": bt_df["buyhold_cum"].iloc[-1],
        "buyhold_sharpe": buyhold_sr,
        "buyhold_mdd": buyhold_mdd,
    }
