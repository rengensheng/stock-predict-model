"""
Inference & Backtesting script.
Supports classification (probabilities) and regression.
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


def backtest_directional(
    df: pd.DataFrame,
    pred_col: str = "pred_prob",
    actual_col: str = "future_return",
    close_col: str = "close",
    fee_rate: float = 0.0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    基于预测概率构造简单策略回测。
    position = +1 if prob > threshold else 0 (可扩展为 -1)
    """
    df = df.copy()
    df["position"] = (df[pred_col] > threshold).astype(float)
    # 若 future_return 列存在，用 future_return 评估策略；否则用日收益近似
    if actual_col in df.columns:
        df["strategy_return"] = df["position"] * df[actual_col] - fee_rate * np.abs(
            df["position"] - df["position"].shift(1).fillna(0)
        )
    else:
        df["strategy_return"] = 0.0
    df["strategy_cum"] = df["strategy_return"].cumsum()

    first_close = df[close_col].iloc[0]
    df["buyhold_cum"] = df[close_col] / first_close - 1

    return df


def evaluate_and_backtest(
    model: nn.Module,
    pipeline: StockDataPipeline,
    split: str = "test",
    device: torch.device = torch.device("cpu"),
    result_dir: str = "./results",
    task: str = "regression",
    threshold: float = 0.5,
) -> Dict[str, float]:
    os.makedirs(result_dir, exist_ok=True)

    if split == "train":
        loader = pipeline.train_loader
        df_raw = pipeline.train_df_raw if hasattr(pipeline, "train_df_raw") else None
        dates = pipeline.train_dates
    elif split == "val":
        loader = pipeline.val_loader
        df_raw = pipeline.val_df_raw if hasattr(pipeline, "val_df_raw") else None
        dates = pipeline.val_dates
    else:
        loader = pipeline.test_loader
        df_raw = pipeline.test_df_raw if hasattr(pipeline, "test_df_raw") else None
        dates = pipeline.test_dates

    metrics = evaluate_model(model, loader, device, task=task)

    if task == "classification":
        probs = metrics["probs"]
        preds = metrics["preds"]
        targets = metrics["targets"]

        # 回测 DataFrame 对齐
        date_indices = list(range(pipeline.seq_len - 1, len(df_raw) if df_raw is not None else 0, pipeline.step))
        if df_raw is not None:
            bt_df = df_raw.iloc[date_indices].copy().reset_index(drop=True)
            bt_df = bt_df.iloc[: len(probs)].copy()
        else:
            bt_df = pd.DataFrame({"future_return": np.zeros(len(probs))})

        bt_df["pred_prob"] = probs
        bt_df["pred_label"] = preds
        bt_df["true_label"] = targets
        if len(dates) >= len(probs):
            bt_df["date"] = dates.iloc[: len(probs)].values
        else:
            bt_df["date"] = pd.date_range(start="2020-01-01", periods=len(probs))

        # 尽量取 close 和 future_return 用于回测
        if "close" not in bt_df.columns:
            bt_df["close"] = 1.0
        if "future_return" not in bt_df.columns:
            bt_df["future_return"] = 0.0

        # Use the optimal threshold found during evaluation
        optimal_th = metrics.get("threshold", threshold)
        bt_df_opt = backtest_directional(bt_df.copy(), threshold=optimal_th)
        bt_df_fixed = backtest_directional(bt_df.copy(), threshold=threshold)

        sr_opt = sharpe_ratio(bt_df_opt["strategy_return"].values)
        mdd_opt = max_drawdown(bt_df_opt["strategy_cum"].values + 1.0)
        sr_fixed = sharpe_ratio(bt_df_fixed["strategy_return"].values)
        mdd_fixed = max_drawdown(bt_df_fixed["strategy_cum"].values + 1.0)

        logger.info(f"========== {split.upper()} SET RESULTS ==========")
        logger.info(f"Loss            : {metrics['loss']:.6f}")
        logger.info(f"AUC             : {metrics['auc']:.4f}")
        logger.info(f"Accuracy        : {metrics['acc']:.4f}")
        logger.info(f"F1              : {metrics['f1']:.4f}")
        logger.info(f"Precision       : {metrics['precision']:.4f}")
        logger.info(f"Recall          : {metrics['recall']:.4f}")
        logger.info(f"Optimal Thresh  : {optimal_th:.4f}")
        logger.info(f"--- Strategy (prob > {optimal_th:.4f}, optimal) ---")
        logger.info(f"Cumulative Ret  : {bt_df_opt['strategy_cum'].iloc[-1]:.4f}")
        logger.info(f"Sharpe Ratio    : {sr_opt:.4f}")
        logger.info(f"Max Drawdown    : {mdd_opt:.4f}")
        logger.info(f"--- Strategy (prob > {threshold:.1f}, fixed) ---")
        logger.info(f"Cumulative Ret  : {bt_df_fixed['strategy_cum'].iloc[-1]:.4f}")
        logger.info(f"Sharpe Ratio    : {sr_fixed:.4f}")
        logger.info(f"Max Drawdown    : {mdd_fixed:.4f}")
        logger.info(f"--- Buy & Hold ---")
        logger.info(f"Cumulative Ret  : {bt_df['buyhold_cum'].iloc[-1]:.4f}")

        csv_path = os.path.join(result_dir, f"backtest_{split}.csv")
        bt_df_opt.to_csv(csv_path, index=False)
        fig_path = os.path.join(result_dir, f"backtest_{split}.png")
        plot_backtest(bt_df_opt, save_path=fig_path, task=task)

        return {
            "loss": metrics["loss"],
            "auc": metrics["auc"],
            "acc": metrics["acc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "optimal_threshold": optimal_th,
            "strategy_cum_ret": bt_df_opt["strategy_cum"].iloc[-1],
            "strategy_sharpe": sr_opt,
            "strategy_mdd": mdd_opt,
        }
    else:
        preds = metrics["preds"]
        targets = metrics["targets"]

        date_indices = list(range(pipeline.seq_len - 1, len(df_raw) if df_raw is not None else 0, pipeline.step))
        if df_raw is not None:
            bt_df = df_raw.iloc[date_indices].copy().reset_index(drop=True)
            bt_df = bt_df.iloc[: len(preds)].copy()
        else:
            bt_df = pd.DataFrame()

        bt_df["pred_return"] = preds
        bt_df["true_return"] = targets
        if len(dates) >= len(preds):
            bt_df["date"] = dates.iloc[: len(preds)].values
        else:
            bt_df["date"] = pd.date_range(start="2020-01-01", periods=len(preds))

        if "close" not in bt_df.columns:
            bt_df["close"] = 1.0
        if "future_return" not in bt_df.columns:
            bt_df["future_return"] = bt_df["true_return"]

        bt_df = backtest_directional(bt_df, pred_col="pred_return", threshold=0.0)

        sr = sharpe_ratio(bt_df["strategy_return"].values)
        mdd = max_drawdown(bt_df["strategy_cum"].values + 1.0)
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

        csv_path = os.path.join(result_dir, f"backtest_{split}.csv")
        bt_df.to_csv(csv_path, index=False)
        fig_path = os.path.join(result_dir, f"backtest_{split}.png")
        plot_backtest(bt_df, save_path=fig_path, task=task)

        return {
            "loss": metrics["loss"],
            "ic": metrics["ic"],
            "rank_ic": metrics["rank_ic"],
            "dir_acc": metrics["dir_acc"],
            "strategy_cum_ret": bt_df["strategy_cum"].iloc[-1],
            "strategy_sharpe": sr,
            "strategy_mdd": mdd,
        }
