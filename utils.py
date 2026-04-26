"""
Utilities: metrics, plotting, logging, seeding
"""

import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch


def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def directional_accuracy(y_true, y_pred):
    """计算方向准确率，输入为 numpy 数组"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    valid = sign_true != 0
    if valid.sum() == 0:
        return 0.0
    return np.mean(sign_true[valid] == sign_pred[valid])


def ic_metric(y_true, y_pred):
    """计算 Pearson IC (信息系数)"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]


def rank_ic_metric(y_true, y_pred):
    """计算 Rank IC (Spearman)"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    import pandas as pd
    s1 = pd.Series(y_true)
    s2 = pd.Series(y_pred)
    return s1.corr(s2, method="spearman")


def sharpe_ratio(returns, annual_factor=252):
    """计算年化夏普比率（假设 returns 是日收益率）"""
    returns = np.asarray(returns)
    if returns.std() == 0:
        return 0.0
    return returns.mean() / (returns.std() + 1e-12) * np.sqrt(annual_factor)


def max_drawdown(cum_returns):
    """计算最大回撤"""
    cum = np.asarray(cum_returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / (peak + 1e-12)
    return drawdown.min()


def plot_training_history(history, save_path=None):
    """history: dict with keys 'train_loss','val_loss','val_ic','val_dir_acc'"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["val_ic"], label="Val IC")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Validation IC")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[2]
    ax.plot(epochs, history["val_dir_acc"], label="Val Dir Acc")
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_title("Validation Directional Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def auc_metric(y_true, y_score):
    """计算 AUC，y_score 为概率或 logit"""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


def f1_metric(y_true, y_pred):
    """计算 F1"""
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return f1_score(y_true, y_pred, zero_division=0)


def precision_metric(y_true, y_pred):
    from sklearn.metrics import precision_score
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return precision_score(y_true, y_pred, zero_division=0)


def recall_metric(y_true, y_pred):
    from sklearn.metrics import recall_score
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return recall_score(y_true, y_pred, zero_division=0)


def plot_training_history(history, save_path=None, task="regression"):
    """history: dict with keys 'train_loss','val_loss', plus task-specific metrics"""
    n_plots = 3 if task == "regression" else 4
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    if task == "regression":
        ax = axes[1]
        ax.plot(epochs, history["val_ic"], label="Val IC")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("Validation IC")
        ax.set_xlabel("Epoch")
        ax.legend()

        ax = axes[2]
        ax.plot(epochs, history["val_dir_acc"], label="Val Dir Acc")
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_title("Validation Directional Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend()
    else:
        ax = axes[1]
        ax.plot(epochs, history["val_auc"], label="Val AUC")
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_title("Validation AUC")
        ax.set_xlabel("Epoch")
        ax.legend()

        ax = axes[2]
        ax.plot(epochs, history["val_acc"], label="Val Acc")
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_title("Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend()

        ax = axes[3]
        ax.plot(epochs, history["val_f1"], label="Val F1")
        ax.set_title("Validation F1")
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_backtest(df_backtest, save_path=None, task="regression"):
    """df_backtest 需要包含 'date', 'strategy_cum', 'buyhold_cum' 列"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(df_backtest["date"], df_backtest["buyhold_cum"], label="Buy & Hold")
    ax.plot(df_backtest["date"], df_backtest["strategy_cum"], label="Strategy")
    ax.set_title("Cumulative Return")
    ax.legend()
    ax.set_ylabel("Cumulative Return")

    ax = axes[1]
    if task == "regression":
        colors = ["green" if p > 0 else "red" for p in df_backtest["pred_return"]]
        ax.bar(df_backtest["date"], df_backtest["pred_return"], color=colors, alpha=0.6, width=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Predicted Return")
        ax.set_ylabel("Predicted Return")
    else:
        colors = ["green" if p > 0.5 else "red" for p in df_backtest["pred_prob"]]
        ax.bar(df_backtest["date"], df_backtest["pred_prob"], color=colors, alpha=0.6, width=1.0)
        ax.axhline(0.5, color="black", linewidth=0.5)
        ax.set_title("Predicted Probability (Up)")
        ax.set_ylabel("Probability")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
