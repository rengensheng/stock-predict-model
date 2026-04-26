"""
Training script with early stopping, LR scheduling, and metric tracking.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_model
from utils import (
    setup_logger,
    set_seed,
    directional_accuracy,
    ic_metric,
    rank_ic_metric,
    plot_training_history,
)

logger = setup_logger("train")


class EarlyStopping:
    """早停：当验证指标不再提升时停止训练。"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value: Optional[float] = None
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda val, best: val < best - min_delta
        elif mode == "max":
            self.is_better = lambda val, best: val > best + min_delta
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, val_metric: float) -> bool:
        if self.best_value is None:
            self.best_value = val_metric
            return False

        if self.is_better(val_metric, self.best_value):
            self.best_value = val_metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


def evaluate_model(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Dict[str, float]:
    """在验证/测试集上评估，返回 loss, IC, RankIC, Directional Accuracy。"""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    avg_loss = total_loss / len(targets)
    ic = ic_metric(targets, preds)
    rank_ic = rank_ic_metric(targets, preds)
    dir_acc = directional_accuracy(targets, preds)

    return {
        "loss": avg_loss,
        "ic": ic,
        "rank_ic": rank_ic,
        "dir_acc": dir_acc,
        "preds": preds,
        "targets": targets,
    }


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    seed: int = 42,
):
    set_seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=True
    )
    early_stopper = EarlyStopping(patience=patience, mode="min")
    writer = SummaryWriter(log_dir)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ic": [],
        "val_dir_acc": [],
    }
    best_val_loss = float("inf")

    logger.info("Starting training ...")
    for epoch in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        avg_train_loss = np.mean(train_losses)

        # ---------- Validation ----------
        val_metrics = evaluate_model(model, val_loader, device)
        avg_val_loss = val_metrics["loss"]
        val_ic = val_metrics["ic"]
        val_dir_acc = val_metrics["dir_acc"]

        # ---------- Logging ----------
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_ic"].append(val_ic)
        history["val_dir_acc"].append(val_dir_acc)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/val_ic", val_ic, epoch)
        writer.add_scalar("Metrics/val_rank_ic", val_metrics["rank_ic"], epoch)
        writer.add_scalar("Metrics/val_dir_acc", val_dir_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val IC: {val_ic:.4f} | "
            f"Val RankIC: {val_metrics['rank_ic']:.4f} | "
            f"Val DirAcc: {val_dir_acc:.4f}"
        )

        # ---------- Scheduler & Checkpoint ----------
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_ic": val_ic,
                },
                ckpt_path,
            )
            logger.info(f"Saved best model to {ckpt_path}")

        if early_stopper(avg_val_loss):
            logger.info("Early stopping triggered.")
            break

    writer.close()

    # 保存训练历史图
    plot_training_history(
        history, save_path=os.path.join(log_dir, "training_history.png")
    )
    logger.info("Training completed.")
    return history
