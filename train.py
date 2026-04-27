"""
Training script with early stopping, LR scheduling, and metric tracking.
Supports both regression and classification.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    setup_logger,
    set_seed,
    directional_accuracy,
    ic_metric,
    rank_ic_metric,
    auc_metric,
    f1_metric,
    precision_metric,
    recall_metric,
    plot_training_history,
)

logger = setup_logger("train")


class FocalLoss(nn.Module):
    """Focal Loss for handling hard-to-classify samples.

    alpha: weight for the positive class (0 < alpha < 1).
           For balanced data, alpha should be close to the negative class ratio
           (i.e. 1 - positive_ratio), so that the minority class gets higher weight.
           If alpha is None, it will be set from data (1 - pos_ratio).
    gamma: focusing parameter. Higher gamma means more focus on hard samples.
    """

    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 1) raw logits
            targets: (batch, 1) binary targets {0, 1}
        """
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        else:
            alpha_weight = 1.0

        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


class EarlyStopping:
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
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str = "regression",
) -> Dict[str, float]:
    model.eval()
    all_logits, all_targets = [], []
    total_loss = 0.0

    if task == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    logits = np.concatenate(all_logits).flatten()
    targets = np.concatenate(all_targets).flatten()

    avg_loss = total_loss / len(targets)

    if task == "classification":
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        labels = targets.astype(int)

        # 使用最优阈值而非固定 0.5
        from sklearn.metrics import roc_curve
        if len(np.unique(labels)) >= 2:
            fpr, tpr, thresholds = roc_curve(labels, probs)
            # 找到使 F1 最大的阈值
            best_threshold = 0.5
            best_f1 = 0.0
            for th in thresholds:
                preds_th = (probs > th).astype(int)
                f1_th = f1_metric(labels, preds_th)
                if f1_th > best_f1:
                    best_f1 = f1_th
                    best_threshold = th
            preds = (probs > best_threshold).astype(int)
        else:
            best_threshold = 0.5
            preds = (probs > 0.5).astype(int)

        return {
            "loss": avg_loss,
            "auc": auc_metric(labels, probs),
            "acc": np.mean(preds == labels),
            "f1": f1_metric(labels, preds),
            "precision": precision_metric(labels, preds),
            "recall": recall_metric(labels, preds),
            "probs": probs,
            "preds": preds,
            "targets": labels,
            "threshold": best_threshold,
        }
    else:
        preds = logits
        return {
            "loss": avg_loss,
            "ic": ic_metric(targets, preds),
            "rank_ic": rank_ic_metric(targets, preds),
            "dir_acc": directional_accuracy(targets, preds),
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
    task: str = "regression",
    use_focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.0,
):
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)

    if task == "classification":
        # 统计训练集正负样本比例
        all_labels = []
        for _, y in train_loader:
            all_labels.extend(y.flatten().tolist())
        all_labels = np.array(all_labels)
        pos_ratio = all_labels.mean()
        neg_ratio = 1.0 - pos_ratio
        logger.info(f"Class balance: pos={pos_ratio:.4f}, neg={neg_ratio:.4f}")

        if use_focal_loss:
            # alpha: weight for positive class. Set to neg_ratio so minority gets more weight.
            # For balanced data (pos≈0.5), alpha≈0.5 — essentially neutral.
            if focal_alpha is None:
                computed_alpha = float(neg_ratio)
            else:
                computed_alpha = focal_alpha
            criterion = FocalLoss(alpha=computed_alpha, gamma=focal_gamma)
            logger.info(f"Using Focal Loss: alpha={computed_alpha:.4f} (pos_weight), gamma={focal_gamma}")

        # Initialize final classification bias to match prior, preventing all-negative logits
        try:
            last_linear = model.fc[-1]  # Sequential last layer
            if isinstance(last_linear, nn.Linear) and last_linear.out_features == 1:
                init_bias = np.log(pos_ratio / (neg_ratio + 1e-6))
                torch.nn.init.constant_(last_linear.bias, init_bias)
                logger.info(f"Initialized output bias to {init_bias:.4f} (log-odds of pos_ratio)")
        except Exception:
            pass
        else:
            pos_weight = torch.tensor(neg_ratio / (pos_ratio + 1e-6), dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info(f"Using BCEWithLogitsLoss: pos_weight={pos_weight.item():.4f}")
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task == "classification":
        # 分类按 AUC 早停
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=patience // 2
        )
        early_stopper = EarlyStopping(patience=patience, mode="max")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 2
        )
        early_stopper = EarlyStopping(patience=patience, mode="min")

    writer = SummaryWriter(log_dir)

    history = {"train_loss": [], "val_loss": []}
    if task == "classification":
        history.update({"val_auc": [], "val_acc": [], "val_f1": []})
        best_metric_name = "auc"
    else:
        history.update({"val_ic": [], "val_dir_acc": []})
        best_metric_name = "loss"

    best_val_metric = float("-inf") if task == "classification" else float("inf")

    logger.info("Starting training ...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        avg_train_loss = np.mean(train_losses)
        val_metrics = evaluate_model(model, val_loader, device, task=task)
        avg_val_loss = val_metrics["loss"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if task == "classification":
            val_auc = val_metrics["auc"]
            val_acc = val_metrics["acc"]
            val_f1 = val_metrics["f1"]
            val_threshold = val_metrics.get("threshold", 0.5)
            history["val_auc"].append(val_auc)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Metrics/val_auc", val_auc, epoch)
            writer.add_scalar("Metrics/val_acc", val_acc, epoch)
            writer.add_scalar("Metrics/val_f1", val_f1, epoch)
            writer.add_scalar("Metrics/val_threshold", val_threshold, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Threshold: {val_threshold:.4f}"
            )

            scheduler.step(val_auc)
            if val_auc > best_val_metric:
                best_val_metric = val_auc
                ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "val_auc": val_auc,
                    },
                    ckpt_path,
                )
                logger.info(f"Saved best model to {ckpt_path}")

            if early_stopper(val_auc):
                logger.info("Early stopping triggered.")
                break
        else:
            val_ic = val_metrics["ic"]
            val_dir_acc = val_metrics["dir_acc"]
            history["val_ic"].append(val_ic)
            history["val_dir_acc"].append(val_dir_acc)

            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Metrics/val_ic", val_ic, epoch)
            writer.add_scalar("Metrics/val_rank_ic", val_metrics["rank_ic"], epoch)
            writer.add_scalar("Metrics/val_dir_acc", val_dir_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | Val IC: {val_ic:.4f} | "
                f"Val RankIC: {val_metrics['rank_ic']:.4f} | Val DirAcc: {val_dir_acc:.4f}"
            )

            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_metric:
                best_val_metric = avg_val_loss
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
    plot_training_history(
        history, save_path=os.path.join(log_dir, "training_history.png"), task=task
    )
    logger.info("Training completed.")
    return history
