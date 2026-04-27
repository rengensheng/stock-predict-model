"""
Main entry: full pipeline from data fetch → train → infer/backtest.
Run: python main.py [--model_type lstm|transformer] [--task classification|regression]
"""

import argparse
import os

import torch

import config as cfg
from data_pipeline import StockDataPipeline
from models import build_model
from train import train_model
from infer import load_best_model, evaluate_and_backtest
from utils import setup_logger

logger = setup_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Stock Return Prediction")
    parser.add_argument(
        "--model_type",
        type=str,
        default=cfg.MODEL_TYPE,
        choices=["lstm", "transformer"],
        help="Model architecture",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=cfg.SYMBOLS,
        help="List of stock symbols, e.g. sz000001 sh600519",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=cfg.TASK,
        choices=["classification", "regression"],
        help="Task type",
    )
    parser.add_argument(
        "--epochs", type=int, default=cfg.EPOCHS, help="Max training epochs"
    )
    parser.add_argument("--lr", type=float, default=cfg.LR, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=cfg.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--seq_len", type=int, default=cfg.SEQ_LEN, help="Sequence length"
    )
    parser.add_argument(
        "--step", type=int, default=cfg.STEP, help="Sliding window step"
    )
    parser.add_argument(
        "--horizon", type=int, default=cfg.HORIZON, help="Prediction horizon"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=cfg.HIDDEN_DIM, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=cfg.NUM_LAYERS, help="Number of layers"
    )
    parser.add_argument("--dropout", type=float, default=cfg.DROPOUT, help="Dropout")
    parser.add_argument(
        "--patience", type=int, default=cfg.PATIENCE, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        default=cfg.USE_ATTENTION,
        help="Use temporal attention in LSTM",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--skip_train", action="store_true", help="Skip training and only run inference"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = cfg.DEVICE
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {args}")

    # ---------------- 1. Data Pipeline ----------------
    logger.info("=" * 50)
    logger.info("BUILDING DATA PIPELINE")
    logger.info("=" * 50)
    pipeline = StockDataPipeline(
        symbols=args.symbols,
        start_date=cfg.START_DATE,
        end_date=cfg.END_DATE,
        horizon=args.horizon,
        seq_len=args.seq_len,
        step=args.step,
        batch_size=args.batch_size,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        task=args.task,
        label_threshold=cfg.LABEL_THRESHOLD,
    )
    pipeline.prepare()

    input_dim = len(pipeline.feature_cols)
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, "scaler.pkl")
    pipeline.save_scaler(scaler_path)

    # ---------------- 2. Model ----------------
    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "task": args.task,
    }
    if args.model_type == "lstm":
        model_kwargs["use_attention"] = args.use_attention
    elif args.model_type == "transformer":
        model_kwargs["nhead"] = cfg.NHEAD
        model_kwargs["dim_feedforward"] = cfg.DIM_FEEDFORWARD

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")

    if not args.skip_train:
        model = build_model(args.model_type, input_dim, **model_kwargs)
        logger.info(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")

        # ---------------- 3. Train ----------------
        logger.info("=" * 50)
        logger.info("STARTING TRAINING")
        logger.info("=" * 50)
        train_model(
            model=model,
            train_loader=pipeline.train_loader,
            val_loader=pipeline.val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=cfg.WEIGHT_DECAY,
            patience=args.patience,
            checkpoint_dir=cfg.CHECKPOINT_DIR,
            log_dir=cfg.LOG_DIR,
            seed=args.seed,
            task=args.task,
            use_focal_loss=cfg.USE_FOCAL_LOSS,
            focal_alpha=cfg.FOCAL_ALPHA,
            focal_gamma=cfg.FOCAL_GAMMA,
            focal_dynamic_gamma=cfg.FOCAL_DYNAMIC_GAMMA,
            focal_gamma_max=cfg.FOCAL_GAMMA_MAX,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
    else:
        logger.info("--skip_train is set, loading existing checkpoint ...")

    # ---------------- 4. Inference & Backtest ----------------
    logger.info("=" * 50)
    logger.info("INFERENCE & BACKTEST")
    logger.info("=" * 50)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_best_model(
        model_type=args.model_type,
        input_dim=input_dim,
        checkpoint_path=checkpoint_path,
        device=device,
        **model_kwargs,
    )

    evaluate_and_backtest(
        model, pipeline, split="val", device=device,
        result_dir=cfg.RESULT_DIR, task=args.task, threshold=cfg.CLASS_THRESHOLD,
    )
    evaluate_and_backtest(
        model, pipeline, split="test", device=device,
        result_dir=cfg.RESULT_DIR, task=args.task, threshold=cfg.CLASS_THRESHOLD,
    )

    logger.info("All done.")


if __name__ == "__main__":
    main()
