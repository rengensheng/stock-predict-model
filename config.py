"""
Stock Prediction Model Configuration
"""

import torch

# ---------------- Data ----------------
SYMBOL = "sz000001"          # 平安银行
START_DATE = "20150101"
END_DATE = "20260101"
HORIZON = 1                # 预测未来5日收益率
SEQ_LEN = 60                 # 序列长度
STEP = 1                     # 滑动窗口步长（1或2可增加样本量）
BATCH_SIZE = 128
NUM_WORKERS = 0              # DataLoader workers

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15            # 自动校验: 1 - train - val

# ---------------- Model ----------------
MODEL_TYPE = "lstm"          # "lstm" or "transformer"
INPUT_DIM = None             # 自动根据特征数设置
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Transformer specific
NHEAD = 4
DIM_FEEDFORWARD = 256

# ---------------- Training ----------------
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10                # 早停耐心值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Paths ----------------
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
RESULT_DIR = "./results"
