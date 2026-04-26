"""
Stock Prediction Model Configuration
"""

import torch

# ---------------- Data ----------------
SYMBOLS = ["sz000001", "sh600519", "sz000858", "sh600036", "sz002415"]
START_DATE = "20150101"
END_DATE = "20260426"
HORIZON = 5                  # 预测未来 N 日涨跌
SEQ_LEN = 60                 # 序列长度
STEP = 1                     # 滑动窗口步长
BATCH_SIZE = 64              # 减小 batch size 增加梯度更新频率
NUM_WORKERS = 0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------- Task ----------------
TASK = "classification"      # "classification" (up/down) or "regression" (return)
CLASS_THRESHOLD = 0.5        # 分类概率阈值
USE_FOCAL_LOSS = True        # 使用 Focal Loss 处理难分类样本
FOCAL_ALPHA = 0.25           # Focal Loss alpha 参数
FOCAL_GAMMA = 2.0            # Focal Loss gamma 参数

# ---------------- Model ----------------
MODEL_TYPE = "lstm"          # "lstm" or "transformer"
INPUT_DIM = None
HIDDEN_DIM = 256             # 增加隐藏层维度
NUM_LAYERS = 3               # 增加层数
DROPOUT = 0.3                # 增加 dropout
USE_ATTENTION = True         # LSTM 是否加 Attention

# Transformer specific
NHEAD = 8
DIM_FEEDFORWARD = 512

# ---------------- Training ----------------
EPOCHS = 100
LR = 5e-4                    # 降低学习率
WEIGHT_DECAY = 1e-3          # 增加正则化
PATIENCE = 15                # 增加早停耐心
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Paths ----------------
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
RESULT_DIR = "./results"
