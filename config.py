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
LABEL_THRESHOLD = 0.01       # 分类任务中，|future_return| < threshold 的样本标记为 ignore
USE_FOCAL_LOSS = True        # 使用 Focal Loss 处理难分类样本
FOCAL_ALPHA = None           # Focal Loss alpha: None=auto from class ratio, or set e.g. 0.25
FOCAL_GAMMA = 2.0            # Focal Loss gamma 参数
LABEL_SMOOTHING = 0.1        # BCEWithLogitsLoss 的标签平滑参数

# ---------------- Model ----------------
MODEL_TYPE = "lstm"          # "lstm" or "transformer"
INPUT_DIM = None
HIDDEN_DIM = 128             # 隐藏层维度
NUM_LAYERS = 2               # 层数
DROPOUT = 0.5                # dropout (提高以对抗过拟合)
USE_ATTENTION = True         # LSTM 是否加 Attention

# Transformer specific
NHEAD = 8
DIM_FEEDFORWARD = 512

# ---------------- Training ----------------
EPOCHS = 100
LR = 5e-4                    # 降低学习率
WEIGHT_DECAY = 1e-2          # 增加正则化
PATIENCE = 15                # 增加早停耐心
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Paths ----------------
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
RESULT_DIR = "./results"
