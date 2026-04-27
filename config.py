"""
Stock Prediction Model Configuration
"""

import torch

# ---------------- Data ----------------
SYMBOLS = ["sh600552"]
START_DATE = "20150101"
END_DATE = "20260426"
HORIZON = 3                 # 预测未来 5 日涨跌（增加预测周期以过滤噪声）
SEQ_LEN = 60                 # 序列长度（增加到 60 天以捕捉更长期的模式）
STEP = 5                     # 滑动窗口步长（增加步长以减少样本间的相关性）
BATCH_SIZE = 128             # 增大 batch size 以稳定训练
NUM_WORKERS = 0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------- Task ----------------
TASK = "classification"      # "classification" (up/down) or "regression" (return)
CLASS_THRESHOLD = 0.5        # 分类概率阈值
LABEL_THRESHOLD = 0.02       # 分类任务中，|future_return| < threshold 的样本标记为 ignore（提高阈值以过滤噪声）
USE_FOCAL_LOSS = True        # 使用 Focal Loss 处理难分类样本
FOCAL_ALPHA = None           # Focal Loss alpha: None=auto from class ratio, or set e.g. 0.25
FOCAL_GAMMA = 2.0            # Focal Loss gamma 基础参数
FOCAL_DYNAMIC_GAMMA = True   # 是否根据涨跌比动态调整 gamma（类别越不平衡，gamma 越大）
FOCAL_GAMMA_MAX = 5.0        # gamma 上限
LABEL_SMOOTHING = 0.1        # BCEWithLogitsLoss 的标签平滑参数

# ---------------- Model ----------------
MODEL_TYPE = "lstm"          # "lstm" or "transformer"
INPUT_DIM = None
HIDDEN_DIM = 256             # 增加隐藏层维度以增强模型表达能力
NUM_LAYERS = 3               # 增加层数
DROPOUT = 0.3                # 适当降低 dropout 以允许模型学习更多模式
USE_ATTENTION = True         # LSTM 是否加 Attention
USE_FEATURE_GROUPING = True  # 是否使用特征分组编码

# Transformer specific
NHEAD = 8
DIM_FEEDFORWARD = 512

# ---------------- Training ----------------
EPOCHS = 150                 # 增加训练轮数
LR = 1e-3                    # 提高学习率
WEIGHT_DECAY = 5e-3          # 适当降低权重衰减
PATIENCE = 20                # 增加早停耐心
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Paths ----------------
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
RESULT_DIR = "./results"
