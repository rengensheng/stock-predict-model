"""
Data Pipeline for A-Share Stock Prediction
Includes: fetch, feature engineering, target generation, sequence creation,
temporal split, rolling standardization (no lookahead).
"""

import os
import pickle
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger("data_pipeline")

# ---------------------------------------------------------------------------
# Optional third-party data sources (soft dependencies)
# ---------------------------------------------------------------------------
try:
    import akshare as ak
except Exception:
    ak = None

try:
    import yfinance as yf
except Exception:
    yf = None


# ---------------------------------------------------------------------------
# Helpers: symbol conversion & synthetic data
# ---------------------------------------------------------------------------

def _to_yfinance_ticker(symbol: str) -> str:
    """
    将 A 股代码转为 yfinance ticker。
    sz000001 -> 000001.SZ,  sh600519 -> 600519.SS
    """
    symbol = symbol.strip().lower()
    if symbol.startswith("sz"):
        return symbol[2:] + ".SZ"
    elif symbol.startswith("sh"):
        return symbol[2:] + ".SS"
    return symbol.upper()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为小写英文。"""
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
        "Date": "date",
        "date": "date",
        "Open": "open",
        "open": "open",
        "Close": "close",
        "close": "close",
        "High": "high",
        "high": "high",
        "Low": "low",
        "low": "low",
        "Volume": "volume",
        "volume": "volume",
        "Amount": "amount",
        "amount": "amount",
        "Turnover": "turnover",
        "turnover": "turnover",
    }
    df = df.rename(columns=col_map)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns after rename: {missing}. "
            f"Original columns: {df.columns.tolist()}."
        )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main fetcher with cascading fallbacks
# ---------------------------------------------------------------------------

def fetch_stock_daily(
    symbol: str = "sz000001",
    start_date: str = "20150101",
    end_date: str = "20231231",
    cache_dir: str = "./data_cache",
) -> pd.DataFrame:
    """
    获取日线数据，带三层回退：
      1) 本地缓存
      2) akshare（免费 A 股）
      3) yfinance（Yahoo Finance，境外可用）
      4) 合成数据（完全离线，保证 demo 可跑）
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"{symbol}_{start_date}_{end_date}.csv"
    )

    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    df = None
    errors = []

    # 1) akshare
    if ak is not None:
        try:
            logger.info(f"Fetching {symbol} from akshare ...")
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            df = _normalize_columns(df)
            logger.info(f"akshare OK, rows={len(df)}")
        except Exception as e:
            errors.append(f"akshare: {e}")
            logger.warning(f"akshare failed: {e}")

    # 2) yfinance
    if df is None and yf is not None:
        try:
            ticker = _to_yfinance_ticker(symbol)
            logger.info(f"Fetching {ticker} from yfinance ...")
            tk = yf.Ticker(ticker)
            df = tk.history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                raise ValueError("yfinance returned empty DataFrame")
            df = df.reset_index()
            df = _normalize_columns(df)
            logger.info(f"yfinance OK, rows={len(df)}")
        except Exception as e:
            errors.append(f"yfinance: {e}")
            logger.warning(f"yfinance failed: {e}")

    if df is None:
        raise RuntimeError(
            f"All live data sources failed for {symbol}. "
            f"Errors: {errors}. "
            f"Please check your network, proxy settings, or install required libraries."
        )

    df.to_csv(cache_file, index=False)
    logger.info(f"Saved cache to {cache_file}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加技术指标与量价特征，严格避免未来函数。
    所有特征在时刻 t 仅使用 [0, t] 信息。
    """
    df = df.copy()

    # 基础收益率
    df["ret_1"] = df["close"].pct_change()

    # 滞后收益率
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)

    # 移动平均与偏离度
    for window in [5, 10, 20, 60]:
        ma = df["close"].rolling(window).mean()
        df[f"ma_{window}"] = ma
        df[f"close_div_ma_{window}"] = df["close"] / (ma + 1e-12) - 1

    # 历史波动率
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["ret_1"].rolling(window).std()

    # 成交量特征
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma5"] + 1e-12)
    if "turnover" not in df.columns:
        df["turnover"] = 0.0

    # ta 库技术指标（内部已做滚动，符合因果性）
    df["rsi_14"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(
        close=df["close"], window=20, window_dev=2
    )
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["close"] + 1e-12)

    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # 价格形态
    df["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
    df["close_position"] = (df["close"] - df["low"]) / (
        df["high"] - df["low"] + 1e-12
    )

    # 剔除因滚动计算产生的初始 NaN
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    未来 horizon 日收益率: (close_{t+horizon} / close_t) - 1
    """
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df.dropna(subset=["future_return"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_sequences(
    data: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
    target_col: str = "future_return",
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口构造 (seq_len, num_features) 样本，支持步长 step 以扩充样本量。
    对于窗口 data[i : i+seq_len]，标签取窗口末尾行 target。
    即：用 [t-seq_len+1, t] 预测未来 horizon 日收益。
    """
    X, y = [], []
    feat_matrix = data[feature_cols].values
    target_vec = data[target_col].values

    for i in range(0, len(data) - seq_len + 1, step):
        X.append(feat_matrix[i : i + seq_len])
        y.append(target_vec[i + seq_len - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class StockDataPipeline:
    """
    端到端数据管道：特征列管理 + 训练/验证/测试划分 + 标准化 + DataLoader 生成。
    """

    def __init__(
        self,
        symbol: str = "sz000001",
        start_date: str = "20150101",
        end_date: str = "20240101",
        horizon: int = 5,
        seq_len: int = 60,
        step: int = 1,
        batch_size: int = 128,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        cache_dir: str = "./data_cache",
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.horizon = horizon
        self.seq_len = seq_len
        self.step = step
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.cache_dir = cache_dir

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def prepare(self) -> "StockDataPipeline":
        """执行完整数据准备流程。"""
        logger.info("Step 1/5: Fetching raw data ...")
        df = fetch_stock_daily(
            self.symbol, self.start_date, self.end_date, self.cache_dir
        )

        logger.info("Step 2/5: Adding features ...")
        df = add_features(df)

        logger.info("Step 3/5: Adding target ...")
        df = add_target(df, horizon=self.horizon)

        # 定义特征列（排除原始价格、标签、日期等）
        exclude = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "future_return",
        }
        self.feature_cols = [c for c in df.columns if c not in exclude]
        logger.info(f"Feature count: {len(self.feature_cols)}")
        logger.info(f"Samples after engineering: {len(df)}")

        logger.info("Step 4/5: Temporal split & rolling standardization ...")
        self._split_and_scale(df)

        logger.info("Step 5/5: Building DataLoaders ...")
        self._build_loaders()

        return self

    def _split_and_scale(self, df: pd.DataFrame):
        """
        按时序划分，并用训练集统计量标准化全部集合。
        验证集/测试集开头多取 seq_len-1 行，用于构造最早期的序列样本。
        """
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # 训练集
        train_df = df.iloc[:train_end].copy()
        # 验证集：需要包含训练集尾部 seq_len-1 行作为历史上下文
        val_df = df.iloc[max(0, train_end - self.seq_len + 1) : val_end].copy()
        # 测试集：同理
        test_df = df.iloc[max(0, val_end - self.seq_len + 1) :].copy()

        # 仅在训练集上 fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[self.feature_cols].values)

        # transform
        train_df.loc[:, self.feature_cols] = self.scaler.transform(
            train_df[self.feature_cols].values
        )
        val_df.loc[:, self.feature_cols] = self.scaler.transform(
            val_df[self.feature_cols].values
        )
        test_df.loc[:, self.feature_cols] = self.scaler.transform(
            test_df[self.feature_cols].values
        )

        # 生成序列（支持步长 step 以扩充样本量）
        self.X_train, self.y_train = create_sequences(
            train_df, self.feature_cols, self.seq_len, step=self.step
        )
        self.X_val, self.y_val = create_sequences(
            val_df, self.feature_cols, self.seq_len, step=self.step
        )
        self.X_test, self.y_test = create_sequences(
            test_df, self.feature_cols, self.seq_len, step=self.step
        )

        # 记录各集对应的日期索引（用于后续回测对齐）
        # 序列末尾日期 = i + seq_len - 1，其中 i 按 step 跳跃
        self.train_dates = train_df["date"].iloc[
            list(range(self.seq_len - 1, len(train_df), self.step))
        ].reset_index(drop=True)
        self.val_dates = val_df["date"].iloc[
            list(range(self.seq_len - 1, len(val_df), self.step))
        ].reset_index(drop=True)
        self.test_dates = test_df["date"].iloc[
            list(range(self.seq_len - 1, len(test_df), self.step))
        ].reset_index(drop=True)

        # 保存分割后的 DataFrame（含原始 close 等），供回测使用
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        logger.info(
            f"Train/Val/Test samples: "
            f"{len(self.y_train)}/{len(self.y_val)}/{len(self.y_test)}"
        )

    def _build_loaders(self):
        train_ds = TensorDataset(
            torch.from_numpy(self.X_train),
            torch.from_numpy(self.y_train).unsqueeze(1),
        )
        val_ds = TensorDataset(
            torch.from_numpy(self.X_val),
            torch.from_numpy(self.y_val).unsqueeze(1),
        )
        test_ds = TensorDataset(
            torch.from_numpy(self.X_test),
            torch.from_numpy(self.y_test).unsqueeze(1),
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def save_scaler(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "feature_cols": self.feature_cols},
                f,
            )
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.feature_cols = obj["feature_cols"]
        logger.info(f"Scaler loaded from {path}")
