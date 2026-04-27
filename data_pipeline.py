"""
Data Pipeline for A-Share Stock Prediction (Multi-Symbol, Classification/Regression)
Includes: batch fetch via baostock, feature engineering, target generation,
sequence creation, temporal split, rolling standardization (no lookahead).
"""

import os
import pickle
import warnings
import atexit
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
# Data source: baostock
# ---------------------------------------------------------------------------
import baostock as bs

_login_done = False

def _ensure_login():
    global _login_done
    if not _login_done:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        _login_done = True
        logger.info("baostock login succeeded")
        atexit.register(bs.logout)


def _to_baostock_code(symbol: str) -> str:
    """sz000001 -> sz.000001"""
    symbol = symbol.strip().lower()
    if symbol.startswith("sz") or symbol.startswith("sh"):
        return symbol[:2] + "." + symbol[2:]
    raise ValueError(f"Invalid symbol format: {symbol}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名，确保包含标准字段。"""
    col_map = {
        "turn": "turnover",
    }
    df = df.rename(columns=col_map)
    # 数值列转 float
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns after rename: {missing}. Original columns: {df.columns.tolist()}."
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_stock_daily(
    symbol: str,
    start_date: str = "20150101",
    end_date: str = "20231231",
    cache_dir: str = "./data_cache",
) -> pd.DataFrame:
    """
    通过 baostock 获取 A 股日线（前复权）。
    start_date / end_date 格式支持 YYYYMMDD 或 YYYY-MM-DD。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_{start_date}_{end_date}.csv")

    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    # 格式化日期
    sd = start_date if "-" in start_date else f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    ed = end_date if "-" in end_date else f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    _ensure_login()
    code = _to_baostock_code(symbol)
    logger.info(f"Fetching {symbol} ({code}) from baostock ...")

    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,turn",
        start_date=sd,
        end_date=ed,
        frequency="d",
        adjustflag="3",  # 前复权
    )

    if rs.error_code != "0":
        raise RuntimeError(f"baostock query failed for {symbol}: {rs.error_msg}")

    data_list = []
    while (rs.error_code == "0") and rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise RuntimeError(f"baostock returned empty data for {symbol}")

    df = pd.DataFrame(data_list, columns=rs.fields)
    df = _normalize_columns(df)
    logger.info(f"baostock OK, rows={len(df)}")

    df.to_csv(cache_file, index=False)
    logger.info(f"Saved cache to {cache_file}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()

    # 多周期滞后收益
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)

    # 累计收益（多周期动量）
    for window in [3, 5, 10, 20]:
        df[f"ret_cum_{window}"] = df["close"] / df["close"].shift(window) - 1

    # 均线和偏离度
    for window in [5, 10, 20, 60]:
        ma = df["close"].rolling(window).mean()
        df[f"ma_{window}"] = ma
        df[f"close_div_ma_{window}"] = df["close"] / (ma + 1e-12) - 1

    # 波动率
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["ret_1"].rolling(window).std()

    # 成交量特征
    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma5"] + 1e-12)
    df["volume_change"] = df["volume"].pct_change()
    if "turnover" not in df.columns:
        df["turnover"] = 0.0
    df["turnover_ma5"] = df["turnover"].rolling(5).mean()
    df["turnover_ratio"] = df["turnover"] / (df["turnover_ma5"] + 1e-12)

    # OBV 资金流向
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ma10"] = obv.rolling(10).mean()
    df["obv_signal"] = obv / (obv.rolling(10).mean() + 1e-12) - 1

    # RSI 多周期
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    df["rsi_6"] = ta.momentum.RSIIndicator(close=df["close"], window=6).rsi()

    # MACD
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # 布林带
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["close"] + 1e-12)
    df["bb_position"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-12)

    # ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["atr_ratio"] = df["atr_14"] / df["close"]

    # ADX 趋势强度
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # CCI
    df["cci_20"] = ta.trend.CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20).cci()

    # Williams %R
    df["williams_r"] = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14).williams_r()

    # 价格形态特征
    df["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-12)
    df["oc_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)  # 蜡烛图实体比例
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)  # 跳空

    # 高低点突破
    df["high_20"] = df["high"].rolling(20).max()
    df["low_20"] = df["low"].rolling(20).min()
    df["break_high"] = (df["close"] - df["high_20"].shift(1)) / (df["high_20"].shift(1) + 1e-12)
    df["break_low"] = (df["close"] - df["low_20"].shift(1)) / (df["low_20"].shift(1) + 1e-12)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_target(df: pd.DataFrame, horizon: int = 5, label_threshold: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df.dropna(subset=["future_return"], inplace=True)
    # 显著涨跌才打标签，小幅波动标记为 -1 (ignore)
    if label_threshold > 0:
        conditions = [
            df["future_return"] > label_threshold,
            df["future_return"] < -label_threshold,
        ]
        choices = [1, 0]
        df["label"] = np.select(conditions, choices, default=-1)
    else:
        df["label"] = (df["future_return"] > 0).astype(int)
    df.reset_index(drop=True, inplace=True)
    return df


def create_sequences(
    data: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
    target_col: str = "future_return",
    step: int = 1,
    task: str = "regression",
    dates: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.Series]]:
    X, y = [], []
    feat_matrix = data[feature_cols].values
    target_vec = data[target_col].values
    valid_indices = []

    for i in range(0, len(data) - seq_len + 1, step):
        target_val = target_vec[i + seq_len - 1]
        # 分类任务中跳过 label == -1 (ignore) 的样本
        if task == "classification" and target_val == -1:
            continue
        X.append(feat_matrix[i : i + seq_len])
        y.append(target_val)
        valid_indices.append(i + seq_len - 1)

    out_dates = dates.iloc[valid_indices].reset_index(drop=True) if dates is not None else None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), out_dates


# ---------------------------------------------------------------------------
# Multi-Symbol Pipeline
# ---------------------------------------------------------------------------

class StockDataPipeline:
    def __init__(
        self,
        symbols: List[str],
        start_date: str = "20150101",
        end_date: str = "20240101",
        horizon: int = 5,
        seq_len: int = 60,
        step: int = 1,
        batch_size: int = 128,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        cache_dir: str = "./data_cache",
        task: str = "regression",
        label_threshold: float = 0.0,
    ):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.horizon = horizon
        self.seq_len = seq_len
        self.step = step
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.cache_dir = cache_dir
        self.task = task
        self.label_threshold = label_threshold

        # 根据任务类型选择目标列
        self.target_col = "label" if task == "classification" else "future_return"

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None

        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = self.y_test = None

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def prepare(self) -> "StockDataPipeline":
        logger.info("Step 1/6: Fetching raw data for all symbols ...")
        all_dfs = []
        for sym in self.symbols:
            try:
                df = fetch_stock_daily(sym, self.start_date, self.end_date, self.cache_dir)
                df["symbol"] = sym
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")
        if not all_dfs:
            raise RuntimeError("No data fetched for any symbol.")

        logger.info("Step 2/6: Adding features per symbol ...")
        processed = []
        for df in all_dfs:
            df = add_features(df)
            df = add_target(df, horizon=self.horizon, label_threshold=self.label_threshold)
            processed.append(df)

        logger.info("Step 3/6: Merging and encoding symbols ...")
        full_df = pd.concat(processed, ignore_index=True)
        full_df.sort_values(["date", "symbol"], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # symbol 数值编码（加入特征列）
        self.symbol_map = {s: i for i, s in enumerate(sorted(full_df["symbol"].unique()))}
        full_df["symbol_id"] = full_df["symbol"].map(self.symbol_map)
        logger.info(f"Symbols: {list(self.symbol_map.keys())}")

        logger.info("Step 3.5/6: Adding cross-sectional features ...")
        full_df = self._add_cross_sectional_features(full_df)

        # 定义特征列
        exclude = {
            "date", "open", "high", "low", "close", "volume", "amount",
            "future_return", "label", "symbol",
        }
        self.feature_cols = [c for c in full_df.columns if c not in exclude]
        logger.info(f"Feature count: {len(self.feature_cols)}")
        logger.info(f"Total rows after engineering: {len(full_df)}")

        logger.info("Step 4/6: Global temporal split & rolling standardization ...")
        self._split_and_scale(full_df)

        logger.info("Step 5/6: Building sequences per symbol and merging ...")
        self._build_sequences()

        logger.info("Step 6/6: Building DataLoaders ...")
        self._build_loaders()

        return self

    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算横截面特征（按日期分组）。"""
        df = df.copy()
        # 当日收益率在横截面上的排名分位
        df["cs_ret_rank"] = df.groupby("date")["ret_1"].rank(pct=True)
        # 当日成交量在横截面上的排名分位
        df["cs_volume_rank"] = df.groupby("date")["volume"].rank(pct=True)
        # 当日成交额在横截面上的排名分位
        if "amount" in df.columns:
            df["cs_amount_rank"] = df.groupby("date")["amount"].rank(pct=True)
        # 相对当日市场平均收益的超额收益
        df["cs_excess_ret"] = df["ret_1"] - df.groupby("date")["ret_1"].transform("mean")
        # 相对当日市场平均成交量的比值
        df["cs_volume_ratio"] = df["volume"] / (df.groupby("date")["volume"].transform("mean") + 1e-12)
        # RSI 横截面排名分位
        if "rsi_14" in df.columns:
            df["cs_rsi_rank"] = df.groupby("date")["rsi_14"].rank(pct=True)
        # 波动率横截面排名分位
        if "volatility_20" in df.columns:
            df["cs_vol_rank"] = df.groupby("date")["volatility_20"].rank(pct=True)
        # 动量横截面排名分位
        if "ret_cum_20" in df.columns:
            df["cs_momentum_rank"] = df.groupby("date")["ret_cum_20"].rank(pct=True)
        return df

    def _split_and_scale(self, df: pd.DataFrame):
        # 按全局日期时序划分
        dates = np.sort(df["date"].unique())
        n = len(dates)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_cutoff = dates[train_end]
        val_cutoff = dates[val_end]

        self.train_cutoff = train_cutoff
        self.val_cutoff = val_cutoff

        train_mask = df["date"] < train_cutoff
        val_mask = (df["date"] >= train_cutoff) & (df["date"] < val_cutoff)
        test_mask = df["date"] >= val_cutoff

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()
        test_df = df.loc[test_mask].copy()

        # 全局 scaler（训练集 fit）
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[self.feature_cols].values)

        for subset in [train_df, val_df, test_df]:
            subset.loc[:, self.feature_cols] = self.scaler.transform(
                subset[self.feature_cols].values
            )

        self.train_df_raw = train_df
        self.val_df_raw = val_df
        self.test_df_raw = test_df

    def _build_sequences(self):
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        train_dates_list, val_dates_list, test_dates_list = [], [], []

        for sym in self.symbols:
            tr = self.train_df_raw[self.train_df_raw["symbol"] == sym]
            va = self.val_df_raw[self.val_df_raw["symbol"] == sym]
            te = self.test_df_raw[self.test_df_raw["symbol"] == sym]

            if len(tr) >= self.seq_len:
                X, y, d = create_sequences(tr, self.feature_cols, self.seq_len, target_col=self.target_col, step=self.step, task=self.task, dates=tr["date"])
                if len(X) > 0:
                    all_X_train.append(X)
                    all_y_train.append(y)
                    train_dates_list.append(d)

            if len(va) >= self.seq_len:
                X, y, d = create_sequences(va, self.feature_cols, self.seq_len, target_col=self.target_col, step=self.step, task=self.task, dates=va["date"])
                if len(X) > 0:
                    all_X_val.append(X)
                    all_y_val.append(y)
                    val_dates_list.append(d)

            if len(te) >= self.seq_len:
                X, y, d = create_sequences(te, self.feature_cols, self.seq_len, target_col=self.target_col, step=self.step, task=self.task, dates=te["date"])
                if len(X) > 0:
                    all_X_test.append(X)
                    all_y_test.append(y)
                    test_dates_list.append(d)

        if not all_X_train:
            raise RuntimeError("No training sequences generated. Check seq_len vs data length.")

        self.X_train = np.concatenate(all_X_train, axis=0)
        self.y_train = np.concatenate(all_y_train, axis=0)
        self.X_val = np.concatenate(all_X_val, axis=0) if all_X_val else np.array([])
        self.y_val = np.concatenate(all_y_val, axis=0) if all_y_val else np.array([])
        self.X_test = np.concatenate(all_X_test, axis=0) if all_X_test else np.array([])
        self.y_test = np.concatenate(all_y_test, axis=0) if all_y_test else np.array([])

        self.train_dates = pd.concat(train_dates_list).reset_index(drop=True) if train_dates_list else pd.Series(dtype="datetime64[ns]")
        self.val_dates = pd.concat(val_dates_list).reset_index(drop=True) if val_dates_list else pd.Series(dtype="datetime64[ns]")
        self.test_dates = pd.concat(test_dates_list).reset_index(drop=True) if test_dates_list else pd.Series(dtype="datetime64[ns]")

        logger.info(
            f"Train/Val/Test sequences: "
            f"{len(self.y_train)}/{len(self.y_val)}/{len(self.y_test)}"
        )

    def _build_loaders(self):
        if len(self.y_train) == 0:
            raise RuntimeError("No training data.")
        train_ds = TensorDataset(
            torch.from_numpy(self.X_train),
            torch.from_numpy(self.y_train).unsqueeze(1).float(),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        if len(self.y_val) > 0:
            val_ds = TensorDataset(
                torch.from_numpy(self.X_val),
                torch.from_numpy(self.y_val).unsqueeze(1).float(),
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
        else:
            self.val_loader = None

        if len(self.y_test) > 0:
            test_ds = TensorDataset(
                torch.from_numpy(self.X_test),
                torch.from_numpy(self.y_test).unsqueeze(1).float(),
            )
            self.test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
        else:
            self.test_loader = None

    def save_scaler(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "feature_cols": self.feature_cols,
                    "symbol_map": self.symbol_map,
                    "train_cutoff": self.train_cutoff,
                    "val_cutoff": self.val_cutoff,
                },
                f,
            )
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.feature_cols = obj["feature_cols"]
        self.symbol_map = obj["symbol_map"]
        self.train_cutoff = obj["train_cutoff"]
        self.val_cutoff = obj["val_cutoff"]
        logger.info(f"Scaler loaded from {path}")
