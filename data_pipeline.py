"""
Data Pipeline for A-Share Stock Prediction (Multi-Symbol, 60-min K-line, T+0 Strategy)
Includes: batch fetch via baostock, feature engineering, target generation,
sequence creation, temporal split, rolling standardization (no lookahead).

Prediction logic: Use past 6 days' 60-min data + current morning's 60-min data
                  to predict afternoon (PM) up/down.
"""

import os
import pickle
import warnings
import atexit
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

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


def _normalize_60min_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一60分钟K线列名，确保包含标准字段和时间列。"""
    col_map = {
        "turn": "turnover",
        "time": "datetime",  # baostock 60分钟线返回 time 字段
    }
    df = df.rename(columns=col_map)
    
    # 数值列转 float
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns after rename: {missing}. Original columns: {df.columns.tolist()}."
        )
    
    # 解析时间
    df["datetime"] = pd.to_datetime(df["datetime"])
    # 确保按时间排序
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 提取日期和交易时段
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    
    # 标识交易时段：上午(AM: 9:30-11:30) 和 下午(PM: 13:00-15:00)
    df["session"] = "AM"
    afternoon_mask = (
        (df["hour"] > 13) | 
        ((df["hour"] == 13) & (df["minute"] >= 0)) |
        ((df["hour"] == 14) & (df["minute"] <= 0))
    )
    df.loc[afternoon_mask, "session"] = "PM"
    
    # 上午最晚时间为11:30，下午最早为13:00
    df["time_order"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    
    return df


def fetch_stock_60min(
    symbol: str,
    start_date: str = "20150101",
    end_date: str = "20231231",
    cache_dir: str = "./data_cache_60min",
) -> pd.DataFrame:
    """
    通过 baostock 获取 A 股 60 分钟线（前复权）。
    start_date / end_date 格式支持 YYYYMMDD 或 YYYY-MM-DD。
    
    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume, 
                                amount, turnover, date, hour, minute, session, time_order
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_60min_{start_date}_{end_date}.csv")

    if os.path.exists(cache_file):
        logger.info(f"Loading cached 60-min data from {cache_file}")
        df = pd.read_csv(cache_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    # 格式化日期
    if "-" in start_date:
        sd = start_date
    else:
        sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    if "-" in end_date:
        ed = end_date
    else:
        ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

    _ensure_login()
    code = _to_baostock_code(symbol)
    logger.info(f"Fetching {symbol} 60-min K-line ({code}) from baostock ...")

    rs = bs.query_history_k_data_plus(
        code,
        "time,open,high,low,close,volume,amount,turn",
        start_date=sd,
        end_date=ed,
        frequency="60",  # 60分钟线
        adjustflag="3",  # 前复权
    )

    if rs.error_code != "0":
        raise RuntimeError(f"baostock query failed for {symbol}: {rs.error_msg}")

    data_list = []
    while (rs.error_code == "0") and rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        raise RuntimeError(f"baostock returned empty 60-min data for {symbol}")

    df = pd.DataFrame(data_list, columns=rs.fields)
    df = _normalize_60min_columns(df)
    logger.info(f"baostock 60-min OK, rows={len(df)}, "
                f"date range: {df['date'].min()} to {df['date'].max()}")

    df.to_csv(cache_file, index=False)
    logger.info(f"Saved cache to {cache_file}")
    return df


def add_features_60min(df: pd.DataFrame, warmup_days: int = 30) -> pd.DataFrame:
    """
    为60分钟K线添加技术指标特征。
    
    Args:
        df: 包含60分钟K线的DataFrame，需包含datetime, open, high, low, close, volume等
        warmup_days: 预热期天数，默认30天（约240根60分钟K线）
    
    注意：60分钟线的指标计算使用60分钟K线数量而不是交易日数
    """
    df = df.copy()
    
    # 计算60分钟收益率
    df["ret_1"] = df["close"].pct_change()

    # ==================== 60分钟量价基础特征 ====================
    # 多周期滞后收益（按60分钟K线数量）
    # 半天约4根K线，1天约8根K线
    for lag in [1, 2, 4, 8, 16, 40]:  # 1h, 2h, 半天, 1天, 2天, 1周
        df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)

    # 累计收益（多周期动量，按60分钟K线数量）
    for window in [4, 8, 16, 32, 80]:  # 半天, 1天, 2天, 1周, 2周
        df[f"ret_cum_{window}"] = df["close"] / df["close"].shift(window) - 1

    # ==================== 均线系统（60分钟周期） ====================
    # 多周期均线及偏离度
    for window in [4, 8, 16, 32, 48, 96, 192]:  # 半天,1天,2天,4天,1周,2周,1月
        ma = df["close"].rolling(window).mean()
        df[f"ma_{window}"] = ma
        df[f"close_div_ma_{window}"] = df["close"] / (ma + 1e-12) - 1

    # 均线多头排列强度（60分钟周期）
    df["ma4_gt_ma8"] = (df["ma_4"] - df["ma_8"]) / (df["close"] + 1e-12)
    df["ma8_gt_ma16"] = (df["ma_8"] - df["ma_16"]) / (df["close"] + 1e-12)
    df["ma16_gt_ma32"] = (df["ma_16"] - df["ma_32"]) / (df["close"] + 1e-12)
    
    df["ma_bullish_score"] = (
        (df["ma_4"] > df["ma_8"]).astype(float) +
        (df["ma_8"] > df["ma_16"]).astype(float) +
        (df["ma_16"] > df["ma_32"]).astype(float) +
        (df["ma_32"] > df["ma_96"]).astype(float)
    ) / 4.0

    # 均线收敛度
    df["ma_spread_4_16"] = abs(df["ma_4"] - df["ma_16"]) / (df["close"] + 1e-12)
    df["ma_spread_16_96"] = abs(df["ma_16"] - df["ma_96"]) / (df["close"] + 1e-12)

    # ==================== 成交量特征（60分钟） ====================
    # 基础量比
    df["volume_ma8"] = df["volume"].rolling(8).mean()  # 1日
    df["volume_ma16"] = df["volume"].rolling(16).mean()  # 2日
    df["volume_ma40"] = df["volume"].rolling(40).mean()  # 1周
    df["volume_ratio_8"] = df["volume"] / (df["volume_ma8"] + 1e-12)
    df["volume_ratio_16"] = df["volume"] / (df["volume_ma16"] + 1e-12)
    df["volume_ratio_40"] = df["volume"] / (df["volume_ma40"] + 1e-12)

    # 成交量变化率
    df["volume_change"] = df["volume"].pct_change()
    df["volume_change_8"] = df["volume"] / df["volume"].shift(8) - 1
    df["volume_change_16"] = df["volume"] / df["volume"].shift(16) - 1

    # 量价关系
    for window in [8, 16, 40]:
        df[f"vol_price_corr_{window}"] = df["volume"].rolling(window).corr(df["ret_1"])

    # 量价信号
    df["vol_down_price_up"] = ((df["volume"] < df["volume"].shift(1)) & 
                                (df["close"] > df["close"].shift(1))).astype(float)
    df["vol_up_price_up"] = ((df["volume"] > df["volume"].shift(1)) & 
                              (df["close"] > df["close"].shift(1))).astype(float)
    df["vol_up_price_down"] = ((df["volume"] > df["volume"].shift(1)) & 
                                (df["close"] < df["close"].shift(1))).astype(float)

    # 换手率特征
    if "turnover" in df.columns:
        df["turnover_ma8"] = df["turnover"].rolling(8).mean()
        df["turnover_ma16"] = df["turnover"].rolling(16).mean()
        df["turnover_ratio_8"] = df["turnover"] / (df["turnover_ma8"] + 1e-12)
        df["turnover_ratio_16"] = df["turnover"] / (df["turnover_ma16"] + 1e-12)
        df["turnover_change"] = df["turnover"].pct_change()

    # OBV 资金流向（60分钟）
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_ma8"] = obv.rolling(8).mean()
    df["obv_ma16"] = obv.rolling(16).mean()
    df["obv_ma40"] = obv.rolling(40).mean()
    df["obv_signal"] = obv / (obv.rolling(16).mean() + 1e-12) - 1
    df["obv_golden_cross"] = ((df["obv_ma8"] > df["obv_ma16"]) & 
                               (df["obv_ma8"].shift(1) <= df["obv_ma16"].shift(1))).astype(float)

    # ==================== 波动率特征（60分钟） ====================
    for window in [4, 8, 16, 40, 80]:
        df[f"volatility_{window}"] = df["ret_1"].rolling(window).std()

    df["volatility_ratio_8_40"] = df["volatility_8"] / (df["volatility_40"] + 1e-12)
    df["volatility_ratio_16_80"] = df["volatility_16"] / (df["volatility_80"] + 1e-12)
    df["vol_vs_avg"] = df["volatility_16"] / (df["volatility_80"] + 1e-12)

    # ==================== 动量/震荡指标（60分钟） ====================
    # RSI 多周期
    for window in [6, 14, 24, 40]:
        df[f"rsi_{window}"] = ta.momentum.RSIIndicator(close=df["close"], window=window).rsi()

    df["rsi_14_ma8"] = df["rsi_14"].rolling(8).mean()
    df["rsi_14_div_ma"] = df["rsi_14"] / (df["rsi_14_ma8"] + 1e-12) - 1
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)

    # MACD（60分钟）
    for fast, slow, signal in [(12, 26, 9), (6, 13, 9), (8, 21, 5)]:
        macd_indicator = ta.trend.MACD(
            close=df["close"], window_slow=slow, window_fast=fast, window_sign=signal
        )
        df[f"macd_{fast}_{slow}"] = macd_indicator.macd()
        df[f"macd_signal_{fast}_{slow}"] = macd_indicator.macd_signal()
        df[f"macd_diff_{fast}_{slow}"] = macd_indicator.macd_diff()

    # 标准 MACD
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["macd_golden_cross"] = ((df["macd"] > df["macd_signal"]) & 
                                (df["macd"].shift(1) <= df["macd_signal"].shift(1))).astype(float)
    df["macd_death_cross"] = ((df["macd"] < df["macd_signal"]) & 
                               (df["macd"].shift(1) >= df["macd_signal"].shift(1))).astype(float)
    df["macd_hist_change"] = df["macd_diff"] - df["macd_diff"].shift(1)

    # Stochastic（60分钟）
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]
    df["kdj_golden_cross"] = ((df["stoch_k"] > df["stoch_d"]) & 
                               (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))).astype(float)
    df["kdj_death_cross"] = ((df["stoch_k"] < df["stoch_d"]) & 
                              (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))).astype(float)

    # CCI（60分钟）
    for window in [14, 20, 40]:
        df[f"cci_{window}"] = ta.trend.CCIIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).cci()

    # Williams %R
    df["williams_r_14"] = ta.momentum.WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"], lbp=14
    ).williams_r()

    # ==================== 趋势强度指标 ====================
    # ADX
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["adx_strong"] = (df["adx"] > 25).astype(float)

    # ==================== 波动率通道指标（60分钟） ====================
    # 布林带
    for window, dev in [(20, 2), (20, 1.5), (40, 2)]:
        bb = ta.volatility.BollingerBands(
            close=df["close"], window=window, window_dev=dev
        )
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        df[f"bb_high_{window}_{dev}"] = bb_high
        df[f"bb_low_{window}_{dev}"] = bb_low
        df[f"bb_width_{window}_{dev}"] = (bb_high - bb_low) / (df["close"] + 1e-12)
        df[f"bb_position_{window}_{dev}"] = (df["close"] - bb_low) / (bb_high - bb_low + 1e-12)

    # ATR
    for window in [7, 14, 21]:
        df[f"atr_{window}"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        ).average_true_range()
        df[f"atr_ratio_{window}"] = df[f"atr_{window}"] / df["close"]

    # ==================== 价格形态特征 ====================
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-12)
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-12)
    df["oc_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # 上下影线
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["close"] + 1e-12)
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["close"] + 1e-12)

    # 高低点突破（60分钟周期）
    for window in [8, 16, 40]:
        df[f"high_{window}"] = df["high"].rolling(window).max()
        df[f"low_{window}"] = df["low"].rolling(window).min()
        df[f"break_high_{window}"] = (df["close"] - df[f"high_{window}"].shift(1)) / (df[f"high_{window}"].shift(1) + 1e-12)
        df[f"break_low_{window}"] = (df["close"] - df[f"low_{window}"].shift(1)) / (df[f"low_{window}"].shift(1) + 1e-12)

    # ==================== 资金流向指标 ====================
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    tp = df["typical_price"]
    raw_money_flow = tp * df["volume"]
    
    tp_diff = tp.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)
    
    for mfi_window in [7, 14]:
        pos_sum = positive_flow.rolling(mfi_window).sum()
        neg_sum = negative_flow.rolling(mfi_window).sum()
        money_ratio = pos_sum / (neg_sum + 1e-12)
        df[f"mfi_{mfi_window}"] = 100 - (100 / (1 + money_ratio))
        df[f"mfi_overbought_{mfi_window}"] = (df[f"mfi_{mfi_window}"] > 80).astype(float)
        df[f"mfi_oversold_{mfi_window}"] = (df[f"mfi_{mfi_window}"] < 20).astype(float)

    # ==================== 日内时序特征 ====================
    # 当日累计涨跌幅
    df["intraday_cum_ret"] = df.groupby("date")["ret_1"].cumsum()
    # 当日收益率
    df["intraday_ret"] = df.groupby("date")["close"].transform("first")
    df["intraday_ret"] = df["close"] / df["intraday_ret"] - 1
    
    # 上午/下午交易时段特征
    df["is_am"] = (df["session"] == "AM").astype(int)
    df["is_pm"] = (df["session"] == "PM").astype(int)
    
    # 当日上午收益率（在上午时段累计到上午收盘）
    am_returns = df[df["session"] == "AM"].groupby("date")["ret_1"].apply(
        lambda x: (1 + x).prod() - 1
    )
    df["am_return"] = df["date"].map(am_returns)
    
    # 前一下午收益率
    pm_returns = df[df["session"] == "PM"].groupby("date")["ret_1"].apply(
        lambda x: (1 + x).prod() - 1
    )
    # 将下午收益率向后移动一天，使其与前一天的日期对应
    pm_returns_shifted = pm_returns.shift(1)
    df["prev_pm_return"] = df["date"].map(pm_returns_shifted)

    # ==================== 市场情绪指标 ====================
    df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
    df["close_to_high"] = df["close"] / (df["high"] + 1e-12)
    df["close_to_low"] = df["close"] / (df["low"] + 1e-12)

    # 替换无穷大值为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 剔除预热期数据（按营业日数）
    if len(df) > warmup_days * 8:  # 每天约8根60分钟K线
        min_date = df["date"].min()
        cutoff_date = min_date + timedelta(days=warmup_days)
        df = df[df["date"] >= cutoff_date].copy()
        logger.info(f"Removed first {warmup_days} warmup days, remaining: {len(df)} rows, "
                     f"{df['date'].nunique()} trading days")
    else:
        logger.warning(f"Insufficient data for warmup removal")
    
    # 填充剩余的 NaN 为 0
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 确保按时间排序
    df = df.sort_values(["date", "time_order"]).reset_index(drop=True)
    
    return df


def add_target_pm(df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """
    为60分钟K线添加预测目标：预测下午(PM)涨跌。
    
    目标定义：
    - 对于上午最后一根K线（11:00-11:30），计算下午(PM)整体收益率
    - PM收益率 = (PM最后一根K线收盘价 / AM最后一根K线收盘价) - 1
    - 如果PM收益率 > threshold，标签=1（涨）
    - 如果PM收益率 < -threshold，标签=0（跌）
    - 否则标签=-1（忽略）
    
    注意：下午的数据只用于计算标签，不作为特征输入。
    """
    df = df.copy()
    
    # 获取每个交易日的日期
    dates = sorted(df["date"].unique())
    
    # 为每个交易日计算PM收益率
    future_pm_returns = {}
    
    for i, date in enumerate(dates):
        day_data = df[df["date"] == date]
        am_data = day_data[day_data["session"] == "AM"]
        pm_data = day_data[day_data["session"] == "PM"]
        
        if len(am_data) == 0 or len(pm_data) == 0:
            future_pm_returns[date] = None
            continue
        
        # AM最后一根K线的收盘价
        am_last_close = am_data.iloc[-1]["close"]
        # PM最后一根K线的收盘价
        pm_last_close = pm_data.iloc[-1]["close"]
        
        # PM收益率
        pm_return = pm_last_close / am_last_close - 1
        future_pm_returns[date] = pm_return
    
    # 为每个交易日创建目标（AM的最后一根K线有目标，其他K线没有）
    df["future_pm_return"] = None
    
    for date in dates:
        if future_pm_returns[date] is None:
            continue
        
        date_mask = df["date"] == date
        am_data = df[date_mask & (df["session"] == "AM")]
        
        if len(am_data) > 0:
            # 只在AM最后一根K线上设置目标
            last_am_idx = am_data.index[-1]
            df.loc[last_am_idx, "future_pm_return"] = future_pm_returns[date]
    
    # 删除没有目标的样本
    df = df.dropna(subset=["future_pm_return"]).copy()
    
    # 生成分类标签
    if threshold > 0:
        conditions = [
            df["future_pm_return"] > threshold,
            df["future_pm_return"] < -threshold,
        ]
        choices = [1, 0]
        df["label"] = np.select(conditions, choices, default=-1)
    else:
        df["label"] = (df["future_pm_return"] > 0).astype(int)
    
    logger.info(f"PM target generated: {len(df)} samples, "
                f"positive: {(df['label']==1).sum()}, "
                f"negative: {(df['label']==0).sum()}, "
                f"ignored: {(df['label']==-1).sum()}")
    
    df.reset_index(drop=True, inplace=True)
    return df


def create_sequences_60min(
    data: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 100,  # 约6天*8 + 上午4根 = 52，取整数100
    target_col: str = "future_pm_return",
    step: int = 1,
    task: str = "classification",
    dates: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.Series]]:
    """
    创建序列数据：使用过去N根60分钟K线预测下午涨跌。
    
    由于我们在add_target_pm中只保留了上午最后一根K线的样本，
    这里直接使用简单的滑动窗口即可。
    """
    X, y = [], []
    feat_matrix = data[feature_cols].values
    target_vec = data[target_col].values
    valid_indices = []

    for i in range(seq_len - 1, len(data), step):
        # 分类任务中跳过 label == -1 (ignore) 的样本
        if task == "classification" and target_vec[i] == -1:
            continue
        
        # 取前seq_len根K线（包括当前）
        X.append(feat_matrix[i - seq_len + 1 : i + 1])
        y.append(target_vec[i])
        valid_indices.append(i)

    out_dates = dates.iloc[valid_indices].reset_index(drop=True) if dates is not None else None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), out_dates


# ---------------------------------------------------------------------------
# Multi-Symbol 60min Pipeline
# ---------------------------------------------------------------------------

class StockDataPipeline:
    """
    60分钟K线数据管道，使用前N天+当日上午数据预测下午涨跌。
    
    Example:
        symbols = ["sh000001", "sz000001", "sz399006"]
        pipeline = StockDataPipeline(
            symbols=symbols,
            start_date="20200101",
            end_date="20241231",
            seq_len=100,  # 约12.5天（每天8根K线）
            batch_size=64,
            task="classification",
            label_threshold=0.001,
        )
        pipeline.prepare()
    """
    def __init__(
        self,
        symbols: List[str],
        start_date: str = "20150101",
        end_date: str = "20241231",
        seq_len: int = 100,  # 60分钟K线序列长度
        step: int = 1,
        batch_size: int = 64,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        cache_dir: str = "./data_cache_60min",
        task: str = "classification",
        label_threshold: float = 0.001,
    ):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.step = step
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.cache_dir = cache_dir
        self.task = task
        self.label_threshold = label_threshold

        # 根据任务类型选择目标列
        self.target_col = "label" if task == "classification" else "future_pm_return"

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
        logger.info("Step 1/6: Fetching 60-min raw data for all symbols ...")
        all_dfs = []
        for sym in self.symbols:
            try:
                df = fetch_stock_60min(sym, self.start_date, self.end_date, self.cache_dir)
                df["symbol"] = sym
                all_dfs.append(df)
                logger.info(f"  {sym}: {len(df)} rows, {df['date'].nunique()} trading days")
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")
        
        if not all_dfs:
            raise RuntimeError("No data fetched for any symbol.")

        logger.info("Step 2/6: Adding 60-min features per symbol ...")
        processed = []
        for df in all_dfs:
            df = add_features_60min(df)
            df = add_target_pm(df, threshold=self.label_threshold)
            processed.append(df)
            logger.info(f"  Processed {df['symbol'].iloc[0]}: {len(df)} labeled samples")

        logger.info("Step 3/6: Merging and encoding symbols ...")
        full_df = pd.concat(processed, ignore_index=True)
        full_df.sort_values(["datetime", "symbol"], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # symbol 数值编码
        self.symbol_map = {s: i for i, s in enumerate(sorted(full_df["symbol"].unique()))}
        full_df["symbol_id"] = full_df["symbol"].map(self.symbol_map)
        logger.info(f"Symbols: {list(self.symbol_map.keys())}")
        logger.info(f"Total labeled samples: {len(full_df)}")

        logger.info("Step 3.5/6: Adding cross-sectional features ...")
        full_df = self._add_cross_sectional_features_60min(full_df)

        # 定义特征列
        exclude = {
            "datetime", "date", "open", "high", "low", "close", "volume", "amount",
            "future_pm_return", "label", "symbol", "hour", "minute", "session", "time_order",
        }
        self.feature_cols = [c for c in full_df.columns if c not in exclude]
        logger.info(f"Feature count: {len(self.feature_cols)}")

        logger.info("Step 4/6: Global temporal split & rolling standardization ...")
        self._split_and_scale(full_df)

        logger.info("Step 5/6: Building sequences and merging ...")
        self._build_sequences()

        logger.info("Step 6/6: Building DataLoaders ...")
        self._build_loaders()

        # 打印统计信息
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        logger.info(f"  Train samples: {len(self.y_train)}")
        if self.y_val is not None and len(self.y_val) > 0:
            logger.info(f"  Val samples: {len(self.y_val)}")
        if self.y_test is not None and len(self.y_test) > 0:
            logger.info(f"  Test samples: {len(self.y_test)}")
        if self.task == "classification":
            for name, y in [("Train", self.y_train), ("Val", self.y_val), ("Test", self.y_test)]:
                if y is not None and len(y) > 0:
                    pos_ratio = (y == 1).sum() / len(y) * 100
                    logger.info(f"  {name} positive ratio: {pos_ratio:.1f}%")
        logger.info("=" * 60)

        return self

    def _add_cross_sectional_features_60min(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算横截面特征（按datetime分组，即同一60分钟时刻）。"""
        df = df.copy()
        
        # 按datetime分组（同一时刻的横截面）
        # 上午收盘时刻比较特殊，我们主要关注这个时刻的横截面
        group_col = "datetime"
        
        # 当日收益率在横截面上的排名分位
        df["cs_ret_rank"] = df.groupby(group_col)["ret_1"].rank(pct=True)
        # 成交量排名分位
        df["cs_volume_rank"] = df.groupby(group_col)["volume"].rank(pct=True)
        if "amount" in df.columns:
            df["cs_amount_rank"] = df.groupby(group_col)["amount"].rank(pct=True)
        # 超额收益
        df["cs_excess_ret"] = df["ret_1"] - df.groupby(group_col)["ret_1"].transform("mean")
        # 成交量相对比值
        df["cs_volume_ratio"] = df["volume"] / (df.groupby(group_col)["volume"].transform("mean") + 1e-12)
        # RSI排名
        if "rsi_14" in df.columns:
            df["cs_rsi_rank"] = df.groupby(group_col)["rsi_14"].rank(pct=True)
        # 波动率排名
        if "volatility_16" in df.columns:
            df["cs_vol_rank"] = df.groupby(group_col)["volatility_16"].rank(pct=True)
        
        return df

    def _split_and_scale(self, df: pd.DataFrame):
        """按datetime全局时序划分，并进行标准化。"""
        # 按datetime排序
        datetimes = np.sort(df["datetime"].unique())
        n = len(datetimes)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_cutoff = datetimes[train_end]
        val_cutoff = datetimes[val_end]

        self.train_cutoff = train_cutoff
        self.val_cutoff = val_cutoff

        logger.info(f"Time split: train < {train_cutoff}, "
                     f"val [{train_cutoff}, {val_cutoff}), "
                     f"test >= {val_cutoff}")

        train_mask = df["datetime"] < train_cutoff
        val_mask = (df["datetime"] >= train_cutoff) & (df["datetime"] < val_cutoff)
        test_mask = df["datetime"] >= val_cutoff

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()
        test_df = df.loc[test_mask].copy()

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # 全局 scaler（训练集 fit）
        self.scaler = StandardScaler()
        train_feature_values = train_df[self.feature_cols].values
        self.scaler.fit(train_feature_values)

        for subset in [train_df, val_df, test_df]:
            subset.loc[:, self.feature_cols] = self.scaler.transform(
                subset[self.feature_cols].values
            )

        self.train_df_raw = train_df
        self.val_df_raw = val_df
        self.test_df_raw = test_df

    def _build_sequences(self):
        """按symbol分别构建序列，然后合并。"""
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        train_dates_list, val_dates_list, test_dates_list = [], [], []

        for sym in self.symbols:
            tr = self.train_df_raw[self.train_df_raw["symbol"] == sym]
            va = self.val_df_raw[self.val_df_raw["symbol"] == sym]
            te = self.test_df_raw[self.test_df_raw["symbol"] == sym]

            if len(tr) >= self.seq_len:
                X, y, d = create_sequences_60min(
                    tr, self.feature_cols, self.seq_len,
                    target_col=self.target_col, step=self.step,
                    task=self.task, dates=tr["datetime"]
                )
                if len(X) > 0:
                    all_X_train.append(X)
                    all_y_train.append(y)
                    train_dates_list.append(d)

            if len(va) >= self.seq_len:
                X, y, d = create_sequences_60min(
                    va, self.feature_cols, self.seq_len,
                    target_col=self.target_col, step=self.step,
                    task=self.task, dates=va["datetime"]
                )
                if len(X) > 0:
                    all_X_val.append(X)
                    all_y_val.append(y)
                    val_dates_list.append(d)

            if len(te) >= self.seq_len:
                X, y, d = create_sequences_60min(
                    te, self.feature_cols, self.seq_len,
                    target_col=self.target_col, step=self.step,
                    task=self.task, dates=te["datetime"]
                )
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

        self.train_dates = pd.concat(train_dates_list).reset_index(drop=True) if train_dates_list else pd.Series()
        self.val_dates = pd.concat(val_dates_list).reset_index(drop=True) if val_dates_list else pd.Series()
        self.test_dates = pd.concat(test_dates_list).reset_index(drop=True) if test_dates_list else pd.Series()

        logger.info(
            f"Sequences - Train: {len(self.y_train)}, Val: {len(self.y_val)}, Test: {len(self.y_test)}"
        )

    def _build_loaders(self):
        """构建DataLoader。"""
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
        """保存scaler和相关配置。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "feature_cols": self.feature_cols,
                    "symbol_map": self.symbol_map,
                    "train_cutoff": self.train_cutoff,
                    "val_cutoff": self.val_cutoff,
                    "seq_len": self.seq_len,
                },
                f,
            )
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        """加载scaler和相关配置。"""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.feature_cols = obj["feature_cols"]
        self.symbol_map = obj["symbol_map"]
        self.train_cutoff = obj["train_cutoff"]
        self.val_cutoff = obj["val_cutoff"]
        if "seq_len" in obj:
            self.seq_len = obj["seq_len"]
        logger.info(f"Scaler loaded from {path}")

    def get_label_distribution(self) -> dict:
        """获取标签分布统计。"""
        result = {}
        for name, y in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            if y is not None and len(y) > 0:
                total = len(y)
                pos = (y == 1).sum()
                neg = (y == 0).sum()
                ign = (y == -1).sum()
                result[name] = {
                    "total": total,
                    "positive": int(pos),
                    "negative": int(neg),
                    "ignored": int(ign),
                    "pos_ratio": float(pos / total) if total > 0 else 0,
                }
        return result


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def create_pipeline(
    symbols: List[str],
    start_date: str = "20200101",
    end_date: str = "20241231",
    seq_len: int = 100,
    batch_size: int = 64,
    task: str = "classification",
) -> StockDataPipeline:
    """
    快速创建60分钟K线预测管道。
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        seq_len: 序列长度（60分钟K线数量）
        batch_size: 批次大小
        task: 任务类型（"classification" 或 "regression"）
    
    Returns:
        StockDataPipeline实例
    """
    return StockDataPipeline(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        seq_len=seq_len,
        batch_size=batch_size,
        task=task,
    )