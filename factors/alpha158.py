# -*- coding: utf-8 -*-
"""
Alpha158 因子库 — 移植自 Qlib Alpha158，纯 pandas 实现

输入: 标准 OHLCV DataFrame (columns: Open, High, Low, Close, Volume)
输出: ~150 列特征 DataFrame

因子组:
  K线特征 (9) × 窗口 = KMID, KLEN, KUP, KLOW, KSFT ...
  动量     (ROC, MA ratio)
  波动率   (STD)
  回归     (BETA, RSQR, RESI)
  价格位置 (MAX, MIN, QTLU, QTLD, RSV)
  时间位置 (IMAX, IMIN, IMXD)
  相关性   (CORR, CORD)
  涨跌占比 (CNTP, CNTN, CNTD)
  涨跌幅度 (SUMP, SUMN, SUMD)
  成交量   (VMA, VSTD, WVMA)
"""

import numpy as np
import pandas as pd


WINDOWS = [5, 10, 20, 30, 60]


def _safe_div(a, b, fill=0.0):
    """Safe division avoiding div-by-zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    if isinstance(result, np.ndarray):
        result = np.where(np.isfinite(result), result, fill)
        return result
    try:
        if not np.isfinite(result):
            return fill
    except (TypeError, ValueError):
        return fill
    return result


def compute_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ~150 Alpha158 features from OHLCV data.

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume.
            Index should be datetime-like. At least 70 rows recommended.

    Returns:
        DataFrame with same index, ~150 feature columns.
        Column names: {group}_{window} (e.g. KMID_5, ROC_20, CORR_60)
    """
    features = {}

    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    # Pre-compute returns
    returns = close.pct_change()
    log_returns = np.log(close / close.shift(1))

    # ── K线特征 ──
    for w in WINDOWS:
        # KMID: (close - open) / open
        features[f'KMID_{w}'] = _safe_div(
            close.rolling(w).mean() - open_.rolling(w).mean(),
            open_.rolling(w).mean()
        )
        # KLEN: (high - low) / open
        features[f'KLEN_{w}'] = _safe_div(
            high.rolling(w).mean() - low.rolling(w).mean(),
            open_.rolling(w).mean()
        )
        # KUP: (high - max(open, close)) / open
        features[f'KUP_{w}'] = _safe_div(
            (high - pd.concat([open_, close], axis=1).max(axis=1)).rolling(w).mean(),
            open_.rolling(w).mean()
        )
        # KLOW: (min(open, close) - low) / open
        features[f'KLOW_{w}'] = _safe_div(
            (pd.concat([open_, close], axis=1).min(axis=1) - low).rolling(w).mean(),
            open_.rolling(w).mean()
        )
        # KSFT: (2*close - high - low) / open
        features[f'KSFT_{w}'] = _safe_div(
            (2 * close - high - low).rolling(w).mean(),
            open_.rolling(w).mean()
        )

    # ── 动量因子 ──
    for w in WINDOWS:
        # ROC: rate of change
        features[f'ROC_{w}'] = _safe_div(close - close.shift(w), close.shift(w))
        # MA ratio: close / MA(w) - 1
        ma = close.rolling(w).mean()
        features[f'MA_{w}'] = _safe_div(close, ma) - 1.0

    # ── 波动率因子 ──
    for w in WINDOWS:
        features[f'STD_{w}'] = _safe_div(close.rolling(w).std(), close)

    # ── 回归因子 (BETA, RSQR, RESI) ──
    for w in WINDOWS:
        beta = pd.Series(np.nan, index=df.index)
        rsqr = pd.Series(np.nan, index=df.index)
        resi = pd.Series(np.nan, index=df.index)

        for i in range(w - 1, len(df)):
            y = close.iloc[i - w + 1:i + 1].values
            x = np.arange(w, dtype=float)
            if len(y) < w or np.std(y) == 0:
                continue
            # Normalize
            if y[0] != 0:
                y_norm = y / y[0] - 1.0
            else:
                y_norm = y - y.mean()
            try:
                coeffs = np.polyfit(x, y_norm, 1)
                beta.iloc[i] = coeffs[0]
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y_norm - y_pred) ** 2)
                ss_tot = np.sum((y_norm - y_norm.mean()) ** 2)
                rsqr.iloc[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                resi.iloc[i] = y_norm[-1] - y_pred[-1]
            except (np.linalg.LinAlgError, ValueError):
                continue

        features[f'BETA_{w}'] = beta
        features[f'RSQR_{w}'] = rsqr
        features[f'RESI_{w}'] = resi

    # ── 价格位置因子 ──
    for w in WINDOWS:
        roll_max = close.rolling(w).max()
        roll_min = close.rolling(w).min()
        roll_range = roll_max - roll_min

        # MAX: (close - min) / (max - min)  [RSV-like]
        features[f'RSV_{w}'] = _safe_div(close - roll_min, roll_range)
        # MAX: close / rolling max - 1
        features[f'MAX_{w}'] = _safe_div(close, roll_max) - 1.0
        # MIN: close / rolling min - 1
        features[f'MIN_{w}'] = _safe_div(close, roll_min) - 1.0
        # Quantile upper: % of days close is below current
        features[f'QTLU_{w}'] = close.rolling(w).apply(
            lambda x: np.mean(x[:-1] < x[-1]) if len(x) > 1 else 0.5, raw=True
        )
        # Quantile lower: % of days close is above current
        features[f'QTLD_{w}'] = close.rolling(w).apply(
            lambda x: np.mean(x[:-1] > x[-1]) if len(x) > 1 else 0.5, raw=True
        )

    # ── 时间位置因子 ──
    for w in WINDOWS:
        # IMAX: days since rolling max / w
        features[f'IMAX_{w}'] = _safe_div(
            close.rolling(w).apply(lambda x: (w - 1) - np.argmax(x), raw=True), w
        )
        # IMIN: days since rolling min / w
        features[f'IMIN_{w}'] = _safe_div(
            close.rolling(w).apply(lambda x: (w - 1) - np.argmin(x), raw=True), w
        )
        # IMXD: IMAX - IMIN (positive = max more recent than min = bullish)
        features[f'IMXD_{w}'] = features[f'IMAX_{w}'] - features[f'IMIN_{w}']

    # ── 相关性因子 ──
    for w in WINDOWS:
        # CORR: correlation between close and volume
        features[f'CORR_{w}'] = close.rolling(w).corr(volume)
        # CORD: correlation between close and log(volume+1)
        log_vol = np.log1p(volume)
        features[f'CORD_{w}'] = close.rolling(w).corr(log_vol)

    # ── 涨跌占比因子 ──
    for w in WINDOWS:
        pos_ret = (returns > 0).astype(float)
        neg_ret = (returns < 0).astype(float)
        # CNTP: proportion of up days
        features[f'CNTP_{w}'] = pos_ret.rolling(w).mean()
        # CNTN: proportion of down days
        features[f'CNTN_{w}'] = neg_ret.rolling(w).mean()
        # CNTD: CNTP - CNTN
        features[f'CNTD_{w}'] = features[f'CNTP_{w}'] - features[f'CNTN_{w}']

    # ── 涨跌幅度因子 ──
    for w in WINDOWS:
        pos_returns = returns.clip(lower=0)
        neg_returns = (-returns).clip(lower=0)
        sump = pos_returns.rolling(w).sum()
        sumn = neg_returns.rolling(w).sum()
        # SUMP: sum of positive returns
        features[f'SUMP_{w}'] = sump
        # SUMN: sum of negative returns
        features[f'SUMN_{w}'] = sumn
        # SUMD: SUMP - SUMN
        features[f'SUMD_{w}'] = sump - sumn

    # ── 成交量因子 ──
    for w in WINDOWS:
        # VMA: volume MA ratio
        vma = volume.rolling(w).mean()
        features[f'VMA_{w}'] = _safe_div(volume, vma) - 1.0
        # VSTD: volume std / volume MA
        features[f'VSTD_{w}'] = _safe_div(volume.rolling(w).std(), vma)
        # WVMA: weighted volume MA (volume-weighted close std)
        vwma_close = (close * volume).rolling(w).sum() / volume.rolling(w).sum().replace(0, np.nan)
        features[f'WVMA_{w}'] = _safe_div(
            ((close - vwma_close) ** 2 * volume).rolling(w).sum(),
            volume.rolling(w).sum()
        ).apply(np.sqrt)
        # VSUMP: sum of volume on up days / total volume
        vol_up = (volume * (returns > 0).astype(float)).rolling(w).sum()
        vol_total = volume.rolling(w).sum()
        features[f'VSUMP_{w}'] = _safe_div(vol_up, vol_total)
        # VSUMN: sum of volume on down days / total volume
        vol_down = (volume * (returns < 0).astype(float)).rolling(w).sum()
        features[f'VSUMN_{w}'] = _safe_div(vol_down, vol_total)
        # VSUMD: VSUMP - VSUMN
        features[f'VSUMD_{w}'] = features[f'VSUMP_{w}'] - features[f'VSUMN_{w}']

    result = pd.DataFrame(features, index=df.index)

    # Replace infinities and clip extreme values
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


# ── 核心因子子集 (~40个)：覆盖每个因子组的代表性因子 ──
# 用于快速评估和 ML 训练（不需要全部 150+ 因子）
CORE_FACTORS = [
    # K线特征 (5)
    'KMID_5', 'KMID_20', 'KLEN_5', 'KLEN_20', 'KSFT_20',
    # 动量 (6)
    'ROC_5', 'ROC_10', 'ROC_20', 'ROC_60', 'MA_5', 'MA_20',
    # 波动率 (3)
    'STD_5', 'STD_20', 'STD_60',
    # 回归 (4)
    'BETA_5', 'BETA_20', 'RSQR_20', 'RESI_20',
    # 价格位置 (5)
    'RSV_5', 'RSV_20', 'MAX_20', 'MIN_20', 'QTLU_20',
    # 时间位置 (3)
    'IMAX_20', 'IMIN_20', 'IMXD_20',
    # 相关性 (2)
    'CORR_20', 'CORD_20',
    # 涨跌占比 (3)
    'CNTP_20', 'CNTN_20', 'CNTD_20',
    # 涨跌幅度 (3)
    'SUMP_20', 'SUMN_20', 'SUMD_20',
    # 成交量 (6)
    'VMA_5', 'VMA_20', 'VSTD_20', 'VSUMP_20', 'VSUMN_20', 'VSUMD_20',
]


def compute_alpha158_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅计算 CORE_FACTORS 子集。
    比 compute_alpha158() 快，适合快速评估和 ML 训练。

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with ~40 core feature columns
    """
    all_features = compute_alpha158(df)
    available = [f for f in CORE_FACTORS if f in all_features.columns]
    return all_features[available]


def get_alpha158_feature_names() -> list[str]:
    """Return list of all Alpha158 feature names (for ML pipeline)."""
    names = []
    for w in WINDOWS:
        for prefix in ['KMID', 'KLEN', 'KUP', 'KLOW', 'KSFT']:
            names.append(f'{prefix}_{w}')
    for w in WINDOWS:
        names.extend([f'ROC_{w}', f'MA_{w}'])
    for w in WINDOWS:
        names.append(f'STD_{w}')
    for w in WINDOWS:
        names.extend([f'BETA_{w}', f'RSQR_{w}', f'RESI_{w}'])
    for w in WINDOWS:
        names.extend([f'RSV_{w}', f'MAX_{w}', f'MIN_{w}', f'QTLU_{w}', f'QTLD_{w}'])
    for w in WINDOWS:
        names.extend([f'IMAX_{w}', f'IMIN_{w}', f'IMXD_{w}'])
    for w in WINDOWS:
        names.extend([f'CORR_{w}', f'CORD_{w}'])
    for w in WINDOWS:
        names.extend([f'CNTP_{w}', f'CNTN_{w}', f'CNTD_{w}'])
    for w in WINDOWS:
        names.extend([f'SUMP_{w}', f'SUMN_{w}', f'SUMD_{w}'])
    for w in WINDOWS:
        names.extend([f'VMA_{w}', f'VSTD_{w}', f'WVMA_{w}', f'VSUMP_{w}', f'VSUMN_{w}', f'VSUMD_{w}'])
    return names
