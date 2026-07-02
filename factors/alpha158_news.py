# -*- coding: utf-8 -*-
"""
新闻 + 宏观特征工程 — 供 ML 训练 & 实时推理

设计原则:
  1. 训练时: 新闻无历史数据 → 用价量异动作为"隐性新闻"代理（可回溯2年）
  2. 推理时: 额外叠加 Finnhub 实时情绪（训练没见过，但 ML 能泛化）
  3. 宏观: FRED 有完整历史，训练/推理均可用
  4. 所有特征做 cross-sectional rank 归一化（-1 到 +1），抗量纲漂移

特征列表 (~12 个):
  价量异动(新闻代理):
    - ABNORMAL_VOLUME:    成交量 vs 20日均量的 z-score（爆量=大事件）
    - ABNORMAL_RETURN:    日收益率 vs 20日波动的 z-score（异常波动=新闻驱动）
    - VOLUME_PRICE_CORR:  5日量价相关（正=资金流入，负=出货）
    - GAP_RATIO:          隔夜跳空 / ATR（大缺口=盘后新闻）
    - INTRADAY_REVERSAL:  日内反转比(close-open)/range（新闻冲击后的消化程度）
  宏观环境:
    - VIX_PERCENTILE:     VIX在过去252日的百分位排名
    - VIX_CHANGE_5D:      VIX 5日变化率
    - YIELD_CURVE:        10Y-2Y利差
    - YIELD_CURVE_CHG20:  利差20日变化
    - RATE_LEVEL:         联邦基金利率水平（归一化）
    - INFLATION_EXP:      10Y盈亏平衡通胀预期
    - CLAIMS_ZSCORE:      初请失业金 z-score（就业突变信号）
"""

import numpy as np
import pandas as pd


def _safe_div(a, b, fill=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return fill if not np.isfinite(result) else result


# ── 价量异动特征（新闻代理，纯 OHLCV，可回溯）─────────────

def compute_news_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 OHLCV 提取价量异动特征，作为新闻事件的代理信号。

    这些特征捕捉"有大事发生"的信号：
    - 成交量突然放大 → 可能有重大新闻
    - 价格异常波动 → 市场对信息的反应
    - 隔夜跳空 → 盘后/盘前新闻
    - 日内反转 → 新闻冲击后的消化

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with ~5 news proxy features
    """
    features = {}

    close = df['Close'].astype(float)
    open_ = df['Open'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    returns = close.pct_change()

    # 1. ABNORMAL_VOLUME: 成交量 z-score (vs 20日均量)
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()
    features['ABNORMAL_VOLUME'] = _safe_div(volume - vol_ma20, vol_std20.clip(lower=1))

    # 2. ABNORMAL_RETURN: 收益率 z-score (vs 20日波动)
    ret_std20 = returns.rolling(20).std()
    features['ABNORMAL_RETURN'] = _safe_div(returns, ret_std20.clip(lower=1e-6))

    # 3. VOLUME_PRICE_CORR: 5日量价相关
    features['VOLUME_PRICE_CORR'] = returns.rolling(5).corr(volume.pct_change())

    # 4. GAP_RATIO: 隔夜跳空 / ATR
    gap = open_ - close.shift(1)
    atr14 = (high - low).rolling(14).mean()
    features['GAP_RATIO'] = _safe_div(gap, atr14.clip(lower=1e-4))

    # 5. INTRADAY_REVERSAL: 日内反转 (close-open) / (high-low)
    intraday_range = (high - low).clip(lower=1e-4)
    features['INTRADAY_REVERSAL'] = (close - open_) / intraday_range

    result = pd.DataFrame(features, index=df.index)
    return result.replace([np.inf, -np.inf], np.nan)


# ── 宏观特征（FRED 历史数据，训练/推理均可用）────────────

def compute_macro_features(
    dates: pd.DatetimeIndex,
    fred_cache: dict | None = None,
) -> pd.DataFrame:
    """
    计算宏观特征，对齐到给定日期序列。

    训练时: 从 FRED 拉历史 → 按日期 forward-fill 对齐
    推理时: 用最新值填充（宏观数据低频，日间变化小）

    Args:
        dates: 需要对齐的日期 index
        fred_cache: 可选的预加载 FRED 数据缓存

    Returns:
        DataFrame with ~7 macro features
    """
    import os
    import requests

    api_key = os.environ.get('FRED_API_KEY', '')
    features = {}

    def _fetch_fred(series_id: str, limit: int = 800) -> pd.Series:
        """获取 FRED 序列，返回 date-indexed Series"""
        if fred_cache and series_id in fred_cache:
            return fred_cache[series_id]
        if not api_key:
            return pd.Series(dtype=float)
        try:
            r = requests.get(
                'https://api.stlouisfed.org/fred/series/observations',
                params={
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': limit,
                },
                timeout=15,
            )
            if r.status_code != 200:
                return pd.Series(dtype=float)
            obs = r.json().get('observations', [])
            data = {}
            for o in obs:
                if o.get('value', '.') not in ('.', ''):
                    try:
                        data[pd.Timestamp(o['date'])] = float(o['value'])
                    except (ValueError, TypeError):
                        continue
            return pd.Series(data, dtype=float).sort_index()
        except Exception:
            return pd.Series(dtype=float)

    def _align(series: pd.Series, name: str) -> pd.Series:
        """将低频 FRED 序列 forward-fill 对齐到 dates"""
        if series.empty:
            return pd.Series(np.nan, index=dates, name=name)
        # 统一去除时区 + normalize 到日期（防止 04:00 vs 00:00 不匹配）
        if hasattr(series.index, 'tz') and series.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_localize(None)
        target = dates
        if hasattr(target, 'tz') and target.tz is not None:
            target = target.tz_localize(None)
        target = target.normalize()
        return series.reindex(target, method='ffill').rename(name)

    # ── 获取 FRED 数据 ──
    vix_raw = _fetch_fred('VIXCLS')
    t10y2y_raw = _fetch_fred('T10Y2Y')
    fedfunds_raw = _fetch_fred('FEDFUNDS')
    t10yie_raw = _fetch_fred('T10YIE')
    icsa_raw = _fetch_fred('ICSA')

    # ── 1. VIX_PERCENTILE: 过去 252 日百分位 ──
    vix_aligned = _align(vix_raw, 'VIX')
    vix_pct = vix_aligned.rolling(252, min_periods=60).rank(pct=True)
    features['VIX_PERCENTILE'] = vix_pct * 2 - 1  # 归一化到 [-1, +1]

    # ── 2. VIX_CHANGE_5D: VIX 5 日变化率 ──
    vix_chg = vix_aligned.pct_change(5)
    # clip 极值, 归一化
    features['VIX_CHANGE_5D'] = vix_chg.clip(-0.5, 0.5) * 2

    # ── 3. YIELD_CURVE: 10Y-2Y 利差 ──
    yc = _align(t10y2y_raw, 'YC')
    # 利差范围大约 -1 到 +3，归一化到 [-1, +1]
    features['YIELD_CURVE'] = (yc.clip(-1.5, 3.0) / 1.5).clip(-1, 1)

    # ── 4. YIELD_CURVE_CHG20: 利差 20 日变化 ──
    yc_chg = yc.diff(20)
    features['YIELD_CURVE_CHG20'] = (yc_chg.clip(-1.0, 1.0))

    # ── 5. RATE_LEVEL: 联邦基金利率（归一化到 [-1, +1]）──
    rate = _align(fedfunds_raw, 'RATE')
    # 0% → -1, 3% → 0, 6% → +1
    features['RATE_LEVEL'] = ((rate - 3.0) / 3.0).clip(-1, 1)

    # ── 6. INFLATION_EXP: 盈亏平衡通胀预期 ──
    infl = _align(t10yie_raw, 'INFL')
    # 2% 是目标，偏离越大信号越强
    # 1% → -1, 2% → 0, 3% → +1
    features['INFLATION_EXP'] = ((infl - 2.0) / 1.0).clip(-1, 1)

    # ── 7. CLAIMS_ZSCORE: 初请失业金 z-score ──
    claims = _align(icsa_raw, 'CLAIMS')
    claims_ma = claims.rolling(52, min_periods=20).mean()
    claims_std = claims.rolling(52, min_periods=20).std()
    features['CLAIMS_ZSCORE'] = (_safe_div(claims - claims_ma, claims_std.clip(lower=1000))).clip(-3, 3) / 3

    # _align 返回的 Series 使用 normalized naive index，
    # 但调用者可能传入 tz-aware dates，所以这里用 .values 重建 DataFrame
    result = pd.DataFrame(
        {k: v.values for k, v in features.items()},
        index=dates,
    )
    return result.replace([np.inf, -np.inf], np.nan)


# ── Finnhub 实时覆盖（仅推理时使用）────────────────────

def compute_realtime_news_features(symbols: list[str]) -> dict[str, dict]:
    """
    从 Finnhub 获取实时新闻特征（仅用于推理/评分，不参与训练）。

    返回 {symbol: {feature_name: value}} 可以直接覆盖到推理特征行。
    设计为：如果 Finnhub 不可用，返回空 dict，ML 用训练时学到的
    价量异动代理特征继续工作。
    """
    try:
        from factors.finnhub_sentiment import compute_sentiment_factors
        sentiment = compute_sentiment_factors(symbols)
    except Exception:
        return {}

    result = {}
    # 计算 universe 级别的统计量（cross-sectional）
    all_sent = [v.get('NEWS_SENTIMENT', 0) for v in sentiment.values()]
    all_vol = [v.get('NEWS_VOL', 0) for v in sentiment.values()]
    sent_mean = np.mean(all_sent) if all_sent else 0
    sent_std = np.std(all_sent) if all_sent else 1
    vol_mean = np.mean(all_vol) if all_vol else 1

    for sym, sf in sentiment.items():
        sent = sf.get('NEWS_SENTIMENT', 0)
        vol = sf.get('NEWS_VOL', 0)

        # 情绪 z-score (cross-sectional)
        sent_z = (sent - sent_mean) / max(sent_std, 0.01)

        # 热度 rank (cross-sectional, 归一化到 [-1, +1])
        buzz_rank = (vol / max(vol_mean, 1) - 1.0)

        result[sym] = {
            'NEWS_SENT_CS': round(np.clip(sent_z, -3, 3) / 3, 4),
            'NEWS_BUZZ_CS': round(np.clip(buzz_rank, -1, 1), 4),
        }

    return result


# ── 统一接口 ──────────────────────────────────────────────

def get_news_proxy_feature_names() -> list[str]:
    return [
        'ABNORMAL_VOLUME', 'ABNORMAL_RETURN', 'VOLUME_PRICE_CORR',
        'GAP_RATIO', 'INTRADAY_REVERSAL',
    ]


def get_macro_feature_names() -> list[str]:
    return [
        'VIX_PERCENTILE', 'VIX_CHANGE_5D',
        'YIELD_CURVE', 'YIELD_CURVE_CHG20',
        'RATE_LEVEL', 'INFLATION_EXP', 'CLAIMS_ZSCORE',
    ]


def get_all_feature_names() -> list[str]:
    return get_news_proxy_feature_names() + get_macro_feature_names()
