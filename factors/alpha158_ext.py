# -*- coding: utf-8 -*-
"""
Alpha158 扩展因子 — 跨资产 + 微观结构 + 高阶统计 + 波动率结构

新增 ~13 个因子，与原 Alpha158 因子拼接后用于 ML 训练。
"""

import numpy as np
import pandas as pd


def _safe_div(a, b, fill=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return result


def compute_extended_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    sector_dfs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    计算扩展因子。

    Args:
        df: 个股 OHLCV DataFrame（index 为 datetime）
        spy_df: SPY 的 OHLCV DataFrame（index 为 datetime），用于跨资产因子。
                如果为 None，跳过跨资产因子。
        sector_dfs: 板块ETF的OHLCV dict {ticker: DataFrame}，用于板块敏感度因子。
                    推荐: XLK(科技), XLF(金融), XLE(能源), XLV(医疗), TLT(长债)

    Returns:
        DataFrame with 扩展因子列，index 与 df 对齐。
    """
    features = {}

    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)
    returns = close.pct_change()

    # ── 跨资产因子（vs SPY）──
    if spy_df is not None and not spy_df.empty:
        spy_close = spy_df['Close'].astype(float).reindex(close.index, method='ffill')
        spy_returns = spy_close.pct_change()

        # RS_SPY: 相对强度 = 个股累计收益 / SPY 累计收益
        for w in [5, 20]:
            stock_cum = returns.rolling(w).sum()
            spy_cum = spy_returns.rolling(w).sum()
            features[f'RS_SPY_{w}'] = _safe_div(stock_cum, spy_cum.abs().clip(lower=1e-6))

        # BETA_SPY: 对 SPY 的 beta
        for w in [20, 60]:
            cov = returns.rolling(w).cov(spy_returns)
            spy_var = spy_returns.rolling(w).var()
            features[f'BETA_SPY_{w}'] = _safe_div(cov, spy_var.clip(lower=1e-10))

        # CORR_SPY: 与 SPY 的相关系数
        features['CORR_SPY_20'] = returns.rolling(20).corr(spy_returns)

    # ── 板块敏感度因子（个股 vs 板块 ETF 相关性 + 相对强度）──
    # 让 ML 自动学习: "加息利空科技(高CORR_XLK)但利好金融(高CORR_XLF)"
    if sector_dfs:
        for etf_ticker, etf_df in sector_dfs.items():
            if etf_df is None or etf_df.empty:
                continue
            etf_close = etf_df['Close'].astype(float).reindex(close.index, method='ffill')
            etf_returns = etf_close.pct_change()
            tag = etf_ticker.upper()
            # 20日滚动相关性: 高值=该股属于这个板块
            features[f'CORR_{tag}_20'] = returns.rolling(20).corr(etf_returns)
            # 20日相对强度: 正=跑赢该板块
            stock_cum = returns.rolling(20).sum()
            etf_cum = etf_returns.rolling(20).sum()
            features[f'RS_{tag}_20'] = _safe_div(stock_cum - etf_cum, etf_cum.abs().clip(lower=1e-6))

    # ── 微观结构因子 ──

    # MFI_14: 资金流量指数
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0)
    neg_mf = raw_mf.where(tp_diff < 0, 0.0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    features['MFI_14'] = (100.0 - 100.0 / (1.0 + mfr)) / 100.0  # 归一化到 0-1

    # OBV_RATIO_20: OBV 相对均线偏离
    obv_direction = np.sign(returns).fillna(0)
    obv = (volume * obv_direction).cumsum()
    obv_ma = obv.rolling(20).mean()
    features['OBV_RATIO_20'] = _safe_div(obv - obv_ma, obv_ma.abs().clip(lower=1e-6))

    # AMIHUD_20: Amihud 非流动性（高=流动性差）
    dollar_volume = close * volume
    amihud_daily = _safe_div(returns.abs(), dollar_volume.clip(lower=1.0))
    features['AMIHUD_20'] = amihud_daily.rolling(20).mean() * 1e6  # 缩放到可用范围

    # ── 高阶统计因子 ──

    # SKEW_20: 收益率偏度
    features['SKEW_20'] = returns.rolling(20).skew()

    # KURT_20: 收益率峰度
    features['KURT_20'] = returns.rolling(20).kurt()

    # AUTOCORR_5: 5 日收益率自相关
    features['AUTOCORR_5'] = returns.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(lag=5) if len(x) >= 20 else 0.0,
        raw=False,
    )

    # ── 波动率结构因子 ──

    # VOL_REGIME: 短期波动率 / 长期波动率
    vol_short = returns.rolling(10).std()
    vol_long = returns.rolling(60).std()
    features['VOL_REGIME'] = _safe_div(vol_short, vol_long.clip(lower=1e-6))

    # REALIZED_SKEW_20: 已实现偏度 = E[r^3] / E[r^2]^1.5
    r2_mean = (returns ** 2).rolling(20).mean()
    r3_mean = (returns ** 3).rolling(20).mean()
    features['REALIZED_SKEW_20'] = _safe_div(r3_mean, (r2_mean ** 1.5).clip(lower=1e-15))

    # RSI_14: 相对强弱指标（补充，alpha158 原版没有）
    gain = returns.clip(lower=0)
    loss = (-returns).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = _safe_div(avg_gain, avg_loss.clip(lower=1e-10))
    features['RSI_14'] = 1.0 - 1.0 / (1.0 + rs)  # 归一化到 0-1

    # ROCP（已有 ROC，这里补充百分比版本方便对齐）
    features['ROCP_5'] = close.pct_change(5)
    features['ROCP_20'] = close.pct_change(20)

    # ── 相对成交量因子（RVOL）──
    # RVOL = 当日成交量 / N日均量，>1.5 表示放量
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    features['RVOL_5'] = _safe_div(volume, vol_ma_5.clip(lower=1.0))
    features['RVOL_20'] = _safe_div(volume, vol_ma_20.clip(lower=1.0))

    # ── 量价共振因子（动量 × RVOL）──
    # 核心逻辑：涨+放量=强确认，涨+缩量=假突破
    features['MOM_RVOL_5'] = features['ROCP_5'] * features['RVOL_5']
    features['MOM_RVOL_20'] = features['ROCP_20'] * features['RVOL_20']

    # ── 动量加速度 ──
    # 正值=短期动量强于长期（突然起势），负值=动量衰减
    features['MOMENTUM_ACCEL'] = features['ROCP_5'] - features['ROCP_20']

    # ── 跨资产领先信号 ──
    # 这些信号来自独立资产类别，与个股 OHLCV 低相关，能显著提升 IC
    if sector_dfs:
        # 1. 债券动量 (TLT) — 利率方向的领先指标
        # 债券跌 = 利率升 → 科技承压; 债券涨 = 避险 → 防御股利好
        tlt_df = sector_dfs.get('TLT')
        if tlt_df is not None and not tlt_df.empty:
            tlt_close = tlt_df['Close'].astype(float).reindex(close.index, method='ffill')
            tlt_ret = tlt_close.pct_change()
            features['TLT_MOM_5'] = tlt_close.pct_change(5)
            features['TLT_MOM_20'] = tlt_close.pct_change(20)
            # 股债背离: 个股涨 + 债券涨 = 异常 (可能是避险涨)
            features['STOCK_BOND_DIVERGE'] = returns.rolling(5).sum() - tlt_ret.rolling(5).sum()

        # 2. 美元强弱 (UUP) — 跨国公司盈利方向
        uup_df = sector_dfs.get('UUP')
        if uup_df is not None and not uup_df.empty:
            uup_close = uup_df['Close'].astype(float).reindex(close.index, method='ffill')
            features['USD_MOM_5'] = uup_close.pct_change(5)
            features['USD_MOM_20'] = uup_close.pct_change(20)

        # 3. VIX 动量 (VIXY) — 恐慌方向
        vixy_df = sector_dfs.get('VIXY')
        if vixy_df is not None and not vixy_df.empty:
            vixy_close = vixy_df['Close'].astype(float).reindex(close.index, method='ffill')
            features['VIX_MOM_5'] = vixy_close.pct_change(5)
            # VIX 急升 = 恐慌加剧, VIX 下降 = 风险偏好回升

        # 4. 高收益债 spread (HYG vs IEF) — 信用风险情绪
        hyg_df = sector_dfs.get('HYG')
        ief_df = sector_dfs.get('IEF')
        if hyg_df is not None and ief_df is not None:
            hyg_close = hyg_df['Close'].astype(float).reindex(close.index, method='ffill')
            ief_close = ief_df['Close'].astype(float).reindex(close.index, method='ffill')
            hyg_ret = hyg_close.pct_change()
            ief_ret = ief_close.pct_change()
            # HYG 跑赢 IEF = risk-on; HYG 跑输 = risk-off
            features['CREDIT_SPREAD_MOM'] = (hyg_ret - ief_ret).rolling(5).sum()

    # 5. 开盘缺口 (Gap) — 日级别的日内代理
    # Gap = (Open - prev Close) / prev Close
    if 'Open' in df.columns:
        prev_close = close.shift(1)
        open_price = df['Open'].astype(float)
        features['GAP_PCT'] = _safe_div(open_price - prev_close, prev_close.clip(lower=0.01)) * 100
        # Gap + Follow-through: 跳空后当天继续涨 = 强势
        features['GAP_FOLLOWTHRU'] = features['GAP_PCT'] * returns

        # 2026-04-24 新增: 针对 AMD +10% 类极端 gap 的特征
        # MEGA_GAP_UP: gap > 3% 标记, 用以区分"普通 gap"vs"mega gap"
        # 历史上两者的 forward return 分布不同, 用 flag 让 ML 学到非线性
        features['MEGA_GAP_UP'] = (features['GAP_PCT'] > 3.0).astype(float)
        features['MEGA_GAP_DN'] = (features['GAP_PCT'] < -3.0).astype(float)
        # GAP × prev 5 日动量: 强动量下的 gap up 可能继续, 弱动量下的 gap 更可能回落
        mom_5 = close.pct_change(5) * 100
        features['GAP_X_MOM5'] = features['GAP_PCT'] * mom_5
        # Gap size 绝对值 (log 变换, 压缩异常值)
        import numpy as _np
        features['GAP_ABS_LOG'] = _np.log1p(features['GAP_PCT'].abs())

    # 6. 日内幅度代理 (High-Low range / Close)
    features['INTRADAY_RANGE'] = _safe_div(high - low, close.clip(lower=0.01))
    # 日内方向: (Close - Open) / (High - Low) — 1=全天涨, -1=全天跌
    if 'Open' in df.columns:
        open_price = df['Open'].astype(float)
        hl_range = (high - low).clip(lower=0.01)
        features['INTRADAY_DIR'] = _safe_div(close - open_price, hl_range)

    result = pd.DataFrame(features, index=df.index)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


# 用于板块敏感度的 ETF 列表 (含跨资产)
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'TLT', 'UUP', 'HYG', 'IEF']
# 跨资产 ETF (VIX3M 在 yfinance 不可用, 只用 VIXY)
CROSS_ASSET_ETFS = ['VIXY']


def get_extended_feature_names(include_spy: bool = True, include_sectors: bool = True) -> list[str]:
    """返回扩展因子的列名列表。"""
    names = []
    if include_spy:
        names.extend([
            'RS_SPY_5', 'RS_SPY_20',
            'BETA_SPY_20', 'BETA_SPY_60',
            'CORR_SPY_20',
        ])
    if include_sectors:
        for etf in SECTOR_ETFS:
            names.append(f'CORR_{etf}_20')
            names.append(f'RS_{etf}_20')
    names.extend([
        'MFI_14', 'OBV_RATIO_20', 'AMIHUD_20',
        'SKEW_20', 'KURT_20', 'AUTOCORR_5',
        'VOL_REGIME', 'REALIZED_SKEW_20',
        'RSI_14', 'ROCP_5', 'ROCP_20',
        'RVOL_5', 'RVOL_20',
        'MOM_RVOL_5', 'MOM_RVOL_20',
        'MOMENTUM_ACCEL',
        # 跨资产领先信号
        'TLT_MOM_5', 'TLT_MOM_20', 'STOCK_BOND_DIVERGE',
        'USD_MOM_5', 'USD_MOM_20',
        'VIX_TERM', 'CREDIT_SPREAD_MOM',
        # 日内代理
        'GAP_PCT', 'GAP_FOLLOWTHRU',
        'MEGA_GAP_UP', 'MEGA_GAP_DN', 'GAP_X_MOM5', 'GAP_ABS_LOG',
        'INTRADAY_RANGE', 'INTRADAY_DIR',
    ])
    return names
