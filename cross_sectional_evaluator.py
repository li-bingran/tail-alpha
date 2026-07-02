#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面评估器 — Top-k/Bottom-k spread + 分桶收益评估

Phase 2 核心组件：评估排序模型的实际盈利能力。

功能:
  1. calc_topk_return() — top-k/bottom-k 收益、spread、成本后 Sharpe
  2. calc_bucket_returns() — 分桶收益单调性检验
"""

import numpy as np
import pandas as pd


def calc_topk_return(
    scored_df: pd.DataFrame,
    k: int = 10,
    cost_bps: float = 10,
    score_col: str = 'score',
    return_col: str = 'future_return',
    date_col: str = 'date',
    symbol_col: str = 'symbol',
) -> dict:
    """
    计算 top-k / bottom-k 日度收益和 spread。

    Args:
        scored_df: 含 date, symbol, score, future_return 的 DataFrame
        k: top/bottom 取多少只
        cost_bps: 双边交易成本 (bps)
        score_col: 分数列名
        return_col: 未来收益列名

    Returns:
        dict with:
          - top_k_return: 平均日度 top-k 收益
          - bottom_k_return: 平均日度 bottom-k 收益
          - spread: top - bottom (gross)
          - spread_net: 扣除成本后的 spread
          - long_short_sharpe: 年化 long-short Sharpe
          - pct_positive_spread: spread > 0 的天数占比
          - n_days: 有效天数
    """
    df = scored_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, score_col, return_col])

    daily_spreads = []
    daily_top = []
    daily_bottom = []

    for date, group in df.groupby(date_col):
        if len(group) < 2 * k:
            continue

        sorted_g = group.sort_values(score_col, ascending=False)
        top = sorted_g.head(k)
        bottom = sorted_g.tail(k)

        top_ret = float(top[return_col].mean())
        bottom_ret = float(bottom[return_col].mean())
        spread = top_ret - bottom_ret

        daily_top.append(top_ret)
        daily_bottom.append(bottom_ret)
        daily_spreads.append(spread)

    if not daily_spreads:
        return {
            'top_k_return': None,
            'bottom_k_return': None,
            'spread': None,
            'spread_net': None,
            'long_short_sharpe': None,
            'pct_positive_spread': None,
            'n_days': 0,
            'k': k,
        }

    spreads = np.array(daily_spreads)
    tops = np.array(daily_top)
    bottoms = np.array(daily_bottom)

    # 成本：每天换仓 long-short，双边各 cost_bps
    # 简化：每天固定扣除 cost_bps * 2（买卖各一次）/ 10000 * 100（转为 %）
    daily_cost_pct = cost_bps * 2 / 10000 * 100

    spread_mean = float(spreads.mean())
    spread_std = float(spreads.std()) if len(spreads) > 1 else 1.0
    spread_net = spread_mean - daily_cost_pct

    # 年化 Sharpe
    if spread_std > 1e-9:
        long_short_sharpe = float((spread_mean - daily_cost_pct) / spread_std * np.sqrt(252))
    else:
        long_short_sharpe = 0.0

    pct_positive = float((spreads > 0).mean() * 100)
    pct_positive_net = float(((spreads - daily_cost_pct) > 0).mean() * 100)

    return {
        'top_k_return': round(float(tops.mean()), 4),
        'bottom_k_return': round(float(bottoms.mean()), 4),
        'spread': round(spread_mean, 4),
        'spread_net': round(spread_net, 4),
        'long_short_sharpe': round(long_short_sharpe, 3),
        'pct_positive_spread': round(pct_positive, 1),
        'pct_positive_spread_net': round(pct_positive_net, 1),
        'n_days': len(daily_spreads),
        'k': k,
        'cost_bps': cost_bps,
    }


def calc_bucket_returns(
    scored_df: pd.DataFrame,
    n_buckets: int = 5,
    score_col: str = 'score',
    return_col: str = 'future_return',
    date_col: str = 'date',
) -> dict:
    """
    分桶收益：按 score 分成 n_buckets 组，验证收益是否单调。

    Args:
        scored_df: 含 date, score, future_return 的 DataFrame
        n_buckets: 分几组

    Returns:
        dict with:
          - bucket_returns: list of {bucket, avg_return, count}
          - is_monotonic: 收益是否从 bucket 1（最低分）到 N（最高分）递增
          - spread_top_bottom: 最高桶 - 最低桶
    """
    df = scored_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[score_col, return_col])

    if len(df) < n_buckets * 10:
        return {'bucket_returns': [], 'is_monotonic': False, 'error': 'insufficient_data'}

    # 每日分桶后平均
    all_bucket_returns = {i: [] for i in range(n_buckets)}

    for date, group in df.groupby(date_col):
        if len(group) < n_buckets * 2:
            continue
        group = group.copy()
        group['bucket'] = pd.qcut(group[score_col], n_buckets, labels=False, duplicates='drop')
        for bucket_id, bucket_group in group.groupby('bucket'):
            all_bucket_returns[int(bucket_id)].append(float(bucket_group[return_col].mean()))

    bucket_summary = []
    for i in range(n_buckets):
        rets = all_bucket_returns.get(i, [])
        bucket_summary.append({
            'bucket': i + 1,
            'label': f'Q{i+1}' + (' (最低)' if i == 0 else ' (最高)' if i == n_buckets - 1 else ''),
            'avg_return': round(float(np.mean(rets)), 4) if rets else None,
            'std_return': round(float(np.std(rets)), 4) if len(rets) > 1 else None,
            'count': len(rets),
        })

    # 检查单调性
    avg_rets = [b['avg_return'] for b in bucket_summary if b['avg_return'] is not None]
    is_monotonic = all(avg_rets[i] <= avg_rets[i+1] for i in range(len(avg_rets)-1)) if len(avg_rets) >= 2 else False

    spread = (avg_rets[-1] - avg_rets[0]) if len(avg_rets) >= 2 else None

    return {
        'bucket_returns': bucket_summary,
        'is_monotonic': is_monotonic,
        'spread_top_bottom': round(spread, 4) if spread is not None else None,
        'n_buckets': n_buckets,
    }


if __name__ == '__main__':
    print('横截面评估器 — 需要输入 scored_df，通常由 ml_pipeline.py 调用')
