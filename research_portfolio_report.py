#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正式组合回测报告 — k=25 多头 / 5 日错峰再平衡 vs SPY。

输入: ml_oos_scores_sp500.parquet（walk-forward OOS 分数，无泄漏）
      alpha158_training_data_sp500.parquet（future_return_1d）
输出: backtest_outputs/portfolio_report_sp500.json
      backtest_outputs/portfolio_report_sp500.png（净值/回撤/月度超额三联图）

口径:
  - 组合 = 每日按分数取 top-25 等权，5 个错峰 tranche（各 1/5 资金）每 5 日再平衡
  - 成本 = 换手部分双边 × cost_bps（默认展示 20bps，JSON 含 10/20/30）
  - 收益对齐: date t 的收益为 t 收盘 → t+1 收盘（future_return_1d），SPY 同口径
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

BASE_DIR = Path(__file__).parent
OUT = BASE_DIR / 'backtest_outputs'
K = 25
REBAL_DAYS = 5
HEADLINE_COST_BPS = 20

# dataviz 验证过的默认调色板
C_STRAT = '#2a78d6'   # series 1
C_SPY = '#1baf7a'     # series 2
C_NEG = '#e34948'
C_POS = '#008300'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
C_MUTED = '#898781'
C_INK = '#0b0b0b'
C_SURFACE = '#fcfcfb'


def nw_t(x, lag=5):
    x = np.asarray(x, float)
    n = x.size
    m = x.mean()
    d = x - m
    lrv = (d ** 2).mean()
    for L in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - L / (lag + 1)) * (d[L:] * d[:-L]).mean()
    return float(m / np.sqrt(max(lrv, 1e-12) / n))


def build_portfolio_returns(df: pd.DataFrame, cost_bps: float) -> pd.Series:
    """5 错峰 tranche 平均的日收益序列（%，已扣费）。"""
    dates = sorted(df['date'].unique())
    by_date = {dt: g.sort_values('score') for dt, g in df.groupby('date')}
    tranche_rets = []
    for offset in range(REBAL_DAYS):
        hold = None
        rows = []
        for i, dt in enumerate(dates):
            g = by_date[dt]
            if len(g) < K * 4:
                continue
            cost_today = 0.0
            if hold is None or (i % REBAL_DAYS) == offset:
                new_hold = set(g.tail(K)['symbol'])
                if hold is not None:
                    turnover = 1 - len(new_hold & hold) / K
                    cost_today = turnover * 2 * cost_bps / 10000 * 100
                hold = new_hold
            held = g[g['symbol'].isin(hold)]
            rows.append({'date': dt, 'ret': held['future_return_1d'].mean() - cost_today})
        tranche_rets.append(pd.DataFrame(rows).set_index('date')['ret'])
    return pd.concat(tranche_rets, axis=1).mean(axis=1).sort_index()


def perf_stats(ret_pct: pd.Series) -> dict:
    r = ret_pct / 100.0
    equity = (1 + r).cumprod()
    years = len(r) / 252
    ann = float(equity.iloc[-1] ** (1 / years) - 1) * 100
    vol = float(r.std() * np.sqrt(252)) * 100
    dd = (equity / equity.cummax() - 1)
    return {
        'total_return_pct': round(float(equity.iloc[-1] - 1) * 100, 1),
        'ann_return_pct': round(ann, 1),
        'ann_vol_pct': round(vol, 1),
        'sharpe': round(ann / vol, 2) if vol > 0 else 0,
        'max_drawdown_pct': round(float(dd.min()) * 100, 1),
        'n_days': int(len(r)),
    }


def main():
    scores = pd.read_parquet(OUT / 'ml_oos_scores_sp500.parquet')
    data = pd.read_parquet(OUT / 'alpha158_training_data_sp500.parquet',
                           columns=['date', 'symbol', 'future_return_1d'])
    for d in (scores, data):
        d['date'] = pd.to_datetime(d['date']).dt.tz_localize(None).dt.normalize()
    df = scores.merge(data, on=['date', 'symbol'], how='inner').dropna()
    print(f'评分×收益: {len(df)} rows, {df["date"].nunique()} days')

    # SPY 同口径（t 收盘 → t+1 收盘）
    import yfinance as yf
    spy_px = yf.Ticker('SPY').history(period='4y')['Close']
    spy_px.index = spy_px.index.tz_localize(None).normalize()
    spy_ret = (spy_px.pct_change().shift(-1) * 100).rename('spy')

    report = {'generated_at': pd.Timestamp.utcnow().isoformat(),
              'config': {'k': K, 'rebal_days': REBAL_DAYS, 'universe': 'sp500_snapshot_501',
                         'score_source': 'ml_oos_scores_sp500 (walk-forward OOS)'},
              'cost_scenarios': {}}

    for cost in (10, 20, 30):
        port = build_portfolio_returns(df, cost)
        spy = spy_ret.reindex(port.index).dropna()
        port_c = port.reindex(spy.index)
        excess = port_c - spy
        stats = {
            'strategy': perf_stats(port_c),
            'spy': perf_stats(spy),
            'excess_ann_pct': round(float(excess.mean() * 252), 1),
            'excess_t_nw': round(nw_t(excess), 2),
        }
        report['cost_scenarios'][f'{cost}bps'] = stats
        print(f'{cost}bps: 策略年化 {stats["strategy"]["ann_return_pct"]}% '
              f'(Sharpe {stats["strategy"]["sharpe"]}, MaxDD {stats["strategy"]["max_drawdown_pct"]}%) '
              f'vs SPY {stats["spy"]["ann_return_pct"]}%  超额 t={stats["excess_t_nw"]}')
        if cost == HEADLINE_COST_BPS:
            h_port, h_spy, h_excess = port_c, spy, excess

    with open(OUT / 'portfolio_report_sp500.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 三联图 ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'text.color': C_INK,
        'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_MUTED,
        'xtick.color': C_MUTED, 'ytick.color': C_MUTED,
        'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': C_SURFACE, 'axes.facecolor': C_SURFACE,
    })

    eq_s = (1 + h_port / 100).cumprod()
    eq_b = (1 + h_spy / 100).cumprod()
    dd = (eq_s / eq_s.cummax() - 1) * 100
    monthly_ex = h_excess.resample('ME').sum()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9.5), sharex=False,
        gridspec_kw={'height_ratios': [3, 1.2, 1.4], 'hspace': 0.38})

    ax1.plot(eq_s.index, eq_s, color=C_STRAT, lw=2, label='Strategy (top-25, 5-day rebalance)')
    ax1.plot(eq_b.index, eq_b, color=C_SPY, lw=2, label='SPY buy & hold')
    ax1.annotate(f'{eq_s.iloc[-1]:.2f}', (eq_s.index[-1], eq_s.iloc[-1]),
                 textcoords='offset points', xytext=(6, 0), color=C_STRAT, fontsize=10)
    ax1.annotate(f'{eq_b.iloc[-1]:.2f}', (eq_b.index[-1], eq_b.iloc[-1]),
                 textcoords='offset points', xytext=(6, -4), color=C_SPY, fontsize=10)
    hs = report['cost_scenarios'][f'{HEADLINE_COST_BPS}bps']
    ax1.set_title(
        f'ML ranking strategy vs SPY (cost {HEADLINE_COST_BPS} bps, walk-forward OOS)\n'
        f'CAGR {hs["strategy"]["ann_return_pct"]}% vs {hs["spy"]["ann_return_pct"]}%  ·  '
        f'Sharpe {hs["strategy"]["sharpe"]} vs {hs["spy"]["sharpe"]}  ·  '
        f'excess NW t = {hs["excess_t_nw"]}',
        fontsize=11, loc='left', color=C_INK)
    ax1.legend(frameon=False, loc='upper left', fontsize=9)
    ax1.set_ylabel('Equity (start = 1)')

    ax2.fill_between(dd.index, dd, 0, color=C_NEG, alpha=0.35, lw=0)
    ax2.plot(dd.index, dd, color=C_NEG, lw=1.2)
    ax2.set_ylabel('Drawdown %')
    ax2.set_title(f'Strategy drawdown (max {hs["strategy"]["max_drawdown_pct"]}%)',
                  fontsize=10, loc='left', color=C_INK)

    colors = [C_POS if v >= 0 else C_NEG for v in monthly_ex]
    ax3.bar(monthly_ex.index, monthly_ex, width=18, color=colors)
    ax3.axhline(0, color=C_AXIS, lw=1)
    ax3.set_ylabel('Monthly excess %')
    ax3.set_title('Monthly excess return vs SPY', fontsize=10, loc='left', color=C_INK)

    fig.savefig(OUT / 'portfolio_report_sp500.png', dpi=150, bbox_inches='tight')
    print(f"\n图表: {OUT / 'portfolio_report_sp500.png'}")
    print(f"报告: {OUT / 'portfolio_report_sp500.json'}")


if __name__ == '__main__':
    main()
