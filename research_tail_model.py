#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尾部针对性模型实验 — 3 分类 (bottom 20% / 中间 / top 20%)。

背景（2026-07-02，见 OPTIMIZATION_BACKLOG.md「简历级验证冲刺」）：
  S&P 500 扩池验证发现回归基线全截面 IC 不显著（NW t=0.96），
  但尾部选择显著（top-25 多头 5 日再平衡 NW t≈2.0-2.5）。
  本实验验证：把模型目标从"全截面排序"改成"只分辨尾部"能否更强。

设计:
  - 标签: 逐日按 future_return_5d 横截面分位打 3 类
    (<=20% → 0, 中间 → 1, >=80% → 2)，天然市场中性（rank 不受当日均值影响）
  - 模型: LightGBM multiclass, 评分 = P(top) - P(bottom)
  - 切分/选因子: 复用 ml_pipeline 的 walk_forward_split / select_features_by_ic
    （同 purge=10 / embargo=5 / top_k=80 / force_include）
  - 评估: 与回归基线（ml_oos_scores_sp500.parquet）在相同日期交集上对比
    逐日 CS-IC、k=5/25/50 多空价差 NW t、k=25 五日错峰再平衡扣费净超额

产出:
  backtest_outputs/tail_model_oos_scores.parquet
  backtest_outputs/tail_model_report.json

仅研究用，不接生产 scoring/下单路径。
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
OUTPUT_DIR = BASE_DIR / 'backtest_outputs'
DATA_PARQUET = OUTPUT_DIR / 'alpha158_training_data_sp500.parquet'
BASELINE_SCORES = OUTPUT_DIR / 'ml_oos_scores_sp500.parquet'

sys.path.insert(0, str(BASE_DIR))
from ml_pipeline import walk_forward_split, select_features_by_ic  # noqa: E402

TAIL_Q = 0.20          # 尾部分位
LABEL_COL = 'future_return_5d'
FORCE_PREFIXES = [
    'VIX_', 'YIELD_', 'RATE_', 'INFLATION_', 'CLAIMS_',
    'ROCP_', 'RVOL_', 'MOM_RVOL_', 'MOMENTUM_ACCEL',
    'RS_SPY_', 'GAP_', 'MEGA_GAP_',
]


def make_tail_labels(df: pd.DataFrame) -> pd.Series:
    """逐日横截面分位 → 3 类标签 (0=bottom, 1=middle, 2=top)。"""
    pct = df.groupby('date')[LABEL_COL].rank(pct=True)
    labels = pd.Series(1, index=df.index, dtype=int)
    labels[pct <= TAIL_Q] = 0
    labels[pct >= 1 - TAIL_Q] = 2
    return labels


def nw_t(x, lag=5):
    """Newey-West t 统计量（Bartlett kernel）。"""
    x = np.asarray(x, float)
    n = x.size
    if n < 20:
        return 0.0
    m = x.mean()
    d = x - m
    lrv = (d ** 2).mean()
    for L in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - L / (lag + 1)) * (d[L:] * d[:-L]).mean()
    return float(m / np.sqrt(max(lrv, 1e-12) / n))


def evaluate_scores(df: pd.DataFrame, name: str) -> dict:
    """df 需含 date/symbol/score/future_return_1d。返回评估指标 dict。"""
    from scipy import stats

    out = {'name': name, 'n_days': int(df['date'].nunique())}
    dates = sorted(df['date'].unique())
    by_date = {dt: g.sort_values('score') for dt, g in df.groupby('date')}

    # 逐日 CS-IC
    daily_ics = []
    for dt in dates:
        g = by_date[dt]
        if len(g) < 50:
            continue
        c, _ = stats.spearmanr(g['score'], g['future_return_1d'])
        if np.isfinite(c):
            daily_ics.append(c)
    out['cs_ic'] = round(float(np.mean(daily_ics)), 4)
    out['cs_ic_t_nw'] = round(nw_t(daily_ics), 2)

    # 多空价差（日频调仓，毛收益）
    spreads = {}
    for k in (5, 25, 50):
        sp = []
        for dt in dates:
            g = by_date[dt]
            if len(g) < k * 4:
                continue
            sp.append(g.tail(k)['future_return_1d'].mean()
                      - g.head(k)['future_return_1d'].mean())
        sp = np.asarray(sp)
        spreads[f'k{k}'] = {
            'daily_mean_pct': round(float(sp.mean()), 4),
            'sharpe': round(float(sp.mean() / sp.std() * np.sqrt(252)), 2),
            't_nw': round(nw_t(sp), 2),
        }
    out['ls_spread'] = spreads

    # k=25 多头 5 日错峰再平衡，扣费净超额
    k = 25
    tranche_ex, tranche_to = [], []
    for offset in range(5):
        hold = None
        rows, turns = [], []
        for i, dt in enumerate(dates):
            g = by_date[dt]
            if len(g) < k * 4:
                continue
            if hold is None or (i % 5) == offset:
                new_hold = set(g.tail(k)['symbol'])
                if hold is not None:
                    turns.append(1 - len(new_hold & hold) / k)
                hold = new_hold
            held = g[g['symbol'].isin(hold)]
            rows.append({'date': dt,
                         'ex': held['future_return_1d'].mean()
                               - g['future_return_1d'].mean()})
        s = pd.DataFrame(rows).set_index('date')['ex']
        tranche_ex.append(s)
        tranche_to.append(np.mean(turns) / 5 if turns else 0.0)
    avg_ex = pd.concat(tranche_ex, axis=1).mean(axis=1)
    avg_to = float(np.mean(tranche_to))
    long25 = {'daily_turnover': round(avg_to, 4)}
    for cost_bps in (10, 20, 30):
        net = avg_ex - avg_to * 2 * cost_bps / 10000 * 100
        long25[f'cost{cost_bps}bps'] = {
            'ann_excess_pct': round(float(net.mean() * 252), 1),
            'sharpe': round(float(net.mean() / net.std() * np.sqrt(252)), 2),
            't_nw': round(nw_t(net.dropna()), 2),
        }
    out['long_k25_5d'] = long25
    return out


def main():
    import lightgbm as lgb
    from factors.alpha158 import get_alpha158_feature_names
    from factors.alpha158_ext import get_extended_feature_names
    from factors.alpha158_news import get_news_proxy_feature_names, get_macro_feature_names

    print('=' * 60)
    print('尾部针对性模型实验 (3-class tail classifier)')
    print('=' * 60)

    df = pd.read_parquet(DATA_PARQUET)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', LABEL_COL, 'future_return_1d'])
    print(f'数据: {df.shape}, {df["symbol"].nunique()} 只, '
          f'{df["date"].nunique()} 天')

    all_names = (get_alpha158_feature_names() + get_extended_feature_names()
                 + get_news_proxy_feature_names() + get_macro_feature_names())
    feature_cols = [c for c in all_names if c in df.columns]

    df['tail_label'] = make_tail_labels(df)

    splits = walk_forward_split(df, train_months=12, purge_days=10, embargo_days=5)
    print(f'walk-forward 窗口: {len(splits)}')

    oos_frames = []
    for idx, split in enumerate(splits):
        train, val, test = split['train'], split['val'], split['test']
        selected = select_features_by_ic(
            train, feature_cols, LABEL_COL, min_ic=0.03, top_k=80,
            force_include_prefixes=FORCE_PREFIXES,
            max_per_prefix_group={'macro': 5},
        )
        if len(selected) < 5:
            print(f'w{idx+1}: 有效因子不足，跳过')
            continue

        clf = lgb.LGBMClassifier(
            objective='multiclass', num_class=3,
            n_estimators=600, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        clf.fit(
            train[selected], train['tail_label'],
            eval_set=[(val[selected], val['tail_label'])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        proba = clf.predict_proba(test[selected])
        score = proba[:, 2] - proba[:, 0]  # P(top) - P(bottom)

        frame = test[['date', 'symbol', 'future_return_1d']].copy()
        frame['score'] = score
        frame['window'] = idx + 1
        oos_frames.append(frame)
        print(f'w{idx+1}/{len(splits)} test={split["test_period"]} '
              f'best_iter={clf.best_iteration_} n_feat={len(selected)}')

    oos = pd.concat(oos_frames, ignore_index=True)
    oos = oos.sort_values(['date', 'symbol', 'window'])
    oos = oos.drop_duplicates(subset=['date', 'symbol'], keep='last')
    oos.to_parquet(OUTPUT_DIR / 'tail_model_oos_scores.parquet', index=False)

    # ── 与回归基线在相同日期交集上对比 ──
    print('\n' + '=' * 60)
    print('评估（与回归基线同口径对比）')
    print('=' * 60)

    base = pd.read_parquet(BASELINE_SCORES)
    base['date'] = pd.to_datetime(base['date']).dt.tz_localize(None).dt.normalize()
    oos_eval = oos.copy()
    oos_eval['date'] = pd.to_datetime(oos_eval['date']).dt.tz_localize(None).dt.normalize()

    ret = df[['date', 'symbol', 'future_return_1d']].copy()
    ret['date'] = ret['date'].dt.tz_localize(None).dt.normalize()
    base = base.merge(ret, on=['date', 'symbol'], how='inner').dropna()

    common_dates = sorted(set(oos_eval['date']) & set(base['date']))
    oos_eval = oos_eval[oos_eval['date'].isin(common_dates)]
    base = base[base['date'].isin(common_dates)]
    print(f'共同评估日: {len(common_dates)}')

    results = {
        'generated_at': pd.Timestamp.utcnow().isoformat(),
        'tail_q': TAIL_Q,
        'n_common_days': len(common_dates),
        'tail_model': evaluate_scores(oos_eval, 'tail_3class'),
        'baseline_regression': evaluate_scores(base, 'regression_ensemble'),
    }

    with open(OUTPUT_DIR / 'tail_model_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for m in ('tail_model', 'baseline_regression'):
        r = results[m]
        k25 = r['long_k25_5d']
        print(f"\n[{r['name']}] CS-IC={r['cs_ic']} (t={r['cs_ic_t_nw']})")
        for k in ('k5', 'k25', 'k50'):
            s = r['ls_spread'][k]
            print(f"  多空 {k}: {s['daily_mean_pct']:+.4f}%/日 "
                  f"Sharpe={s['sharpe']} t={s['t_nw']}")
        for c in ('cost10bps', 'cost20bps', 'cost30bps'):
            v = k25[c]
            print(f"  多头k25/5d {c}: 净年化={v['ann_excess_pct']:+.1f}% "
                  f"Sharpe={v['sharpe']} t={v['t_nw']}")
    print(f"\nReport: {OUTPUT_DIR / 'tail_model_report.json'}")


if __name__ == '__main__':
    main()
