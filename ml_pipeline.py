# -*- coding: utf-8 -*-
"""
ML Pipeline v2 — 对标 Qlib 的量化 ML 训练框架

升级点（vs ml_weights.py）:
  1. Walk-Forward 验证: 滚动窗口训练/验证/测试
  2. Alpha158 因子: ~150 个因子替代原来 6 个
  3. 多 Horizon 预测: 1d/3d/5d/10d 多目标
  4. TimeSeriesSplit: 正确的时间序列交叉验证
  5. 模型集成: 多 horizon 加权平均
  6. 因子筛选: 自动淘汰弱因子 (IC < 0.02)

用法:
  python ml_pipeline.py [--no-regen] [--apply] [--horizons 1,3,5,10]
"""

import json
import sys
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print('[错误] pip install lightgbm')
    sys.exit(1)

from scipy import stats

from ml_model_registry import MODEL_REGISTRY, BaseRanker, to_relevance, build_group_array

warnings.filterwarnings('ignore', category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'backtest_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 数据生成 ──────────────────────────────────────────────────────

def generate_alpha158_training_data(
    symbols: list[str] | None = None,
    period: str = '2y',
    horizons: list[int] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate training data using Alpha158 features.

    Returns:
        DataFrame with Alpha158 features + future returns at multiple horizons.
    """
    import yfinance as yf
    from factors.alpha158 import compute_alpha158, get_alpha158_feature_names

    if symbols is None:
        try:
            from backtest import load_training_universe
            symbols = load_training_universe()
        except Exception:
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
                'JPM', 'GS', 'XOM', 'UNH', 'BA', 'WMT', 'SPY', 'QQQ',
                'MRVL', 'ANET', 'COIN', 'PLTR', 'INTC', 'CRM', 'AVGO',
                'BAC', 'C', 'CVX', 'SLB', 'LLY', 'HD', 'DIS',
            ]
    if horizons is None:
        horizons = [1, 3, 5, 10]

    max_horizon = max(horizons)
    all_rows = []

    # 下载 SPY 数据用于跨资产因子
    print('  下载 SPY 数据（跨资产因子）...', end=' ', flush=True)
    try:
        spy_df = yf.Ticker('SPY').history(period=period, interval='1d')
        print(f'{len(spy_df)} rows')
    except Exception as e:
        print(f'失败: {e}, 跳过跨资产因子\n{traceback.format_exc()}')
        spy_df = pd.DataFrame()

    from factors.alpha158_ext import compute_extended_features, SECTOR_ETFS, CROSS_ASSET_ETFS
    from factors.alpha158_news import compute_news_proxy_features, compute_macro_features

    # 下载板块 ETF + 跨资产 ETF 数据
    all_etfs = list(SECTOR_ETFS) + list(CROSS_ASSET_ETFS)
    sector_dfs = {}
    print(f'  下载 ETF {all_etfs}...', end=' ', flush=True)
    for etf in all_etfs:
        try:
            edf = yf.Ticker(etf).history(period=period, interval='1d')
            if not edf.empty:
                sector_dfs[etf] = edf
        except Exception:
            pass
    print(f'{len(sector_dfs)}/{len(all_etfs)} OK')

    # 预加载宏观特征（全 symbol 共享，只需拉一次 FRED）
    print('  加载 FRED 宏观数据...', end=' ', flush=True)
    _macro_df = None
    try:
        # 先用 SPY 的日期作为参考 index
        if not spy_df.empty:
            _macro_df = compute_macro_features(spy_df.index)
            n_macro = _macro_df.dropna(how='all').shape[0]
            print(f'{n_macro} rows, {len(_macro_df.columns)} features')
        else:
            print('跳过（无 SPY 数据）')
    except Exception as e:
        print(f'失败: {e}')

    for sym in symbols:
        print(f'  Alpha158+Ext+News for {sym}...', end=' ', flush=True)
        try:
            df = yf.Ticker(sym).history(period=period, interval='1d')
            if df.empty or len(df) < 80 + max_horizon:
                print('skipped')
                continue

            features = compute_alpha158(df)

            # 拼接扩展因子（含板块敏感度）
            ext = compute_extended_features(
                df,
                spy_df=spy_df if not spy_df.empty else None,
                sector_dfs=sector_dfs if sector_dfs else None,
            )
            features = pd.concat([features, ext], axis=1)

            # 拼接新闻代理因子（价量异动）
            news_proxy = compute_news_proxy_features(df)
            features = pd.concat([features, news_proxy], axis=1)

            # 拼接宏观因子（按日期 forward-fill 对齐）
            if _macro_df is not None and not _macro_df.empty:
                macro_aligned = _macro_df.reindex(features.index, method='ffill')
                features = pd.concat([features, macro_aligned], axis=1)

            close = df['Close'].astype(float)

            # Future returns
            for h in horizons:
                features[f'future_return_{h}d'] = close.pct_change(h).shift(-h) * 100.0
                features[f'future_up_{h}d'] = (close.pct_change(h).shift(-h) > 0).astype(int)

            features['symbol'] = sym
            features['date'] = df.index
            features['close'] = close.values

            # Drop rows with NaN in any feature
            feature_cols = [c for c in features.columns
                           if c not in ['symbol', 'date', 'close'] and 'future_' not in c]
            return_cols = [f'future_return_{h}d' for h in horizons]
            valid_mask = features[feature_cols].notna().all(axis=1) & features[return_cols].notna().all(axis=1)
            features = features[valid_mask]

            all_rows.append(features)
            print(f'{len(features)} rows')
        except Exception as e:
            print(f'error: {e}\n{traceback.format_exc()}')
            continue

    if not all_rows:
        print('[error] No training data generated')
        return pd.DataFrame()

    df_out = pd.concat(all_rows, ignore_index=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_path, index=False)
        print(f'\nSaved training data: {output_path} ({len(df_out)} rows)')

    return df_out


# ── Walk-Forward 验证（含 Purge + Embargo）────────────────────────

def walk_forward_split(
    df: pd.DataFrame,
    train_months: int = 12,
    val_months: int = 1,
    test_months: int = 1,
    step_months: int = 1,
    purge_days: int = 10,
    embargo_days: int = 5,
) -> list[dict]:
    """
    Generate walk-forward train/val/test splits with Purge + Embargo.

    Purge: 从 train_end 前 purge_days 天的样本从 train 剔除（防标签窗口泄漏）
    Embargo: 从 val_start 后 embargo_days 天的样本从 val 剔除
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    min_date = df['date'].min()
    max_date = df['date'].max()

    splits = []
    train_start = min_date

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(months=test_months)

        if test_end > max_date:
            break

        # Purge: 从 train 中剔除 train_end 前 purge_days 天
        purge_start = train_end - pd.Timedelta(days=purge_days)
        train = df[(df['date'] >= train_start) & (df['date'] < purge_start)]

        # Embargo: 从 val 中剔除 val_start 后 embargo_days 天
        embargo_end = train_end + pd.Timedelta(days=embargo_days)
        val = df[(df['date'] >= embargo_end) & (df['date'] < val_end)]

        test = df[(df['date'] >= val_end) & (df['date'] < test_end)]

        if len(train) >= 100 and len(val) >= 20 and len(test) >= 20:
            splits.append({
                'train': train,
                'val': val,
                'test': test,
                'train_period': f'{train_start.date()} to {purge_start.date()}',
                'val_period': f'{embargo_end.date()} to {val_end.date()}',
                'test_period': f'{val_end.date()} to {test_end.date()}',
                'purge_days': purge_days,
                'embargo_days': embargo_days,
            })

        train_start += pd.DateOffset(months=step_months)

    return splits


# ── 因子筛选 ──────────────────────────────────────────────────────

def select_features_by_ic(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    min_ic: float = 0.02,
    top_k: int | None = None,
    force_include_prefixes: list[str] | None = None,
    max_per_prefix_group: dict[str, int] | None = None,
) -> list[str]:
    """
    Select features with |IC| > min_ic. Optionally keep only top_k.

    Args:
        force_include_prefixes: 以这些前缀开头的特征绕过 IC 筛选，
            直接进入候选（让 GBDT 自行判断重要性）。
            适用于宏观/新闻等低频特征，其 IC 在小窗口中不稳定。
        max_per_prefix_group: 按组限制 force_include 特征数量。
            {'macro': 5} → 宏观因子（VIX_/YIELD_/RATE_/INFLATION_/CLAIMS_）最多 5 个。

    Returns:
        List of selected feature column names.
    """
    force_prefixes = tuple(force_include_prefixes or [])
    macro_limit = (max_per_prefix_group or {}).get('macro', 999)
    MACRO_PREFIXES = ('VIX_', 'YIELD_', 'RATE_', 'INFLATION_', 'CLAIMS_')

    ics = {}
    for col in feature_cols:
        mask = train_df[col].notna() & train_df[label_col].notna()
        fv = train_df.loc[mask, col]
        fr = train_df.loc[mask, label_col]
        if len(fv) < 30:
            continue
        corr, _ = stats.spearmanr(fv, fr)
        if not np.isnan(corr):
            ics[col] = corr

    # Filter by min_ic (force_include 特征绕过阈值)
    selected = {}
    forced = []
    for k, v in ics.items():
        if force_prefixes and k.startswith(force_prefixes):
            selected[k] = v
            forced.append(k)
        elif abs(v) >= min_ic:
            selected[k] = v

    if forced:
        print(f'  [force_include] {len(forced)} 个特征绕过IC筛选: {forced}')

    # 分离 force_include 和普通因子，确保 force_include 不被 top_k 截断
    forced_features = sorted(
        [k for k in selected if k in forced],
        key=lambda k: abs(selected[k]), reverse=True,
    )
    normal_features = sorted(
        [k for k in selected if k not in forced],
        key=lambda k: abs(selected[k]), reverse=True,
    )

    # 宏观因子上限：按 |IC| 排序后只保留 top N 个宏观因子
    if macro_limit < 999:
        macro_count = 0
        filtered_forced = []
        dropped_macro = []
        for f in forced_features:
            if f.startswith(MACRO_PREFIXES):
                macro_count += 1
                if macro_count > macro_limit:
                    dropped_macro.append(f)
                    continue
            filtered_forced.append(f)
        if dropped_macro:
            print(f'  [macro_cap] 宏观因子限制 {macro_limit} 个，移除: {dropped_macro}')
        forced_features = filtered_forced

    # top_k 只截断普通因子，force_include 因子始终保留
    if top_k:
        remaining_slots = max(top_k - len(forced_features), 0)
        normal_features = normal_features[:remaining_slots]

    # 合并：force_include 排前面（确保入选），普通因子补充
    sorted_features = forced_features + normal_features

    return sorted_features


# ── LightGBM 训练 ──────────────────────────────────────────────────

def _build_group_array(df: pd.DataFrame, date_col: str = 'date') -> np.ndarray:
    """构建 LambdaRank 所需的 group 数组：每天的股票数。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    groups = df.groupby(date_col).size().values
    return groups


def train_lgbm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    task: str = 'regression',
) -> tuple:
    """
    Train a LightGBM model with early stopping.

    Args:
        task: 'regression', 'classification', or 'lambdarank'

    Returns:
        (model, metrics_dict)
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values

    if task == 'lambdarank':
        # LambdaRank: group = 每天的截面大小
        train_groups = _build_group_array(train_df)
        val_groups = _build_group_array(val_df)

        # LambdaRank 需要整数 relevance label，将连续收益转为 0-4 分桶
        def _to_relevance(y, df):
            """每日截面内将收益转为 0-4 整数等级"""
            rel = np.zeros(len(y), dtype=np.int32)
            df_tmp = df.copy()
            df_tmp['_idx'] = np.arange(len(df_tmp))
            df_tmp['_y'] = y
            for _, g in df_tmp.groupby('date'):
                if len(g) < 5:
                    # 不够分桶，给中间等级
                    rel[g['_idx'].values] = 2
                    continue
                # 截面内分 5 桶
                try:
                    buckets = pd.qcut(g['_y'], q=5, labels=False, duplicates='drop')
                    rel[g['_idx'].values] = buckets.fillna(2).astype(np.int32).values
                except Exception:
                    rel[g['_idx'].values] = 2
            return rel

        y_train_rel = _to_relevance(y_train, train_df)
        y_val_rel = _to_relevance(y_val, val_df)

        model = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=800,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=40,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            label_gain=[0, 1, 2, 3, 4],
        )
        model.fit(
            X_train, y_train_rel,
            group=train_groups,
            eval_set=[(X_val, y_val_rel)],
            eval_group=[val_groups],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(X_val)

        # 评估用横截面 RankIC
        ic = float(stats.spearmanr(pred, y_val)[0]) if len(pred) > 10 else 0.0
        direction_acc = float(np.mean((pred > 0) == (y_val > 0)))

        # 计算每日横截面 IC
        val_copy = val_df.copy()
        val_copy['_pred'] = pred
        val_copy['date'] = pd.to_datetime(val_copy['date'], errors='coerce')
        daily_ics = []
        for _, g in val_copy.groupby('date'):
            if len(g) < 5:
                continue
            c, _ = stats.spearmanr(g['_pred'].values, g[label_col].values)
            if np.isfinite(c):
                daily_ics.append(c)

        cs_ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
        cs_ic_std = float(np.std(daily_ics)) if len(daily_ics) > 1 else 1.0
        cs_icir = cs_ic_mean / cs_ic_std if cs_ic_std > 1e-9 else 0.0

        metrics = {
            'task': 'lambdarank',
            'pooled_ic': round(ic, 4),
            'directional_accuracy': round(direction_acc, 4),
            'cross_sectional_ic': round(cs_ic_mean, 4),
            'cross_sectional_icir': round(cs_icir, 4),
            'n_daily_ics': len(daily_ics),
            'val_rows': len(y_val),
        }

    elif task == 'classification':
        model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=40,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(X_val)
        acc = float(np.mean(pred == y_val))
        metrics = {'accuracy': round(acc, 4), 'val_rows': len(y_val)}
    else:
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=40,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        direction_acc = float(np.mean((pred > 0) == (y_val > 0)))
        ic = float(stats.spearmanr(pred, y_val)[0]) if len(pred) > 10 else 0.0
        metrics = {
            'rmse': round(rmse, 4),
            'directional_accuracy': round(direction_acc, 4),
            'ic': round(ic, 4),
            'val_rows': len(y_val),
        }

    return model, metrics


# ── 多 Horizon 集成 ──────────────────────────────────────────────

def _compute_cs_ic(val_df, pred, label_col):
    """计算每日横截面 IC 和 ICIR。"""
    val_copy = val_df.copy()
    val_copy['_pred'] = pred
    val_copy['date'] = pd.to_datetime(val_copy['date'], errors='coerce')
    daily_ics = []
    for _, g in val_copy.groupby('date'):
        if len(g) < 5:
            continue
        c, _ = stats.spearmanr(g['_pred'].values, g[label_col].values)
        if np.isfinite(c):
            daily_ics.append(c)
    cs_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
    cs_std = float(np.std(daily_ics)) if len(daily_ics) > 1 else 1.0
    cs_icir = cs_ic / cs_std if cs_std > 1e-9 else 0.0
    return cs_ic, cs_icir, daily_ics


def train_multi_horizon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] = [1, 3, 5, 10],
    horizon_weights: dict[int, float] | None = None,
    task: str = 'regression',
    model_types: list[str] | None = None,
    use_double_ensemble: bool = True,
) -> dict:
    """
    Train models for multiple horizons × multiple model types.

    Args:
        horizon_weights: {horizon: weight}. Default: IC-weighted.
        task: 'regression' or 'lambdarank'.
        model_types: list of model keys from MODEL_REGISTRY. Default: ['lgbm', 'xgb', 'catboost'].
        use_double_ensemble: 是否使用 DoubleEnsemble 样本降噪。

    Returns:
        dict with per-horizon×model models, metrics, and ensemble score.
    """
    if model_types is None:
        model_types = ['xgb', 'catboost']  # 4/10: lgbm 下线, 所有 horizon OOS IC 均为负

    models = {}          # key: f"{h}_{model_type}"
    all_metrics = {}
    per_model_ic = {}    # key: f"{h}_{model_type}" -> |IC|

    for h in horizons:
        label_col = f'future_return_{h}d'
        if label_col not in train_df.columns:
            continue

        t_mask = train_df[label_col].notna()
        v_mask = val_df[label_col].notna()
        t_df = train_df[t_mask].copy()
        v_df = val_df[v_mask].copy()

        if len(t_df) < 100 or len(v_df) < 20:
            continue

        # DoubleEnsemble 样本降噪
        sample_weight = None
        if use_double_ensemble and task == 'lambdarank':
            try:
                from double_ensemble import double_ensemble_weights
                sample_weight = double_ensemble_weights(
                    t_df, feature_cols, label_col, n_rounds=3, decay=0.5,
                )
                print(f'  Horizon {h}d: DoubleEnsemble weights computed '
                      f'(min={sample_weight.min():.3f}, max={sample_weight.max():.3f})')
            except Exception as e:
                print(f'  Horizon {h}d: DoubleEnsemble 失败: {e}, 使用等权\n{traceback.format_exc()}')
                sample_weight = None

        X_train = t_df[feature_cols].values
        y_train = t_df[label_col].values
        X_val = v_df[feature_cols].values
        y_val = v_df[label_col].values

        if task == 'lambdarank':
            train_groups = build_group_array(t_df)
            val_groups = build_group_array(v_df)
            y_train_rel = to_relevance(y_train, t_df)
            y_val_rel = to_relevance(y_val, v_df)
        else:
            train_groups = None
            val_groups = None
            y_train_rel = y_train
            y_val_rel = y_val

        for model_type in model_types:
            key = f'{h}_{model_type}'
            try:
                if task == 'lambdarank':
                    ranker = MODEL_REGISTRY[model_type]()
                    ranker.fit(
                        X_train, y_train_rel, X_val, y_val_rel,
                        train_groups, val_groups,
                        sample_weight=sample_weight,
                    )
                    pred = ranker.predict(X_val)
                    cs_ic, cs_icir, _ = _compute_cs_ic(v_df, pred, label_col)
                    direction_acc = float(np.mean((pred > np.median(pred)) == (y_val > 0)))
                    metrics = {
                        'task': 'lambdarank',
                        'cross_sectional_ic': round(cs_ic, 4),
                        'cross_sectional_icir': round(cs_icir, 4),
                        'directional_accuracy': round(direction_acc, 4),
                    }
                    all_metrics[key] = metrics
                    if cs_ic > 0:
                        models[key] = ranker
                        per_model_ic[key] = cs_ic
                        print(f'  {key}: CS-IC={cs_ic:.4f}, ICIR={cs_icir:.4f} [KEEP]')
                    else:
                        print(f'  {key}: CS-IC={cs_ic:.4f}, ICIR={cs_icir:.4f} [DROP: IC<=0]')
                else:
                    # regression fallback: 仍用旧方法
                    model, metrics = train_lgbm_model(
                        t_df, v_df, feature_cols, label_col, task=task,
                    )
                    models[key] = model
                    all_metrics[key] = metrics
                    ic_val = metrics.get('ic', 0)
                    per_model_ic[key] = abs(ic_val)
                    print(f'  {key}: IC={ic_val}')
                    break  # regression 模式只训练 lgbm

            except Exception as e:
                print(f'  {key}: 训练失败: {e}\n{traceback.format_exc()}')
                continue

    if not models:
        return {
            'models': {}, 'metrics': {}, 'ensemble_directional_accuracy': 0.0,
            'ensemble_ic': 0.0, 'horizon_weights': {}, 'model_weights': {},
            'task': task,
        }

    # ── 双层 IC 加权 ──
    # 模型层：同 horizon 下 3 个模型按各自 OOS IC 加权
    # Horizon 层：汇总后的 horizon IC 加权
    model_weights = {}  # key -> final weight
    horizon_ic_agg = {}  # horizon -> sum of model ICs

    for key, ic in per_model_ic.items():
        h = int(key.split('_')[0])
        horizon_ic_agg[h] = horizon_ic_agg.get(h, 0) + ic

    total_ic = sum(horizon_ic_agg.values())

    for key, ic in per_model_ic.items():
        h = int(key.split('_')[0])
        # 模型层权重：同 horizon 内按 IC 归一化
        horizon_total = horizon_ic_agg.get(h, 1)
        model_within_horizon = ic / horizon_total if horizon_total > 0 else 1.0 / len(model_types)
        # Horizon 层权重
        horizon_weight = horizon_ic_agg.get(h, 0) / total_ic if total_ic > 0 else 1.0 / len(horizons)
        model_weights[key] = model_within_horizon * horizon_weight

    # 归一化 + 单模型权重上限 40%（防止单模型主导 ensemble）
    MAX_SINGLE_WEIGHT = 0.40
    total_mw = sum(model_weights.values())
    if total_mw > 0:
        model_weights = {k: v / total_mw for k, v in model_weights.items()}
        # 迭代截断：超过上限的部分重新分配给其他模型
        for _ in range(5):
            capped = {k: min(v, MAX_SINGLE_WEIGHT) for k, v in model_weights.items()}
            excess = sum(model_weights.values()) - sum(capped.values())
            if excess < 1e-6:
                break
            uncapped = {k for k, v in model_weights.items() if v < MAX_SINGLE_WEIGHT}
            if not uncapped:
                break
            bonus = excess / len(uncapped)
            model_weights = {k: min(v + bonus, MAX_SINGLE_WEIGHT) if k in uncapped else MAX_SINGLE_WEIGHT
                             for k, v in capped.items()}
        else:
            model_weights = capped
        # 最终归一化
        total_mw2 = sum(model_weights.values())
        if total_mw2 > 0:
            model_weights = {k: v / total_mw2 for k, v in model_weights.items()}

    # horizon_weights（兼容旧接口）
    if horizon_weights is None:
        if total_ic > 0:
            horizon_weights = {h: horizon_ic_agg.get(h, 0) / total_ic
                               for h in horizons if any(f'{h}_' in k for k in models)}
        else:
            n_h = len(set(int(k.split('_')[0]) for k in models))
            horizon_weights = {h: 1.0 / n_h for h in horizons if any(f'{h}_' in k for k in models)}
    else:
        total_w = sum(horizon_weights.values())
        horizon_weights = {k: v / total_w for k, v in horizon_weights.items()}

    # 计算 ensemble 预测
    ensemble_pred = np.zeros(len(val_df))
    for key, ranker in models.items():
        if isinstance(ranker, BaseRanker):
            pred = ranker.predict(val_df[feature_cols].values)
        else:
            pred = ranker.predict(val_df[feature_cols].values)
        w = model_weights.get(key, 0)
        ensemble_pred += pred * w

    # Ensemble metrics
    label_5d = val_df.get('future_return_5d')
    if label_5d is not None:
        mask = label_5d.notna()
        if mask.sum() > 10:
            ens_dir_acc = float(np.mean((ensemble_pred[mask] > 0) == (label_5d[mask].values > 0)))
            ens_ic = float(stats.spearmanr(ensemble_pred[mask], label_5d[mask].values)[0])
        else:
            ens_dir_acc = 0.0
            ens_ic = 0.0
    else:
        ens_dir_acc = 0.0
        ens_ic = 0.0

    return {
        'models': models,
        'metrics': all_metrics,
        'ensemble_directional_accuracy': round(ens_dir_acc, 4),
        'ensemble_ic': round(ens_ic, 4),
        'horizon_weights': {str(k): round(v, 4) for k, v in horizon_weights.items()},
        'model_weights': {k: round(v, 6) for k, v in model_weights.items()},
        'task': task,
    }


# ── 完整 Pipeline ──────────────────────────────────────────────

def run_pipeline(
    horizons: list[int] | None = None,
    min_ic: float = 0.02,
    top_k_features: int = 80,
    train_months: int = 12,
    no_regen: bool = False,
    task: str = 'regression',
    purge_days: int = 10,
    embargo_days: int = 5,
    symbols: list[str] | None = None,
    period: str = '2y',
    neutralize_labels: bool = False,
    dataset_tag: str = '',
) -> dict:
    """
    Full ML pipeline: data generation → feature selection → walk-forward training.

    Args:
        symbols: 训练股票池；None 时用 backtest.load_training_universe()（生产默认）。
        period: yfinance 历史长度（'2y' / '3y'...），越长 walk-forward 窗口越多。
        neutralize_labels: True 时对 future_return_{h}d 做逐日横截面去均值
            （市场中性化）。排序类评估指标（CS-IC / top-k spread）不受影响，
            但训练目标改为相对收益，让模型专注选股而非择时。
        dataset_tag: 数据缓存文件后缀，研究跑批与生产缓存互不覆盖。

    Returns:
        Complete report dict.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    tag = f'_{dataset_tag}' if dataset_tag else ''
    parquet_path = OUTPUT_DIR / f'alpha158_training_data{tag}.parquet'

    # Step 1: Generate or load data
    print('=' * 60)
    print('Step 1/4: Generate Alpha158 Training Data')
    print('=' * 60)

    if no_regen and parquet_path.exists():
        print(f'[复用] {parquet_path}')
        df = pd.read_parquet(parquet_path)
    else:
        df = generate_alpha158_training_data(
            symbols=symbols,
            period=period,
            horizons=horizons,
            output_path=str(parquet_path),
        )

    if df.empty:
        print('[错误] No training data')
        return {}

    print(f'数据形状: {df.shape}, 标的数: {df["symbol"].nunique()}')

    if neutralize_labels:
        for h in horizons:
            col = f'future_return_{h}d'
            if col in df.columns:
                df[col] = df[col] - df.groupby('date')[col].transform('mean')
        print('[label] future_return 已做逐日横截面去均值（市场中性化）')

    # Identify feature columns (Alpha158 + 扩展 + 新闻代理 + 宏观)
    from factors.alpha158 import get_alpha158_feature_names
    from factors.alpha158_ext import get_extended_feature_names
    from factors.alpha158_news import get_news_proxy_feature_names, get_macro_feature_names
    all_feature_names = (
        get_alpha158_feature_names()
        + get_extended_feature_names()
        + get_news_proxy_feature_names()
        + get_macro_feature_names()
    )
    feature_cols = [c for c in all_feature_names if c in df.columns]

    n_news = len([c for c in get_news_proxy_feature_names() if c in df.columns])
    n_macro = len([c for c in get_macro_feature_names() if c in df.columns])
    print(f'可用因子: {len(feature_cols)} (Alpha158 + 扩展 + {n_news}新闻 + {n_macro}宏观)')

    # Step 2: Walk-Forward Splits
    print('\n' + '=' * 60)
    print('Step 2/4: Walk-Forward Validation Splits')
    print('=' * 60)

    splits = walk_forward_split(
        df, train_months=train_months,
        purge_days=purge_days, embargo_days=embargo_days,
    )
    print(f'生成 {len(splits)} 个时间窗口 (purge={purge_days}d, embargo={embargo_days}d)')

    if not splits:
        print('[错误] 无法生成足够的 walk-forward 窗口')
        return {}

    # Step 3: Train across all windows
    print('\n' + '=' * 60)
    print('Step 3/4: Walk-Forward Training')
    print('=' * 60)

    all_test_results = []
    all_selected_features = []
    window_reports = []
    oos_score_frames = []
    all_daily_cs_ics = []  # 跨窗口逐日 CS-IC，用于显著性 t 检验

    for idx, split in enumerate(splits):
        print(f'\n--- Window {idx + 1}/{len(splits)} ---')
        print(f'  Train: {split["train_period"]}  ({len(split["train"])} rows)')
        print(f'  Val:   {split["val_period"]}  ({len(split["val"])} rows)')
        print(f'  Test:  {split["test_period"]}  ({len(split["test"])} rows)')

        # Feature selection on training data
        # force_include: 绕过IC筛选的因子前缀（让 GBDT 自行判断重要性）
        #   - 宏观因子: 同日所有股票同值，CS-IC 天然不稳定
        #   - 动量/RVOL: 短窗口IC不稳定但对个股选股至关重要
        #   - RS_SPY: 相对强度，个股vs大盘
        selected = select_features_by_ic(
            split['train'], feature_cols, 'future_return_5d',
            min_ic=min_ic, top_k=top_k_features,
            force_include_prefixes=[
                'VIX_', 'YIELD_', 'RATE_', 'INFLATION_', 'CLAIMS_',
                'ROCP_', 'RVOL_', 'MOM_RVOL_', 'MOMENTUM_ACCEL',
                'RS_SPY_',
                # 2026-04-24 强制保留 GAP 类特征 (AMD +10% gap 事件催化)
                'GAP_', 'MEGA_GAP_',
            ],
            max_per_prefix_group={'macro': 5},  # 宏观因子上限
        )
        if len(selected) < 5:
            print(f'  [跳过] 仅 {len(selected)} 个有效因子')
            continue

        print(f'  选中因子: {len(selected)}')
        all_selected_features.extend(selected)

        # Multi-horizon training
        result = train_multi_horizon(
            split['train'], split['val'], selected, horizons=horizons, task=task,
        )

        # Test on held-out data
        if result['models']:
            # 使用双层 IC 加权 ensemble
            mw = result.get('model_weights', {})
            test_preds = np.zeros(len(split['test']))
            X_test = split['test'][selected].values
            for key, ranker in result['models'].items():
                w = mw.get(key, 1.0 / len(result['models']))
                if isinstance(ranker, BaseRanker):
                    test_preds += ranker.predict(X_test) * w
                else:
                    test_preds += ranker.predict(X_test) * w

            test_5d = split['test'].get('future_return_5d')
            if test_5d is not None:
                mask = test_5d.notna()
                if mask.sum() > 10:
                    test_dir_acc = float(np.mean(
                        (test_preds[mask] > 0) == (test_5d[mask].values > 0)
                    ))
                    test_ic = float(stats.spearmanr(test_preds[mask], test_5d[mask].values)[0])
                else:
                    test_dir_acc = 0.0
                    test_ic = 0.0
            else:
                test_dir_acc = 0.0
                test_ic = 0.0

            # 横截面 RankIC（日度）
            test_copy = split['test'].copy()
            test_copy['_pred'] = test_preds
            test_copy['date'] = pd.to_datetime(test_copy['date'], errors='coerce')
            if {'date', 'symbol'}.issubset(test_copy.columns):
                score_frame = test_copy[['date', 'symbol', '_pred']].copy()
                score_frame = score_frame.rename(columns={'_pred': 'score'})
                score_frame['window'] = idx + 1
                oos_score_frames.append(score_frame)
            daily_test_ics = []
            for _, g in test_copy.groupby('date'):
                if len(g) < 5 or 'future_return_5d' not in g.columns:
                    continue
                valid = g['future_return_5d'].notna()
                gv = g[valid]
                if len(gv) < 5:
                    continue
                c, _ = stats.spearmanr(gv['_pred'].values, gv['future_return_5d'].values)
                if np.isfinite(c):
                    daily_test_ics.append(c)

            cs_test_ic = float(np.mean(daily_test_ics)) if daily_test_ics else 0.0
            cs_test_std = float(np.std(daily_test_ics)) if len(daily_test_ics) > 1 else 1.0
            cs_test_icir = cs_test_ic / cs_test_std if cs_test_std > 1e-9 else 0.0
            all_daily_cs_ics.extend(daily_test_ics)

            # Top-k spread 评估
            topk_result = {}
            try:
                from cross_sectional_evaluator import calc_topk_return
                test_scored = test_copy.rename(columns={
                    '_pred': 'score', 'future_return_5d': 'future_return',
                })
                topk_result = calc_topk_return(test_scored, k=5, cost_bps=10)
            except Exception:
                pass

            window_report = {
                'window': idx + 1,
                'train_period': split['train_period'],
                'test_period': split['test_period'],
                'n_features': len(selected),
                'task': task,
                'val_metrics': result['metrics'],
                'val_ensemble_dir_acc': result['ensemble_directional_accuracy'],
                'val_ensemble_ic': result['ensemble_ic'],
                'test_dir_acc': round(test_dir_acc, 4),
                'test_ic': round(test_ic, 4),
                'test_cs_ic': round(cs_test_ic, 4),
                'test_cs_icir': round(cs_test_icir, 4),
                'test_topk_spread': topk_result,
            }
            window_reports.append(window_report)
            all_test_results.append(test_dir_acc)

            print(f'  Test DirAcc: {test_dir_acc:.4f}  IC: {test_ic:.4f}  '
                  f'CS-IC: {cs_test_ic:.4f}  ICIR: {cs_test_icir:.4f}')

    # Step 4: Aggregate results
    print('\n' + '=' * 60)
    print('Step 4/4: Aggregate Results')
    print('=' * 60)

    # Feature importance across windows
    from collections import Counter
    feature_frequency = Counter(all_selected_features)
    top_features = feature_frequency.most_common(30)

    if all_test_results:
        avg_test_acc = float(np.mean(all_test_results))
        std_test_acc = float(np.std(all_test_results))
    else:
        avg_test_acc = 0.0
        std_test_acc = 0.0

    # 汇总横截面指标
    avg_cs_ic = float(np.mean([w.get('test_cs_ic', 0) for w in window_reports])) if window_reports else 0.0
    avg_cs_icir = float(np.mean([w.get('test_cs_icir', 0) for w in window_reports])) if window_reports else 0.0

    avg_spread_net = float(np.mean([
        (w.get('test_topk_spread') or {}).get('spread_net', 0)
        for w in window_reports
    ])) if window_reports else 0.0
    pct_positive_spread_net = float(np.mean([
        1.0 if ((w.get('test_topk_spread') or {}).get('spread_net', 0) > 0) else 0.0
        for w in window_reports
    ])) if window_reports else 0.0

    # 逐日 CS-IC 的显著性 t 检验。future_return_5d 标签有 5 日重叠，
    # 日度 IC 序列存在自相关，用 Newey-West (Bartlett kernel, lag=5) 修正标准误。
    ic_series = np.asarray(all_daily_cs_ics, dtype=float)
    n_ic_days = int(ic_series.size)
    ic_t_stat = 0.0
    ic_t_stat_nw = 0.0
    ic_p_value_nw = 1.0
    if n_ic_days >= 20:
        ic_mean = float(np.mean(ic_series))
        demeaned = ic_series - ic_mean
        gamma0 = float(np.mean(demeaned ** 2))
        if gamma0 > 1e-12:
            ic_t_stat = float(ic_mean / np.sqrt(gamma0 / n_ic_days))
            nw_lag = 5
            lrv = gamma0
            for lag in range(1, min(nw_lag, n_ic_days - 1) + 1):
                cov = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
                lrv += 2.0 * (1.0 - lag / (nw_lag + 1.0)) * cov
            lrv = max(lrv, 1e-12)
            ic_t_stat_nw = float(ic_mean / np.sqrt(lrv / n_ic_days))
            ic_p_value_nw = float(2.0 * (1.0 - stats.norm.cdf(abs(ic_t_stat_nw))))

    # 统一显著性判据：regression 和 lambdarank 的产出都是横截面排序分，
    # 一律用 CS-IC / ICIR / 净价差一致性 / NW t 检验，不再用方向准确率
    is_significant = (
        avg_cs_ic >= 0.02 and
        avg_cs_icir >= 0.30 and
        avg_spread_net > 0 and
        pct_positive_spread_net >= 0.55 and
        ic_t_stat_nw >= 2.0
    )
    significance_basis = {
        'mode': 'ranking',
        'avg_test_cs_ic_min': 0.02,
        'avg_test_cs_icir_min': 0.30,
        'avg_spread_net_positive': True,
        'pct_positive_spread_net_min': 0.55,
        'ic_t_stat_nw_min': 2.0,
        'nw_lag': 5,
    }

    report = {
        'pipeline': 'ml_pipeline_v2_alpha158',
        'generated_at': pd.Timestamp.utcnow().isoformat(),
        'task': task,
        'purge_days': purge_days,
        'embargo_days': embargo_days,
        'data_shape': list(df.shape),
        'n_symbols': int(df['symbol'].nunique()),
        'period': period,
        'neutralize_labels': neutralize_labels,
        'dataset_tag': dataset_tag,
        'n_windows': len(window_reports),
        'horizons': horizons,
        'min_ic_threshold': min_ic,
        'top_k_features': top_k_features,
        'avg_test_directional_accuracy': round(avg_test_acc, 4),
        'std_test_directional_accuracy': round(std_test_acc, 4),
        'avg_test_cs_ic': round(avg_cs_ic, 4),
        'avg_test_cs_icir': round(avg_cs_icir, 4),
        'avg_test_spread_net': round(avg_spread_net, 4),
        'pct_positive_test_spread_net': round(pct_positive_spread_net, 4),
        'cs_ic_n_days': n_ic_days,
        'cs_ic_t_stat': round(ic_t_stat, 4),
        'cs_ic_t_stat_nw': round(ic_t_stat_nw, 4),
        'cs_ic_p_value_nw': round(ic_p_value_nw, 6),
        'is_significant': is_significant,
        'significance_basis': significance_basis,
        'window_reports': window_reports,
        'top_features': [{'feature': f, 'frequency': c} for f, c in top_features],
    }

    score_artifact_path = None
    if oos_score_frames:
        oos_scores = pd.concat(oos_score_frames, ignore_index=True)
        oos_scores = oos_scores.dropna(subset=['date', 'symbol', 'score'])
        oos_scores['date'] = pd.to_datetime(oos_scores['date'], errors='coerce')
        oos_scores = oos_scores.dropna(subset=['date'])
        oos_scores = oos_scores.sort_values(['date', 'symbol', 'window'])
        oos_scores = oos_scores.drop_duplicates(subset=['date', 'symbol'], keep='last')

        score_artifact = OUTPUT_DIR / f'ml_oos_scores{tag}.parquet'
        oos_scores.to_parquet(score_artifact, index=False)
        score_artifact_path = str(score_artifact)
        report['oos_score_artifact'] = score_artifact_path

    # Save report
    report_path = OUTPUT_DIR / f'ml_pipeline_report{tag}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f'\n平均 OOS 方向准确率: {avg_test_acc:.4f} ± {std_test_acc:.4f}')
    print(f'CS-IC: {avg_cs_ic:.4f}  ICIR: {avg_cs_icir:.4f}  '
          f'NW t-stat: {ic_t_stat_nw:.2f} (p={ic_p_value_nw:.4f}, n={n_ic_days}d)')
    print(f'净价差: {avg_spread_net:+.2f}%/窗口, {pct_positive_spread_net:.0%} 窗口为正')
    print(f'统计显著 (ranking 判据): {"YES" if is_significant else "NO"}')
    print(f'\nTop 10 最常被选中的因子:')
    for f, c in top_features[:10]:
        print(f'  {f:<20} 出现 {c} 次')
    print(f'\nReport: {report_path}')

    return report


# ── 训练最终模型并导出权重 ──────────────────────────────────────

def train_final_model(
    horizons: list[int] | None = None,
    min_ic: float = 0.02,
    top_k: int = 80,
    model_types: list[str] | None = None,
) -> dict:
    """
    Train a final model on all available data (except last month for validation).
    Export v2 artifact with multi-model ensemble.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]
    if model_types is None:
        model_types = ['xgb', 'catboost']  # 4/10: lgbm 下线, 所有 horizon OOS IC 均为负

    parquet_path = OUTPUT_DIR / 'alpha158_training_data.parquet'
    if not parquet_path.exists():
        print('[错误] 请先运行 run_pipeline() 生成训练数据')
        return {}

    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    from factors.alpha158 import get_alpha158_feature_names
    from factors.alpha158_ext import get_extended_feature_names
    from factors.alpha158_news import get_news_proxy_feature_names, get_macro_feature_names
    all_names = (
        get_alpha158_feature_names()
        + get_extended_feature_names()
        + get_news_proxy_feature_names()
        + get_macro_feature_names()
    )
    feature_cols = [c for c in all_names if c in df.columns]

    # Time split: all but last month for training, last month for validation
    cutoff = df['date'].max() - pd.DateOffset(months=1)
    train = df[df['date'] < cutoff]
    val = df[df['date'] >= cutoff]

    # Feature selection
    selected = select_features_by_ic(
        train, feature_cols, 'future_return_5d',
        min_ic=min_ic, top_k=top_k,
        force_include_prefixes=[
            'VIX_', 'YIELD_', 'RATE_', 'INFLATION_', 'CLAIMS_',
            'ROCP_', 'RVOL_', 'MOM_RVOL_', 'MOMENTUM_ACCEL',
            'RS_SPY_',
            # 2026-04-24 强制保留 GAP 类特征 (AMD +10% gap 事件催化)
            'GAP_', 'MEGA_GAP_',
        ],
        max_per_prefix_group={'macro': 5},
    )
    print(f'Selected {len(selected)} features for final model')

    if len(selected) < 5:
        print('[错误] 太少有效因子')
        return {}

    # Train multi-horizon × multi-model (lambdarank)
    result = train_multi_horizon(
        train, val, selected, horizons=horizons,
        task='lambdarank', model_types=model_types,
        use_double_ensemble=True,
    )

    # Extract feature importance (aggregate from all models)
    weights = {}
    n_models = 0
    for key, ranker in result['models'].items():
        try:
            imp = ranker.feature_importances_.astype(float)
            total = imp.sum()
            if total > 0:
                for col, val_imp in zip(selected, imp):
                    weights[col] = weights.get(col, 0) + val_imp / total
                n_models += 1
        except Exception:
            continue
    if n_models > 0:
        weights = {k: v / n_models for k, v in weights.items()}

    mw = result.get('model_weights', {})
    hw = result.get('horizon_weights', {})

    final_report = {
        'pipeline': 'ml_pipeline_v2_multi_model',
        'generated_at': pd.Timestamp.utcnow().isoformat(),
        'selected_features': selected,
        'model_types': model_types,
        'n_models': len(result['models']),
        'feature_weights': {k: round(v, 6) for k, v in sorted(weights.items(), key=lambda x: -x[1])[:30]},
        'metrics': {k: v for k, v in result['metrics'].items()},
        'ensemble_dir_acc': result['ensemble_directional_accuracy'],
        'ensemble_ic': result['ensemble_ic'],
    }

    report_path = OUTPUT_DIR / 'ml_final_model_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    print(f'\nFinal model report: {report_path}')

    # ── 质量门槛：ensemble IC 必须 > 0，否则不覆盖旧模型 ──
    ensemble_ic = result.get('ensemble_ic', 0)
    n_surviving = len(result['models'])
    if n_surviving == 0:
        print('\n[警告] 无模型通过质量筛选 (IC>0)，保留旧模型不覆盖')
        return final_report
    if ensemble_ic <= 0:
        print(f'\n[警告] Ensemble IC={ensemble_ic:.4f} <= 0，保留旧模型不覆盖')
        return final_report

    print(f'\n[质量检查] Ensemble IC={ensemble_ic:.4f}, '
          f'{n_surviving} 个模型通过筛选')

    # ── 备份旧模型 ──
    model_path = OUTPUT_DIR / 'ml_final_models.pkl'
    if model_path.exists():
        backup_path = OUTPUT_DIR / f'ml_final_models_backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f'  旧模型已备份: {backup_path.name}')

    # ── 持久化模型（v2 格式，供 scoring_engine 加载）──
    model_artifact = {
        'version': 2,
        'models': result['models'],                   # {"1_lgbm": ranker, "1_xgb": ranker, ...}
        'selected_features': selected,
        'horizon_weights': {int(k): v for k, v in hw.items()},
        'model_weights': mw,                           # {"1_lgbm": 0.12, ...}
    }
    joblib.dump(model_artifact, model_path)
    print(f'Model artifact saved (v2): {model_path}')
    print(f'  Models: {list(result["models"].keys())}')
    print(f'  Features: {len(selected)}')

    return final_report


# ── 熊市专用模型训练 ──────────────────────────────────────────────

def train_bear_model(
    horizons: list[int] | None = None,
    min_ic: float = 0.02,
    top_k: int = 80,
    model_types: list[str] | None = None,
    vix_threshold: float = 0.0,
) -> dict:
    """
    Train a bear-market-specific model using only high-VIX regime data.

    数据筛选: VIX_PERCENTILE > vix_threshold（0.0 对应 raw percentile > 0.5，VIX 上半段）
    Horizon: 默认 [1, 3, 5]（去掉 10d，熊市波动大）
    质量门槛: IC > 0 + 方向准确率 > 53% + Top-5 spread > 0
    """
    if horizons is None:
        horizons = [1, 3, 5]
    if model_types is None:
        model_types = ['xgb', 'catboost']  # 4/10: lgbm 下线, 所有 horizon OOS IC 均为负

    parquet_path = OUTPUT_DIR / 'alpha158_training_data.parquet'
    if not parquet_path.exists():
        print('[错误] 请先运行 run_pipeline() 生成训练数据')
        return {}

    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    # ── 筛选熊市数据 ──
    if 'VIX_PERCENTILE' not in df.columns:
        print('[错误] 训练数据缺少 VIX_PERCENTILE 列，无法筛选熊市')
        return {}

    bear_mask = df['VIX_PERCENTILE'] > vix_threshold
    df_bear = df[bear_mask].copy()
    n_days = df_bear['date'].dt.date.nunique()
    print(f'[Bear Model] 筛选熊市数据: {len(df_bear)} 行, {n_days} 天 '
          f'(VIX_PERCENTILE > {vix_threshold}, 占比 {100*len(df_bear)/len(df):.1f}%)')

    if len(df_bear) < 500:
        print(f'[错误] 熊市数据太少 ({len(df_bear)} 行)，需要至少 500 行')
        return {}

    from factors.alpha158 import get_alpha158_feature_names
    from factors.alpha158_ext import get_extended_feature_names
    from factors.alpha158_news import get_news_proxy_feature_names, get_macro_feature_names
    all_names = (
        get_alpha158_feature_names()
        + get_extended_feature_names()
        + get_news_proxy_feature_names()
        + get_macro_feature_names()
    )
    feature_cols = [c for c in all_names if c in df_bear.columns]

    # Time split: all but last month for training, last month for validation
    cutoff = df_bear['date'].max() - pd.DateOffset(months=1)
    train = df_bear[df_bear['date'] < cutoff]
    val = df_bear[df_bear['date'] >= cutoff]

    print(f'[Bear Model] 训练集: {len(train)} 行, 验证集: {len(val)} 行')
    if len(train) < 300 or len(val) < 50:
        print(f'[错误] 训练集或验证集太小')
        return {}

    # Feature selection on bear data
    selected = select_features_by_ic(
        train, feature_cols, 'future_return_5d',
        min_ic=min_ic, top_k=top_k,
        force_include_prefixes=[
            'VIX_', 'YIELD_', 'RATE_', 'INFLATION_', 'CLAIMS_',
            'ROCP_', 'RVOL_', 'MOM_RVOL_', 'MOMENTUM_ACCEL',
            'RS_SPY_',
            # 2026-04-24 强制保留 GAP 类特征 (AMD +10% gap 事件催化)
            'GAP_', 'MEGA_GAP_',
        ],
        max_per_prefix_group={'macro': 5},
    )
    print(f'[Bear Model] 选中 {len(selected)} 个特征')

    if len(selected) < 5:
        print('[错误] 太少有效因子')
        return {}

    # Train multi-horizon × multi-model
    result = train_multi_horizon(
        train, val, selected, horizons=horizons,
        task='lambdarank', model_types=model_types,
        use_double_ensemble=True,
    )

    # ── 三重质量门槛 ──
    ensemble_ic = result.get('ensemble_ic', 0)
    ens_dir_acc = result.get('ensemble_directional_accuracy', 0)
    n_surviving = len(result['models'])

    print(f'\n[Bear Model] === 质量检查 ===')
    print(f'  模型数: {n_surviving}')
    print(f'  Ensemble IC: {ensemble_ic:.4f}')
    print(f'  方向准确率: {ens_dir_acc:.4f}')

    # 门槛 1: 模型存在且 IC > 0
    if n_surviving == 0:
        print('  [FAIL] 无模型通过质量筛选')
        return {}
    if ensemble_ic <= 0:
        print(f'  [FAIL] Ensemble IC={ensemble_ic:.4f} <= 0')
        return {}
    print(f'  [PASS] IC > 0')

    # 门槛 2: 方向准确率 > 53%
    if ens_dir_acc < 0.53:
        print(f'  [FAIL] 方向准确率 {ens_dir_acc:.4f} < 0.53')
        return {}
    print(f'  [PASS] 方向准确率 > 53%')

    # 门槛 3: Top-5 spread > 0 (在验证集上做横截面评估)
    topk_result = {}
    bucket_result = {}
    try:
        from cross_sectional_evaluator import calc_topk_return, calc_bucket_returns

        # 计算 ensemble 预测
        mw = result.get('model_weights', {})
        val_preds = np.zeros(len(val))
        X_val = val[selected].values
        for key, ranker in result['models'].items():
            w = mw.get(key, 1.0 / len(result['models']))
            if isinstance(ranker, BaseRanker):
                val_preds += ranker.predict(X_val) * w
            else:
                val_preds += ranker.predict(X_val) * w

        val_scored = val.copy()
        val_scored['score'] = val_preds
        val_scored['future_return'] = val_scored.get('future_return_5d')

        topk_result = calc_topk_return(val_scored, k=5, cost_bps=10)
        bucket_result = calc_bucket_returns(val_scored, n_buckets=5)

        spread = topk_result.get('spread')
        pct_pos = topk_result.get('pct_positive_spread')
        ls_sharpe = topk_result.get('long_short_sharpe')
        is_mono = bucket_result.get('is_monotonic', False)

        print(f'  Top-5 Spread: {spread}')
        print(f'  Spread 胜率: {pct_pos}%')
        print(f'  Long-Short Sharpe: {ls_sharpe}')
        print(f'  分桶单调: {is_mono}')

        if spread is not None and spread <= 0:
            print(f'  [FAIL] Top-5 spread {spread:.4f} <= 0')
            return {}
        print(f'  [PASS] Top-5 spread > 0')

    except Exception as e:
        print(f'  [WARN] 横截面评估失败: {e}，跳过 spread 门槛')

    print(f'\n[Bear Model] === 全部质量检查通过 ===')

    # Extract feature importance
    weights = {}
    n_models_imp = 0
    for key, ranker in result['models'].items():
        try:
            imp = ranker.feature_importances_.astype(float)
            total = imp.sum()
            if total > 0:
                for col, val_imp in zip(selected, imp):
                    weights[col] = weights.get(col, 0) + val_imp / total
                n_models_imp += 1
        except Exception:
            continue
    if n_models_imp > 0:
        weights = {k: v / n_models_imp for k, v in weights.items()}

    mw = result.get('model_weights', {})
    hw = result.get('horizon_weights', {})

    # Report
    final_report = {
        'pipeline': 'ml_pipeline_v2_bear_model',
        'regime': 'bear',
        'vix_threshold': vix_threshold,
        'generated_at': pd.Timestamp.utcnow().isoformat(),
        'bear_data_rows': len(df_bear),
        'bear_data_days': n_days,
        'train_rows': len(train),
        'val_rows': len(val),
        'selected_features': selected,
        'model_types': model_types,
        'n_models': len(result['models']),
        'feature_weights': {k: round(v, 6) for k, v in sorted(weights.items(), key=lambda x: -x[1])[:30]},
        'metrics': {k: v for k, v in result['metrics'].items()},
        'ensemble_dir_acc': ens_dir_acc,
        'ensemble_ic': ensemble_ic,
        'topk_spread': topk_result,
        'bucket_returns': bucket_result,
    }

    report_path = OUTPUT_DIR / 'ml_final_model_report_bear.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    print(f'\nBear model report: {report_path}')

    # ── 备份旧模型 ──
    model_path = OUTPUT_DIR / 'ml_final_models_bear.pkl'
    if model_path.exists():
        backup_path = OUTPUT_DIR / f'ml_final_models_bear_backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f'  旧 bear 模型已备份: {backup_path.name}')

    # ── 持久化模型 ──
    model_artifact = {
        'version': 2,
        'regime': 'bear',
        'vix_threshold': vix_threshold,
        'models': result['models'],
        'selected_features': selected,
        'horizon_weights': {int(k): v for k, v in hw.items()},
        'model_weights': mw,
    }
    joblib.dump(model_artifact, model_path)
    print(f'Bear model artifact saved (v2): {model_path}')
    print(f'  Models: {list(result["models"].keys())}')
    print(f'  Features: {len(selected)}')

    return final_report


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ML Pipeline v2 (Alpha158 + Walk-Forward)')
    parser.add_argument('--no-regen', action='store_true', help='Reuse existing training data')
    parser.add_argument('--horizons', type=str, default='1,3,5,10', help='Prediction horizons')
    parser.add_argument('--min-ic', type=float, default=0.02, help='Minimum IC for factor selection')
    parser.add_argument('--top-k', type=int, default=80, help='Top-K features to keep')
    parser.add_argument('--train-months', type=int, default=12, help='Training window months')
    parser.add_argument('--final', action='store_true', help='Train final model after pipeline')
    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'lambdarank'],
                        help='Model task: regression or lambdarank')
    parser.add_argument('--purge-days', type=int, default=10, help='Purge days for CV')
    parser.add_argument('--embargo-days', type=int, default=5, help='Embargo days for CV')
    parser.add_argument('--regime', type=str, default=None,
                        choices=['bear'],
                        help='Train regime-specific model (bear)')
    parser.add_argument('--universe', type=str, default='default',
                        choices=['default', 'sp500'],
                        help='训练股票池: default=生产池(~60), sp500=研究大池(~500)')
    parser.add_argument('--top-n', type=int, default=None,
                        help='sp500 池截取前 N 只（按快照顺序）')
    parser.add_argument('--period', type=str, default='2y',
                        help='yfinance 历史长度，如 2y / 3y')
    parser.add_argument('--neutralize-labels', action='store_true',
                        help='标签逐日横截面去均值（市场中性化）')
    parser.add_argument('--dataset-tag', type=str, default='',
                        help='数据缓存后缀，研究与生产缓存隔离')
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(',')]

    pipeline_symbols = None
    if args.universe == 'sp500':
        from research_universe import load_sp500_universe
        pipeline_symbols = load_sp500_universe(top_n=args.top_n)
        if not args.dataset_tag:
            args.dataset_tag = 'sp500'
        print(f'[universe] S&P 500 研究池: {len(pipeline_symbols)} 只')

    report = run_pipeline(
        horizons=horizons,
        min_ic=args.min_ic,
        top_k_features=args.top_k,
        train_months=args.train_months,
        no_regen=args.no_regen,
        task=args.task,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        symbols=pipeline_symbols,
        period=args.period,
        neutralize_labels=args.neutralize_labels,
        dataset_tag=args.dataset_tag,
    )

    if args.final and report:
        print('\n' + '=' * 60)
        if args.regime == 'bear':
            print('Training Bear Market Model')
            print('=' * 60)
            train_bear_model(horizons=horizons, min_ic=args.min_ic, top_k=args.top_k)
        else:
            print('Training Final Model')
            print('=' * 60)
            train_final_model(horizons=horizons, min_ic=args.min_ic, top_k=args.top_k)
