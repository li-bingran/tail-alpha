# -*- coding: utf-8 -*-
"""
DoubleEnsemble 样本降噪

原理：
1. 浅 LightGBM（depth=3）在 train 上快速拟合
2. 残差大的样本（噪声）降权（指数衰减）
3. 重复 n_rounds 轮

返回 sample_weights，归一化到均值=1。
"""

import numpy as np
import pandas as pd


def double_ensemble_weights(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    n_rounds: int = 3,
    decay: float = 0.5,
) -> np.ndarray:
    """
    计算 DoubleEnsemble 样本权重。

    Args:
        train_df: 训练数据
        feature_cols: 特征列
        label_col: 标签列
        n_rounds: 迭代轮数
        decay: 残差衰减系数（越小降噪越激进）

    Returns:
        sample_weights: shape=(n_samples,), 均值=1 的权重数组
    """
    import lightgbm as lgb

    X = train_df[feature_cols].values
    y = train_df[label_col].values

    n_samples = len(y)
    weights = np.ones(n_samples, dtype=np.float64)

    for round_idx in range(n_rounds):
        # 浅模型快速拟合
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            subsample=0.8,
            colsample_bytree=0.6,
            n_jobs=-1,
            random_state=42 + round_idx,
            verbose=-1,
        )
        model.fit(X, y, sample_weight=weights)

        # 计算残差
        pred = model.predict(X)
        residuals = np.abs(y - pred)

        # 残差归一化到 [0, 1]
        r_min, r_max = residuals.min(), residuals.max()
        if r_max > r_min:
            norm_residuals = (residuals - r_min) / (r_max - r_min)
        else:
            norm_residuals = np.zeros_like(residuals)

        # 指数衰减：残差大的样本降权
        round_weights = np.exp(-decay * norm_residuals)

        # 累积权重
        weights *= round_weights

    # 归一化到均值=1
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    else:
        weights = np.ones(n_samples, dtype=np.float64)

    return weights
