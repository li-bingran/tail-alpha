# -*- coding: utf-8 -*-
"""
ML Model Registry — 统一 LightGBM / XGBoost / CatBoost 训练和预测接口

所有模型使用 LambdaRank 目标（排序学习），超参数量级一致。
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


def to_relevance(y: np.ndarray, df: pd.DataFrame, n_buckets: int = 5) -> np.ndarray:
    """
    每日截面内将连续收益转为 0-(n_buckets-1) 整数等级。
    三种模型共用此逻辑。
    """
    rel = np.zeros(len(y), dtype=np.int32)
    df_tmp = df.copy()
    df_tmp['_idx'] = np.arange(len(df_tmp))
    df_tmp['_y'] = y
    for _, g in df_tmp.groupby('date'):
        if len(g) < n_buckets:
            rel[g['_idx'].values] = n_buckets // 2
            continue
        try:
            buckets = pd.qcut(g['_y'], q=n_buckets, labels=False, duplicates='drop')
            rel[g['_idx'].values] = buckets.fillna(n_buckets // 2).astype(np.int32).values
        except Exception:
            rel[g['_idx'].values] = n_buckets // 2
    return rel


def build_group_array(df: pd.DataFrame, date_col: str = 'date') -> np.ndarray:
    """构建 LambdaRank 所需的 group 数组：每天的股票数。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    return df.groupby(date_col).size().values


class BaseRanker(ABC):
    """统一模型接口。"""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_groups: np.ndarray,
        val_groups: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """训练模型，返回验证集指标。"""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测排序分数。"""
        ...

    @abstractmethod
    def get_model_object(self) -> Any:
        """返回底层模型对象。"""
        ...

    @property
    def feature_importances_(self) -> np.ndarray:
        """特征重要性。"""
        obj = self.get_model_object()
        if hasattr(obj, 'feature_importances_'):
            return obj.feature_importances_
        return np.array([])


class LGBMRanker(BaseRanker):
    """LightGBM LambdaRank 包装。"""

    def __init__(self):
        self._model = None
        self._feature_names: list[str] | None = None

    def fit(self, X_train, y_train, X_val, y_val, train_groups, val_groups,
            sample_weight=None):
        import lightgbm as lgb
        self._model = lgb.LGBMRanker(
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
            label_gain=list(range(5)),
        )
        fit_params = dict(
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        self._model.fit(X_train, y_train, **fit_params)
        # 记录 feature names 用于 predict 时消除 warnings
        if hasattr(self._model, 'feature_name_'):
            self._feature_names = self._model.feature_name_
        return {}

    def predict(self, X):
        if isinstance(X, np.ndarray) and self._feature_names and X.shape[1] == len(self._feature_names):
            X = pd.DataFrame(X, columns=self._feature_names)
        return self._model.predict(X)

    def get_model_object(self):
        return self._model


class XGBRanker(BaseRanker):
    """XGBoost Ranking 包装。"""

    def __init__(self):
        self._model = None

    def fit(self, X_train, y_train, X_val, y_val, train_groups, val_groups,
            sample_weight=None):
        import xgboost as xgb
        self._model = xgb.XGBRanker(
            objective='rank:ndcg',
            n_estimators=800,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            tree_method='hist',
        )
        fit_params = dict(
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            verbose=False,
        )
        # XGBRanker 的 sample_weight 需要按 group 聚合（每组一个权重）
        if sample_weight is not None and train_groups is not None:
            group_weights = []
            offset = 0
            for g_size in train_groups:
                group_weights.append(float(sample_weight[offset:offset + g_size].mean()))
                offset += g_size
            fit_params['sample_weight'] = np.array(group_weights)
        self._model.fit(X_train, y_train, **fit_params)
        return {}

    def predict(self, X):
        return self._model.predict(X)

    def get_model_object(self):
        return self._model


class CatBoostRanker(BaseRanker):
    """CatBoost YetiRank 包装。"""

    def __init__(self):
        self._model = None

    def fit(self, X_train, y_train, X_val, y_val, train_groups, val_groups,
            sample_weight=None):
        from catboost import CatBoost, Pool

        # CatBoost 需要展开 group 为 group_id 数组
        def _groups_to_ids(groups):
            ids = []
            for gid, size in enumerate(groups):
                ids.extend([gid] * size)
            return np.array(ids)

        train_pool = Pool(
            data=X_train, label=y_train,
            group_id=_groups_to_ids(train_groups),
            weight=sample_weight,
        )
        val_pool = Pool(
            data=X_val, label=y_val,
            group_id=_groups_to_ids(val_groups),
        )

        self._model = CatBoost(dict(
            loss_function='YetiRank',
            iterations=800,
            learning_rate=0.02,
            depth=6,
            subsample=0.8,
            rsm=0.6,  # colsample_bytree equivalent
            l2_leaf_reg=0.1,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
        ))
        self._model.fit(train_pool, eval_set=val_pool)
        return {}

    def predict(self, X):
        return self._model.predict(X).flatten()

    def get_model_object(self):
        return self._model

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is not None:
            return np.array(self._model.get_feature_importance())
        return np.array([])


class NormalizedRegressor:
    """包装回归模型, predict 时自动归一化到旧 ranker 分数范围。

    旧 LambdaRank ranker 输出 std ≈ 0.32, 新 XGBRegressor 预测 future_return_%
    输出 std ≈ 1.64。归一化后确保下游阈值 (buy_th, sell_th, score*80) 不失效。
    """
    def __init__(self, model, normalizer=5.0):
        self._model = model
        self._normalizer = normalizer

    def predict(self, X):
        return self._model.predict(X) / self._normalizer

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    def get_model_object(self):
        return self._model


MODEL_REGISTRY = {
    'lgbm': LGBMRanker,
    'xgb': XGBRanker,
    'catboost': CatBoostRanker,
}
