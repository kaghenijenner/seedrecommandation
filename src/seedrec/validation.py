"""Spatial and temporal (grouped) cross-validation.

The proposal (sec. 3.6.4, 3.8) warns that random splits overstate performance when
agro-ecological observations are spatially or temporally autocorrelated. These helpers
re-estimate predictive performance with grouped folds so the honest generalization gap is
visible next to the optimistic random-CV numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import RANDOM_SEED
from .data import feature_columns
from .model import build_preprocessor, candidate_models
from sklearn.pipeline import Pipeline


def _season_year(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("_", n=1).str[0]


def grouped_cv_metrics(df: pd.DataFrame, group_col: str, model_name: str = "xgboost", n_splits: int = 5) -> dict:
    """Out-of-fold metrics using GroupKFold so whole groups are held out together.

    ``group_col`` may be any column (e.g. ``agro_ecological_zone`` or ``district``) or the
    special value ``"season_year"`` for hold-out-season validation.
    """
    x = df[feature_columns()]
    y = df["is_suitable"].to_numpy()

    if group_col == "season_year":
        groups = _season_year(df["season"]).to_numpy()
    else:
        groups = df[group_col].astype(str).to_numpy()

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return {"group_col": group_col, "n_groups": int(n_groups), "note": "too few groups for grouped CV"}
    splits = min(n_splits, n_groups)

    models = candidate_models()
    estimator = models.get(model_name) or next(iter(models.values()))
    pipe = Pipeline([("preprocess", build_preprocessor(df)), ("model", estimator)])

    oof = np.full(len(y), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=splits)
    for train_idx, test_idx in gkf.split(x, y, groups):
        fitted = clone(pipe)
        fitted.fit(x.iloc[train_idx], pd.Series(y[train_idx]))
        model = fitted.named_steps["model"]
        if hasattr(model, "predict_proba"):
            oof[test_idx] = fitted.predict_proba(x.iloc[test_idx])[:, 1]
        else:
            oof[test_idx] = fitted.predict(x.iloc[test_idx]).astype(float)

    mask = ~np.isnan(oof)
    pred = (oof[mask] >= 0.5).astype(int)
    y_eval = y[mask]
    auc = float(roc_auc_score(y_eval, oof[mask])) if len(np.unique(y_eval)) > 1 else None
    return {
        "group_col": group_col,
        "n_groups": int(n_groups),
        "folds": int(splits),
        "model": model_name,
        "accuracy": float(accuracy_score(y_eval, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred)),
        "f1": float(f1_score(y_eval, pred, zero_division=0)),
        "roc_auc": auc,
    }


def all_grouped_metrics(df: pd.DataFrame, model_name: str = "xgboost") -> dict:
    """Leave-zone-out, leave-district-out, and hold-out-season validation in one call."""
    return {
        "leave_zone_out": grouped_cv_metrics(df, "agro_ecological_zone", model_name),
        "leave_district_out": grouped_cv_metrics(df, "district", model_name),
        "hold_out_season": grouped_cv_metrics(df, "season_year", model_name),
    }
