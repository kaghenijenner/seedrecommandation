from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, RANDOM_SEED
from .data import feature_columns


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict
    model_name: str


def _xgboost_classifier():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
    except Exception:
        return None


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    features = feature_columns()
    categorical = [col for col in features if not is_numeric_dtype(df[col])]
    numeric = [col for col in features if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def candidate_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=RANDOM_SEED),
    }
    xgb = _xgboost_classifier()
    if xgb is not None:
        models["xgboost"] = xgb
    return models


def _evaluate_pipeline(pipe: Pipeline, x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    split_count = min(4, groups.nunique())
    cv = GroupKFold(n_splits=split_count)
    fold_scores = []
    for train_idx, test_idx in cv.split(x, y, groups):
        fitted = clone(pipe)
        fitted.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(x.iloc[test_idx])
        fold = {
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "balanced_accuracy": _balanced_accuracy_no_warning(y.iloc[test_idx].to_numpy(), pred),
            "precision": precision_score(y.iloc[test_idx], pred, zero_division=0),
            "recall": recall_score(y.iloc[test_idx], pred, zero_division=0),
            "f1": f1_score(y.iloc[test_idx], pred, zero_division=0),
            "roc_auc": np.nan,
        }
        if y.iloc[test_idx].nunique() > 1 and hasattr(fitted.named_steps["model"], "predict_proba"):
            fold["roc_auc"] = roc_auc_score(y.iloc[test_idx], fitted.predict_proba(x.iloc[test_idx])[:, 1])
        fold_scores.append(fold)

    return {metric: float(np.nanmean([fold[metric] for fold in fold_scores])) for metric in fold_scores[0]}


def _balanced_accuracy_no_warning(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def train_best_model(df: pd.DataFrame) -> TrainingResult:
    x = df[feature_columns()]
    y = df["is_suitable"]
    groups = df["district"]

    results = {}
    pipelines = {}
    for name, estimator in candidate_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor(df)), ("model", estimator)])
        metrics = _evaluate_pipeline(pipe, x, y, groups)
        results[name] = metrics
        pipelines[name] = pipe

    best_name = max(results, key=lambda name: (results[name].get("f1", 0), results[name].get("balanced_accuracy", 0)))
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(x, y)

    metrics = {
        "selected_model": best_name,
        "model_comparison": results,
        "training_rows": int(len(df)),
        "positive_rate": float(y.mean()),
    }
    return TrainingResult(best_pipeline, metrics, best_name)


def save_model(result: TrainingResult, path=MODELS_DIR / "seedrec_pipeline.joblib") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, path)


def load_model(path=MODELS_DIR / "seedrec_pipeline.joblib") -> Pipeline:
    return joblib.load(path)


def predict_suitability(pipe: Pipeline, rows: pd.DataFrame) -> np.ndarray:
    x = rows[feature_columns()]
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    return pipe.predict(x)
