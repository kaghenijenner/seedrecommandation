from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, RANDOM_SEED
from .data import feature_columns


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict
    model_name: str


def _xgboost_classifier(**overrides):
    try:
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
        )
        params.update(overrides)
        return XGBClassifier(**params)
    except Exception:
        return None


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    features = feature_columns()
    categorical = [col for col in features if not is_numeric_dtype(df[col])]
    numeric = [col for col in features if col not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def candidate_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, l2_regularization=1.0, random_state=RANDOM_SEED
        ),
    }
    xgb = _xgboost_classifier()
    if xgb is not None:
        models["xgboost"] = xgb
    return models


def _tune_xgboost(preprocessor: ColumnTransformer, x: pd.DataFrame, y: pd.Series):
    """Light randomized search so the proposal's hero model (XGBoost) is competitive."""
    base = _xgboost_classifier()
    if base is None:
        return None, {}
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", base)])
    param_dist = {
        "model__n_estimators": [300, 400, 600],
        "model__max_depth": [5, 6, 8, 10],
        "model__learning_rate": [0.05, 0.1, 0.15, 0.2],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__min_child_weight": [1, 2, 4],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=16,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x, y)
    tuned = search.best_estimator_.named_steps["model"]
    best_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    return tuned, best_params


def _model_scores(fitted: Pipeline, x: pd.DataFrame) -> np.ndarray:
    model = fitted.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return fitted.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = fitted.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return fitted.predict(x).astype(float)


def _best_threshold_metrics(y_true: pd.Series, scores: np.ndarray) -> dict:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_metrics = None

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        metrics = {
            "decision_threshold": float(threshold),
            "accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": _balanced_accuracy_no_warning(y_true.to_numpy(), pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        }
        if best_metrics is None or (
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1"],
        ) > (
            best_metrics["accuracy"],
            best_metrics["balanced_accuracy"],
            best_metrics["f1"],
        ):
            best_metrics = metrics

    return best_metrics


def _evaluate_pipeline(pipe: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    oof_scores = np.empty(len(y), dtype=float)
    for train_idx, test_idx in cv.split(x, y):
        fitted = clone(pipe)
        fitted.fit(x.iloc[train_idx], y.iloc[train_idx])
        scores = _model_scores(fitted, x.iloc[test_idx])
        oof_scores[test_idx] = scores
        pred = (scores >= 0.5).astype(int)
        fold = {
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "balanced_accuracy": _balanced_accuracy_no_warning(y.iloc[test_idx].to_numpy(), pred),
            "precision": precision_score(y.iloc[test_idx], pred, zero_division=0),
            "recall": recall_score(y.iloc[test_idx], pred, zero_division=0),
            "f1": f1_score(y.iloc[test_idx], pred, zero_division=0),
            "roc_auc": np.nan,
        }
        if y.iloc[test_idx].nunique() > 1 and not np.allclose(scores, scores[0]):
            fold["roc_auc"] = roc_auc_score(y.iloc[test_idx], scores)
        fold_scores.append(fold)

    tuned = _best_threshold_metrics(y, oof_scores)
    metrics = {metric: float(np.nanmean([fold[metric] for fold in fold_scores])) for metric in fold_scores[0]}
    metrics.update(tuned)
    return metrics


def _balanced_accuracy_no_warning(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in (0, 1):
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def _selection_score(metrics: dict) -> float:
    """Robust composite that does not reward the majority class only (proposal sec. 3.8)."""
    auc = metrics.get("roc_auc")
    auc = 0.5 if auc is None or np.isnan(auc) else auc
    return float(np.mean([auc, metrics.get("balanced_accuracy", 0.0), metrics.get("f1", 0.0)]))


def _select_best(results: dict) -> str:
    """Pick the strongest model, preferring the proposal's hero (XGBoost) on near-ties.

    XGBoost is the methodology's designated model and is fully SHAP-compatible, so when it is
    within ~1% of the best composite we deploy it rather than a marginally higher tree ensemble.
    """
    best = max(_selection_score(metrics) for metrics in results.values())
    near = [name for name, metrics in results.items() if best - _selection_score(metrics) <= 1e-2]
    for preferred in ("xgboost", "hist_gradient_boosting", "extra_trees", "random_forest", "logistic_regression"):
        if preferred in near:
            return preferred
    return near[0]


def _calibration_assessment(estimator, preprocessor: ColumnTransformer, x: pd.DataFrame, y: pd.Series) -> dict:
    """Out-of-fold Brier before/after calibration; apply calibration only if it helps.

    On near-deterministic suitability labels the tree ensemble is already sharp, so isotonic
    calibration can degrade both Brier and the within-case ranking. We therefore assess
    isotonic and sigmoid calibration and keep whichever (including 'none') minimises Brier
    (proposal sec. 2.7.1 calibration & uncertainty).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    out = {"brier_uncalibrated": None, "brier_isotonic": None, "brier_sigmoid": None, "applied": "none"}
    try:
        uncal = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
        out["brier_uncalibrated"] = float(brier_score_loss(y, cross_val_predict(uncal, x, y, cv=cv, method="predict_proba")[:, 1]))
        for method in ("isotonic", "sigmoid"):
            cal = Pipeline(
                [("preprocess", clone(preprocessor)), ("model", CalibratedClassifierCV(clone(estimator), cv=3, method=method))]
            )
            out[f"brier_{method}"] = float(brier_score_loss(y, cross_val_predict(cal, x, y, cv=cv, method="predict_proba")[:, 1]))
    except Exception:
        return out

    options = {
        "none": out["brier_uncalibrated"],
        "isotonic": out["brier_isotonic"],
        "sigmoid": out["brier_sigmoid"],
    }
    options = {key: value for key, value in options.items() if value is not None}
    if options:
        out["applied"] = min(options, key=options.get)
    return out


def train_best_model(df: pd.DataFrame) -> TrainingResult:
    x = df[feature_columns()]
    y = df["is_suitable"]

    preprocessor = build_preprocessor(df)
    models = candidate_models()

    # Tune XGBoost so it is a genuine contender rather than an under-configured default.
    tuned_xgb, xgb_params = _tune_xgboost(preprocessor, x, y)
    if tuned_xgb is not None:
        models["xgboost"] = tuned_xgb

    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
        results[name] = _evaluate_pipeline(pipe, x, y)

    best_name = _select_best(results)
    best_estimator = clone(models[best_name])

    calibration = _calibration_assessment(best_estimator, preprocessor, x, y)

    # Apply calibration only when it improves the Brier score; otherwise keep the sharp,
    # ranking-friendly, SHAP-friendly base estimator as the final "model" step.
    if calibration.get("applied", "none") != "none":
        final_model = CalibratedClassifierCV(clone(best_estimator), cv=5, method=calibration["applied"])
    else:
        final_model = clone(best_estimator)
    final_pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", final_model)])
    final_pipeline.fit(x, y)

    metrics = {
        "selected_model": best_name,
        "selection_score": _selection_score(results[best_name]),
        "decision_threshold": results[best_name]["decision_threshold"],
        "calibration": calibration,
        "xgboost_best_params": xgb_params,
        "model_comparison": results,
        "training_rows": int(len(df)),
        "positive_rate": float(y.mean()),
    }
    return TrainingResult(final_pipeline, metrics, best_name)


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
