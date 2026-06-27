from __future__ import annotations

import numpy as np
import pandas as pd


# Graded relevance for ranking quality (proposal sec. 2.5.3, 3.8).
RELEVANCE = {"suitable": 2, "moderately_suitable": 1, "unsuitable": 0}


def _relevance(series: pd.Series) -> np.ndarray:
    return series.map(RELEVANCE).fillna(0).astype(float).to_numpy()


def ranking_metrics(recommendations: pd.DataFrame, k: int = 3) -> dict:
    grouped = recommendations.groupby("case_id", sort=False)
    top_k_hits = []
    precision_values = []
    recall_values = []
    ndcg_values = []
    rr_values = []

    for _, group in grouped:
        ordered = group.sort_values("rank")
        rel_all = _relevance(ordered["suitability_class"])
        relevant_binary = (rel_all >= 2).astype(int)
        relevant_total = max(1, int(relevant_binary.sum()))

        top = ordered.head(k)
        rel_k = _relevance(top["suitability_class"])
        hits = (rel_k >= 2).astype(int)

        top_k_hits.append(float(hits.any()))
        precision_values.append(float(hits.mean()) if len(hits) else 0.0)
        recall_values.append(float(hits.sum() / relevant_total))

        gains = (2 ** rel_k - 1) / np.log2(np.arange(2, len(rel_k) + 2))
        ideal_rel = np.sort(rel_all)[::-1][:k]
        ideal = (2 ** ideal_rel - 1) / np.log2(np.arange(2, len(ideal_rel) + 2))
        ndcg_values.append(float(gains.sum() / ideal.sum()) if ideal.sum() else 0.0)

        relevant_positions = np.where(relevant_binary >= 1)[0]
        rr_values.append(float(1.0 / (relevant_positions[0] + 1)) if len(relevant_positions) else 0.0)

    return {
        f"top_{k}_accuracy": float(np.mean(top_k_hits)),
        f"precision_at_{k}": float(np.mean(precision_values)),
        f"recall_at_{k}": float(np.mean(recall_values)),
        f"ndcg_at_{k}": float(np.mean(ndcg_values)),
        "mrr": float(np.mean(rr_values)),
    }


def ranking_metrics_at_ks(recommendations: pd.DataFrame, ks: tuple[int, ...] = (3, 5)) -> dict:
    """Combined ranking metrics at several cut-offs (MRR is shared across k)."""
    out: dict = {}
    for k in ks:
        out.update(ranking_metrics(recommendations, k=k))
    return out


def fairness_summary(recommendations: pd.DataFrame, group_col: str, k: int = 3) -> pd.DataFrame:
    """Ranking quality per sub-group (proposal sec. 3.8.3 fairness/inclusiveness check).

    ``recommendations`` must carry ``group_col`` (the pipeline merges it from the case context).
    """
    if group_col not in recommendations.columns:
        return pd.DataFrame()
    rows = []
    for value, group in recommendations.groupby(group_col, dropna=False):
        metrics = ranking_metrics(group, k=k)
        metrics[group_col] = value
        metrics["cases"] = int(group["case_id"].nunique())
        rows.append(metrics)
    return pd.DataFrame(rows)


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["agro_ecological_zone", "resource_level"], dropna=False)
        .agg(
            cases=("case_id", "nunique"),
            rows=("case_id", "size"),
            suitable_rate=("is_suitable", "mean"),
            mean_suitability=("suitability_score", "mean"),
        )
        .reset_index()
    )
