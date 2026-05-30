from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(recommendations: pd.DataFrame, k: int = 3) -> dict:
    grouped = recommendations.groupby("case_id", sort=False)
    top_k_hits = []
    precision_values = []
    recall_values = []
    ndcg_values = []

    for _, group in grouped:
        ranked = group.sort_values("rank").head(k)
        relevant_total = max(1, int(group["suitability_class"].eq("suitable").sum()))
        hits = ranked["suitability_class"].eq("suitable").astype(int).to_numpy()
        top_k_hits.append(float(hits.any()))
        precision_values.append(float(hits.mean()))
        recall_values.append(float(hits.sum() / relevant_total))
        gains = (2 ** hits - 1) / np.log2(np.arange(2, len(hits) + 2))
        ideal_hits = np.sort(group["suitability_class"].eq("suitable").astype(int).to_numpy())[::-1][:k]
        ideal = (2 ** ideal_hits - 1) / np.log2(np.arange(2, len(ideal_hits) + 2))
        ndcg_values.append(float(gains.sum() / ideal.sum()) if ideal.sum() else 0.0)

    return {
        f"top_{k}_accuracy": float(np.mean(top_k_hits)),
        f"precision_at_{k}": float(np.mean(precision_values)),
        f"recall_at_{k}": float(np.mean(recall_values)),
        f"ndcg_at_{k}": float(np.mean(ndcg_values)),
    }


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

