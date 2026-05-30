from __future__ import annotations

import pandas as pd

from .data import feature_columns
from .explain import agronomic_drivers, explanation_sentence
from .model import predict_suitability


def rank_recommendations(pipe, modelling_df: pd.DataFrame, case_id: str, top_k: int = 5) -> pd.DataFrame:
    case_rows = modelling_df[modelling_df["case_id"].eq(case_id)].copy()
    if case_rows.empty:
        raise ValueError(f"No rows found for case_id={case_id}")

    case_rows["model_probability"] = predict_suitability(pipe, case_rows)
    case_rows["availability_penalty"] = case_rows["available"].map({"yes": 0.0, "unknown": 0.05, "no": 0.12}).fillna(0.05)
    case_rows["recommendation_score"] = (case_rows["model_probability"] - case_rows["availability_penalty"]).clip(0, 1)
    case_rows = case_rows.sort_values(["recommendation_score", "suitability_score"], ascending=False).reset_index(drop=True)
    case_rows["rank"] = range(1, len(case_rows) + 1)

    drivers = case_rows.apply(lambda row: agronomic_drivers(row)[0], axis=1)
    cautions = case_rows.apply(lambda row: agronomic_drivers(row)[1], axis=1)
    case_rows["positive_drivers"] = drivers.map(lambda items: " | ".join(items))
    case_rows["cautionary_drivers"] = cautions.map(lambda items: " | ".join(items))
    case_rows["explanation"] = case_rows.apply(explanation_sentence, axis=1)

    columns = [
        "case_id",
        "district",
        "season",
        "crop",
        "rank",
        "variety_name",
        "recommendation_score",
        "suitability_score",
        "suitability_class",
        "available",
        "supplier",
        "availability_note",
        "data_confidence",
        "positive_drivers",
        "cautionary_drivers",
        "explanation",
    ]
    return case_rows[columns].head(top_k)


def build_case_from_inputs(modelling_df: pd.DataFrame, district: str, season: str, input_access: str, production_goal: str, resource_level: str) -> str:
    matching = modelling_df[
        modelling_df["district"].eq(district)
        & modelling_df["season"].eq(season)
        & modelling_df["input_access"].eq(input_access)
        & modelling_df["production_goal"].eq(production_goal)
        & modelling_df["resource_level"].eq(resource_level)
    ]
    if not matching.empty:
        return str(matching.iloc[0]["case_id"])
    district_cases = modelling_df[modelling_df["district"].eq(district)]
    if district_cases.empty:
        raise ValueError(f"No demo case is available for district={district}")
    return str(district_cases.iloc[0]["case_id"])

