from __future__ import annotations

import numpy as np
import pandas as pd


POSITIVE_TEMPLATES = {
    "rainfall": "rainfall conditions fit the variety adaptation range",
    "altitude": "the elevation is within the variety adaptation range",
    "maturity": "the maturity period fits the season length",
    "drought": "drought tolerance is useful for the local drought risk",
    "disease": "disease resistance strengthens suitability",
    "yield": "yield potential supports the farmer's production goal",
    "inputs": "input requirements match the farmer's resource level",
    "vegetation": "vegetation condition indicates a favourable growing environment",
}

CAUTION_TEMPLATES = {
    "rainfall": "rainfall is outside the preferred variety range",
    "altitude": "elevation is outside the preferred variety range",
    "maturity": "the variety may mature too late for the season length",
    "drought": "drought risk may challenge this variety",
    "disease": "disease resistance is relatively weak",
    "inputs": "the variety may require more inputs than the farmer can access",
    "availability": "seed availability is not confirmed locally",
}


def agronomic_drivers(row: pd.Series) -> tuple[list[str], list[str]]:
    drivers: list[str] = []
    cautions: list[str] = []

    if row["min_rainfall_mm"] <= row["mean_rainfall_mm"] <= row["max_rainfall_mm"]:
        drivers.append(POSITIVE_TEMPLATES["rainfall"])
    else:
        cautions.append(CAUTION_TEMPLATES["rainfall"])

    if row["min_altitude_m"] <= row["elevation_m"] <= row["max_altitude_m"]:
        drivers.append(POSITIVE_TEMPLATES["altitude"])
    else:
        cautions.append(CAUTION_TEMPLATES["altitude"])

    if row["maturity_days"] <= row["season_length_days"] + 15:
        drivers.append(POSITIVE_TEMPLATES["maturity"])
    else:
        cautions.append(CAUTION_TEMPLATES["maturity"])

    if row["drought_tolerance"] == "high" and row["drought_index"] >= 0.35:
        drivers.append(POSITIVE_TEMPLATES["drought"])
    elif row["drought_tolerance"] == "low" and row["drought_index"] >= 0.35:
        cautions.append(CAUTION_TEMPLATES["drought"])

    if row["disease_resistance"] == "high":
        drivers.append(POSITIVE_TEMPLATES["disease"])
    elif row["disease_resistance"] == "low":
        cautions.append(CAUTION_TEMPLATES["disease"])

    if row["yield_potential_t_ha"] >= 5 and row["production_goal"] in {"market", "food_and_market"}:
        drivers.append(POSITIVE_TEMPLATES["yield"])

    levels = {"low": 1, "medium": 2, "high": 3}
    if levels[row["input_access"]] >= levels[row["input_requirement"]]:
        drivers.append(POSITIVE_TEMPLATES["inputs"])
    else:
        cautions.append(CAUTION_TEMPLATES["inputs"])

    if row["ndvi"] >= 0.6:
        drivers.append(POSITIVE_TEMPLATES["vegetation"])

    if row["available"] != "yes":
        cautions.append(CAUTION_TEMPLATES["availability"])

    return drivers[:4], cautions[:3]


def explanation_sentence(row: pd.Series) -> str:
    drivers, cautions = agronomic_drivers(row)
    driver_text = "; ".join(drivers) if drivers else "available data gives moderate support for this variety"
    caution_text = "; ".join(cautions) if cautions else "no major caution was detected in the demo data"
    return f"Recommended because {driver_text}. Caution: {caution_text}."


def shap_summary(pipe, rows: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap

        transformed = pipe.named_steps["preprocess"].transform(rows)
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        importance = np.abs(values).mean(axis=0)
        return pd.DataFrame({"feature": names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        return fallback_feature_importance(pipe)


def fallback_feature_importance(pipe) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(names))
    return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)

