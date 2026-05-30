from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from .config import DATA_RAW


ORDERED_LEVELS = {"low": 1, "medium": 2, "high": 3}


def load_raw_data(raw_dir=DATA_RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = raw_dir
    varieties = pd.read_csv(raw_dir / "seed_varieties.csv")
    locations = pd.read_csv(raw_dir / "location_profiles.csv")
    farmers = pd.read_csv(raw_dir / "farmer_context.csv")
    availability = pd.read_csv(raw_dir / "seed_availability.csv")
    return varieties, locations, farmers, availability


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [name for name in df.columns if is_string_dtype(df[name])]:
        df[col] = df[col].astype(str).str.strip()
    return df


def _score_row(row: pd.Series) -> float:
    score = 50.0

    if row["min_rainfall_mm"] <= row["mean_rainfall_mm"] <= row["max_rainfall_mm"]:
        score += 18
    else:
        distance = min(abs(row["mean_rainfall_mm"] - row["min_rainfall_mm"]), abs(row["mean_rainfall_mm"] - row["max_rainfall_mm"]))
        score -= min(18, distance / 25)

    if row["min_altitude_m"] <= row["elevation_m"] <= row["max_altitude_m"]:
        score += 12
    else:
        distance = min(abs(row["elevation_m"] - row["min_altitude_m"]), abs(row["elevation_m"] - row["max_altitude_m"]))
        score -= min(12, distance / 60)

    drought = ORDERED_LEVELS[row["drought_tolerance"]]
    disease = ORDERED_LEVELS[row["disease_resistance"]]
    inputs = ORDERED_LEVELS[row["input_access"]]
    requirement = ORDERED_LEVELS[row["input_requirement"]]

    score += drought * row["drought_index"] * 10
    score += disease * 3
    score += min(row["yield_potential_t_ha"], 7) * 3
    score += max(0, 125 - row["maturity_days"]) / 5
    score += row["ndvi"] * 8
    score += max(0, inputs - requirement) * 5
    score -= max(0, requirement - inputs) * 7

    if row["drainage"] == "poor" and row["soil_type"] in {"sandy", "sandy_loam"}:
        score -= 3
    if row["production_goal"] == "market":
        score += row["yield_potential_t_ha"] * 1.5
    if row["production_goal"] == "food_security":
        score += max(0, 120 - row["maturity_days"]) / 6

    return float(np.clip(score, 0, 100))


def build_modelling_dataset(
    varieties: pd.DataFrame,
    locations: pd.DataFrame,
    farmers: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    varieties = normalize_text_columns(varieties)
    locations = normalize_text_columns(locations)
    farmers = normalize_text_columns(farmers)
    availability = normalize_text_columns(availability)

    base = farmers.merge(locations, on=["district", "season"], how="left", validate="many_to_one")
    rows = base.merge(varieties, on="crop", how="left")
    rows = rows.merge(availability, on=["district", "variety_id"], how="left")

    rows["available"] = rows["available"].fillna("unknown")
    rows["availability_note"] = rows["availability_note"].fillna("Availability not confirmed")
    rows["supplier"] = rows["supplier"].fillna("Unknown")
    rows["suitability_score"] = rows.apply(_score_row, axis=1).round(2)
    rows["suitability_class"] = pd.cut(
        rows["suitability_score"],
        bins=[-1, 59.99, 74.99, 100],
        labels=["unsuitable", "moderately_suitable", "suitable"],
    ).astype(str)
    rows["is_suitable"] = (rows["suitability_score"] >= 75).astype(int)

    weak_fields = [
        "mean_rainfall_mm",
        "soil_ph",
        "organic_matter_pct",
        "ndvi",
        "available",
    ]
    rows["data_confidence"] = np.where(rows[weak_fields].isna().any(axis=1), "medium", "high")
    rows.loc[rows["available"].eq("unknown"), "data_confidence"] = "medium"
    return rows


def feature_columns() -> list[str]:
    return [
        "district",
        "agro_ecological_zone",
        "zardi_zone",
        "season",
        "farm_size_acres",
        "input_access",
        "production_goal",
        "resource_level",
        "market_preference",
        "mean_rainfall_mm",
        "mean_temperature_c",
        "drought_index",
        "season_length_days",
        "soil_type",
        "soil_ph",
        "organic_matter_pct",
        "drainage",
        "elevation_m",
        "ndvi",
        "market_access",
        "maturity_days",
        "drought_tolerance",
        "disease_resistance",
        "yield_potential_t_ha",
        "min_rainfall_mm",
        "max_rainfall_mm",
        "min_altitude_m",
        "max_altitude_m",
        "input_requirement",
    ]
