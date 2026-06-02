from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from .config import DATA_RAW


ORDERED_LEVELS = {"low": 1, "medium": 2, "high": 3}


def _ordinal_level(series: pd.Series) -> pd.Series:
    return series.map(ORDERED_LEVELS).fillna(2).astype(float)


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
    observed_percentile = row.get("target_yield_percentile_within_crop_0_100")
    score = 50.0

    mean_rainfall = row.get("mean_rainfall_mm", np.nan)
    min_rainfall = row.get("min_rainfall_mm", np.nan)
    max_rainfall = row.get("max_rainfall_mm", np.nan)
    elevation = row.get("elevation_m", np.nan)
    min_altitude = row.get("min_altitude_m", np.nan)
    max_altitude = row.get("max_altitude_m", np.nan)
    drought_index = row.get("drought_index", 0)
    ndvi = row.get("ndvi", np.nan)

    if pd.notna(mean_rainfall) and pd.notna(min_rainfall) and pd.notna(max_rainfall) and min_rainfall <= mean_rainfall <= max_rainfall:
        score += 18
    elif pd.notna(mean_rainfall) and pd.notna(min_rainfall) and pd.notna(max_rainfall):
        distance = min(abs(mean_rainfall - min_rainfall), abs(mean_rainfall - max_rainfall))
        score -= min(18, distance / 25)

    if pd.notna(elevation) and pd.notna(min_altitude) and pd.notna(max_altitude) and min_altitude <= elevation <= max_altitude:
        score += 12
    elif pd.notna(elevation) and pd.notna(min_altitude) and pd.notna(max_altitude):
        distance = min(abs(elevation - min_altitude), abs(elevation - max_altitude))
        score -= min(12, distance / 60)

    drought = ORDERED_LEVELS.get(row["drought_tolerance"], 2)
    disease = ORDERED_LEVELS.get(row["disease_resistance"], 2)
    inputs = ORDERED_LEVELS.get(row["input_access"], 2)
    requirement = ORDERED_LEVELS.get(row["input_requirement"], 2)

    score += drought * (0 if pd.isna(drought_index) else drought_index) * 10
    score += disease * 3
    score += min(row["yield_potential_t_ha"], 7) * 3
    score += max(0, 125 - row["maturity_days"]) / 5
    score += (0 if pd.isna(ndvi) else ndvi) * 8
    score += max(0, inputs - requirement) * 5
    score -= max(0, requirement - inputs) * 7

    if row["drainage"] == "poor" and row["soil_type"] in {"sandy", "sandy_loam"}:
        score -= 3
    if row["production_goal"] == "market":
        score += row["yield_potential_t_ha"] * 1.5
    if row["production_goal"] == "food_security":
        score += max(0, 120 - row["maturity_days"]) / 6

    score = float(np.clip(score, 0, 100))
    if pd.notna(observed_percentile):
        score = 0.65 * float(observed_percentile) + 0.35 * score
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

    rainfall_in_range = rows["mean_rainfall_mm"].between(rows["min_rainfall_mm"], rows["max_rainfall_mm"], inclusive="both")
    altitude_in_range = rows["elevation_m"].between(rows["min_altitude_m"], rows["max_altitude_m"], inclusive="both")
    rows["rainfall_in_range"] = rainfall_in_range.astype(float)
    rows["rainfall_range_distance_mm"] = np.where(
        rainfall_in_range,
        0.0,
        np.minimum((rows["mean_rainfall_mm"] - rows["min_rainfall_mm"]).abs(), (rows["mean_rainfall_mm"] - rows["max_rainfall_mm"]).abs()),
    )
    rows["altitude_in_range"] = altitude_in_range.astype(float)
    rows["altitude_range_distance_m"] = np.where(
        altitude_in_range,
        0.0,
        np.minimum((rows["elevation_m"] - rows["min_altitude_m"]).abs(), (rows["elevation_m"] - rows["max_altitude_m"]).abs()),
    )
    rows["input_access_level"] = _ordinal_level(rows["input_access"])
    rows["input_requirement_level"] = _ordinal_level(rows["input_requirement"])
    rows["input_level_gap"] = rows["input_access_level"] - rows["input_requirement_level"]
    rows["input_level_shortfall"] = (rows["input_requirement_level"] - rows["input_access_level"]).clip(lower=0)
    rows["input_level_surplus"] = (rows["input_access_level"] - rows["input_requirement_level"]).clip(lower=0)
    rows["drought_tolerance_level"] = _ordinal_level(rows["drought_tolerance"])
    rows["disease_resistance_level"] = _ordinal_level(rows["disease_resistance"])
    rows["drought_x_index"] = rows["drought_tolerance_level"] * rows["drought_index"].fillna(0)
    rows["yield_potential_capped"] = rows["yield_potential_t_ha"].clip(upper=7)
    rows["maturity_urgency"] = (125 - rows["maturity_days"]).clip(lower=0)
    rows["market_goal"] = (rows["production_goal"] == "market").astype(int)
    rows["food_security_goal"] = (rows["production_goal"] == "food_security").astype(int)
    rows["poor_drainage_sandy_penalty"] = ((rows["drainage"] == "poor") & rows["soil_type"].isin({"sandy", "sandy_loam"})).astype(int)

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
        "climate_t2m_max_mean_c",
        "climate_t2m_min_mean_c",
        "climate_relative_humidity_pct",
        "climate_daily_rainfall_mean_mm",
        "climate_annualized_rainfall_mm",
        "climate_surface_pressure_kpa",
        "climate_wind_speed_m_s",
        "climate_solar_radiation_mj_m2_day",
        "drought_index",
        "season_length_days",
        "soil_type",
        "soil_ph",
        "organic_matter_pct",
        "isda_ph_0_20",
        "isda_ph_20_50",
        "isda_carbon_organic_0_20_g_kg",
        "isda_carbon_organic_20_50_g_kg",
        "isda_nitrogen_total_0_20_g_kg",
        "isda_nitrogen_total_20_50_g_kg",
        "isda_aluminium_extractable_0_20_ppm",
        "isda_aluminium_extractable_20_50_ppm",
        "isda_phosphorous_extractable_0_20_ppm",
        "isda_phosphorous_extractable_20_50_ppm",
        "isda_potassium_extractable_0_20_ppm",
        "isda_potassium_extractable_20_50_ppm",
        "isda_magnesium_extractable_0_20_ppm",
        "isda_magnesium_extractable_20_50_ppm",
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
        "rainfall_in_range",
        "rainfall_range_distance_mm",
        "altitude_in_range",
        "altitude_range_distance_m",
        "input_access_level",
        "input_requirement_level",
        "input_level_gap",
        "input_level_shortfall",
        "input_level_surplus",
        "drought_tolerance_level",
        "disease_resistance_level",
        "drought_x_index",
        "yield_potential_capped",
        "maturity_urgency",
        "market_goal",
        "food_security_goal",
        "poor_drainage_sandy_penalty",
    ]
