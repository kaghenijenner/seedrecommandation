from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW, PROJECT_ROOT

DATASETS_DIR = PROJECT_ROOT / "datasets"
MISSING_DATASETS_DIR = PROJECT_ROOT / "missingdatasets"


def _level_from_bool(value) -> str:
    if pd.isna(value):
        return "medium"
    return "high" if bool(value) else "medium"


def _disease_level(notes: str) -> str:
    text = str(notes).lower()
    if "resistant" in text or "tolerant" in text:
        return "high"
    return "medium"


def _input_requirement(row: pd.Series) -> str:
    potential = row.get("variety_yield_potential_mean_t_ha")
    if pd.notna(potential) and potential >= 7:
        return "high"
    if pd.notna(potential) and potential >= 3:
        return "medium"
    return "low"


def _season_code(row: pd.Series) -> str:
    return f"{int(row['planting_year'])}_{str(row['season_name']).strip()}"


def _district_to_zone(region: str) -> str:
    mapping = {
        "Central": "Lake Victoria Crescent and Central Uganda",
        "Eastern": "Eastern and Mt Elgon farming systems",
        "Northern": "Northern Uganda farming systems",
        "Western": "Western and South Western Highlands",
    }
    return mapping.get(str(region), "Uganda mixed farming systems")


def _soil_type(ph: float, carbon: float) -> str:
    if pd.isna(ph):
        return "unknown"
    if ph < 5.5:
        return "acidic_loam"
    if pd.notna(carbon) and carbon >= 30:
        return "organic_loam"
    return "loam"


def _drainage(anomaly_pct: float) -> str:
    if pd.isna(anomaly_pct):
        return "moderate"
    if anomaly_pct < 75:
        return "poor"
    if anomaly_pct > 110:
        return "good"
    return "moderate"


def _availability(row: pd.Series) -> str:
    return "yes" if row.get("licensed_company_count", 0) and row["licensed_company_count"] > 0 else "unknown"


def _extract_altitude_bounds(notes: str) -> tuple[int, int]:
    text = str(notes)
    match = re.search(r"(\d{3,4})\s*[-–]\s*(\d{3,4})\s*masl", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    if "below 1200" in text.lower():
        return 0, 1200
    return 700, 2200


def _load_ndvi_by_district(missing_dir: Path) -> pd.DataFrame:
    path = missing_dir / "uganda_ndvi_by_district.csv"
    if not path.exists():
        return pd.DataFrame(columns=["district", "ndvi"])
    ndvi = pd.read_csv(path)
    return (
        ndvi.dropna(subset=["district", "mean"])
        .assign(district=lambda df: df["district"].astype(str).str.strip())
        .groupby("district", as_index=False)["mean"]
        .mean()
        .rename(columns={"mean": "ndvi"})
    )


def _load_climate_by_district(missing_dir: Path) -> pd.DataFrame:
    path = missing_dir / "uganda_climate_data.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "district",
                "climate_t2m_mean_c",
                "climate_t2m_max_mean_c",
                "climate_t2m_min_mean_c",
                "climate_relative_humidity_pct",
                "climate_daily_rainfall_mean_mm",
                "climate_annualized_rainfall_mm",
                "climate_surface_pressure_kpa",
                "climate_wind_speed_m_s",
                "climate_solar_radiation_mj_m2_day",
            ]
        )

    climate = pd.read_csv(path, parse_dates=["date"])
    climate = climate.dropna(subset=["district"])
    summary = (
        climate.assign(district=lambda df: df["district"].astype(str).str.strip())
        .groupby("district", as_index=False)
        .agg(
            climate_t2m_mean_c=("T2M", "mean"),
            climate_t2m_max_mean_c=("T2M_MAX", "mean"),
            climate_t2m_min_mean_c=("T2M_MIN", "mean"),
            climate_relative_humidity_pct=("RH2M", "mean"),
            climate_daily_rainfall_mean_mm=("PRECTOTCORR", "mean"),
            climate_surface_pressure_kpa=("PS", "mean"),
            climate_wind_speed_m_s=("WS2M", "mean"),
            climate_solar_radiation_mj_m2_day=("ALLSKY_SFC_SW_DWN", "mean"),
        )
    )
    summary["climate_annualized_rainfall_mm"] = summary["climate_daily_rainfall_mean_mm"] * 365.25
    numeric_cols = [col for col in summary.columns if col != "district"]
    summary[numeric_cols] = summary[numeric_cols].round(4)
    return summary


def _load_soilgrids_by_district(missing_dir: Path) -> pd.DataFrame:
    path = missing_dir / "uganda_soilgrids_data.csv"
    if not path.exists():
        return pd.DataFrame(columns=["district"])

    soil = pd.read_csv(path)
    if "value" not in soil.columns or soil["value"].notna().sum() == 0:
        return pd.DataFrame(columns=["district"])

    soil = soil.dropna(subset=["district", "soil_property", "depth_cm", "value"]).copy()
    soil["district"] = soil["district"].astype(str).str.strip()
    soil["soil_property"] = soil["soil_property"].astype(str).str.lower().str.strip()
    soil["depth_cm"] = soil["depth_cm"].astype(str).str.replace("-", "_", regex=False).str.replace("cm", "", regex=False)
    pivot = (
        soil.pivot_table(index="district", columns=["soil_property", "depth_cm"], values="value", aggfunc="mean")
        .reset_index()
    )
    pivot.columns = [
        "district" if col[0] == "district" else f"soilgrids_{col[0]}_{col[1]}cm"
        for col in pivot.columns.to_flat_index()
    ]
    numeric_cols = [col for col in pivot.columns if col != "district"]
    pivot[numeric_cols] = pivot[numeric_cols].round(4)
    return pivot


ISDA_ENRICHMENT_COLUMNS = [
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
]


def _load_isda_enrichment_by_district(missing_dir: Path) -> pd.DataFrame:
    candidates = [
        missing_dir / "new" / "uganda_isda_soil_data.csv",
        PROJECT_ROOT / "uganda_isda_soil_data.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["district", *ISDA_ENRICHMENT_COLUMNS])

    soil = pd.read_csv(path)
    required = {"district", "soil_property", "depth_cm", "value"}
    if not required.issubset(soil.columns) or soil["value"].notna().sum() == 0:
        return pd.DataFrame(columns=["district", *ISDA_ENRICHMENT_COLUMNS])

    property_units = {
        "ph": "",
        "carbon_organic": "g_kg",
        "nitrogen_total": "g_kg",
        "aluminium_extractable": "ppm",
        "phosphorous_extractable": "ppm",
        "potassium_extractable": "ppm",
        "magnesium_extractable": "ppm",
    }
    soil = soil.dropna(subset=["district", "soil_property", "depth_cm", "value"]).copy()
    soil["district"] = soil["district"].astype(str).str.strip()
    soil["soil_property"] = soil["soil_property"].astype(str).str.lower().str.strip()
    soil["depth_cm"] = soil["depth_cm"].astype(str).str.replace("-", "_", regex=False)
    soil = soil[soil["soil_property"].isin(property_units)]
    pivot = soil.pivot_table(index="district", columns=["soil_property", "depth_cm"], values="value", aggfunc="mean").reset_index()
    flat_columns = []
    for col in pivot.columns.to_flat_index():
        if col[0] == "district":
            flat_columns.append("district")
            continue
        unit = property_units.get(col[0], "")
        suffix = f"_{unit}" if unit else ""
        flat_columns.append(f"isda_{col[0]}_{col[1]}{suffix}")
    pivot.columns = flat_columns
    for col in ISDA_ENRICHMENT_COLUMNS:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot[ISDA_ENRICHMENT_COLUMNS] = pivot[ISDA_ENRICHMENT_COLUMNS].round(4)
    return pivot[["district", *ISDA_ENRICHMENT_COLUMNS]]


def _resource_level(mean_quintile: float) -> str:
    if pd.isna(mean_quintile):
        return "unknown"
    if mean_quintile <= 2.25:
        return "low"
    if mean_quintile <= 3.75:
        return "medium"
    return "high"


def _market_preference(urban_share: float) -> str:
    if pd.isna(urban_share):
        return "unknown"
    if urban_share >= 0.5:
        return "high"
    if urban_share >= 0.2:
        return "medium"
    return "low"


def _input_access(mean_quintile: float, urban_share: float) -> str:
    if pd.isna(mean_quintile):
        return "medium"
    if mean_quintile >= 4 or (mean_quintile >= 3 and urban_share >= 0.35):
        return "high"
    if mean_quintile <= 2:
        return "low"
    return "medium"


def _load_unps_district_context(missing_dir: Path) -> pd.DataFrame:
    pov_path = missing_dir / "UGA_2019_UNPS_v03_M_CSV" / "pov2019_20.csv"
    geo_path = missing_dir / "UGA_2019_UNPS_v03_M_CSV" / "HH" / "gsec1.csv"
    if not pov_path.exists() or not geo_path.exists():
        return pd.DataFrame(
            columns=[
                "district",
                "farm_size_acres",
                "input_access",
                "resource_level",
                "market_preference",
                "unps_mean_quintile",
                "unps_poverty_rate",
                "unps_urban_share",
            ]
        )

    pov = pd.read_csv(pov_path, low_memory=False)
    geo = pd.read_csv(geo_path, usecols=["hhid", "district"], low_memory=False).rename(columns={"district": "district_name"})
    joined = pov.merge(geo, on="hhid", how="left")
    joined = joined.dropna(subset=["district_name"])
    summary = (
        joined.groupby("district_name", as_index=False)
        .agg(
            unps_mean_quintile=("quints", "mean"),
            unps_poverty_rate=("poor_2020", "mean"),
            unps_urban_share=("urban", "mean"),
            unps_mean_household_size=("hsize", "mean"),
        )
    )
    summary["district"] = summary["district_name"].astype(str).str.title().str.strip()
    summary = summary.drop(columns=["district_name"])
    summary["farm_size_acres"] = np.nan
    summary["resource_level"] = summary["unps_mean_quintile"].map(_resource_level)
    summary["market_preference"] = summary["unps_urban_share"].map(_market_preference)
    summary["input_access"] = [
        _input_access(quintile, urban)
        for quintile, urban in zip(summary["unps_mean_quintile"], summary["unps_urban_share"])
    ]
    return summary


def prepare_real_raw_files(
    datasets_dir: Path = DATASETS_DIR,
    raw_dir: Path = DATA_RAW,
    missing_dir: Path = MISSING_DATASETS_DIR,
) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(datasets_dir / "uganda_seed_candidate_variety_scoring_table.csv")
    candidates = candidates.dropna(subset=["standard_crop", "variety_name", "admin_2", "season_name", "planting_year"])
    candidates["crop"] = candidates["standard_crop"].str.strip().str.lower()
    candidates["district"] = candidates["admin_2"].str.strip()
    candidates["season"] = candidates.apply(_season_code, axis=1)

    rainfall_bounds = (
        candidates.groupby("crop")["season_rainfall_total_mm"]
        .quantile([0.1, 0.9])
        .unstack()
        .rename(columns={0.1: "min_rainfall_mm", 0.9: "max_rainfall_mm"})
        .reset_index()
    )

    variety_base = (
        candidates.sort_values(["crop", "variety_name"])
        .drop_duplicates(["crop", "variety_name"])
        .merge(rainfall_bounds, on="crop", how="left")
    )
    altitude_bounds = variety_base["adaptation_notes"].apply(_extract_altitude_bounds)
    varieties = pd.DataFrame(
        {
            "variety_id": variety_base["nvin"].fillna(variety_base["variety_name"]).astype(str),
            "variety_name": variety_base["variety_name"],
            "crop": variety_base["crop"],
            "maturity_days": variety_base["variety_maturity_mean_days"].fillna(
                variety_base[["maturity_min_days", "maturity_max_days"]].mean(axis=1)
            ),
            "drought_tolerance": variety_base["variety_drought_tolerant"].map(_level_from_bool),
            "disease_resistance": variety_base["disease_resistance_notes"].map(_disease_level),
            "yield_potential_t_ha": variety_base["variety_yield_potential_mean_t_ha"].fillna(
                variety_base["yield_potential_max_t_ha"]
            ),
            "min_rainfall_mm": variety_base["min_rainfall_mm"].round(2),
            "max_rainfall_mm": variety_base["max_rainfall_mm"].round(2),
            "min_altitude_m": [bounds[0] for bounds in altitude_bounds],
            "max_altitude_m": [bounds[1] for bounds in altitude_bounds],
            "input_requirement": variety_base.apply(_input_requirement, axis=1),
            "notes": variety_base["adaptation_notes"].fillna("") + " " + variety_base["disease_resistance_notes"].fillna(""),
        }
    )
    varieties.to_csv(raw_dir / "seed_varieties.csv", index=False)

    ndvi_by_district = _load_ndvi_by_district(missing_dir)
    climate_by_district = _load_climate_by_district(missing_dir)
    soilgrids_by_district = _load_soilgrids_by_district(missing_dir)
    isda_enrichment_by_district = _load_isda_enrichment_by_district(missing_dir)
    location_base = (
        candidates.drop_duplicates(["district", "season"])
        .merge(ndvi_by_district, on="district", how="left")
        .merge(climate_by_district, on="district", how="left")
        .merge(soilgrids_by_district, on="district", how="left")
        .merge(isda_enrichment_by_district, on="district", how="left")
    )
    rainfall = location_base["season_rainfall_total_mm"]
    drought_index = (1 - (rainfall - rainfall.min()) / (rainfall.max() - rainfall.min())).clip(0, 1)
    locations = pd.DataFrame(
        {
            "district": location_base["district"],
            "agro_ecological_zone": location_base["admin_1"].map(_district_to_zone),
            "zardi_zone": location_base["admin_1"],
            "season": location_base["season"],
            "mean_rainfall_mm": location_base["season_rainfall_total_mm"].round(2),
            "mean_temperature_c": location_base["climate_t2m_mean_c"],
            "drought_index": drought_index.round(3),
            "season_length_days": np.where(location_base["season_name"].str.lower().eq("first"), 120, 115),
            "soil_type": [
                _soil_type(ph, carbon)
                for ph, carbon in zip(location_base["ph_topsoil_0_20cm_mean"], location_base["carbon_organic_topsoil_0_20cm_mean"])
            ],
            "soil_ph": location_base["isda_ph_0_20"].combine_first(location_base["ph_topsoil_0_20cm_mean"]).round(3),
            "organic_matter_pct": location_base["carbon_organic_topsoil_0_20cm_mean"].round(3),
            "drainage": location_base["season_anomaly_pct_mean"].map(_drainage),
            "elevation_m": 1200,
            "ndvi": location_base["ndvi"].round(4),
            "market_access": "unknown",
            "climate_t2m_max_mean_c": location_base["climate_t2m_max_mean_c"],
            "climate_t2m_min_mean_c": location_base["climate_t2m_min_mean_c"],
            "climate_relative_humidity_pct": location_base["climate_relative_humidity_pct"],
            "climate_daily_rainfall_mean_mm": location_base["climate_daily_rainfall_mean_mm"],
            "climate_annualized_rainfall_mm": location_base["climate_annualized_rainfall_mm"],
            "climate_surface_pressure_kpa": location_base["climate_surface_pressure_kpa"],
            "climate_wind_speed_m_s": location_base["climate_wind_speed_m_s"],
            "climate_solar_radiation_mj_m2_day": location_base["climate_solar_radiation_mj_m2_day"],
        }
    )
    for col in soilgrids_by_district.columns:
        if col != "district":
            locations[col] = location_base[col]
    for col in ISDA_ENRICHMENT_COLUMNS:
        locations[col] = location_base[col]
    locations.to_csv(raw_dir / "location_profiles.csv", index=False)

    district_context = _load_unps_district_context(missing_dir)
    farmer_base = candidates.drop_duplicates("row_id").merge(district_context, on="district", how="left")
    farmers = pd.DataFrame(
        {
            "case_id": farmer_base["sample_id"],
            "district": farmer_base["district"],
            "crop": farmer_base["crop"],
            "season": farmer_base["season"],
            "farm_size_acres": farmer_base["farm_size_acres"],
            "input_access": farmer_base["input_access"].fillna("medium"),
            "production_goal": "food_and_market",
            "resource_level": farmer_base["resource_level"].fillna("unknown"),
            "gender": "unknown",
            "market_preference": farmer_base["market_preference"].fillna("unknown"),
            "target_yield_percentile_within_crop_0_100": farmer_base["target_yield_percentile_within_crop_0_100"],
            "target_yield_t_ha": farmer_base["target_yield_t_ha"],
            "unps_mean_quintile": farmer_base["unps_mean_quintile"],
            "unps_poverty_rate": farmer_base["unps_poverty_rate"],
            "unps_urban_share": farmer_base["unps_urban_share"],
        }
    )
    farmers.to_csv(raw_dir / "farmer_context.csv", index=False)

    availability_base = candidates.drop_duplicates(["district", "variety_name"])
    availability = pd.DataFrame(
        {
            "district": availability_base["district"],
            "variety_id": availability_base["nvin"].fillna(availability_base["variety_name"]).astype(str),
            "available": availability_base.apply(_availability, axis=1),
            "supplier": availability_base["licensed_companies"].fillna("NARO or licensed seed source not specified"),
            "availability_note": availability_base["seed_source_notes"].fillna("District-specific availability not confirmed"),
        }
    )
    availability.to_csv(raw_dir / "seed_availability.csv", index=False)

    return {
        "seed_varieties": len(varieties),
        "location_profiles": len(locations),
        "farmer_context": len(farmers),
        "seed_availability": len(availability),
    }


def main() -> None:
    counts = prepare_real_raw_files()
    for name, count in counts.items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
