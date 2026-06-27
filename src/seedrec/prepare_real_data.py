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


def _input_requirement_from_yield(yield_t_ha: float) -> str:
    if pd.notna(yield_t_ha) and yield_t_ha >= 7:
        return "high"
    if pd.notna(yield_t_ha) and yield_t_ha >= 3:
        return "medium"
    return "low"


MODELLED_CROPS = ("maize", "beans", "groundnuts", "rice")


def _normalize_name_key(name: object) -> str:
    """Collapse a variety name to a comparison key (e.g. 'Chinuga-1' == 'Chinuga 1')."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _release_year_from_nvin(nvin: object) -> float:
    match = re.search(r"/(\d{4})/", str(nvin))
    if match:
        return float(match.group(1))
    return np.nan


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
        missing_dir / "uganda_isda_soil_data.csv",
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


def _build_variety_catalogue(candidates: pd.DataFrame, datasets_dir: Path) -> pd.DataFrame:
    """Union of measured-trait varieties and the full NARO licensed catalogue.

    The measured set (rich agronomic traits from the NARO compendium) is kept as-is.
    Every other licensed variety for the modelled crops is added with crop-level proxy
    traits and flagged via ``trait_source='imputed'`` so downstream code can lower its
    data confidence. ``release_year`` and ``licensed_company_count`` are real per-variety
    signals that keep imputed varieties distinguishable beyond crop averages.
    """
    rainfall_bounds = (
        candidates.groupby("crop")["season_rainfall_total_mm"]
        .quantile([0.1, 0.9])
        .unstack()
        .rename(columns={0.1: "min_rainfall_mm", 0.9: "max_rainfall_mm"})
        .reset_index()
    )

    # --- measured varieties (full traits) ---
    measured_base = (
        candidates.sort_values(["crop", "variety_name"])
        .drop_duplicates(["crop", "variety_name"])
        .merge(rainfall_bounds, on="crop", how="left")
    )
    measured_altitude = measured_base["adaptation_notes"].apply(_extract_altitude_bounds)
    measured = pd.DataFrame(
        {
            "variety_id": measured_base["nvin"].fillna(measured_base["variety_name"]).astype(str),
            "variety_name": measured_base["variety_name"].astype(str).str.strip(),
            "crop": measured_base["crop"],
            "maturity_days": measured_base["variety_maturity_mean_days"].fillna(
                measured_base[["maturity_min_days", "maturity_max_days"]].mean(axis=1)
            ),
            "drought_tolerance": measured_base["variety_drought_tolerant"].map(_level_from_bool),
            "disease_resistance": measured_base["disease_resistance_notes"].map(_disease_level),
            "yield_potential_t_ha": measured_base["variety_yield_potential_mean_t_ha"].fillna(
                measured_base["yield_potential_max_t_ha"]
            ),
            "min_rainfall_mm": measured_base["min_rainfall_mm"].round(2),
            "max_rainfall_mm": measured_base["max_rainfall_mm"].round(2),
            "min_altitude_m": [bounds[0] for bounds in measured_altitude],
            "max_altitude_m": [bounds[1] for bounds in measured_altitude],
            "input_requirement": measured_base.apply(_input_requirement, axis=1),
            "notes": (
                measured_base["adaptation_notes"].fillna("") + " " + measured_base["disease_resistance_notes"].fillna("")
            ).str.strip(),
            "release_year": measured_base["release_reference_year"].fillna(
                measured_base["nvin"].map(_release_year_from_nvin)
            ),
            "licensed_company_count": measured_base["licensed_company_count"].fillna(0),
            "supplier": measured_base["licensed_companies"].fillna("NARO licensed seed source"),
            "trait_source": "measured",
        }
    )
    measured["name_key"] = measured["variety_name"].map(_normalize_name_key)

    # --- crop-level proxy statistics for imputed varieties ---
    crop_stats = measured.groupby("crop").agg(
        crop_maturity=("maturity_days", "median"),
        crop_yield=("yield_potential_t_ha", "median"),
        crop_min_rainfall=("min_rainfall_mm", "median"),
        crop_max_rainfall=("max_rainfall_mm", "median"),
        crop_min_alt=("min_altitude_m", "median"),
        crop_max_alt=("max_altitude_m", "median"),
    )

    # --- licensed catalogue (names + licensing; traits imputed) ---
    lic = pd.read_csv(datasets_dir / "uganda_naro_licensed_seed_varieties_2023.csv")
    lic["crop"] = lic["crop"].astype(str).str.strip().str.lower()
    lic = lic[lic["crop"].isin(MODELLED_CROPS)].copy()
    lic["variety_name"] = lic["variety_name"].astype(str).str.strip()
    lic["name_key"] = lic["variety_name"].map(_normalize_name_key)

    measured_keys = set(zip(measured["crop"], measured["name_key"]))
    lic = lic[~lic.apply(lambda r: (r["crop"], r["name_key"]) in measured_keys, axis=1)].copy()
    lic = lic.merge(crop_stats, on="crop", how="left")

    imputed = pd.DataFrame(
        {
            "variety_id": lic["nvin"].fillna(lic["variety_name"]).astype(str),
            "variety_name": lic["variety_name"],
            "crop": lic["crop"],
            "maturity_days": lic["crop_maturity"],
            "drought_tolerance": "medium",
            "disease_resistance": "medium",
            "yield_potential_t_ha": lic["crop_yield"],
            "min_rainfall_mm": lic["crop_min_rainfall"].round(2),
            "max_rainfall_mm": lic["crop_max_rainfall"].round(2),
            "min_altitude_m": lic["crop_min_alt"],
            "max_altitude_m": lic["crop_max_alt"],
            "input_requirement": lic["crop_yield"].map(_input_requirement_from_yield),
            "notes": "NARO-licensed variety; agronomic traits estimated from crop averages (no measured trait record).",
            "release_year": lic["nvin"].map(_release_year_from_nvin),
            "licensed_company_count": lic["licensed_company_count"].fillna(0),
            "supplier": lic["licensed_companies"].fillna("NARO licensed seed source"),
            "trait_source": "imputed",
        }
    )
    imputed["name_key"] = imputed["variety_name"].map(_normalize_name_key)

    catalogue = pd.concat([measured, imputed], ignore_index=True)
    catalogue = catalogue.drop_duplicates("variety_id", keep="first").drop(columns=["name_key"])
    catalogue["release_year"] = catalogue["release_year"].astype(float)
    catalogue["licensed_company_count"] = catalogue["licensed_company_count"].fillna(0).astype(int)
    return catalogue.reset_index(drop=True)


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

    varieties_full = _build_variety_catalogue(candidates, datasets_dir)
    variety_columns = [
        "variety_id",
        "variety_name",
        "crop",
        "maturity_days",
        "drought_tolerance",
        "disease_resistance",
        "yield_potential_t_ha",
        "min_rainfall_mm",
        "max_rainfall_mm",
        "min_altitude_m",
        "max_altitude_m",
        "input_requirement",
        "notes",
        "release_year",
        "licensed_company_count",
        "trait_source",
    ]
    varieties_full[variety_columns].to_csv(raw_dir / "seed_varieties.csv", index=False)

    # The NDVI / climate / soil / UNPS enrichment files ship inside `datasets/`; fall back to it
    # when the optional `missingdatasets/` override folder is absent.
    enrichment_dir = missing_dir if missing_dir.exists() else datasets_dir

    ndvi_by_district = _load_ndvi_by_district(enrichment_dir)
    climate_by_district = _load_climate_by_district(enrichment_dir)
    soilgrids_by_district = _load_soilgrids_by_district(enrichment_dir)
    isda_enrichment_by_district = _load_isda_enrichment_by_district(enrichment_dir)
    location_base = (
        candidates.drop_duplicates(["district", "season"])
        .merge(ndvi_by_district, on="district", how="left")
        .merge(climate_by_district, on="district", how="left")
        .merge(soilgrids_by_district, on="district", how="left")
        .merge(isda_enrichment_by_district, on="district", how="left")
    )
    rainfall = location_base["season_rainfall_total_mm"]
    rainfall_range = rainfall.max() - rainfall.min()
    if rainfall_range and not pd.isna(rainfall_range):
        drought_index = (1 - (rainfall - rainfall.min()) / rainfall_range).clip(0, 1)
    else:
        drought_index = pd.Series(0.5, index=location_base.index)
        
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

    district_context = _load_unps_district_context(enrichment_dir)
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

    has_license = varieties_full["licensed_company_count"] > 0
    variety_availability = pd.DataFrame(
        {
            "variety_id": varieties_full["variety_id"],
            "available": np.where(has_license, "yes", "unknown"),
            "supplier": varieties_full["supplier"].fillna("NARO licensed seed source"),
            "availability_note": np.where(
                has_license,
                varieties_full["licensed_company_count"].astype(int).astype(str)
                + " licensed seed company(ies) listed nationally",
                "Licensing/availability not confirmed",
            ),
        }
    )
    districts = pd.DataFrame({"district": sorted(locations["district"].dropna().unique())})
    availability = (
        variety_availability.merge(districts, how="cross")
        [["district", "variety_id", "available", "supplier", "availability_note"]]
    )
    availability.to_csv(raw_dir / "seed_availability.csv", index=False)

    return {
        "seed_varieties": len(varieties_full),
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
