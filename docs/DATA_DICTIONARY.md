# Data Dictionary

The current implementation uses verified Uganda district/crop data, NARO candidate-variety data, and optional enrichment files from `missingdatasets`.

## Unit of Analysis

`location + season + crop + seed variety + farmer context`

## Core Variables

| Variable | Category | Description |
| --- | --- | --- |
| `case_id` | Farmer context | Anonymous recommendation case identifier. |
| `district` | Location | District used as the spatial unit. |
| `agro_ecological_zone` | Location | Agro-ecological context for grouped evaluation. |
| `zardi_zone` | Location | Zonal agricultural research context. |
| `season` | Time | Agricultural season or year-season identifier. |
| `crop` | Crop | Crop name. |
| `variety_id` | Variety | Seed variety identifier. |
| `variety_name` | Variety | Human-readable seed variety name. |
| `mean_rainfall_mm` | Climate | Seasonal rainfall estimate. |
| `mean_temperature_c` | Climate | Seasonal mean temperature. |
| `climate_t2m_max_mean_c` | Climate | District mean of daily maximum temperature where climate data is available. |
| `climate_t2m_min_mean_c` | Climate | District mean of daily minimum temperature where climate data is available. |
| `climate_relative_humidity_pct` | Climate | District mean relative humidity where climate data is available. |
| `climate_daily_rainfall_mean_mm` | Climate | District mean daily rainfall from the climate file where available. |
| `climate_annualized_rainfall_mm` | Climate | Annualized rainfall proxy derived from daily rainfall mean. |
| `climate_surface_pressure_kpa` | Climate | District mean surface pressure where climate data is available. |
| `climate_wind_speed_m_s` | Climate | District mean wind speed where climate data is available. |
| `climate_solar_radiation_mj_m2_day` | Climate | District mean solar radiation where climate data is available. |
| `drought_index` | Climate | Drought-risk proxy from 0 to 1. |
| `season_length_days` | Climate | Estimated growing season length. |
| `soil_type` | Soil | Dominant local soil textural class. |
| `soil_ph` | Soil | Soil acidity/alkalinity indicator. |
| `organic_matter_pct` | Soil | Organic matter percentage. |
| `drainage` | Soil | Drainage class. |
| `elevation_m` | Terrain | Elevation above sea level. |
| `ndvi` | Remote sensing | Vegetation condition indicator. |
| `maturity_days` | Variety trait | Days to maturity. |
| `drought_tolerance` | Variety trait | Low, medium, or high drought tolerance. |
| `disease_resistance` | Variety trait | Low, medium, or high disease resistance. |
| `yield_potential_t_ha` | Variety trait | Yield potential in tonnes per hectare. |
| `input_requirement` | Variety trait | Low, medium, or high input requirement. |
| `input_access` | Farmer context | Farmer access to seed, fertilizer, chemicals, and advisory inputs. |
| `production_goal` | Farmer context | Food security, market, or mixed goal. |
| `resource_level` | Farmer context | Overall resource profile. |
| `market_preference` | Farmer context | Market-access/preference proxy. |
| `unps_mean_quintile` | Farmer context | District-level mean welfare quintile from UNPS where matched. |
| `unps_poverty_rate` | Farmer context | District-level poverty-rate proxy from UNPS where matched. |
| `unps_urban_share` | Farmer context | District-level urban household share from UNPS where matched. |
| `available` | Seed market | Whether seed availability/licensing information is present. |
| `suitability_score` | Label | Suitability proxy score from observed yield percentile plus agronomic compatibility. |
| `suitability_class` | Label | Unsuitable, moderately suitable, or suitable. |
| `data_confidence` | Governance | High or medium confidence flag based on missing/uncertain fields. |
