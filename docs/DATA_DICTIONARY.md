# Data Dictionary

The first implementation uses maize demo data. Replace these records with validated institutional or field datasets during the research implementation.

## Unit of Analysis

`location + season + crop + seed variety + farmer context`

## Core Variables

| Variable | Category | Description |
| --- | --- | --- |
| `case_id` | Farmer context | Anonymous recommendation case identifier. |
| `district` | Location | District used as the demo spatial unit. |
| `agro_ecological_zone` | Location | Agro-ecological context for grouped evaluation. |
| `zardi_zone` | Location | Zonal agricultural research context. |
| `season` | Time | Agricultural season or year-season identifier. |
| `crop` | Crop | Crop name, initially maize. |
| `variety_id` | Variety | Seed variety identifier. |
| `variety_name` | Variety | Human-readable seed variety name. |
| `mean_rainfall_mm` | Climate | Seasonal rainfall estimate. |
| `mean_temperature_c` | Climate | Seasonal mean temperature. |
| `drought_index` | Climate | Demo drought-risk indicator from 0 to 1. |
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
| `yield_potential_t_ha` | Variety trait | Demo yield potential in tonnes per hectare. |
| `input_requirement` | Variety trait | Low, medium, or high input requirement. |
| `input_access` | Farmer context | Farmer access to seed, fertilizer, chemicals, and advisory inputs. |
| `production_goal` | Farmer context | Food security, market, or mixed goal. |
| `resource_level` | Farmer context | Overall resource profile. |
| `available` | Seed market | Whether seed is locally available in demo data. |
| `suitability_score` | Label | Synthetic expert-style suitability score from 0 to 100. |
| `suitability_class` | Label | Unsuitable, moderately suitable, or suitable. |
| `data_confidence` | Governance | High or medium confidence flag based on missing/uncertain fields. |

