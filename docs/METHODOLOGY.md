# Methodology Implementation Notes

## Design Science Artefact

The implemented artefact has two parts:

1. A reproducible data and modelling pipeline.
2. A small Streamlit decision-support prototype.

The artefact is designed to be evaluated by predictive performance, ranking quality, explanation plausibility, and stakeholder usefulness.

## Data Integration

The pipeline integrates four CSV inputs:

- `seed_varieties.csv`
- `location_profiles.csv`
- `farmer_context.csv`
- `seed_availability.csv`

The integration workflow standardizes text fields, joins farmer cases to location profiles, expands each case across candidate seed varieties, joins local seed availability, and creates a modelling-ready dataset.

## Model Development

The system trains multiple candidate classifiers:

- Logistic Regression baseline.
- Random Forest baseline.
- XGBoost when installed.

The target is `is_suitable`, derived from the synthetic suitability score in the demo dataset. In the final research dataset, replace this with observed suitability, yield-performance class, or expert-labelled suitability.

## Recommendation Output

The system generates a ranked list with:

- Variety name.
- Suitability and recommendation scores.
- Rank.
- Positive drivers.
- Cautionary drivers.
- Seed availability note.
- Data confidence level.

## Explainability

The current explanation layer combines:

- Agronomic rule-based local explanations.
- SHAP summary when supported.
- Feature importance fallback when SHAP is unavailable.

LIME should be added during final experimentation for selected local cases.

## Validation

The implementation supports:

- Grouped cross-validation by district.
- Classification metrics.
- Ranking metrics.
- Agro-ecological and resource-level summary tables.

For the dissertation, add stakeholder review forms and expert plausibility scoring.

