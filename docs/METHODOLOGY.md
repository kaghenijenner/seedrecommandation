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

### Variety catalogue

The variety catalogue is the union of (a) NARO compendium varieties with measured agronomic
traits and (b) the full NARO licensed seed catalogue for the modelled crops (maize, beans,
groundnuts, rice). Varieties without a measured trait record receive crop-level proxy traits
and are flagged with `trait_source = "imputed"`; this is surfaced as lower `data_confidence`
(proposal sec. 2.4.5, 3.4.1). Two real per-variety signals — `release_year` (parsed from the
NVIN) and `licensed_company_count` — keep imputed varieties distinguishable. This raises the
candidate pool from 2 to roughly 11–20 varieties per case so the ranking task is meaningful.

## Model Development

The system trains and compares several candidate classifiers:

- Logistic Regression and Random Forest / Extra Trees baselines.
- HistGradientBoosting.
- XGBoost (the proposal's hero model), tuned with a light `RandomizedSearchCV`.

Selection uses a robust composite (ROC-AUC + balanced accuracy + F1) rather than raw accuracy,
with a tie-break that prefers XGBoost when models are effectively tied. Probability calibration
(isotonic / sigmoid) is **assessed** out-of-fold and only **applied** when it improves the Brier
score; on the near-deterministic suitability labels the tree ensemble is already sharp, so
calibration is reported but typically not applied (proposal sec. 2.7.1).

The target is `is_suitable`, derived from the synthetic suitability score. In the final research
dataset, replace this with observed suitability, yield-performance class, or expert-labelled
suitability.

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

Explanations are layered (proposal sec. 2.6.3):

1. **Plain-language (farmer):** a friendly sentence per recommendation. An optional Gemini LLM
   phrasing is available behind `SEEDREC_USE_GEMINI=1`; the default is a deterministic, offline
   agronomic sentence.
2. **Agronomic (extension officer):** rule-based positive drivers and cautions.
3. **Technical (researcher):** global SHAP importance (`feature_importance.csv` +
   `shap_global_summary.png`), per-recommendation local SHAP contributions, and LIME local
   explanations. SHAP/feature-importance fall back gracefully if a model is unsupported, and the
   SHAP helpers unwrap a calibrated classifier to the underlying tree when needed.

## Validation

The implementation reports two complementary views:

- **Random-split CV** (optimistic; suitability-label reconstruction) in `model_comparison`.
- **Grouped / spatial-temporal CV** (`validation_grouped`): leave-zone-out, leave-district-out,
  and hold-out-season via `GroupKFold` (proposal sec. 3.6.4, 3.8). The gap between the two makes
  the spatial-autocorrelation effect visible, exactly as the proposal argues.

Other outputs: classification metrics, ranking metrics (top-k, precision@k, recall@k, nDCG@k, MRR
at k=3 and 5), calibration (Brier), agro-ecological/resource summary, and fairness-style ranking
splits by resource level and zone (`fairness_*.csv`, proposal sec. 3.8.3).

