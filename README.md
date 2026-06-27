# Explainable Seed Variety Recommendation System

This project implements a research-backed prototype for selecting suitable seed varieties for Ugandan smallholder farming. It covers maize, beans, groundnuts, and rice using verified Uganda district/crop data (NARO varieties and licensing, CHIRPS rainfall, iSDA soil, NDVI, NASA POWER climate, UNPS welfare context), with synthetic suitability labels where variety-level trial outcomes are not yet available. The candidate set is the full NARO-licensed catalogue, so each location is ranked against many varieties rather than a token few.

## Research Goal

Build an explainable machine learning system that integrates agro-ecological, climate, soil, remote-sensing, seed-variety, farmer-context, and seed-availability data to produce ranked seed variety recommendations with human-readable explanations.

The first version is a decision-support artefact, not a replacement for agronomists, extension workers, or farmer judgement.

## Unit of Analysis

Each modelling row represents:

`location + season + crop + seed variety + farmer context`

## Repository Structure

- `data/raw`: source datasets or synthetic demo inputs.
- `data/processed`: integrated modelling-ready datasets.
- `src/seedrec`: reusable Python package for data integration, modelling, ranking, and explainability.
- `app`: Streamlit prototype.
- `models`: trained preprocessing and model artefacts.
- `reports`: evaluation outputs, explanations, and dissertation-ready tables.
- `docs`: methodology, data dictionary, and todo list.
- `tests`: automated checks for core behaviour.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m seedrec.prepare_real_data
.venv/bin/python -m seedrec.pipeline
streamlit run app/streamlit_app.py
```

If `xgboost`, `shap`, or `lime` are unavailable in your environment, the pipeline falls back to scikit-learn models and built-in feature-contribution explanations where possible.

Plain-language explanations are deterministic and offline by default. To use the optional Gemini LLM phrasing in the app, set `SEEDREC_USE_GEMINI=1` (and `GEMINI_API_KEY`); it is intentionally off by default so the pipeline and app stay fast and offline.

The `datasets` folder is treated as the verified source bundle. Run `python -m seedrec.prepare_real_data` whenever those source files change; it rebuilds the project inputs in `data/raw`.

If a `missingdatasets` folder exists, the preparation step also uses supported enrichment files such as district NDVI, NASA-style climate/weather summaries, SoilGrids values when populated, and UNPS district welfare/resource context.

## Main Outputs

Running the pipeline creates:

- `data/raw/*.csv` real project inputs generated from `datasets`
- `data/processed/modelling_dataset.csv`
- `models/seedrec_pipeline.joblib`
- `reports/evaluation_metrics.json` (model comparison, calibration, ranking@{3,5}+MRR, grouped/spatial-temporal validation)
- `reports/sample_recommendations.csv`
- `reports/feature_importance.csv` and `reports/shap_global_summary.png` (global SHAP)
- `reports/fairness_resource_level.csv`, `reports/fairness_zone.csv`
- `reports/explanation_report.md` (layered farmer / extension / SHAP+LIME example)

## Success Criteria

- The system ranks candidate varieties (maize, beans, groundnuts, rice) for a district/season/farmer context against the full licensed catalogue.
- The farmer interface is attractive and plain-language: a hero "top recommendation" card, friendly icon-led inputs, and confidence/availability badges — usable by non-technical users including women in farming.
- Each recommendation has layered explanations: a plain-language reason, agronomic drivers/cautions for extension officers, and SHAP/LIME technical attributions.
- Evaluation covers predictive performance, ranking quality (top-k, nDCG, MRR), probability calibration, and spatial/temporal generalization (leave-zone-out, leave-district-out, hold-out-season).
- The implementation runs on verified Uganda district/crop data while clearly documenting imputed variety traits (lower confidence) and missing variety-level trial outcomes.