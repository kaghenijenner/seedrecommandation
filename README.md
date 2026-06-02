# Explainable Seed Variety Recommendation System

This project implements a research-backed prototype for selecting suitable seed varieties for Ugandan smallholder farming. The first version targets maize and demonstrates the full workflow with anonymized synthetic data so the pipeline can run before real institutional datasets are available.

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

The `datasets` folder is treated as the verified source bundle. Run `python -m seedrec.prepare_real_data` whenever those source files change; it rebuilds the project inputs in `data/raw`.

If a `missingdatasets` folder exists, the preparation step also uses supported enrichment files such as district NDVI, NASA-style climate/weather summaries, SoilGrids values when populated, and UNPS district welfare/resource context.

## Main Outputs

Running the pipeline creates:

- `data/raw/*.csv` real project inputs generated from `datasets`
- `data/processed/modelling_dataset.csv`
- `models/seedrec_pipeline.joblib`
- `reports/evaluation_metrics.json`
- `reports/sample_recommendations.csv`
- `reports/explanation_report.md`

## Success Criteria

- The system ranks candidate maize varieties for a district/season/farmer context.
- The main recommendation table stays simple for non-technical users and shows only the rank and variety name by default.
- Detailed scoring, confidence, drivers, cautions, and availability notes appear in the explanation section for each variety.
- The evaluation covers predictive performance, ranking quality, and explanation plausibility.
- The implementation can be demonstrated with verified Uganda district/crop data while clearly documenting missing variety-level trial outcomes.