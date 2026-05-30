# Project Todo List

## Stage 1: Foundation

- [x] Create project repository structure.
- [x] Write project README.
- [x] Define first target crop as maize.
- [x] Define the unit of analysis.
- [x] Create initial data dictionary.
- [x] List required datasets and possible sources.
- [ ] Replace demo data with verified real datasets.

## Stage 2: Data Collection

- [x] Add demo seed variety trait data.
- [x] Add demo climate/location profile data.
- [x] Add demo soil and terrain data.
- [x] Add demo remote-sensing indicator data.
- [x] Add demo farmer-context data.
- [x] Add demo seed availability data.
- [ ] Record final source, date, license, coverage, and limitations for each real dataset.

## Stage 3: Data Preparation

- [x] Clean and standardize demo datasets.
- [x] Merge datasets into a modelling-ready table.
- [x] Create synthetic suitability labels and scores.
- [x] Create data confidence indicators.
- [x] Save processed dataset versions through the pipeline.
- [ ] Add real-data quality report once final datasets are obtained.

## Stage 4: Modelling

- [x] Implement logistic regression baseline.
- [x] Implement Random Forest baseline.
- [x] Implement optional XGBoost candidate model.
- [x] Compare models using grouped validation.
- [x] Save trained model and preprocessing pipeline.
- [ ] Tune hyperparameters on final dataset.

## Stage 5: Recommendation Logic

- [x] Generate candidate varieties for each case.
- [x] Score each candidate variety.
- [x] Rank varieties by suitability.
- [x] Apply seed availability penalty without hiding biological suitability.
- [x] Add data confidence level.
- [x] Produce final recommendation output format.

## Stage 6: Explainability

- [x] Add agronomic text explanations.
- [x] Add SHAP summary hook with fallback feature importance.
- [x] Add local positive and cautionary drivers.
- [ ] Add LIME report generation after dependencies are installed and final data is available.
- [ ] Conduct expert plausibility review.

## Stage 7: Evaluation

- [x] Add predictive evaluation metrics.
- [x] Add ranking metrics.
- [x] Add grouped agro-ecological/resource summary.
- [ ] Add robustness checks across seasons when multi-season data is available.
- [ ] Conduct stakeholder review.
- [ ] Document final limitations.

## Stage 8: Prototype

- [x] Choose Streamlit for the prototype.
- [x] Build input form for farmer/location conditions.
- [x] Connect prototype to model pipeline.
- [x] Display ranked recommendations.
- [x] Display explanations and caution notes.
- [x] Add low-confidence and demo-data warning.
- [ ] Capture prototype screenshots for dissertation.

## Stage 9: Final Packaging

- [x] Prepare code artefacts for pipeline, model, explanations, and demo.
- [x] Prepare documentation skeleton.
- [ ] Prepare final evaluation report from real data.
- [ ] Prepare dissertation result tables and figures.
- [ ] Prepare final presentation slides.
- [ ] Archive final code, data versions, model files, and reports.

