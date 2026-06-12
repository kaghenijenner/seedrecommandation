# Explanation Report

Explanations are layered (proposal sec. 2.6.3): a plain-language line for farmers, agronomic drivers/cautions for extension officers, and SHAP/LIME feature attributions for technical reviewers.

## Sample recommendation

- District: Kalangala
- Crop: beans
- Variety: NAROBEAN 7
- Recommendation score: 1.000
- Model probability: 1.000
- Data confidence: high

### 1. Plain-language (farmer)

This variety suits your farm because the elevation is within the variety adaptation range; the maturity period fits the season length; drought tolerance is useful for the local drought risk; disease resistance strengthens suitability. Things to watch: rainfall is outside the preferred variety range.

### 2. Agronomic (extension officer)

- Positive drivers: the elevation is within the variety adaptation range | the maturity period fits the season length | drought tolerance is useful for the local drought risk | disease resistance strengthens suitability
- Cautions: rainfall is outside the preferred variety range
- Seed availability: 3 licensed seed company(ies) listed nationally

### 3. Technical (SHAP / LIME)

SHAP local contributions (feature: signed contribution to suitability):

- mean rainfall mm: +3.343
- rainfall range distance mm: +2.322
- organic matter pct: +1.048
- soil ph: +0.969
- market preference low: +0.779
- input level gap: +0.756

LIME local explanation (reason: weight toward suitability):

- market preference=low: +0.166
- organic matter pct > 35.69: +0.163
- mean rainfall mm > 846.16: +0.127
- soil ph <= 5.93: +0.055
- drainage=moderate: +0.047
- drought x index <= 1.48: +0.035
- disease resistance=high: -0.027
- soil type=acidic loam: +0.021

## Global drivers

See `reports/feature_importance.csv` and `reports/shap_global_summary.png` for the global SHAP feature importance across all cases.
