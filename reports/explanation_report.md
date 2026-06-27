# Explanation Report

Explanations are layered (proposal sec. 2.6.3): a plain-language line for farmers, agronomic drivers/cautions for extension officers, and SHAP/LIME feature attributions for technical reviewers.

## Sample recommendation

- District: Kalangala
- Crop: beans
- Variety: NAROBEAN 6
- Recommendation score: 0.999
- Model probability: 0.999
- Data confidence: high

### 1. Plain-language (farmer)

This variety suits your farm because the elevation is within the variety adaptation range; the maturity period fits the season length; drought tolerance is useful for the local drought risk; disease resistance strengthens suitability. Things to watch: rainfall is outside the preferred variety range.

### 2. Agronomic (extension officer)

- Positive drivers: the elevation is within the variety adaptation range | the maturity period fits the season length | drought tolerance is useful for the local drought risk | disease resistance strengthens suitability
- Cautions: rainfall is outside the preferred variety range
- Seed availability: 1 licensed seed company(ies) listed nationally

### 3. Technical (SHAP / LIME)

SHAP local contributions (feature: signed contribution to suitability):

- mean rainfall mm: +1.751
- rainfall range distance mm: +1.573
- soil ph: +0.748
- drought index: +0.710
- organic matter pct: +0.660
- market preference low: +0.605

LIME local explanation (reason: weight toward suitability):

- organic matter pct > 35.69: +0.130
- market preference=low: +0.101
- soil ph <= 5.93: +0.083
- mean rainfall mm > 846.16: +0.048
- season=2009 First: -0.038
- drought x index <= 1.48: +0.030
- zardi zone=Central: -0.025
- soil type=acidic loam: +0.024

## Global drivers

See `reports/feature_importance.csv` and `reports/shap_global_summary.png` for the global SHAP feature importance across all cases.
