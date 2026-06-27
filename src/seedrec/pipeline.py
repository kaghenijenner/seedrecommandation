from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-seedrec")

import pandas as pd

from .config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from .data import build_modelling_dataset, load_raw_data
from .evaluate import fairness_summary, group_summary, ranking_metrics_at_ks
from .explain import lime_explanation, save_shap_summary_figure, shap_summary, technical_explanation
from .model import save_model, train_best_model
from .recommend import rank_recommendations
from .validation import all_grouped_metrics


def _write_explanation_report(path, sample, technical, lime_items) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Explanation Report\n\n")
        fh.write(
            "Explanations are layered (proposal sec. 2.6.3): a plain-language line for farmers, "
            "agronomic drivers/cautions for extension officers, and SHAP/LIME feature attributions "
            "for technical reviewers.\n\n"
        )
        fh.write("## Sample recommendation\n\n")
        fh.write(f"- District: {sample['district']}\n")
        fh.write(f"- Crop: {sample['crop']}\n")
        fh.write(f"- Variety: {sample['variety_name']}\n")
        fh.write(f"- Recommendation score: {sample['recommendation_score']:.3f}\n")
        fh.write(f"- Model probability: {sample['model_probability']:.3f}\n")
        fh.write(f"- Data confidence: {sample['data_confidence']}\n\n")

        fh.write("### 1. Plain-language (farmer)\n\n")
        fh.write(f"{sample['explanation']}\n\n")

        fh.write("### 2. Agronomic (extension officer)\n\n")
        fh.write(f"- Positive drivers: {sample['positive_drivers'] or 'None recorded'}\n")
        fh.write(f"- Cautions: {sample['cautionary_drivers'] or 'None recorded'}\n")
        fh.write(f"- Seed availability: {sample['availability_note']}\n\n")

        fh.write("### 3. Technical (SHAP / LIME)\n\n")
        if technical and technical.get("shap"):
            fh.write("SHAP local contributions (feature: signed contribution to suitability):\n\n")
            for feature, value in technical["shap"]:
                fh.write(f"- {feature}: {value:+.3f}\n")
            fh.write("\n")
        else:
            fh.write("SHAP local contributions unavailable in this environment.\n\n")
        if lime_items:
            fh.write("LIME local explanation (reason: weight toward suitability):\n\n")
            for reason, weight in lime_items:
                fh.write(f"- {reason}: {weight:+.3f}\n")
            fh.write("\n")
        else:
            fh.write("LIME local explanation unavailable in this environment.\n\n")

        fh.write("## Global drivers\n\n")
        fh.write("See `reports/feature_importance.csv` and `reports/shap_global_summary.png` for the ")
        fh.write("global SHAP feature importance across all cases.\n")


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    varieties, locations, farmers, availability = load_raw_data()
    modelling_df = build_modelling_dataset(varieties, locations, farmers, availability)
    modelling_df.to_csv(DATA_PROCESSED / "modelling_dataset.csv", index=False)

    result = train_best_model(modelling_df)
    save_model(result)

    all_recommendations = []
    for case_id in sorted(modelling_df["case_id"].unique()):
        recs = rank_recommendations(result.pipeline, modelling_df, case_id, top_k=6)
        all_recommendations.append(recs)
    recommendations = pd.concat(all_recommendations, ignore_index=True)
    recommendations.to_csv(REPORTS_DIR / "sample_recommendations.csv", index=False)

    # Annotate recommendations with case context for fairness analysis.
    case_meta = modelling_df.drop_duplicates("case_id")[["case_id", "resource_level", "agro_ecological_zone"]]
    recommendations_ctx = recommendations.merge(case_meta, on="case_id", how="left")

    metrics = result.metrics
    metrics["ranking_random_split"] = ranking_metrics_at_ks(recommendations, ks=(3, 5))
    metrics["validation_grouped"] = all_grouped_metrics(modelling_df, model_name="xgboost")
    with open(REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    group_summary(modelling_df).to_csv(REPORTS_DIR / "group_summary.csv", index=False)
    fairness_summary(recommendations_ctx, "resource_level", k=3).to_csv(REPORTS_DIR / "fairness_resource_level.csv", index=False)
    fairness_summary(recommendations_ctx, "agro_ecological_zone", k=3).to_csv(REPORTS_DIR / "fairness_zone.csv", index=False)

    shap_summary(result.pipeline, modelling_df.head(400)).head(25).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_shap_summary_figure(result.pipeline, modelling_df.head(400), REPORTS_DIR / "shap_global_summary.png")

    # Showcase a confident, well-attributed recommendation so SHAP/LIME are meaningful.
    sample = recommendations.sort_values("recommendation_score", ascending=False).iloc[0]
    technical = technical_explanation(result.pipeline, modelling_df, sample["case_id"], sample["variety_id"], with_lime=False)
    sample_row = modelling_df[
        (modelling_df["case_id"].eq(sample["case_id"])) & (modelling_df["variety_id"].eq(sample["variety_id"]))
    ]
    lime_items = lime_explanation(result.pipeline, modelling_df, sample_row.iloc[0]) if not sample_row.empty else None
    _write_explanation_report(REPORTS_DIR / "explanation_report.md", sample, technical, lime_items)

    print("Pipeline complete")
    print(f"Selected model: {result.model_name}")
    print(f"Calibration applied: {metrics['calibration'].get('applied')}")
    print(f"Ranking (random split): {metrics['ranking_random_split']}")
    print(f"Leave-zone-out: {metrics['validation_grouped']['leave_zone_out']}")
    print(f"Processed data: {DATA_PROCESSED / 'modelling_dataset.csv'}")
    print(f"Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
