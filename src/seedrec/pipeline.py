from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-seedrec")

from .config import DATA_PROCESSED, REPORTS_DIR
from .data import build_modelling_dataset, load_raw_data
from .evaluate import group_summary, ranking_metrics
from .explain import shap_summary
from .model import save_model, train_best_model
from .recommend import rank_recommendations


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
    recommendations = __import__("pandas").concat(all_recommendations, ignore_index=True)
    recommendations.to_csv(REPORTS_DIR / "sample_recommendations.csv", index=False)

    metrics = result.metrics
    metrics["ranking"] = ranking_metrics(recommendations, k=3)
    with open(REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    group_summary(modelling_df).to_csv(REPORTS_DIR / "group_summary.csv", index=False)
    shap_summary(result.pipeline, modelling_df.head(20)).head(20).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    with open(REPORTS_DIR / "explanation_report.md", "w", encoding="utf-8") as fh:
        fh.write("# Explanation Report\n\n")
        fh.write("This report summarizes the explanation layer for the demo model.\n\n")
        fh.write("## Sample Recommendation\n\n")
        sample = recommendations.iloc[0]
        fh.write(f"- District: {sample['district']}\n")
        fh.write(f"- Variety: {sample['variety_name']}\n")
        fh.write(f"- Score: {sample['recommendation_score']:.3f}\n")
        fh.write(f"- Explanation: {sample['explanation']}\n\n")
        fh.write("## Notes\n\n")
        fh.write("SHAP is used when installed and compatible with the selected model. Otherwise, the project writes model feature importance as a fallback.\n")

    print("Pipeline complete")
    print(f"Selected model: {result.model_name}")
    print(f"Processed data: {DATA_PROCESSED / 'modelling_dataset.csv'}")
    print(f"Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
