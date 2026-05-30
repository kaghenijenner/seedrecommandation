from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seedrec.config import DATA_PROCESSED, MODELS_DIR
from seedrec.data import build_modelling_dataset, load_raw_data
from seedrec.model import load_model, save_model, train_best_model
from seedrec.recommend import build_case_from_inputs, rank_recommendations


st.set_page_config(page_title="Seed Variety Recommender", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    processed = DATA_PROCESSED / "modelling_dataset.csv"
    if processed.exists():
        return pd.read_csv(processed)
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed, index=False)
    return df


@st.cache_resource
def load_or_train_model(df: pd.DataFrame):
    model_path = MODELS_DIR / "seedrec_pipeline.joblib"
    if model_path.exists():
        return load_model(model_path)
    result = train_best_model(df)
    save_model(result, model_path)
    return result.pipeline


df = load_dataset()
pipe = load_or_train_model(df)

st.title("Explainable Seed Variety Recommendation")
st.caption("Maize demo for Ugandan smallholder farming contexts")

with st.sidebar:
    district = st.selectbox("District", sorted(df["district"].unique()))
    season = st.selectbox("Season", sorted(df["season"].unique()))
    input_access = st.selectbox("Input access", ["low", "medium", "high"], index=1)
    production_goal = st.selectbox("Production goal", ["food_security", "food_and_market", "market"], index=1)
    resource_level = st.selectbox("Resource level", ["low", "medium", "high"], index=1)
    top_k = st.slider("Number of recommendations", 3, 6, 5)

case_id = build_case_from_inputs(df, district, season, input_access, production_goal, resource_level)
recommendations = rank_recommendations(pipe, df, case_id, top_k=top_k)

st.subheader(f"Ranked maize varieties for {district}")
st.dataframe(
    recommendations[
        [
            "rank",
            "variety_name",
            "recommendation_score",
            "suitability_class",
            "available",
            "data_confidence",
            "availability_note",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Explanation")
for _, row in recommendations.iterrows():
    with st.expander(f"#{row['rank']} {row['variety_name']}"):
        st.write(row["explanation"])
        st.write(f"Positive drivers: {row['positive_drivers'] or 'None recorded'}")
        st.write(f"Cautions: {row['cautionary_drivers'] or 'None recorded'}")
        st.write(f"Seed availability: {row['availability_note']}")

st.info("This is a research prototype using synthetic demo data. Replace the raw CSV files with verified institutional datasets before field use.")

