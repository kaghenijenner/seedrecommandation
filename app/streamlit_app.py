from __future__ import annotations

import sys
from pathlib import Path
import os

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import seedrec.config as config
from seedrec.config import DATA_PROCESSED, MODELS_DIR
from seedrec.data import build_modelling_dataset, load_raw_data
from seedrec.model import load_model, save_model, train_best_model
from seedrec.recommend import build_case_from_inputs, rank_recommendations, recommendation_table_columns
from seedrec.seasons import current_season_phase, latest_season_code_for_phase, season_phase_options


st.set_page_config(page_title="Seed Variety Recommender", layout="wide")

if os.getenv("GEMINI_API_KEY"):
    st.sidebar.success("Gemini explanations are enabled")
else:
    st.sidebar.info("Set GEMINI_API_KEY to enable Gemini explanations")


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

with st.sidebar:
    crop = st.selectbox("Crop", sorted(df["crop"].unique()))
    crop_df = df[df["crop"].eq(crop)]
    district = st.selectbox("District", sorted(crop_df["district"].unique()))
    district_df = crop_df[crop_df["district"].eq(district)]
    season_choices = season_phase_options(district_df)
    default_phase = current_season_phase()
    season_phase = st.selectbox("Season phase", season_choices, index=season_choices.index(default_phase) if default_phase in season_choices else 0)
    season = latest_season_code_for_phase(district_df, season_phase)
    input_access = st.selectbox("Input access", sorted(district_df["input_access"].dropna().unique()))
    production_goal = st.selectbox("Production goal", sorted(district_df["production_goal"].dropna().unique()))
    resource_level = st.selectbox("Resource level", sorted(district_df["resource_level"].dropna().unique()))
    top_k = st.slider("Number of recommendations", 3, 6, 5)
    show_scoring = st.checkbox("Show scoring details (model probability & availability penalty)", value=False)
    min_threshold_default = float(getattr(config, "MIN_RECOMMENDATION_THRESHOLD", 0.0))
    min_threshold = st.slider("Minimum recommendation score", 0.0, 1.0, min_threshold_default, 0.01)

    st.caption(f"Current calendar phase defaults to {default_phase}; using the latest available historical record for {season_phase}.")

case_id = build_case_from_inputs(df, district, season, input_access, production_goal, resource_level, crop=crop)
recommendations = rank_recommendations(pipe, df, case_id, top_k=top_k, min_threshold=min_threshold)

st.subheader(f"Ranked {crop} varieties for {district}")
if recommendations.empty:
    st.info("No strong recommendation is available for this case; all candidate seeds were scored at or below zero after availability adjustment.")
else:
    display_cols = recommendation_table_columns(show_scoring=show_scoring)
    st.dataframe(recommendations[display_cols], width="stretch", hide_index=True)

st.subheader("Explanation")
if recommendations.empty:
    st.caption("There are no recommended seeds to explain for this case.")
else:
    for _, row in recommendations.iterrows():
        with st.expander(f"#{row['rank']} {row['variety_name']}"):
            st.write(row["explanation"])
            st.write(f"Recommendation score: {row['recommendation_score']:.2f}")
            st.write(f"Model probability: {row['model_probability']:.2f}")
            st.write(f"Availability penalty: {row['availability_penalty']:.2f}")
            st.write(f"Positive drivers: {row['positive_drivers'] or 'None recorded'}")
            st.write(f"Cautions: {row['cautionary_drivers'] or 'None recorded'}")
            st.write(f"Seed availability: {row['availability_note']}")
