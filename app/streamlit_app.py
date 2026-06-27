from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import seedrec.config as config
from seedrec.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from seedrec.data import build_modelling_dataset, load_raw_data
from seedrec.explain import technical_explanation
from seedrec.model import load_model, save_model, train_best_model
from seedrec.recommend import build_case_from_inputs, rank_recommendations
from seedrec.seasons import current_season_phase, latest_season_code_for_phase, season_phase_options

st.set_page_config(page_title="Best Seeds for My Farm", page_icon="🌱", layout="wide")


# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
        html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox { font-family: 'Nunito', sans-serif; }
        .block-container { padding-top: 1.4rem; max-width: 1150px; }
        h1, h2, h3 { color: #1b4d20; font-weight: 800; }

        .app-hero {
            background: linear-gradient(135deg, #2e7d32 0%, #43a047 55%, #7cb342 100%);
            color: #ffffff; border-radius: 22px; padding: 26px 30px; margin-bottom: 8px;
            box-shadow: 0 10px 26px rgba(46,125,50,0.28);
        }
        .app-hero h1 { color: #ffffff; margin: 0 0 4px 0; font-size: 2.0rem; }
        .app-hero p { color: #eaf6ea; margin: 0; font-size: 1.05rem; }

        .pick-card {
            background: #ffffff; border-radius: 20px; padding: 24px 26px; margin: 14px 0 6px 0;
            border-left: 10px solid #2e7d32; box-shadow: 0 8px 22px rgba(31,42,34,0.10);
        }
        .pick-card .crown { font-size: 0.95rem; color: #1b7e3a; font-weight: 800; letter-spacing: .4px; }
        .pick-card .vname { font-size: 1.9rem; font-weight: 800; color: #14331a; margin: 2px 0 6px 0; }
        .pick-card .why { font-size: 1.12rem; color: #33413a; line-height: 1.55; }
        .pick-card .where { font-size: 1.0rem; color: #4a5a50; margin-top: 10px; }

        .badge {
            display: inline-block; padding: 5px 13px; border-radius: 999px; font-weight: 700;
            font-size: 0.85rem; margin: 4px 8px 4px 0;
        }
        .badge-high { background: #e3f4e4; color: #1b7e3a; border: 1px solid #b6e0bb; }
        .badge-medium { background: #fdf3da; color: #9a6b00; border: 1px solid #f2d98f; }
        .badge-low { background: #eceff1; color: #51616b; border: 1px solid #cfd8dc; }
        .badge-avail { background: #e7f0fb; color: #1259a8; border: 1px solid #bcd6f4; }
        .match-pill { background: #14331a; color: #fff; padding: 6px 14px; border-radius: 999px; font-weight: 800; }

        .opt-name { font-size: 1.28rem; font-weight: 800; color: #14331a; }
        .opt-why { font-size: 1.02rem; color: #3a473f; }
        .soft-note { color:#6a7670; font-size:0.92rem; }
        section[data-testid="stSidebar"] { background: #f3f7ec; }
        section[data-testid="stSidebar"] h2 { font-size: 1.15rem; }
        div[data-baseweb="select"] > div { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

INPUT_ACCESS_LABELS = {
    "low": "Hard to get seed & fertiliser",
    "medium": "Some access to inputs",
    "high": "Good access to inputs",
    "unknown": "Not sure",
}
RESOURCE_LABELS = {
    "low": "Small budget",
    "medium": "Medium budget",
    "high": "Larger budget",
    "unknown": "Not sure",
}


def confidence_badge(level: str) -> str:
    level = str(level).lower()
    mapping = {
        "high": ("High confidence", "badge-high"),
        "medium": ("Medium confidence", "badge-medium"),
        "low": ("Lower confidence", "badge-low"),
    }
    text, css = mapping.get(level, ("Confidence unknown", "badge-low"))
    return f'<span class="badge {css}">{text}</span>'


def availability_badge(available: str) -> str:
    available = str(available).lower()
    mapping = {
        "yes": "Sold by listed seed companies",
        "unknown": "Ask your local seed seller",
        "no": "May be hard to find locally",
    }
    return f'<span class="badge badge-avail">{mapping.get(available, "Check local availability")}</span>'


def match_label(score: float) -> str:
    pct = int(round(float(score) * 100))
    if pct >= 80:
        return f"{pct}% match"
    if pct >= 60:
        return f"{pct}% match"
    return f"{pct}% match"


# --------------------------------------------------------------------------------------
# Data + model
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    processed = DATA_PROCESSED / "modelling_dataset.csv"
    if processed.exists():
        return pd.read_csv(processed)
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed, index=False)
    return df


@st.cache_resource(show_spinner="Training the recommendation model (first run only)…")
def load_or_train_model(_df: pd.DataFrame):
    model_path = MODELS_DIR / "seedrec_pipeline.joblib"
    if model_path.exists():
        return load_model(model_path)
    result = train_best_model(_df)
    save_model(result)
    return result.pipeline


@st.cache_data(show_spinner=False)
def cached_technical(case_id, variety_id, with_lime, _pipe, _df):
    return technical_explanation(_pipe, _df, case_id, variety_id, with_lime=with_lime)


def friendly_select(label, values, label_map, help_text=None, default=None):
    values = [v for v in values if pd.notna(v)]
    index = values.index(default) if default in values else 0
    return st.selectbox(
        label,
        values,
        index=index,
        format_func=lambda v: label_map.get(str(v), str(v).title()),
        help=help_text,
    )


def render_drivers(positive: str, cautions: str) -> None:
    pos = [item for item in str(positive).split(" | ") if item and item != "nan"]
    cau = [item for item in str(cautions).split(" | ") if item and item != "nan"]
    if pos:
        st.markdown("**What helps this variety here**")
        for item in pos:
            st.markdown(f"- {item}")
    if cau:
        st.markdown("**Things to watch**")
        for item in cau:
            st.markdown(f"- {item}")
    if not pos and not cau:
        st.caption("No strong drivers were detected for this case.")


def render_technical(case_id, variety_id, pipe, df, key: str) -> None:
    # Compute SHAP/LIME lazily (only on request) so the app stays fast and responsive.
    state_key = f"tech_{key}"
    if st.button("Compute SHAP / LIME for this variety", key=f"btn_{key}"):
        st.session_state[state_key] = True
    if not st.session_state.get(state_key):
        st.caption("Click the button to compute the technical (SHAP/LIME) explanation for this variety.")
        return
    with_lime = st.checkbox("Also compute LIME explanation (slower)", key=f"lime_{key}", value=False)
    with st.spinner("Reading the model…"):
        tech = cached_technical(case_id, variety_id, with_lime, pipe, df)
    shap_items = tech.get("shap") if tech else None
    if shap_items:
        st.markdown("**SHAP - how each factor pushed the suitability score**")
        for feature, value in shap_items:
            arrow = "▲" if value >= 0 else "▼"
            st.markdown(f"{arrow} **{feature}** ({value:+.2f})")
    else:
        st.caption("SHAP contributions are not available in this environment.")
    if with_lime:
        lime_items = tech.get("lime") if tech else None
        if lime_items:
            st.markdown("**LIME - local reasons for this single recommendation**")
            for reason, weight in lime_items:
                arrow = "▲" if weight >= 0 else "▼"
                st.markdown(f"{arrow} {reason} ({weight:+.2f})")
        else:
            st.caption("LIME explanation is not available in this environment.")


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
inject_css()
df = load_dataset()
pipe = load_or_train_model(df)
default_goal = str(df["production_goal"].mode().iloc[0]) if "production_goal" in df.columns else "food_and_market"

with st.sidebar:
    st.markdown("## Tell us about your farm")
    st.caption("Answer a few simple questions and we will suggest the best seeds for you.")

    crop = friendly_select(
        "Which crop do you want to plant?",
        sorted(df["crop"].unique()),
        {c: f"{c.title()}" for c in df["crop"].unique()},
        help_text="Pick the crop you have already decided to grow.",
    )
    crop_df = df[df["crop"].eq(crop)]

    district = st.selectbox("Where is your farm (district)?", sorted(crop_df["district"].unique()))
    district_df = crop_df[crop_df["district"].eq(district)]

    season_choices = season_phase_options(district_df)
    default_phase = current_season_phase()
    season_phase = st.selectbox(
        "Which planting season?",
        season_choices,
        index=season_choices.index(default_phase) if default_phase in season_choices else 0,
        help="Uganda has two main rainy seasons. We use the latest data for that season.",
    )
    season = latest_season_code_for_phase(district_df, season_phase)
    if season is None:
        st.warning(f"No data yet for {season_phase} in {district}. Please choose another season.")
        st.stop()

    input_access = friendly_select(
        "Can you buy inputs (seed, fertiliser)?",
        sorted(district_df["input_access"].dropna().unique()),
        INPUT_ACCESS_LABELS,
        help_text="This helps us pick varieties that match what you can afford and access.",
    )
    resource_level = friendly_select(
        "What is your farming budget?",
        sorted(district_df["resource_level"].dropna().unique()),
        RESOURCE_LABELS,
    )
    top_k = st.slider("How many seed options to show?", 3, 6, 5)

    with st.expander("Advanced (for technical users)"):
        show_scoring = st.checkbox("Show scoring numbers", value=False)
        min_threshold_default = float(getattr(config, "MIN_RECOMMENDATION_THRESHOLD", 0.0))
        min_threshold = st.slider("Minimum recommendation score", 0.0, 1.0, min_threshold_default, 0.01)

st.markdown(
    f"""
    <div class="app-hero">
        <h1>Best Seeds for My Farm</h1>
        <p>Friendly, explainable seed-variety advice for Ugandan smallholder farmers, built for everyone in farming.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

case_id = build_case_from_inputs(df, district, season, input_access, default_goal, resource_level, crop=crop)
recommendations = rank_recommendations(pipe, df, case_id, top_k=top_k, min_threshold=min_threshold)

st.markdown(f"### Best {crop.title()} seeds for {district} - {season_phase}")

if recommendations.empty:
    st.info(
        "We could not find a confident seed suggestion for these answers. "
        "Try a different season or ask your local extension officer."
    )
    st.stop()

top = recommendations.iloc[0]
st.markdown(
    f"""
    <div class="pick-card">
        <div class="crown">OUR TOP RECOMMENDATION</div>
        <div class="vname">{top['variety_name']}</div>
        <span class="match-pill">{match_label(top['recommendation_score'])}</span>
        {confidence_badge(top['data_confidence'])}
        {availability_badge(top['available'])}
        <div class="why"><b>Why this is good for you:</b> {top['explanation']}</div>
        <div class="where"><b>Where to get it:</b> {top['supplier']} &nbsp;·&nbsp; {top['availability_note']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("More detail on the top pick (for extension officers)"):
    render_drivers(top["positive_drivers"], top["cautionary_drivers"])
    if show_scoring:
        st.markdown(
            f"<span class='soft-note'>Model probability {top['model_probability']:.2f} · "
            f"availability penalty {top['availability_penalty']:.2f} · "
            f"final score {top['recommendation_score']:.2f}</span>",
            unsafe_allow_html=True,
        )
with st.expander("Technical explanation for the top pick (SHAP / LIME)"):
    render_technical(top["case_id"], top["variety_id"], pipe, df, key="top")

others = recommendations.iloc[1:]
if not others.empty:
    st.markdown("### Other good options")
    for _, row in others.iterrows():
        with st.container(border=True):
            head, badges = st.columns([3, 2])
            with head:
                st.markdown(f"<span class='opt-name'>#{int(row['rank'])} · {row['variety_name']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='opt-why'>{row['explanation']}</span>", unsafe_allow_html=True)
            with badges:
                st.markdown(f"<span class='match-pill'>{match_label(row['recommendation_score'])}</span>", unsafe_allow_html=True)
                st.markdown(confidence_badge(row["data_confidence"]) + availability_badge(row["available"]), unsafe_allow_html=True)
            st.markdown(f"<span class='soft-note'>{row['supplier']} · {row['availability_note']}</span>", unsafe_allow_html=True)
            with st.expander("More detail (for extension officers)"):
                render_drivers(row["positive_drivers"], row["cautionary_drivers"])
                if show_scoring:
                    st.markdown(
                        f"<span class='soft-note'>Model probability {row['model_probability']:.2f} · "
                        f"final score {row['recommendation_score']:.2f}</span>",
                        unsafe_allow_html=True,
                    )
            with st.expander("Technical explanation (SHAP / LIME)"):
                render_technical(row["case_id"], row["variety_id"], pipe, df, key=str(row["variety_id"]))

with st.expander("How the model decides (global view)"):
    image_path = REPORTS_DIR / "shap_global_summary.png"
    if image_path.exists():
        st.image(str(image_path), caption="Most influential factors across all cases (global SHAP importance)")
    else:
        st.caption("Run `python -m seedrec.pipeline` to generate the global importance chart.")
    st.markdown(
        "This tool combines **climate, soil, seed traits, farmer context and local seed availability**, "
        "predicts how suitable each variety is, then ranks the options and explains every choice."
    )