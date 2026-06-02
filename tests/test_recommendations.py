from datetime import date

import pandas as pd
import pytest

from seedrec.data import build_modelling_dataset, load_raw_data
from seedrec.model import train_best_model
from seedrec.recommend import rank_recommendations, recommendation_table_columns
from seedrec.seasons import current_season_phase, latest_season_code_for_phase


def test_modelling_dataset_has_expected_rows():
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    expected_rows = sum(
        len(farmers[farmers["crop"].eq(crop)]) * len(varieties[varieties["crop"].eq(crop)])
        for crop in farmers["crop"].unique()
    )
    assert len(df) == expected_rows
    assert {"suitability_score", "suitability_class", "data_confidence"}.issubset(df.columns)


def test_recommendation_output_has_required_fields():
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    result = train_best_model(df)
    recs = rank_recommendations(result.pipeline, df, str(df.iloc[0]["case_id"]), top_k=2)
    assert len(recs) == 2
    assert recs["rank"].tolist() == [1, 2]
    assert recs["explanation"].str.len().min() > 20
    assert {"recommendation_score", "available", "data_confidence"}.issubset(recs.columns)


def test_current_season_phase_defaults_to_first_season_in_june():
    assert current_season_phase(date(2026, 6, 2)) == "First season"


def test_latest_season_code_for_phase_chooses_most_recent_year():
    district_df = pd.DataFrame({"season": ["2008_First", "2009_First", "2008_Second", "2009_Second"]})
    assert latest_season_code_for_phase(district_df, "First season") == "2009_First"


def test_recommendations_drop_zero_scores(monkeypatch):
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    case_id = str(df.iloc[0]["case_id"])

    def fake_predict_suitability(pipe, rows):
        return pd.Series([0.2] + [0.0] * (len(rows) - 1), index=rows.index)

    monkeypatch.setattr("seedrec.recommend.predict_suitability", fake_predict_suitability)
    recs = rank_recommendations(object(), df, case_id, top_k=5)
    assert not recs.empty
    assert (recs["recommendation_score"] > 0).all()
    assert recs.iloc[0]["recommendation_score"] == pytest.approx(0.2)


def test_recommendation_table_columns_are_simple_by_default():
    assert recommendation_table_columns() == ["rank", "variety_name"]


def test_recommendation_table_columns_can_show_scoring():
    assert recommendation_table_columns(show_scoring=True) == [
        "rank",
        "variety_name",
        "model_probability",
        "availability_penalty",
        "recommendation_score",
        "available",
    ]
