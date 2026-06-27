from datetime import date

import pandas as pd
import pytest

from seedrec.data import build_modelling_dataset, load_raw_data
from seedrec.evaluate import ranking_metrics_at_ks
from seedrec.model import predict_suitability, train_best_model
from seedrec.recommend import rank_recommendations, recommendation_table_columns
from seedrec.seasons import current_season_phase, latest_season_code_for_phase
from seedrec.validation import grouped_cv_metrics


@pytest.fixture(scope="module")
def raw():
    return load_raw_data()


@pytest.fixture(scope="module")
def full_df(raw):
    varieties, locations, farmers, availability = raw
    return build_modelling_dataset(varieties, locations, farmers, availability)


@pytest.fixture(scope="module")
def small_df(full_df):
    cases = full_df["case_id"].drop_duplicates().sample(60, random_state=0)
    return full_df[full_df["case_id"].isin(cases)].reset_index(drop=True)


@pytest.fixture(scope="module")
def trained(small_df):
    return train_best_model(small_df), small_df


def test_modelling_dataset_has_expected_rows(raw, full_df):
    varieties, locations, farmers, availability = raw
    expected_rows = sum(
        len(farmers[farmers["crop"].eq(crop)]) * len(varieties[varieties["crop"].eq(crop)])
        for crop in farmers["crop"].unique()
    )
    assert len(full_df) == expected_rows
    assert {"suitability_score", "suitability_class", "data_confidence", "trait_source"}.issubset(full_df.columns)


def test_catalogue_was_expanded(raw):
    varieties = raw[0]
    assert len(varieties) > 8
    assert (varieties.groupby("crop")["variety_name"].count() >= 5).all()
    assert {"trait_source", "release_year", "licensed_company_count"}.issubset(varieties.columns)
    assert varieties["trait_source"].isin({"measured", "imputed"}).all()


def test_many_candidates_per_case(full_df):
    # The original catalogue gave only 2 candidates/case, which made ranking meaningless.
    assert full_df.groupby("case_id").size().min() > 2


def test_recommendation_output_has_required_fields(trained):
    result, sdf = trained
    recs = rank_recommendations(result.pipeline, sdf, str(sdf.iloc[0]["case_id"]), top_k=3)
    assert len(recs) == 3
    assert recs["rank"].tolist() == [1, 2, 3]
    assert "variety_id" in recs.columns
    assert recs["explanation"].str.len().min() > 20
    assert recs["model_probability"].between(0, 1).all()
    assert {"recommendation_score", "available", "data_confidence"}.issubset(recs.columns)


def test_ranking_metrics_are_not_degenerate(trained):
    result, sdf = trained
    case_ids = sdf["case_id"].unique()[:30]
    all_recs = pd.concat(
        [rank_recommendations(result.pipeline, sdf, cid, top_k=6) for cid in case_ids],
        ignore_index=True,
    )
    metrics = ranking_metrics_at_ks(all_recs, ks=(3, 5))
    assert {"top_3_accuracy", "ndcg_at_3", "ndcg_at_5", "mrr"}.issubset(metrics)
    # "any hit in top-k" must be >= "fraction of hits in top-k"; both in [0, 1].
    assert metrics["top_3_accuracy"] >= metrics["precision_at_3"] - 1e-9
    assert 0.0 < metrics["ndcg_at_3"] <= 1.0
    assert 0.0 < metrics["mrr"] <= 1.0


def test_grouped_validation_runs(small_df):
    out = grouped_cv_metrics(small_df, "district", model_name="xgboost")
    assert out["group_col"] == "district"
    assert ("accuracy" in out) or ("note" in out)


def test_current_season_phase_defaults_to_first_season_in_june():
    assert current_season_phase(date(2026, 6, 2)) == "First season"


def test_latest_season_code_for_phase_chooses_most_recent_year():
    district_df = pd.DataFrame({"season": ["2008_First", "2009_First", "2008_Second", "2009_Second"]})
    assert latest_season_code_for_phase(district_df, "First season") == "2009_First"


def test_recommendations_drop_zero_scores(monkeypatch, full_df):
    case_id = str(full_df.iloc[0]["case_id"])

    def fake_predict_suitability(pipe, rows):
        return pd.Series([0.2] + [0.0] * (len(rows) - 1), index=rows.index)

    monkeypatch.setattr("seedrec.recommend.predict_suitability", fake_predict_suitability)
    recs = rank_recommendations(object(), full_df, case_id, top_k=5)
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
