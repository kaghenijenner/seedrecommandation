from seedrec.data import build_modelling_dataset, load_raw_data
from seedrec.model import train_best_model
from seedrec.recommend import rank_recommendations


def test_modelling_dataset_has_expected_rows():
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    assert len(df) == len(farmers) * len(varieties)
    assert {"suitability_score", "suitability_class", "data_confidence"}.issubset(df.columns)


def test_recommendation_output_has_required_fields():
    varieties, locations, farmers, availability = load_raw_data()
    df = build_modelling_dataset(varieties, locations, farmers, availability)
    result = train_best_model(df)
    recs = rank_recommendations(result.pipeline, df, "c001", top_k=3)
    assert len(recs) == 3
    assert recs["rank"].tolist() == [1, 2, 3]
    assert recs["explanation"].str.len().min() > 20
    assert {"recommendation_score", "available", "data_confidence"}.issubset(recs.columns)

