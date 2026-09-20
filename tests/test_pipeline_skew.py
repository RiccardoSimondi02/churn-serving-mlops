import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.columns import ALL_FEATURES
from src.inference.pipeline import build_pipeline
from src.training.train import load_dataset, predict_scores

TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def fitted_pipeline_and_val():
    df = load_dataset()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42), scale_numeric=True
    )
    pipeline.fit(train_df[ALL_FEATURES], train_df["Churn"])
    return pipeline, val_df[ALL_FEATURES]


def test_single_row_matches_batch(fitted_pipeline_and_val):
    pipeline, X_val = fitted_pipeline_and_val
    batch_scores = predict_scores(pipeline, X_val)

    for i in range(10):
        single_score = predict_scores(pipeline, X_val.iloc[[i]])[0]
        assert np.isclose(single_score, batch_scores[i], rtol=0, atol=TOLERANCE)


def test_record_built_from_a_dict_matches_batch(fitted_pipeline_and_val):
    pipeline, X_val = fitted_pipeline_and_val
    batch_scores = predict_scores(pipeline, X_val)

    record = X_val.iloc[0].to_dict()
    as_frame = pd.DataFrame([record], columns=ALL_FEATURES)

    score = predict_scores(pipeline, as_frame)[0]
    assert np.isclose(score, batch_scores[0], rtol=0, atol=TOLERANCE)


def test_column_order_does_not_change_the_score(fitted_pipeline_and_val):
    pipeline, X_val = fitted_pipeline_and_val
    batch_scores = predict_scores(pipeline, X_val)

    record = X_val.iloc[0].to_dict()
    shuffled_columns = list(reversed(ALL_FEATURES))
    as_frame = pd.DataFrame([record], columns=shuffled_columns)

    score = predict_scores(pipeline, as_frame)[0]
    assert np.isclose(score, batch_scores[0], rtol=0, atol=TOLERANCE)
