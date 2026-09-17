import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

from src.features.columns import NUMERIC_FEATURES, CATEGORICAL_FEATURES, NULLABLE_NUMERIC_FEATURE

def clean(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    return df


def predict_scores(pipeline, X):
    proba = pipeline.predict_proba(X)
    classes_ = pipeline.named_steps["model"].classes_
    positive_idx = list(classes_).index(1)
    y_score = proba[:, positive_idx]
    return y_score

def build_pipeline(estimator, scale_numeric):
    total_charges_pipeline = Pipeline(steps=[
    ("cleaner", FunctionTransformer(clean, feature_names_out="one-to-one")),
    ("scale", StandardScaler() if scale_numeric else "passthrough")
    ])

    preprocess = ColumnTransformer(transformers=[
        ("numeric", StandardScaler() if scale_numeric else "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("total_charges", total_charges_pipeline, [NULLABLE_NUMERIC_FEATURE]),
    ])

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", estimator),
    ])
    return pipeline




