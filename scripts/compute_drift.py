

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from src.api.storage import build_engine
from src.features.columns import (
    ALL_FEATURES,
    CATEGORICAL_VALUES,
    NULLABLE_NUMERIC_FEATURE,
)
from src.inference.model_loader import return_model_info
from src.inference.pipeline import predict_scores
from src.training.train import load_dataset

N_BINS = 10
WINDOWS = [1, 7] # days per window, each window labelled with its last day
NUMERIC_MONITORED = ["tenure", "MonthlyCharges", NULLABLE_NUMERIC_FEATURE, "score"]
CATEGORICAL_MONITORED = {**CATEGORICAL_VALUES, "SeniorCitizen": [0, 1]}

def fit_numeric_bins(ref_values, n_bins=N_BINS):
    _, edges = pd.qcut(ref_values.dropna(), q=n_bins, retbins = True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges

def numeric_shares(values, edges):
    binned = pd.cut(values, bins=edges)
    shares = binned.value_counts(normalize=False, sort=False)
    nan_counter = pd.isna(values).sum()
    shares["missing"] = nan_counter
    shares = shares/len(values)
    return shares

def categorical_shares(values, categories):
    return values.value_counts(normalize=True).reindex(categories, fill_value=0)

def psi(p_ref, q_cur, n_ref, n_cur):
    # empty bins count as half an observation of their own sample: eps = 1/(2n)
    p = np.clip(p_ref.to_numpy(dtype=float), 1 / (2 * n_ref), None)
    q = np.clip(q_cur.to_numpy(dtype=float), 1 / (2 * n_cur), None)
    return float(np.sum((q - p) * np.log(q / p)))

def load_reference(model):
    df = load_dataset()
    ref = df[df["split"] == "val"].copy()

    ref["score"] = predict_scores(model, ref[ALL_FEATURES])
    ref[NULLABLE_NUMERIC_FEATURE] = pd.to_numeric(ref[NULLABLE_NUMERIC_FEATURE], errors="coerce")
    return ref

def load_traffic(engine):
    columns = ", ".join(f'"{c}"' for c in ALL_FEATURES)
    query = f"""
        SELECT event_time::date AS day, probability AS score, {columns}
        FROM predictions
    """
    return pd.read_sql(text(query), engine)

def compute_metrics(traffic, window_days, n_ref, numeric_bins, p_numerical, p_categorical):
    days = sorted(traffic["day"].unique())
    # skip incomplete windows at the start of the traffic
    first_full = days[0] + pd.Timedelta(days=window_days - 1)

    result = []
    for end_day in days:
        if end_day < first_full:
            continue
        start_day = end_day - pd.Timedelta(days=window_days - 1)
        window_df = traffic[(traffic["day"] >= start_day) & (traffic["day"] <= end_day)]
        n_cur = len(window_df)

        for feature in NUMERIC_MONITORED:
            q_numerical = numeric_shares(window_df[feature], numeric_bins[feature])
            psi_num = psi(p_numerical[feature], q_numerical, n_ref, n_cur)
            result.append({"day": end_day, "feature": feature, "window_days": window_days,
                           "psi": psi_num, "n_rows": n_cur})

        for feature in CATEGORICAL_MONITORED:
            categories = CATEGORICAL_MONITORED[feature]
            q_categorical = categorical_shares(window_df[feature], categories)
            psi_cat = psi(p_categorical[feature], q_categorical, n_ref, n_cur)
            result.append({"day": end_day, "feature": feature, "window_days": window_days,
                           "psi": psi_cat, "n_rows": n_cur})
    return result

def write_metrics(engine, metrics):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE drift_metrics"))
        metrics.to_sql("drift_metrics", conn, if_exists="append", index=False)

def main():
    load_dotenv()
    engine = build_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT model_version, count(*) FROM predictions GROUP BY model_version"
        )).all()

    if len(rows) != 1:
        raise SystemExit(f"expected one model version in predictions, found: {rows}")

    traffic_version = rows[0][0]
    model, version, _ = return_model_info()

    if str(traffic_version) != str(version):
        raise SystemExit(
            f"traffic was scored by model v{traffic_version}, champion is v{version}"
        )

    traffic = load_traffic(engine)
    traffic["day"] = pd.to_datetime(traffic["day"])
    ref = load_reference(model)
    numeric_bins = {f: fit_numeric_bins(ref[f]) for f in NUMERIC_MONITORED}
    p_numerical = {}
    p_categorical = {}

    for feature in NUMERIC_MONITORED:
        p_numerical[feature] = numeric_shares(ref[feature],numeric_bins[feature])

    for feature in CATEGORICAL_MONITORED:
        categories = CATEGORICAL_MONITORED[feature]
        p_categorical[feature] = categorical_shares(ref[feature], categories)


    result = []
    for window_days in WINDOWS:
        result += compute_metrics(traffic, window_days, len(ref),
                                  numeric_bins, p_numerical, p_categorical)

    metrics = pd.DataFrame(result)
    write_metrics(engine, metrics)
    print(metrics.groupby("window_days").agg(rows=("psi", "size"), days=("day", "nunique")))





if __name__ == "__main__":
    main()