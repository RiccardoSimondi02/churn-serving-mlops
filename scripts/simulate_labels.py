import argparse
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from src.api.storage import build_engine
from src.training.train import load_dataset

OBSERVATION_DAYS = 90

def tables_are_empty(engine):
    with engine.connect() as conn:
        labels_count = conn.execute(text("SELECT count(*) FROM labels")).scalar()
    return labels_count == 0

def reset_tables(engine):
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE labels"))
        conn.commit()

def load_predictions(engine):
    query = "SELECT request_id, customer_id, event_time FROM predictions ORDER BY request_id"
    return pd.read_sql(text(query), engine)

def attach_truth(predictions, dataset):
    """Join each prediction to its ground-truth Churn label from the dataset.

    Left-joins on customer_id.
    """
    truth = dataset[["customerID", "Churn"]].rename(
        columns={"customerID": "customer_id", "Churn": "label"})
    merged = predictions.merge(truth, on="customer_id", how="left", validate="many_to_one")
    missing = merged["label"].isna().sum()
    assert missing == 0, f"{missing} predictions have no customer in the dataset"
    return merged

def assign_available_at(df, rng):
    """Simulate when each label becomes known, modelling label-arrival latency.

    Churners (label == 1) resolve on a random day within the observation window,
    since churn can be observed as soon as it happens; non-churners are only
    confirmed once the full OBSERVATION_DAYS window has elapsed with no churn.
    available_at = event_time + that delay.
    """
    churn_delay = rng.integers(1, OBSERVATION_DAYS + 1, size=len(df))
    delay_days = np.where(df["label"] == 1, churn_delay, OBSERVATION_DAYS)
    df = df.copy()
    df["available_at"] = df["event_time"] + pd.to_timedelta(delay_days, unit="D")
    return df

def write_labels(engine, labels):
    """Insert the generated labels."""
    with engine.begin() as conn:
        labels[["request_id", "label", "available_at"]].to_sql(
            "labels", conn, if_exists="append", index=False)
        n_labels = conn.execute(text("SELECT count(*) FROM labels")).scalar()
        n_predictions = conn.execute(text("SELECT count(*) FROM predictions")).scalar()
    assert n_labels == n_predictions, f"{n_labels} labels for {n_predictions} predictions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    load_dotenv()
    engine = build_engine()

    if args.reset:
            reset_tables(engine)
    elif not tables_are_empty(engine):
        print("labels not empty. Rerun with --reset to clear it.", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset()
    predictions = load_predictions(engine)
    df_merged = attach_truth(predictions, dataset)
    df_with_labels = assign_available_at(df_merged, rng)
    write_labels(engine, df_with_labels)


if __name__ == "__main__":
    main()

