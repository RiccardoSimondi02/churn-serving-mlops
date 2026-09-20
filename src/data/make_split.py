from pathlib import Path
from urllib.error import URLError

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.columns import DATA_URL


def segment_churn_rate(segment_df):
    return len(segment_df[segment_df["Churn"] == "Yes"]) / len(segment_df) * 100

EXCLUDED = 60
ROOT = Path(__file__).resolve().parents[2]

try:
    df = pd.read_csv(DATA_URL)
except (URLError, OSError) as e:
    raise RuntimeError(f"could not download the dataset from {DATA_URL}: {e}") from e



segment_excluded = df[df["tenure"] >= EXCLUDED]
population = df[df["tenure"] < EXCLUDED]

population_remaining, eval_frozen = train_test_split(population, stratify= population["Churn"], random_state= 42, test_size = 0.125)
population_remaining, test = train_test_split(population_remaining, stratify= population_remaining["Churn"], random_state= 42, test_size = 0.125 / (1 - 0.125))
train, val = train_test_split(population_remaining, stratify= population_remaining["Churn"], random_state= 42, test_size = 0.15 / (1 - 0.25))

manifest_df = pd.concat([
    train[["customerID"]].assign(split="train"),
    val[["customerID"]].assign(split="val"),
    test[["customerID"]].assign(split="test"),
    eval_frozen[["customerID"]].assign(split="eval_frozen"),
    segment_excluded[["customerID"]].assign(split="excluded")
], ignore_index=True)

manifest_df.to_csv(ROOT / "data" / "processed" / "split_manifest.csv", index=False)

splits = {
    "train": train,
    "val": val,
    "test": test,
    "eval_frozen": eval_frozen,
}

for split_df in splits.items():
    print("churn rate: " + str(round(segment_churn_rate(split_df), 2)) + "  len: " + str(len(split_df)))

