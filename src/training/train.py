
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics

from src.inference.pipeline import build_pipeline
from src.features.columns import ALL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

ESTIMATORS = {
    "logreg": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "logreg_balanced": lambda seed: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed
    ),
    "gb": lambda seed: GradientBoostingClassifier(random_state=seed),
}

def predict_scores(pipeline, X):
    proba = pipeline.predict_proba(X)
    classes_ = pipeline.named_steps["model"].classes_
    positive_idx = list(classes_).index(1)
    y_score = proba[:, positive_idx]
    return y_score



def load_dataset(): 
    df = pd.read_csv(ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    manifest_df = pd.read_csv(ROOT / "data" / "processed" / "split_manifest.csv")

    raw_len = len(df)
    merged_df = pd.merge(df, manifest_df, on="customerID", validate="one_to_one")
    assert len(merged_df) == raw_len, "error during the merge"


    merged_df["Churn"] = merged_df["Churn"].map({"Yes": 1, "No": 0})
    assert merged_df["Churn"].isna().sum() == 0, "unexpected value inside the target"

    return merged_df

def evaluate(y_score, y_true):
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score)
    return {"pr_auc": pr_auc,"prevalence": y_true.mean(),"n_sample": len(y_true)}
    

def find_threshold(y_score, y_true, min_precision):
    best = 0
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)

    above_threshold = precision[:-1] >= min_precision
    if not np.any(above_threshold):
        raise ValueError(
            f"no threshold reaches min_precision={min_precision} "
        )
    best = np.argmax(above_threshold)

    return precision[best], recall[best], thresholds[best]


def run_training(estimator, scale_numeric, min_precision):
    df = load_dataset()
    train_df = df[df["split"] == "train"]
    X_train = train_df[ALL_FEATURES]
    y_train = train_df["Churn"]
    pipeline = build_pipeline(estimator, scale_numeric)
    pipeline.fit(X_train, y_train)

    val_df = df[df["split"] == "val"]
    X_val = val_df[ALL_FEATURES]
    y_val = val_df["Churn"]
    y_score = predict_scores(pipeline, X_val)
    metrics = evaluate(y_score, y_val)
    precision, recall, threshold = find_threshold(y_score, y_val, min_precision)
    params = estimator.get_params()
    random_state = params.get("random_state")

    log_param = {"estimator": type(estimator).__name__, "scale_numeric": scale_numeric, "min_precision": min_precision,  "random_state": random_state}
    log_metric =  {"pr_auc": metrics["pr_auc"],"prevalence": metrics["prevalence"],"n_sample": metrics["n_sample"], "precision": precision, "recall": recall, "threshold": threshold}
    return log_param, log_metric, pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a churn model and evaluate it on the validation split."
    )
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS), default="logreg")
    parser.add_argument(
        "--scale-numeric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardise numeric features: needed by logistic regression, useless for trees.",
    )
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Base name for the saved artifacts. Defaults to <estimator>_<scaled|raw>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.estimator}_{'scaled' if args.scale_numeric else 'raw'}"

    estimator = ESTIMATORS[args.estimator](args.seed)
    log_param, log_metric, pipeline = run_training(
        estimator, args.scale_numeric, args.min_precision
    )

    print(f"run: {run_name}")
    for name, value in log_param.items():
        print(f"  param  {name}: {value}")
    for name, value in log_metric.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"  metric {name}: {formatted}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{run_name}.joblib")

    # The threshold travels with the model: a .joblib alone does not say at which
    # cut-off it is meant to be used.
    with open(MODELS_DIR / f"{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"run_name": run_name, "params": log_param, "metrics": log_metric},
            f,
            indent=2,
            default=float,
        )

    print(f"saved: {MODELS_DIR / run_name}.joblib and .json")

    



if __name__ == "__main__":
    main()