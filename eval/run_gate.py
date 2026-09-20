
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.features.columns import ALL_FEATURES
from src.inference.pipeline import predict_scores
from src.training.train import ESTIMATORS, evaluate, load_dataset, run_training

# Measured 2026-09-20 on eval_frozen: bootstrap 95% band (1000 resamples, seed=42,
# see bootstrap_pr_auc below) is [0.6453, 0.7754], median 0.7172. Reproduce it with
# `python -m eval.run_gate --bootstrap`. Set just under the band's lower edge, so
# the gate fires on a real drift in pr_auc, not on ordinary sampling noise.
GATE_MIN_PR_AUC = 0.64


def bootstrap_pr_auc(y_score, y_true, n_iterations=1000, seed=42):
    y_true = np.asarray(y_true)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        scores[i] = average_precision_score(y_true[idx], y_score[idx])

    p2_5, p50, p97_5 = np.percentile(scores, [2.5, 50, 97.5])

    print(f"bootstrap pr_auc (eval_frozen, {n_iterations} iterations)")
    print(f"  2.5th percentile:  {p2_5:.4f}")
    print(f"  50th percentile:   {p50:.4f}")
    print(f"  97.5th percentile: {p97_5:.4f}")

    return p2_5, p50, p97_5





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap",action="store_true")
    args = parser.parse_args()

    SCALE_NUMERIC = False
    MIN_PRECISION = 0.60

    log_param, log_metric, pipeline, sample = run_training(ESTIMATORS["gb"](42), SCALE_NUMERIC, MIN_PRECISION)

    df = load_dataset()

    df_eval_frozen = df[df["split"] == "eval_frozen"]
    X_eval_frozen = df_eval_frozen[ALL_FEATURES]
    y_eval_frozen = df_eval_frozen["Churn"]

    y_score = predict_scores(pipeline, X_eval_frozen)

    if args.bootstrap:
        bootstrap_pr_auc(y_score, y_eval_frozen)
        return

    metrics = evaluate(y_score, y_eval_frozen)

    predictions = (y_score >= log_metric["threshold"]).astype(int)

    precision = precision_score(y_eval_frozen, predictions)
    recall = recall_score(y_eval_frozen, predictions)

    print("eval_frozen gate")
    print(f"  n_sample:   {metrics['n_sample']}")
    print(f"  prevalence: {metrics['prevalence']:.4f}")
    print(f"  pr_auc:     {metrics['pr_auc']:.4f}")
    print(f"  precision:  {precision:.4f}  (threshold={log_metric['threshold']:.4f})")
    print(f"  recall:     {recall:.4f}")

    diff = metrics["pr_auc"] - GATE_MIN_PR_AUC
    if diff >= 0:
        print(f"  gate:       PASS  pr_auc {metrics['pr_auc']:.4f} >= min {GATE_MIN_PR_AUC:.4f} (+{diff:.4f})")
    else:
        print(f"  gate:       FAIL  pr_auc {metrics['pr_auc']:.4f} < min {GATE_MIN_PR_AUC:.4f} ({diff:.4f})")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

