import argparse
from pathlib import Path
import mlflow
import numpy as np

from src.features.columns import ALL_FEATURES, ID_COLUMN
from src.inference.pipeline import predict_scores


MODEL_NAME = "churn-classifier"
ALIAS = "champion"
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("churn-model-selection")

    client = mlflow.MlflowClient()

    run = client.get_run(args.run_id)
    model_uri = f"models:/{run.data.tags['logged_model_id']}"

    existing = client.search_model_versions(f"name='{MODEL_NAME}'")
    reused = next((mv for mv in existing if mv.source == model_uri), None)

    if reused is not None:
        version = reused.version
        print(f"version {version} already registered, promoting it")
    else:
        result = mlflow.register_model(model_uri, MODEL_NAME)
        version = result.version

        threshold = run.data.metrics["threshold"]
        client.set_model_version_tag(MODEL_NAME, version, "threshold", str(threshold))

        # Close the loop: the artifact just registered must reproduce, row by row, the
        # scores the run produced.
        sample = mlflow.load_table("verification_sample.json", run_ids=[args.run_id])

        candidate = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
        scores_now = predict_scores(candidate, sample[ALL_FEATURES])
        expected = sample["expected_score"].to_numpy()

        mismatched = ~np.isclose(scores_now, expected, rtol=0, atol=1e-9)
        if mismatched.any():
            offenders = sample.loc[mismatched, ID_COLUMN].tolist()[:5]
            raise SystemExit(
                f"aborted: version {version} does not reproduce the scores logged by run "
                f"{args.run_id} on {mismatched.sum()} of {len(sample)} rows "
                f"(first offenders: {offenders})"
            )

        print(f"verified: version {version} reproduces {len(sample)} logged scores")

    if args.promote:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)

    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
    except mlflow.exceptions.MlflowException:
        champion_version = None

    print(
        f"version {version} registered, promoted={args.promote}, "
        f"champion now points to version {champion_version}"
    )


if __name__ == "__main__":
    main()