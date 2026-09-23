import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import text

from src.api.storage import build_engine
from src.features.columns import ALL_FEATURES
from src.training.train import load_dataset

RAMP_START_DAY = 20
TARGET = 0.7 # percentage of shifted over the day's total at the end
TIMES_SPAN = 60

def tables_are_empty(engine):
    with engine.connect() as conn:
        predictions_count = conn.execute(text("SELECT count(*) FROM predictions")).scalar()
        request_log_count = conn.execute(text("SELECT count(*) FROM request_log")).scalar()
    return predictions_count == 0 and request_log_count == 0

def reset_tables(engine):
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE predictions, request_log"))
        conn.commit()

def build_payload(row, event_time):
    body = json.loads(row[ALL_FEATURES].to_json())
    body["customer_id"] = str(row["customerID"])
    raw_total = body["TotalCharges"]
    body["TotalCharges"] = None if raw_total is None or str(raw_total).strip() == "" else float(raw_total)
    body["event_time"] = event_time.isoformat()
    return body

async def send_request(client, semaphore, payload):
    async with semaphore:
        try:
            response = await client.post("/predict", json=payload)
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            print(f"request failed: {e}")
            return False

async def run(seed, base_url, concurrency):
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:

        tasks = []
        semaphore = asyncio.Semaphore(concurrency)
        rng = np.random.default_rng(seed)
        simulated_date = datetime.now(UTC) - timedelta(days=TIMES_SPAN)

        dataset = load_dataset()
        pool_normal = dataset[dataset["split"] == "test"]
        pool_shifted = dataset[dataset["split"] == "excluded"]

        for d in range(TIMES_SPAN):
            if d >= RAMP_START_DAY:
                frac = (d - RAMP_START_DAY) / ((TIMES_SPAN - 1) - RAMP_START_DAY) * TARGET
            else:
                frac = 0.0

            base = int(rng.integers(50, 80))
            if simulated_date.weekday() >= 5:  # saturday/sunday
                base = int(base * 0.5)

            for _ in range(base):
                pool= pool_shifted if rng.random() < frac else pool_normal
                row = pool.iloc[rng.integers(0, len(pool))]

                seconds_into_day = int(rng.integers(0, 86400))
                event_time = simulated_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC) + timedelta(seconds=seconds_into_day)
                payload = build_payload(row, event_time)
                tasks.append(asyncio.create_task(send_request(client, semaphore, payload)))

            simulated_date += timedelta(days=1)
        results = await asyncio.gather(*tasks)
        sent = len(results)
        succeded = sum(results)
        print(f"{sent} sent, {succeded} succeded")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    engine = build_engine()

    if args.reset:
        reset_tables(engine)
    elif not tables_are_empty(engine):
        print("predictions/request_log not empty. Rerun with --reset to clear it.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(seed=args.seed, base_url=args.base_url, concurrency=args.concurrency))

if __name__ == "__main__":
    main()

