from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx


def pretty(title: str, payload: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(payload, indent=2, default=str))


def require_ok(response: httpx.Response, step: str) -> dict:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"\n[{step}] failed with status {response.status_code}")
        print(response.text)
        raise SystemExit(1) from exc

    try:
        return response.json()
    except Exception as exc:
        print(f"\n[{step}] did not return JSON")
        print(response.text)
        raise SystemExit(1) from exc


def poll_job(client: httpx.Client, base_url: str, job_id: str, timeout_s: int, interval_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_payload = None

    while time.time() < deadline:
        response = client.get(f"{base_url}/jobs/{job_id}")
        last_payload = require_ok(response, "poll_job")
        status = last_payload.get("status")
        print(f"[poll] job_id={job_id} status={status}")
        if status in {"completed", "failed"}:
            return last_payload
        time.sleep(interval_s)

    print("\nTimed out while waiting for async job completion.")
    if last_payload is not None:
        pretty("Last job payload", last_payload)
    raise SystemExit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Live smoke test for Alpha Analyst pipeline")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to test")
    parser.add_argument("--company-name", default=None, help="Optional company name")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip /ingest step")
    parser.add_argument("--timeout", type=int, default=180, help="Async polling timeout in seconds")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()

    analysis_payload = {
        "ticker": args.ticker,
        "company_name": args.company_name,
        "include_filing_analysis": True,
        "include_news_sentiment": True,
        "max_news_articles": 5,
    }

    ingestion_payload = {
        "ticker": args.ticker,
        "filing_type": "10-K",
        "force_refresh": False,
    }

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        root = require_ok(client.get(f"{args.base_url}/"), "root")
        pretty("Root", root)

        health = require_ok(client.get(f"{args.base_url}/health"), "health")
        pretty("Health", health)

        if not args.skip_ingest:
            ingest = require_ok(
                client.post(f"{args.base_url}/ingest", json=ingestion_payload),
                "ingest",
            )
            pretty("Ingest", ingest)

        sync_analysis = require_ok(
            client.post(f"{args.base_url}/analyze", json=analysis_payload),
            "analyze_sync",
        )
        pretty("Synchronous analysis", sync_analysis)

        async_analysis = require_ok(
            client.post(f"{args.base_url}/analyze/async", json=analysis_payload),
            "analyze_async",
        )
        pretty("Async analysis accepted", async_analysis)

        job_id = async_analysis["job_id"]
        final_job = poll_job(
            client=client,
            base_url=args.base_url,
            job_id=job_id,
            timeout_s=args.timeout,
            interval_s=args.interval,
        )
        pretty("Async final job state", final_job)

        stats = require_ok(client.get(f"{args.base_url}/stats"), "stats")
        pretty("Stats", stats)

        metrics = client.get(f"{args.base_url}/metrics")
        if metrics.status_code != 200:
            print("\n[metrics] endpoint failed")
            print(metrics.text)
            raise SystemExit(1)
        print("\n" + "=" * 80)
        print("Metrics")
        print("=" * 80)
        print("Metrics endpoint returned data successfully.")

    print("\nLive smoke test finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
