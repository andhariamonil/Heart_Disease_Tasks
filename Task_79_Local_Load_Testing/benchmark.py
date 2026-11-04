"""Bounded localhost-only benchmark for Task 78's inference API."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

LOCAL_API = 'http://127.0.0.1:8000'
MAX_WORKERS = 32
MAX_REQUESTS = 1000
FEATURES = {'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 130, 'chol': 240, 'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2}


def send_request(token: str) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        response = requests.post(f'{LOCAL_API}/predict', headers={'Authorization': f'Bearer {token}'}, json={'features': FEATURES}, timeout=10)
        return response.status_code, time.perf_counter() - started
    except requests.RequestException:
        return 0, time.perf_counter() - started


def run_benchmark(request_count: int = 200, worker_count: int = 16, token: str | None = None) -> dict[str, float | int]:
    if not 1 <= request_count <= MAX_REQUESTS:
        raise ValueError(f'request_count must be between 1 and {MAX_REQUESTS}')
    if not 1 <= worker_count <= MAX_WORKERS:
        raise ValueError(f'worker_count must be between 1 and {MAX_WORKERS}')
    token = token or os.environ.get('API_BEARER_TOKEN', '')
    if len(token) < 32:
        raise ValueError('Provide a local test bearer token with at least 32 characters')
    requests.get(f'{LOCAL_API}/healthz', timeout=3).raise_for_status()
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(send_request, token) for _ in range(request_count)]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    latencies = [latency for _, latency in results]
    return {
        'requests': request_count,
        'workers': worker_count,
        'successes': sum(status == 200 for status, _ in results),
        'errors': sum(status != 200 for status, _ in results),
        'elapsed_seconds': elapsed,
        'requests_per_second': request_count / elapsed if elapsed else 0.0,
        'median_latency_ms': statistics.median(latencies) * 1000,
        'p95_latency_ms': sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--workers', type=int, default=16)
    arguments = parser.parse_args()
    print(run_benchmark(arguments.requests, arguments.workers))