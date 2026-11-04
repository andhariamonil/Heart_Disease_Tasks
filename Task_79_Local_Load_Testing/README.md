# Heart Disease Prediction Using Machine Learning

## Task 79: Bounded Local API Load Testing

### Objective

Measure response status, throughput, median latency, and p95 latency for Task 78's local inference API. The harness is fixed to `127.0.0.1`, supports up to 1,000 total requests, and caps concurrency at 32 workers to limit accidental service impact.

This is a local smoke/benchmark test, not a test of thousands of simultaneous connections. For larger load tests, use an authorized cloud load-testing service against a dedicated staging environment with agreed quotas, monitoring, and stop conditions.

### Run

Start the Task 78 API locally, then set a local test bearer token and run:

```bash
source .venv/bin/activate
export API_BEARER_TOKEN='a-local-test-token-of-at-least-32-chars'
streamlit run Task_79_Local_Load_Testing/app.py
```

Or run the CLI benchmark:

```bash
python Task_79_Local_Load_Testing/benchmark.py --requests 200 --workers 16
```

No notebook is included because this is an operational HTTP load-test utility; the runnable app and CLI script are the useful artifacts.