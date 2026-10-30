# Heart Disease Prediction Using Machine Learning

## Task 69: Custom Prediction Caching & Concurrent Request Optimization

### Objective

Display machine learning model predictions dynamically by designing and implementing a thread-safe, custom Least Recently Used (LRU) in-memory caching engine from scratch, ensuring smooth and instantaneous UI updates without unnecessary page reloading, and efficiently serving concurrent prediction requests under multi-threaded load.

Key Architectural Capabilities:
1. **Thread-Safe Custom LRU Cache Engine**:
   - Built from first principles using an internal `OrderedDict` guarded by `threading.RLock`.
   - Feature vector fingerprinting: implements deterministic rounding and 64-bit hashing of patient attributes, enabling $O(1)$ retrieval (< 0.003 ms) for repeated or near-identical clinical profiles.
   - Comprehensive telemetry tracking: hits, misses, hit ratio %, eviction events, and memory footprint.
2. **Smooth, Non-Reloading UI Updates**:
   - Leverages persistent `st.session_state` and reactive container swapping (`st.empty()`) in Streamlit.
   - Prevents full application redraws or state loss when clinicians adjust continuous parameter sliders.
3. **High-Concurrency Multi-Threaded Stress Testing**:
   - Evaluates system performance under simulated concurrent user traffic using Python's `concurrent.futures.ThreadPoolExecutor` (up to 100 concurrent worker threads).
   - Measures end-to-end throughput (requests per second), latency percentiles ($p50, p95, p99$), cache hit rate dynamics, and validates thread safety under race conditions.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_69_Custom_Prediction_Caching_Concurrency/app.py
```

Run all cells in `Task_69_Custom_Prediction_Caching_Concurrency/Task_69_Custom_Prediction_Caching_Concurrency.ipynb` for the standalone notebook.

