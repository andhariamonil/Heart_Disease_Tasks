# Heart Disease Prediction Using Machine Learning

## Task 52: Model Runtime, Memory, and Performance Benchmark

### Objective

Measure fitting and inference time with `perf_counter`, estimate additional peak Python-traced memory with `tracemalloc`, and compare these costs with held-out accuracy/F1 for scratch model families. Use the results to discuss practical model complexity and deployment trade-offs.

Memory numbers are process- and allocator-dependent approximations, and timings are hardware-specific. Benchmark again on the target deployment environment.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_52_Model_Runtime_Memory_Benchmark/app.py
```

Run all cells in `Task_52_Model_Runtime_Memory_Benchmark/Task_52_Model_Runtime_Memory_Benchmark.ipynb` for the standalone notebook.