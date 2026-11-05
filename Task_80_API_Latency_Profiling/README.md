# Heart Disease Prediction Using Machine Learning

## Task 80: API Latency and Inference Profiling

### Objective

Profile validation, scaling, inference, explanation, serialization, model-artifact loading, peak Python-traced memory, and batch inference. Compare FP32 with opt-in FP16 and symmetric INT8 inference and quantify output probability error.

The Task 78 API loads the versioned artifact once per worker, exposes a `Server-Timing` header, and supports `MODEL_QUANTIZATION=none|fp16|int8`. Prediction payloads are deliberately not cached because they contain sensitive health features. FP16/INT8 should only be enabled after validation on representative data; timing is hardware-dependent.

### Run

From the `Heart_Disease_Tasks` directory:

```bash
source .venv/bin/activate
streamlit run Task_80_API_Latency_Profiling/app.py
```

For deployment, set `MODEL_QUANTIZATION` in the ECS task definition environment and compare error/latency on the target hardware before rollout.