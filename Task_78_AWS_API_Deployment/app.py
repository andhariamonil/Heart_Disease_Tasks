import hmac
import os
import sys
import time
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, FiniteFloat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, validate_input_row


MODEL_PATH = Path(os.environ.get('MODEL_ARTIFACT_PATH', '/app/model.joblib'))
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parents[1] / 'Task_47_Versioned_Model_Export' / 'artifacts' / 'heart_logistic_v1.0.0.joblib'
if not MODEL_PATH.is_file():
    raise RuntimeError(f'Model artifact not found: {MODEL_PATH}')

ARTIFACT = joblib.load(MODEL_PATH)
if ARTIFACT.get('schema_version') != 1 or ARTIFACT.get('model_version') != '1.0.0':
    raise RuntimeError('Unsupported model artifact version')
FEATURES = ARTIFACT['feature_names']
MEANS = np.asarray(ARTIFACT['preprocessing']['means'], dtype=float)
SCALES = np.asarray(ARTIFACT['preprocessing']['scales'], dtype=float)
MODEL = LogisticRegressionScratch(**ARTIFACT['parameters'])
MODEL.weights = np.asarray(ARTIFACT['state']['weights'], dtype=float)
MODEL.bias = float(ARTIFACT['state']['bias'])
MODEL_QUANTIZATION = os.environ.get('MODEL_QUANTIZATION', 'none').lower()
if MODEL_QUANTIZATION not in {'none', 'fp16', 'int8'}:
    raise RuntimeError("MODEL_QUANTIZATION must be 'none', 'fp16', or 'int8'")
MODEL_WEIGHTS = MODEL.weights.copy()
WEIGHT_SCALE = 1.0
INPUT_SCALE = 1.0 / 32.0
if MODEL_QUANTIZATION == 'fp16':
    MODEL_WEIGHTS = MODEL.weights.astype(np.float16)
elif MODEL_QUANTIZATION == 'int8':
    WEIGHT_SCALE = max(float(np.max(np.abs(MODEL.weights))) / 127.0, 1e-12)
    MODEL_WEIGHTS = np.clip(np.round(MODEL.weights / WEIGHT_SCALE), -127, 127).astype(np.int8)
MODEL_BIAS = np.float16(MODEL.bias) if MODEL_QUANTIZATION == 'fp16' else MODEL.bias
THRESHOLD = float(ARTIFACT.get('decision_threshold', 0.5))
CATEGORY_CODES = {'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2], 'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3, 4], 'thal': [0, 1, 2, 3]}
HARD_RANGES = {'age': (18, 100), 'trestbps': (60, 260), 'chol': (80, 700), 'thalach': (40, 250), 'oldpeak': (-5, 15)}

app = FastAPI(title='Heart Model Inference API', version=ARTIFACT['model_version'])
bearer = HTTPBearer(auto_error=False)


@app.middleware('http')
async def add_server_timing(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers['Server-Timing'] = f'api;dur={(time.perf_counter() - started) * 1000:.3f}'
    return response


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: dict[str, FiniteFloat]


def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
    expected = os.environ.get('API_BEARER_TOKEN', '')
    if len(expected) < 32:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='API authentication is not configured')
    if credentials is None or credentials.scheme.lower() != 'bearer' or not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='A valid bearer token is required', headers={'WWW-Authenticate': 'Bearer'})
    return True


@app.get('/healthz')
def health_check():
    return {'status': 'ok', 'model_version': ARTIFACT['model_version'], 'quantization': MODEL_QUANTIZATION}


@app.post('/predict')
def predict(request: PredictionRequest, _authenticated: Annotated[bool, Depends(authenticate)]):
    supplied = request.features
    missing = [feature for feature in FEATURES if feature not in supplied]
    extra = [feature for feature in supplied if feature not in FEATURES]
    if missing or extra:
        raise HTTPException(status_code=422, detail={'message': 'Feature names must exactly match the versioned model schema', 'missing': missing, 'extra': extra, 'expected_features': FEATURES})
    ordered = {feature: supplied[feature] for feature in FEATURES}
    parsed, issues, suggestions = validate_input_row(ordered, FEATURES, CATEGORY_CODES, HARD_RANGES, {})
    errors = [issue for issue in issues if issue['level'] == 'error']
    if errors:
        raise HTTPException(status_code=422, detail={'message': 'Feature validation failed', 'issues': errors, 'suggestions': suggestions})
    vector = np.array([[parsed[feature] for feature in FEATURES]], dtype=float)
    scaled = (vector - MEANS) / SCALES
    if MODEL_QUANTIZATION == 'fp16':
        logit = float(np.dot(scaled[0].astype(np.float16), MODEL_WEIGHTS).astype(np.float32) + MODEL_BIAS)
        probability = float(1 / (1 + np.exp(-np.clip(logit, -35, 35))))
    elif MODEL_QUANTIZATION == 'int8':
        quantized_input = np.clip(np.round(scaled[0] / INPUT_SCALE), -127, 127).astype(np.int8)
        integer_dot = int(quantized_input.astype(np.int32) @ MODEL_WEIGHTS.astype(np.int32))
        logit = integer_dot * INPUT_SCALE * WEIGHT_SCALE + MODEL.bias
        probability = float(1 / (1 + np.exp(-np.clip(logit, -35, 35))))
    else:
        probability = float(MODEL.predict_proba(scaled)[0])
    explanation_weights = MODEL_WEIGHTS.astype(float) * WEIGHT_SCALE if MODEL_QUANTIZATION == 'int8' else MODEL_WEIGHTS.astype(float)
    contributions = explanation_weights * scaled[0]
    top = np.argsort(np.abs(contributions))[::-1][:3]
    confidence = max(probability, 1 - probability)
    return {
        'model_version': ARTIFACT['model_version'],
        'predicted_class': int(probability >= THRESHOLD),
        'positive_probability': probability,
        'confidence': confidence,
        'explanation': [{'feature': FEATURES[index], 'logit_contribution': float(contributions[index])} for index in top],
        'disclaimer': 'Educational model output only; not a diagnosis.',
    }