import hmac
import os
import sys
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
THRESHOLD = float(ARTIFACT.get('decision_threshold', 0.5))
CATEGORY_CODES = {'sex': [0, 1], 'cp': [0, 1, 2, 3], 'fbs': [0, 1], 'restecg': [0, 1, 2], 'exang': [0, 1], 'slope': [0, 1, 2], 'ca': [0, 1, 2, 3, 4], 'thal': [0, 1, 2, 3]}
HARD_RANGES = {'age': (18, 100), 'trestbps': (60, 260), 'chol': (80, 700), 'thalach': (40, 250), 'oldpeak': (-5, 15)}

app = FastAPI(title='Heart Model Inference API', version=ARTIFACT['model_version'])
bearer = HTTPBearer(auto_error=False)


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
    return {'status': 'ok', 'model_version': ARTIFACT['model_version']}


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
    probability = float(MODEL.predict_proba(scaled)[0])
    contributions = MODEL.weights * scaled[0]
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