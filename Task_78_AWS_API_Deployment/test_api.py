import os
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MODEL_ARTIFACT_PATH', str(ROOT / 'Task_47_Versioned_Model_Export' / 'artifacts' / 'heart_logistic_v1.0.0.joblib'))
os.environ.setdefault('API_BEARER_TOKEN', 'local-test-token-with-at-least-thirty-two-characters')

from app import FEATURES, app

client = TestClient(app)
TOKEN = os.environ['API_BEARER_TOKEN']


def payload():
    example = {'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 130, 'chol': 240, 'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2}
    return {'features': {name: float(example[name]) for name in FEATURES}}


def test_health_check_is_available():
    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_predict_requires_authentication():
    assert client.post('/predict', json=payload()).status_code == 401


def test_predict_returns_versioned_probability():
    response = client.post('/predict', headers={'Authorization': f'Bearer {TOKEN}'}, json=payload())
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result['positive_probability'] <= 1
    assert result['model_version'] == '1.0.0'


def test_predict_rejects_incomplete_feature_schema():
    invalid = payload()
    invalid['features'].pop(FEATURES[0])
    assert client.post('/predict', headers={'Authorization': f'Bearer {TOKEN}'}, json=invalid).status_code == 422


def test_predict_rejects_invalid_category():
    invalid = payload()
    invalid['features']['cp'] = 99
    response = client.post('/predict', headers={'Authorization': f'Bearer {TOKEN}'}, json=invalid)
    assert response.status_code == 422
    assert 'suggestions' in response.json()['detail']