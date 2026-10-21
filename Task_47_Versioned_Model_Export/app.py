import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_ml import LogisticRegressionScratch, load_data, metrics, standardize, standardize_fit, stratified_split


ARTIFACT_SCHEMA = 1
MODEL_VERSION = '1.0.0'


def artifact_bytes_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore_model(payload):
    if payload.get('schema_version') != ARTIFACT_SCHEMA:
        raise ValueError('Unsupported artifact schema version')
    supported_classes = {'manual_ml.LogisticRegressionScratch', 'ManualLogistic'}
    if payload.get('model_class') not in supported_classes:
        raise ValueError('Unsupported model class')
    model = LogisticRegressionScratch(**payload['parameters'])
    model.weights = np.asarray(payload['state']['weights'], dtype=float)
    model.bias = float(payload['state']['bias'])
    if model.weights.shape != (payload['feature_count'],):
        raise ValueError('Artifact weight count does not match feature metadata')
    return model


st.title('Task 47: Versioned Model Export')
csv_path = Path(__file__).with_name('heart.csv')
if not csv_path.exists():
    csv_path = ROOT / 'Task_21_Decision_Tree_Feature_Importance' / 'heart.csv'
_, _, features, frame = load_data(csv_path)
frame = frame.drop_duplicates().reset_index(drop=True)
X, y = frame[features].to_numpy(float), frame['target'].to_numpy(int)
X_development, X_test, y_development, y_test = stratified_split(X, y, 0.2, seed=47)
X_train, X_validation, y_train, y_validation = stratified_split(X_development, y_development, 0.2, seed=48)
means, scales = standardize_fit(X_train)
X_train_scaled = standardize(X_train, means, scales)
X_validation_scaled = standardize(X_validation, means, scales)

parameters = {'learning_rate': 0.06, 'iterations': 1400, 'l2': 0.02}
selection_model = LogisticRegressionScratch(**parameters).fit(X_train_scaled, y_train)
validation_result = metrics(y_validation, selection_model.predict(X_validation_scaled))
threshold = 0.5

# Refit after model selection and package only explicit, versioned state.
final_means, final_scales = standardize_fit(X_development)
development_scaled = standardize(X_development, final_means, final_scales)
test_scaled = standardize(X_test, final_means, final_scales)
final_model = LogisticRegressionScratch(**parameters).fit(development_scaled, y_development)
test_metrics = metrics(y_test, final_model.predict(test_scaled, threshold))
payload = {
    'schema_version': ARTIFACT_SCHEMA,
    'model_version': MODEL_VERSION,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'model_class': 'manual_ml.LogisticRegressionScratch',
    'parameters': parameters,
    'state': {'weights': final_model.weights.tolist(), 'bias': float(final_model.bias)},
    'feature_names': features,
    'feature_count': len(features),
    'target_name': 'target',
    'decision_threshold': threshold,
    'preprocessing': {'kind': 'standardize', 'means': final_means.tolist(), 'scales': final_scales.tolist()},
    'dataset_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
}

artifact_dir = Path(__file__).with_name('artifacts')
artifact_dir.mkdir(exist_ok=True)
artifact_path = artifact_dir / f'heart_logistic_v{MODEL_VERSION}.joblib'
temporary_path = artifact_path.with_suffix('.tmp.joblib')
joblib.dump(payload, temporary_path, compress=('zlib', 3), protocol=5)
os.replace(temporary_path, artifact_path)
loaded_payload = joblib.load(artifact_path)
restored_model = restore_model(loaded_payload)
loaded_threshold = loaded_payload.get('decision_threshold', loaded_payload.get('threshold', 0.5))
restored_predictions = restored_model.predict(test_scaled, loaded_threshold)
round_trip_match = bool(np.array_equal(restored_predictions, final_model.predict(test_scaled, threshold)))
artifact_hash = artifact_bytes_digest(artifact_path)

st.write(f'Validation F1 before final refit: {validation_result["f1"]:.3f}')
st.write('Final held-out metrics:', {key: round(value, 3) for key, value in test_metrics.items() if key in {'accuracy', 'precision', 'recall', 'f1'}})
st.write(f'Model version: {MODEL_VERSION}; schema: {ARTIFACT_SCHEMA}; round-trip predictions match: {round_trip_match}')
st.write(f'Compressed artifact: {artifact_path.name} ({artifact_path.stat().st_size:,} bytes)')
st.write(f'SHA-256: {artifact_hash}')
st.download_button('Download versioned model artifact', data=artifact_path.read_bytes(), file_name=artifact_path.name, mime='application/octet-stream')
st.caption('The artifact stores explicit model state and preprocessing metadata, not an opaque estimator object. Keep the compatible manual_ml.py implementation and recorded runtime versions with deployments. Only load trusted joblib artifacts.')