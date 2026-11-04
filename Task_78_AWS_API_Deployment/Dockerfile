FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ARTIFACT_PATH=/app/model.joblib

WORKDIR /app
COPY Task_78_AWS_API_Deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY manual_ml.py ./manual_ml.py
COPY Task_78_AWS_API_Deployment/app.py ./app.py
COPY Task_47_Versioned_Model_Export/artifacts/heart_logistic_v1.0.0.joblib ./model.joblib

USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]