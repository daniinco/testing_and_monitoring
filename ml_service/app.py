from typing import Any
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Response

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.features import to_dataframe
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


MODEL = Model()

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['endpoint'],
                            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
PREPROCESS_LATENCY = Histogram('preprocess_duration_seconds', 'Preprocessing latency',
                               buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
INFERENCE_LATENCY = Histogram('inference_duration_seconds', 'Model inference latency',
                              buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
PREDICTION_VALUE = Counter('model_predictions_total', 'Model predictions', ['prediction'])
PREDICTION_PROBA = Histogram('model_prediction_probability', 'Model prediction probabilities',
                             buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_UPDATE_COUNT = Counter('model_updates_total', 'Number of model updates')
CURRENT_MODEL_RUN_ID = Gauge('current_model_run_id_info', 'Current model run_id', ['run_id'])
FEATURE_VALUES = Histogram('feature_value', 'Feature values', ['feature'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    yield
    # add any teardown logic here if needed


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}
    
    @app.get('/metrics')
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = MODEL.get().model
        if model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        try:
            df = to_dataframe(request, needed_columns=MODEL.features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail='Preprocessing failed')

        try:
            probability = model.predict_proba(df)[0][1]
        except Exception as e:
            raise HTTPException(status_code=500, detail='Model inference failed')
        prediction = int(probability >= 0.5)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        MODEL.set(run_id=run_id)
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
