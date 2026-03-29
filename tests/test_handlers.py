import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from ml_service.app import create_app, MODEL
from ml_service.model import ModelData
from ml_service import config


req = {
    'age': 26, 'workclass': 'State-gov', 'fnlwgt': 77516,
    'education': 'HS-grad', 'education.num': 13,
    'marital.status': 'Never-married', 'occupation': 'Craft-repair',
    'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Male',
    'capital.gain': 0, 'capital.loss': 0,
    'hours.per.week': 40, 'native.country': 'United-States',
}
bad_req = {
    'age': "booba", 'workclass': 'State-gov', 'fnlwgt': 77516,
    'education': 'HS-grad', 'education.num': 13,
    'marital.status': 'Never-married', 'occupation': 'Craft-repair',
    'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Male',
    'capital.gain': 0, 'capital.loss': 0,
    'hours.per.week': 40, 'native.country': 'United-States',
}


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'
    assert r.json()['run_id'] == config.default_run_id()


def test_predict_success(client):
    r = client.post('/predict', json=req)
    assert r.status_code == 200
    body = r.json()
    assert body['prediction'] is not None
    assert 0.0 <= body['probability'] <= 1.0


def test_missing(client):
    r = client.post('/predict', json={'age': 2})
    assert r.status_code == 422


def test_invalid(client):
    r = client.post('/predict', json=bad_req)
    assert r.status_code == 422


def test_update_model(client):
    r = client.post('/updateModel', json={'run_id': '8990717746ed4cfda04aaabd43c8bad5'})
    assert r.status_code == 200
    assert r.json()['run_id'] == '8990717746ed4cfda04aaabd43c8bad5'
