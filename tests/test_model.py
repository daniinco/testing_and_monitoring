import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.ensemble import RandomForestClassifier

from ml_service.model import Model, ModelData


def test_model_initial():
    model = Model()
    data = model.get()
    assert data.model is None
    assert data.run_id is None


def test_feat_not_loaded():
    model = Model()
    with pytest.raises(RuntimeError):
        _ = model.features

def test_random_forest():
    model = Model()
    model.set(run_id='edfad2bc1f1e4681a4174ee5bb09bd35')
    data = model.get()
    assert data.run_id == 'edfad2bc1f1e4681a4174ee5bb09bd35'
    assert len(model.features) == 6

    df = pd.DataFrame([["White",
      "Male",
      "United-States",
      "Craft-repair",
      "HS-grad",
      0]], columns=["race", "sex", "native.country", "occupation", 
    "education", "capital.gain"])
    probability = data.model.predict_proba(df)[0][1]


def test_empty_run_id():
    model = Model()
    with pytest.raises(ValueError):
        model.set(run_id='')
