import pytest
import pandas as pd

from ml_service.features import to_dataframe, FEATURE_COLUMNS
from ml_service.schemas import PredictRequest


FULL_REQUEST = PredictRequest(
    age=1,
    workclass='',
    fnlwgt=1,
    education='',
    **{'education.num': 0, 'marital.status': '',
       'capital.gain': 0, 'capital.loss': 0, 'hours.per.week': 0,
       'native.country': ''},
    occupation='',
    race='',
    sex='',
    relationship=""
)


def test_all():
    df = to_dataframe(FULL_REQUEST)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_needed():
    needed = ['age', 'sex', 'education.num']
    df = to_dataframe(FULL_REQUEST, needed_columns=needed)
    assert list(df.columns) == needed
    assert df['age'].iloc[0] == 1


def test_missing():
    req = PredictRequest(age=1)
    with pytest.raises(ValueError, match='Missing required features'):
        to_dataframe(req, needed_columns=['age', 'sex'])
