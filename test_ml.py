import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope="module")
def census_df():
    """
    Load a small sample of the census dataset for tests.
    Using a small subset keeps tests fast.
    """
    project_root = os.getcwd()
    data_path = os.path.join(project_root, "data", "census.csv")
    df = pd.read_csv(data_path)

    # Use a small sample to keep unit tests quick
    return df.sample(n=200, random_state=42)


@pytest.fixture(scope="module")
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_process_data_returns_expected_types(census_df, cat_features):
    """
    Ensure process_data returns numpy arrays and fitted encoder/lb when training=True.
    """
    X, y, encoder, lb = process_data(
        census_df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0


def test_train_model_is_logistic_regression(census_df, cat_features):
    """
    Ensure train_model returns the expected model type (LogisticRegression).
    """
    X, y, encoder, lb = process_data(
        census_df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)


def test_compute_model_metrics_known_case():
    """
    Metrics should match expected output for a known simple case.
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    # precision = TP/(TP+FP) = 1/(1+0) = 1.0
    # recall    = TP/(TP+FN) = 1/(1+1) = 0.5
    # f1        = 2PR/(P+R) = 2*1*0.5/(1+0.5) = 0.6667
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 0.5)
    assert np.isclose(fbeta, 2 * 1.0 * 0.5 / (1.0 + 0.5), atol=1e-4)


def test_inference_output_shape(census_df, cat_features):
    """
    Inference should return predictions with the same number of rows as X.
    """
    X, y, encoder, lb = process_data(
        census_df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
