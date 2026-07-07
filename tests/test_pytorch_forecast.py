import pytest
import numpy as np
import pandas as pd
from src.engines.pytorch_forecast import PyTorchTimeSeriesRegressor, TORCH_AVAILABLE

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_ts_regressor_lstm():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = PyTorchTimeSeriesRegressor(model_type='lstm', epochs=2, batch_size=10, hidden_size=16)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_ts_regressor_tcn():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = PyTorchTimeSeriesRegressor(model_type='tcn', epochs=2, batch_size=10, hidden_size=16)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (100,)
