import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from free_mlops.test_models import test_single_prediction
from free_mlops.test_models import test_batch_prediction
from free_mlops.test_models import load_test_data_from_uploaded_file
from free_mlops.test_models import save_test_results


def test_single_prediction_success(tmp_path):
    # Criar modelo mock
    mock_model = Mock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_model.classes_ = [0, 1]
    
    model_path = tmp_path / "model.pkl"
    
    with patch('free_mlops.test_models.load_model', return_value=mock_model):
        result = test_single_prediction(
            model_path=model_path,
            input_data={"feature1": 1.0, "feature2": "A"}
        )
    
    assert result["success"] is True
    assert result["prediction"] == 1
    assert result["probabilities"] == {"0": 0.2, "1": 0.8}
    assert result["input_data"] == {"feature1": 1.0, "feature2": "A"}


def test_single_prediction_error(tmp_path):
    # Criar modelo mock que falha
    mock_model = Mock()
    mock_model.predict.side_effect = Exception("Model error")
    
    model_path = tmp_path / "model.pkl"
    
    with patch('free_mlops.test_models.load_model', return_value=mock_model):
        result = test_single_prediction(
            model_path=model_path,
            input_data={"feature1": 1.0}
        )
    
    assert result["success"] is False
    assert "Model error" in result["error"]
    assert result["input_data"] == {"feature1": 1.0}


def test_batch_prediction_success(tmp_path):
    # Criar modelo mock
    mock_model = Mock()
    mock_model.predict.return_value = [1, 0, 1]
    mock_model.predict_proba.return_value = [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]]
    mock_model.classes_ = [0, 1]
    
    model_path = tmp_path / "model.pkl"
    batch_data = [
        {"feature1": 1.0, "feature2": "A"},
        {"feature1": 2.0, "feature2": "B"},
        {"feature1": 3.0, "feature2": "C"}
    ]
    
    with patch('free_mlops.test_models.load_model', return_value=mock_model):
        result = test_batch_prediction(
            model_path=model_path,
            batch_data=batch_data
        )
    
    assert result["success"] is True
    assert result["predictions"] == [1, 0, 1]
    assert result["batch_size"] == 3
    assert len(result["probabilities"]) == 3
    assert result["input_data"] == batch_data


def test_batch_prediction_error(tmp_path):
    # Criar modelo mock que falha
    mock_model = Mock()
    mock_model.predict.side_effect = Exception("Batch error")
    
    model_path = tmp_path / "model.pkl"
    batch_data = [{"feature1": 1.0}]
    
    with patch('free_mlops.test_models.load_model', return_value=mock_model):
        result = test_batch_prediction(
            model_path=model_path,
            batch_data=batch_data
        )
    
    assert result["success"] is False
    assert "Batch error" in result["error"]
    assert result["batch_size"] == 1


def test_load_test_data_from_uploaded_file(tmp_path):
    # Criar CSV de teste
    csv_content = "feature1,feature2,target\n1.0,A,0\n2.0,B,1\n3.0,C,0"
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(csv_content)
    
    # Mock do uploaded file
    mock_file = Mock()
    mock_file.getvalue.return_value = csv_content.encode()
    
    result = load_test_data_from_uploaded_file(mock_file, sample_size=2)
    
    assert len(result) == 2
    assert result[0] == {"feature1": 1.0, "feature2": "A", "target": 0}
    assert result[1] == {"feature1": 2.0, "feature2": "B", "target": 1}


def test_save_test_results(tmp_path):
    results = {
        "success": True,
        "prediction": 1,
        "probabilities": {"0": 0.2, "1": 0.8}
    }
    
    output_path = tmp_path / "results" / "test_results.json"
    save_test_results(results, output_path)
    
    assert output_path.exists()
    saved_content = output_path.read_text(encoding="utf-8")
    saved_results = json.loads(saved_content)
    
    assert saved_results == results
