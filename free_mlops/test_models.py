from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from typing import Any, Literal

from free_mlops.service import load_model


def test_single_prediction(
    model_path: Path,
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """Testa uma única predição com o modelo carregado."""
    model = load_model(model_path)
    
    # Converter dict para DataFrame
    df = pd.DataFrame([input_data])
    
    try:
        prediction = model.predict(df)[0]
        
        # Tentar obter probabilidades se for classificação
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba_dict = model.predict_proba(df)[0]
                if hasattr(model, "classes_"):
                    classes = model.classes_
                    proba = {str(cls): float(prob) for cls, prob in zip(classes, proba_dict)}
                else:
                    proba = [float(p) for p in proba_dict]
            except Exception:
                pass
        
        return {
            "success": True,
            "prediction": prediction,
            "probabilities": proba,
            "input_data": input_data,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "input_data": input_data,
        }


def test_batch_prediction(
    model_path: Path,
    batch_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Testa predição em lote com o modelo carregado."""
    model = load_model(model_path)
    
    # Converter lista de dicts para DataFrame
    df = pd.DataFrame(batch_data)
    
    try:
        predictions = model.predict(df).tolist()
        
        # Tentar obter probabilidades se for classificação
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                proba_matrix = model.predict_proba(df)
                if hasattr(model, "classes_"):
                    classes = model.classes_
                    probabilities = []
                    for proba_row in proba_matrix:
                        proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, proba_row)}
                        probabilities.append(proba_dict)
                else:
                    probabilities = proba_matrix.tolist()
            except Exception:
                pass
        
        return {
            "success": True,
            "predictions": predictions,
            "probabilities": probabilities,
            "batch_size": len(batch_data),
            "input_data": batch_data,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "batch_size": len(batch_data),
            "input_data": batch_data,
        }


def load_test_data_from_csv(csv_path: Path, sample_size: int | None = None) -> list[dict[str, Any]]:
    """Carrega dados de teste de um arquivo CSV."""
    df = pd.read_csv(csv_path)
    
    if sample_size and sample_size < len(df):
        df = df.head(sample_size)
    
    return df.to_dict(orient="records")


def load_test_data_from_uploaded_file(uploaded_file, sample_size: int | None = None) -> list[dict[str, Any]]:
    """Carrega dados de teste de um arquivo uploaded no Streamlit."""
    import tempfile
    
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    
    try:
        return load_test_data_from_csv(tmp_path, sample_size)
    finally:
        # Limpar arquivo temporário
        tmp_path.unlink(missing_ok=True)


def save_test_results(results: dict[str, Any], output_path: Path) -> None:
    """Salva resultados do teste em um arquivo JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# Funções auxiliares para testes (não são fixtures)
def _test_single_prediction_helper(model_path: Path, input_data: dict[str, Any]) -> dict[str, Any]:
    """Função auxiliar para testar single prediction."""
    return test_single_prediction(model_path, input_data)


def _test_batch_prediction_helper(model_path: Path, batch_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Função auxiliar para testar batch prediction."""
    return test_batch_prediction(model_path, batch_data)
