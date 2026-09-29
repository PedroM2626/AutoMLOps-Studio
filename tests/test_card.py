from src.utils.helpers import generate_model_card, get_consumption_code, get_cv_consumption_code
import mlflow
import os
import re


def test_model_card_generation_path():
    card = generate_model_card(
        model_name="logistic_regression",
        params={"C": 1.0, "solver": "lbfgs"},
        metrics={"accuracy": 0.91, "f1": 0.90},
        feature_names=["age", "income", "balance"],
        task_type="classification",
        duration=12.5,
    )

    assert "Model Card: logistic_regression" in card
    assert "accuracy" in card
    assert "age" in card


def _snippet_uri(code):
    return re.search(r'mlflow\.set_tracking_uri\("([^"]*)"\)', code).group(1)


def test_consumption_snippet_uses_the_store_in_use_not_the_environment(monkeypatch):
    # A job can be handed an explicit URI that never reached the environment, so a snippet
    # built from os.getenv would tell the user to look in the wrong place.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///decoy.db")
    active = mlflow.get_tracking_uri()

    code = get_consumption_code("random_forest", "abc123", "classification", feature_names=["f1"])

    assert _snippet_uri(code) == active.replace("\\", "/")
    assert "decoy.db" not in code
    assert 'runs:/abc123/random_forest' in code
    compile(code, "<snippet>", "exec")


def test_consumption_snippet_never_carries_uri_credentials(monkeypatch):
    # Basic-auth credentials belong to the connection, not to a snippet the user pastes
    # into a notebook or shares.
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "https://pedro:tok3nSecret@dagshub.com/me/exp.mlflow")

    code = get_consumption_code("xgb", "run1", "regression")
    cv_code = get_cv_consumption_code("cv_model", "run2", "image_classification", "resnet18")

    for snippet in (code, cv_code):
        assert "tok3nSecret" not in snippet
        assert "pedro:" not in snippet
        assert "dagshub.com/me/exp.mlflow" in snippet
        # The redaction note must not end up inside the quoted URI, which would silently
        # point the loader at a store that does not exist.
        assert _snippet_uri(snippet) == "https://dagshub.com/me/exp.mlflow"
        assert "Credentials were removed" in snippet
        compile(snippet, "<snippet>", "exec")


def test_windows_path_store_produces_runnable_snippet(monkeypatch):
    # Locks the escape bug: "file:///C:\Users\me\mlruns" inside a double-quoted string is a
    # SyntaxError (\U is not a valid escape), and the snippet is meant to be copy-pasted.
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "file:///C:\\Users\\me\\mlruns")

    code = get_consumption_code("knn", "run9", "classification")

    assert _snippet_uri(code) == "file:///C:/Users/me/mlruns"
    compile(code, "<snippet>", "exec")

