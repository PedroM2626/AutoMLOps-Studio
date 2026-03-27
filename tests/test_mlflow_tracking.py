from src.tracking import mlflow as mlflow_module


def test_register_model_from_run_uses_resolved_artifact(monkeypatch):
    calls = {}

    monkeypatch.setattr(mlflow_module, "_resolve_logged_model_artifact_path", lambda run_id: "my_model_artifact")

    def fake_register_model(model_uri, model_name):
        calls["uri"] = model_uri
        calls["name"] = model_name

    monkeypatch.setattr(mlflow_module.mlflow, "register_model", fake_register_model)

    ok = mlflow_module.register_model_from_run("abc123", "my-model")

    assert ok is True
    assert calls["uri"] == "runs:/abc123/my_model_artifact"
    assert calls["name"] == "my-model"
