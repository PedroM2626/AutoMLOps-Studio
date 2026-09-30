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


def test_load_registered_model_reads_the_flavour_from_the_artifact():
    """The registry holds sklearn pipelines and PyTorch models side by side. Loading every
    one of them through the sklearn loader returned None for each vision model."""
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    from src.tracking.mlflow import load_registered_model, register_model_from_run

    with mlflow.start_run(run_name="tabular") as run:
        sklearn_uri = mlflow.sklearn.log_model(
            LogisticRegression().fit([[0.0], [1.0]], [0, 1]), name="model").model_uri
        tab_run = run.info.run_id
    with mlflow.start_run(run_name="vision") as run:
        torch_uri = mlflow.pytorch.log_model(nn.Linear(1, 2), name="model").model_uri
        vision_run = run.info.run_id

    assert register_model_from_run(tab_run, "tabular_model") is True
    assert register_model_from_run(vision_run, "vision_model") is True

    tabular = load_registered_model("tabular_model", "1")
    vision = load_registered_model("vision_model", "1")

    assert isinstance(tabular, LogisticRegression)
    assert isinstance(vision, nn.Module), "a PyTorch artifact must come back as a module"
    assert list(vision.parameters()), "the playground reads the device off the parameters"
    assert tabular.predict([[1.0]]).tolist() == [1]
