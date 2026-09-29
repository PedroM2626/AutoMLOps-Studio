"""Serving and export deliverables: bundle contents, the zip, a live uvicorn service and
the ONNX artifact are exercised as real artifacts over real HTTP."""
import json
import os
import socket
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.core.api_exporter import (
    _validate_ref,
    build_bundle_dir,
    export_model_api,
    start_local_service,
    stop_local_service,
)
from src.engines.classical import AutoMLTrainer, export_model_to_onnx

MODEL_NAME = "serving_probe_model"


def _post(url, payload, expect_status=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode("utf-8") or "{}")
        if expect_status is not None and error.code == expect_status:
            return error.code, body
        raise


def _port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2.0)
        return probe.connect_ex((host, port)) == 0


@pytest.fixture
def registered_model():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    import mlflow

    rng = np.random.default_rng(5)
    rows = 120
    frame = pd.DataFrame({
        "f_num": rng.normal(0.0, 1.0, rows),
        "f_cat": rng.choice(["a", "b", "c"], rows),
    })
    target = (frame["f_num"] + (frame["f_cat"] == "c").astype(float) * 0.5 > 0.05).astype(int).to_numpy()

    pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), ["f_num"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["f_cat"]),
        ])),
        ("model", RandomForestClassifier(n_estimators=8, random_state=0)),
    ])
    pipeline.fit(frame, target)

    # Logged inside an active run, the way MLFlowTracker.log_experiment does it: outside a
    # run the registered version carries no run_id, which is not a state the product creates.
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            pipeline, name="model", registered_model_name=MODEL_NAME, input_example=frame.head(2)
        )
    versions = mlflow.MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(entry.version) for entry in versions)
    return MODEL_NAME, str(latest)


def test_bundle_is_self_containing_and_valid_python(registered_model, tmp_path):
    name, version = registered_model
    dest = tmp_path / "bundle"
    dest.mkdir()

    build_bundle_dir(name, version, str(dest))

    assert (dest / "app.py").exists()
    assert (dest / "Dockerfile").exists()
    assert (dest / "requirements.txt").exists()
    assert (dest / "model").is_dir()

    source = (dest / "app.py").read_text(encoding="utf-8")
    compile(source, "bundled app.py", "exec")
    # An unsubstituted placeholder would ship a literally-braced string in the title.
    assert "{model_name}" not in source and "{version}" not in source
    assert name in source

    served = [
        line.strip()
        for line in (dest / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The model's own requirements are merged in, and they may arrive pinned (mlflow==3.x),
    # so match on the package name rather than the bare string.
    for wanted in ("fastapi", "uvicorn", "pandas"):
        assert any(req == wanted or req.startswith(f"{wanted}==") or req.startswith(f"{wanted}~")
                   for req in served), f"{wanted} missing from {served}"


def test_bundle_rejects_unsafe_references(tmp_path):
    # A separator is what turns these into a path; the charset rule is what blocks it.
    for unsafe_name in ("../escape", "dir/escape", "with spaces", "tabs\tinside", ""):
        with pytest.raises(ValueError):
            _validate_ref(unsafe_name, "1")
    for unsafe_version in ("../1", "1/2", "1 2", ""):
        with pytest.raises(ValueError):
            _validate_ref(MODEL_NAME, unsafe_version)


def test_export_model_api_produces_an_openable_zip(registered_model):
    name, version = registered_model

    zip_path = export_model_api(name, version)

    assert zip_path.endswith(".zip") and os.path.exists(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        assert "app.py" in members and "Dockerfile" in members and "requirements.txt" in members
        assert any(member.startswith("model/") for member in members)
        assert archive.read("app.py").decode("utf-8").count("async def predict") == 1
    os.remove(zip_path)


def test_local_service_answers_health_and_predicts(registered_model):
    name, version = registered_model

    handle = start_local_service(name, version, timeout=300.0)
    try:
        assert handle["health"]["model_loaded"] is True
        assert handle["health"]["status"] == "Healthy"

        status, body = _post(
            f"{handle['url']}/predict",
            [{"f_num": 0.9, "f_cat": "c"}, {"f_num": -0.5, "f_cat": "a"}],
        )
        assert status == 200
        assert len(body["predictions"]) == 2
        assert set(body["predictions"]).issubset({0, 1})

        # A row missing a required feature is the caller's fault: it must not be reported
        # as a successful prediction, and the response must not carry a server traceback.
        status, error_body = _post(f"{handle['url']}/predict", [{"f_num": 0.1}], expect_status=400)
        assert status == 400
        assert "trace" not in error_body

        status, _ = _post(f"{handle['url']}/predict", [], expect_status=400)
        assert status == 400
    finally:
        stop_local_service(handle)

    assert handle["process"].poll() is not None
    port = int(handle["url"].rsplit(":", 1)[1])
    assert not _port_open("127.0.0.1", port)


def test_local_service_reports_the_service_log_when_it_never_starts(registered_model, monkeypatch):
    import src.core.api_exporter as exporter

    name, version = registered_model
    monkeypatch.setattr(
        exporter, "APP_CODE", 'raise RuntimeError("bundle could not start")\n' + exporter.APP_CODE
    )

    with pytest.raises(RuntimeError) as raised:
        exporter.start_local_service(name, version, timeout=25.0)
    assert "bundle could not start" in str(raised.value)


def test_onnx_export_matches_the_sklearn_predictions(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, (60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=6, random_state=0).fit(X, y)
    onnx_path = tmp_path / "model.onnx"

    export_model_to_onnx(clf, np.zeros((1, 4)), str(onnx_path))

    assert onnx_path.exists() and onnx_path.stat().st_size > 0
    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    labels = np.asarray(session.run(None, {"float_input": X.astype("float32")})[0]).ravel()

    np.testing.assert_allclose(labels, clf.predict(X), rtol=1e-4, atol=1e-4)


def test_onnx_export_reports_a_missing_model(tmp_path):
    trainer = AutoMLTrainer(task_type="classification")
    trainer.best_model = None

    with pytest.raises(ValueError, match="No best model"):
        trainer.export_best_model_to_onnx(np.zeros((1, 3)), str(tmp_path / "x.onnx"))

    with pytest.raises(ValueError, match="No model"):
        export_model_to_onnx(None, np.zeros((1, 3)), str(tmp_path / "y.onnx"))


def test_trainer_exports_its_fitted_champion(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(2)
    X = rng.normal(0.0, 1.0, (60, 4))
    y = (X[:, 2] > 0).astype(int)
    trainer = AutoMLTrainer(task_type="classification")
    trainer.best_model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    out = trainer.export_best_model_to_onnx(X[:1], str(tmp_path / "champion.onnx"))

    assert os.path.exists(out) and os.path.getsize(out) > 0
