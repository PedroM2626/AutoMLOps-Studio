"""The whitebox notebook is a deliverable: its cells must run, not just look plausible."""
import json

import numpy as np
import pandas as pd
import pytest

from src.core.notebook_generator import WhiteboxNotebookGenerator, _py_literal


@pytest.fixture
def dataset(tmp_path):
    frame = pd.DataFrame({
        "f_num": np.arange(80, dtype=float),
        "f_cat": ["a", "b", "c", "d"] * 20,
        "target": [0, 1] * 40,
    })
    path = tmp_path / "nb_data.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _code_cells(notebook_path):
    with open(notebook_path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        cells.append("".join(source) if isinstance(source, list) else source)
    return cells


def _generate(dataset, tmp_path, name, task="classification", metric="accuracy", **overrides):
    best_params = overrides.pop("best_params", {"model_name": "random_forest", "rf_max_depth": 4})
    generator = WhiteboxNotebookGenerator(
        config={"task": task, "target": "target", "optimization_metric": metric},
        best_params=best_params,
        feature_names=overrides.pop("feature_names", ["f_num", "f_cat"]),
        dataset_path=dataset,
        **overrides,
    )
    return _code_cells(generator.generate(str(tmp_path / name)))


def test_every_generated_cell_is_valid_python(dataset, tmp_path):
    cells = _generate(dataset, tmp_path, "nb.ipynb")

    assert len(cells) >= 5
    for index, source in enumerate(cells):
        compile(source, f"<cell {index}>", "exec")


def test_generated_notebook_runs_end_to_end(dataset, tmp_path, monkeypatch):
    import matplotlib
    monkeypatch.setenv("MPLBACKEND", "Agg")
    matplotlib.use("Agg", force=True)

    cells = _generate(
        dataset, tmp_path, "nb_run.ipynb",
        best_params={"model_name": "random_forest", "rf_max_depth": 4, "task_type": "classification"},
    )

    namespace = {"__name__": "__main__"}
    for index, source in enumerate(cells):
        try:
            exec(compile(source, f"<cell {index}>", "exec"), namespace)
        except Exception as error:  # noqa: BLE001 - the failure below is the report
            pytest.fail(f"generated notebook died in cell {index}: {type(error).__name__}: {error}")

    assert namespace["model"].__class__.__name__ == "RandomForestClassifier"
    assert namespace["model"].get_params()["max_depth"] == 4
    assert len(namespace["preds"]) == len(namespace["y_test_proc"])
    assert 0.0 <= float(namespace["score"]) <= 1.0


def test_generated_cells_survive_windows_paths_and_quoted_names(tmp_path):
    awkward_path = r"C:\Users\me\data's set\train.csv"
    awkward_target = "y'weird"
    generator = WhiteboxNotebookGenerator(
        config={"task": "regression", "target": awkward_target, "optimization_metric": "rmse"},
        best_params={"model_name": "ridge", "rg_alpha": 0.5},
        feature_names=["f"],
        dataset_path=awkward_path,
    )

    cells = _code_cells(generator.generate(str(tmp_path / "awkward.ipynb")))

    for index, source in enumerate(cells):
        compile(source, f"<cell {index}>", "exec")

    loading = next(cell for cell in cells if "read_csv" in cell)
    literal = next(line for line in loading.splitlines() if "read_csv" in line)
    namespace = {}
    # Only the path string is evaluated: the file itself does not exist in this test.
    extracted = literal.split("pd.read_csv(", 1)[1].rsplit(")", 1)[0]
    assert eval(extracted) == awkward_path
    assert f"target_col = {json.dumps(awkward_target)}" in loading


def test_py_literal_round_trips_types_and_special_floats():
    payload = {"a": 1, "b": 0.5, "c": "x", "d": None, "e": True, "f": [1, "2"], "g": (3,)}
    assert eval(_py_literal(payload)) == payload

    assert _py_literal(float("nan")) == "float('nan')"
    assert _py_literal(float("inf")) == "float('inf')"
    assert _py_literal(float("-inf")) == "float('-inf')"
    assert _py_literal(np.int64(7)) == "7"
    assert _py_literal(np.float64(1.5)) == "1.5"
    assert _py_literal(set()) == "set()"
    assert eval(_py_literal({"nested": {"k": [1.0, None, "s"]}})) == {"nested": {"k": [1.0, None, "s"]}}
