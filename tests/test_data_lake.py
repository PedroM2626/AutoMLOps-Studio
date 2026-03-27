from pathlib import Path

import pandas as pd
import pytest

from src.core.data_lake import DataLake


def test_data_lake_save_load_and_delete(tmp_path: Path):
    lake = DataLake(tmp_path / "lake")
    frame = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    saved_path = Path(lake.save_dataframe(frame, "demo_dataset"))

    assert saved_path.exists()
    assert lake.list_datasets() == ["demo_dataset"]

    versions = lake.list_versions("demo_dataset")
    assert len(versions) == 1

    loaded = lake.load_version("demo_dataset", versions[0])
    assert loaded.to_dict(orient="list") == frame.to_dict(orient="list")

    assert lake.delete_version("demo_dataset", versions[0]) is True
    assert lake.list_datasets() == []


def test_data_lake_rejects_path_traversal(tmp_path: Path):
    lake = DataLake(tmp_path / "lake")

    with pytest.raises(ValueError):
        lake.list_versions("../escape")