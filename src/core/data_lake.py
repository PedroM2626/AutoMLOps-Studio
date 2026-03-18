from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import io
import zipfile

import pandas as pd


@dataclass(frozen=True)
class DatasetVersion:
    dataset: str
    version: str
    path: Path
    modified_at: datetime


class DataLake:
    """Simple filesystem-backed dataset catalog used by the UI layers."""

    def __init__(self, base_path: str | Path = "./data_lake"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[str]:
        datasets = []
        for child in self.base_path.iterdir():
            if child.is_dir() and child.name != "monitoring":
                datasets.append(child.name)
        return sorted(datasets)

    def list_versions(self, dataset_name: str) -> list[str]:
        dataset_dir = self.base_path / dataset_name
        if not dataset_dir.exists():
            return []
        versions = [path.name for path in dataset_dir.iterdir() if path.is_file()]
        return sorted(versions, reverse=True)

    def get_version_info(self, dataset_name: str, version: str) -> DatasetVersion:
        version_path = self._resolve_version_path(dataset_name, version)
        stat = version_path.stat()
        return DatasetVersion(
            dataset=dataset_name,
            version=version,
            path=version_path,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

    def load_version(self, dataset_name: str, version: str, nrows: int | None = None) -> pd.DataFrame:
        version_path = self._resolve_version_path(dataset_name, version)
        suffix = version_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(version_path, nrows=nrows)
        if suffix == ".json":
            try:
                return pd.read_json(version_path, lines=True)
            except ValueError:
                return pd.read_json(version_path)
        if suffix == ".parquet":
            return pd.read_parquet(version_path)
        if suffix == ".txt":
            lines = version_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if nrows is not None:
                lines = lines[:nrows]
            return pd.DataFrame({"text": lines})
        if suffix == ".zip":
            with zipfile.ZipFile(version_path) as archive:
                members = archive.namelist()
            if nrows is not None:
                members = members[:nrows]
            return pd.DataFrame({"archive_member": members})

        raise ValueError(f"Unsupported dataset version format: {suffix}")

    def save_raw_file(self, payload: bytes, dataset_name: str, original_name: str) -> str:
        dataset_slug = self._slugify(dataset_name)
        dataset_dir = self.base_path / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_name).suffix or ".bin"
        version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
        version_path = dataset_dir / version_name
        version_path.write_bytes(payload)
        return str(version_path)

    def delete_version(self, dataset_name: str, version: str) -> bool:
        version_path = self._resolve_version_path(dataset_name, version)
        if not version_path.exists():
            return False

        version_path.unlink()
        dataset_dir = version_path.parent
        if dataset_dir.exists() and not any(dataset_dir.iterdir()):
            dataset_dir.rmdir()
        return True

    def save_dataframe(self, frame: pd.DataFrame, dataset_name: str, file_name: str | None = None) -> str:
        file_name = file_name or "dataset.csv"
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        return self.save_raw_file(buffer.getvalue().encode("utf-8"), dataset_name, file_name)

    def _resolve_version_path(self, dataset_name: str, version: str) -> Path:
        version_path = self.base_path / dataset_name / version
        if not version_path.exists():
            raise FileNotFoundError(f"Dataset version not found: {dataset_name}/{version}")
        return version_path

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
        return cleaned.strip("_") or "dataset"