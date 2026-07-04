from __future__ import annotations

from pathlib import Path

from dataops.contracts import DatasetReleaseManifest


class ReleaseRegistry:
    def __init__(self, root: Path | str = Path("artifacts/dataops")) -> None:
        self.root = Path(root)
        self.releases_path = self.root / "releases.jsonl"
        self.active_dir = self.root / "active"

    def append_release(self, manifest: DatasetReleaseManifest) -> None:
        if self.get_release(manifest.dataset_name, manifest.dataset_version) is not None:
            raise ValueError(f"release already exists: {manifest.release_id}")

        self.releases_path.parent.mkdir(parents=True, exist_ok=True)
        with self.releases_path.open("a", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json())
            handle.write("\n")

    def read_releases(self) -> list[DatasetReleaseManifest]:
        if not self.releases_path.exists():
            return []
        releases: list[DatasetReleaseManifest] = []
        for line in self.releases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                releases.append(DatasetReleaseManifest.model_validate_json(line))
        return releases

    def get_release(self, dataset_name: str, dataset_version: str) -> DatasetReleaseManifest | None:
        for release in self.read_releases():
            if release.dataset_name == dataset_name and release.dataset_version == dataset_version:
                return release
        return None

    def pin_active(self, pointer_name: str, manifest: DatasetReleaseManifest) -> None:
        if self.get_release(manifest.dataset_name, manifest.dataset_version) is None:
            raise ValueError(f"release is not registered: {manifest.release_id}")

        self.active_dir.mkdir(parents=True, exist_ok=True)
        self._pointer_path(pointer_name).write_text(
            "\n".join(
                [
                    f"dataset_name: {manifest.dataset_name}",
                    f"dataset_version: {manifest.dataset_version}",
                    f"release_type: {manifest.release_type}",
                    f"release_status: {manifest.release_status}",
                    f"release_id: {manifest.release_id}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def read_active(self, pointer_name: str) -> DatasetReleaseManifest | None:
        path = self._pointer_path(pointer_name)
        if not path.exists():
            return None

        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key, value = line.split(": ", 1)
            values[key] = value

        return self.get_release(values["dataset_name"], values["dataset_version"])

    def _pointer_path(self, pointer_name: str) -> Path:
        return self.active_dir / f"{pointer_name}.yaml"
