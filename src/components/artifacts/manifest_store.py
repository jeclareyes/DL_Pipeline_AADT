"""JSON-backed manifest store for artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ManifestStore:
    """Read and write artifact bundle manifests."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self._cache: Dict[str, Any] | None = None

    def read(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest at {self.manifest_path} must contain a JSON object.")
        self._cache = data
        return data

    def write(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError("Manifest data must be a dictionary.")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        self._cache = data

