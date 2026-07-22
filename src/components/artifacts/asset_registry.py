"""Registry helpers for bundle manifest assets."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .manifest_store import ManifestStore


class AssetRegistry:
    """Read and query manifest entries without materializing assets."""

    def __init__(self, manifest_store: ManifestStore):
        self.manifest_store = manifest_store

    def get_manifest(self) -> Dict[str, Any]:
        return self.manifest_store.read()

    def get_base_artifact_entry(self) -> Dict[str, Any]:
        manifest = self.get_manifest()
        base_entry = manifest.get("base_artifact")
        if not isinstance(base_entry, dict):
            raise KeyError("Manifest does not contain a base_artifact entry.")
        return base_entry

    def find_route_set(
        self,
        spec_id: str,
        required_fingerprint: str | None = None,
        minimum_k_active: int | None = None,
    ) -> Optional[Dict[str, Any]]:
        manifest = self.get_manifest()
        route_sets = manifest.get("route_sets", {})
        if not isinstance(route_sets, dict):
            raise ValueError("manifest['route_sets'] must be a dictionary.")
        asset = route_sets.get(spec_id)
        if not isinstance(asset, dict):
            return None
        if required_fingerprint is not None and asset.get("fingerprint") != required_fingerprint:
            return None
        if minimum_k_active is not None:
            metadata = asset.get("metadata", {})
            available_k = metadata.get("k_generate")
            if not isinstance(available_k, int) or available_k < minimum_k_active:
                return None
        return asset

    def find_assignment_set(
        self,
        spec_id: str,
        required_fingerprint: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        manifest = self.get_manifest()
        assignment_sets = manifest.get("assignment_sets", {})
        if not isinstance(assignment_sets, dict):
            raise ValueError("manifest['assignment_sets'] must be a dictionary.")
        asset = assignment_sets.get(spec_id)
        if not isinstance(asset, dict):
            return None
        if required_fingerprint is not None and asset.get("fingerprint") != required_fingerprint:
            return None
        return asset

