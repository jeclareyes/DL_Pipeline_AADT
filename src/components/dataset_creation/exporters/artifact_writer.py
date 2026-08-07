from __future__ import annotations

import platform
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .manifest_writer import save_manifest


class DatasetManifestBuilder:
    """
    Construye el manifest JSON asegurando que no haya objetos pesados,
    y que los elementos estén organizados en:
    config, data, metadata, validations, reports.
    """
    
    @staticmethod
    def build(
        dataset_name: str,
        config: Any,
        data: dict[str, Any],
        metadata: dict[str, Any],
        reports: dict[str, Any] | None,
        validations: dict[str, Any] | None,
        master_artifact_path: str,
    ) -> dict[str, Any]:
        # 1. Resolver la configuración
        resolved_config = _config_to_container(config)
        
        # 2. Generar resúmenes para 'data'
        data_summary = {}
        for key, value in data.items():
            data_summary[key] = DatasetManifestBuilder._summarize_data_object(value)
            
        data_summary["master_artifact"] = {
            "path": master_artifact_path,
            "format": "joblib"
        }
        
        # 3. Limpiar metadata de objetos en runtime (ej. rng)
        clean_metadata = {k: v for k, v in metadata.items() if k != "rng"}
        
        # 4. Generar fingerprint/hashes básicos
        config_hash = DatasetManifestBuilder._generate_hash(resolved_config)
        
        manifest = {
            "schema_version": "1.0",
            "artifact_type": "dataset_creation_manifest",
            "artifact_id": f"{dataset_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "networkx_version": __import__("networkx").__version__ if _has_networkx() else None,
            },
            "reproducibility": {
                "config_hash": config_hash,
                "data_fingerprint": "v1", # TODO: Implement real fingerprint if needed
            },
            "config": resolved_config,
            "flow_columns": clean_metadata.get("flow_columns", {}),
            "data": data_summary,
            "metadata": clean_metadata,
            "validations": validations or {},
            "reports": reports or {},
        }
        return manifest

    @staticmethod
    def _summarize_data_object(obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "rows": obj.shape[0],
                "columns": list(obj.columns)
            }
        elif isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        elif hasattr(obj, "number_of_nodes"): # networkx graph
            return {
                "__type__": "networkx_graph",
                "num_nodes": obj.number_of_nodes(),
                "num_edges": obj.number_of_edges(),
            }
        elif isinstance(obj, dict):
            # Podría ser routes_by_od u otros mapeos.
            return {
                "__type__": "dict",
                "keys_count": len(obj)
            }
        return str(type(obj))

    @staticmethod
    def _generate_hash(data_dict: dict) -> str:
        # Simple hash of the string representation
        return hashlib.md5(str(data_dict).encode("utf-8")).hexdigest()


def save_master_artifact(
    *,
    dataset_name: str,
    config,
    data: dict[str, Any],
    metadata: dict[str, Any],
    info_dir: str | Path,
    reports: dict[str, Any] | None = None,
    validations: dict[str, Any] | None = None,
) -> dict[str, str]:
    info_path = Path(info_dir)
    info_path.mkdir(parents=True, exist_ok=True)

    master_path = info_path / "dataset_master.joblib"
    manifest_path = info_path / "dataset_manifest.json"

    # Construir el JSON manifest a través del Builder
    manifest_artifact = DatasetManifestBuilder.build(
        dataset_name=dataset_name,
        config=config,
        data=data,
        metadata=metadata,
        reports=reports,
        validations=validations,
        master_artifact_path=str(master_path)
    )

    # El joblib almacena el payload completo ejecutable
    master_artifact = {
        "schema_version": "1.0",
        "artifact_type": "dataset_master",
        "artifact_id": manifest_artifact["artifact_id"],
        "dataset_name": dataset_name,
        "created_at": manifest_artifact["created_at"],
        "config_hash": manifest_artifact["reproducibility"]["config_hash"],
        "data_fingerprint": manifest_artifact["reproducibility"]["data_fingerprint"],
        "config": manifest_artifact["config"],
        "flow_columns": manifest_artifact["flow_columns"],
        "data": data,
        "metadata": manifest_artifact["metadata"],
        "validations": manifest_artifact["validations"],
    }

    joblib.dump(master_artifact, master_path)
    save_manifest(manifest_artifact, manifest_path)

    return {
        "master_artifact_path": str(master_path),
        "manifest_path": str(manifest_path),
    }


def _config_to_container(config: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if hasattr(config, "__dataclass_fields__"):
        return OmegaConf.to_container(OmegaConf.structured(config), resolve=True)
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def _has_networkx() -> bool:
    try:
        __import__("networkx")
    except ModuleNotFoundError:
        return False
    return True
