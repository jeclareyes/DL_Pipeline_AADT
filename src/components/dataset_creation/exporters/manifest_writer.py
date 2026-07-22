from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import is_dataclass, asdict

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

def to_serializable(obj: Any) -> Any:
    """
    Convierte objetos complejos del pipeline a tipos compatibles con JSON.
    """

    if isinstance(obj, (DictConfig, ListConfig)):
        return to_serializable(OmegaConf.to_container(obj, resolve=True))

    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]

    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "shape": list(obj.shape),
            "columns": obj.columns.tolist(),
        }

    if isinstance(obj, pd.Series):
        return obj.tolist()

    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if isinstance(obj, np.random.Generator):
        return {
            "__type__": "numpy.random.Generator",
            "bit_generator": obj.bit_generator.__class__.__name__,
        }

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Path):
        return str(obj)

    try:
        import networkx as nx
        from networkx.classes.coreviews import AdjacencyView, AtlasView
    except ModuleNotFoundError:
        nx = None
        AdjacencyView = ()
        AtlasView = ()

    if nx is not None and isinstance(obj, nx.Graph):
        return {
            "__type__": "networkx_graph",
            "num_nodes": obj.number_of_nodes(),
            "num_edges": obj.number_of_edges(),
            "is_directed": obj.is_directed(),
        }

    if isinstance(obj, (AdjacencyView, AtlasView)):
        return to_serializable(dict(obj))

    if is_dataclass(obj):
        return to_serializable(asdict(obj))

    if hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())

    if hasattr(obj, "dict"):
        return to_serializable(obj.dict())

    if hasattr(obj, "__dict__"):
        return to_serializable(vars(obj))

    return obj

def to_serializable_deprecated(obj: Any) -> Any:
    """
    Convierte objetos complejos usados en el pipeline a estructuras compatibles
    con JSON: dict, list, str, int, float, bool o None.
    """

    # OmegaConf/Hydra no siempre es directamente serializable por json.dumps.
    # Primero se convierte a contenedores estándar de Python y luego se vuelve
    # a pasar por esta misma función para limpiar valores internos.
    if isinstance(obj, (DictConfig, ListConfig)):
        return to_serializable(OmegaConf.to_container(obj, resolve=True))

    # Los diccionarios pueden contener valores no serializables en cualquier nivel.
    # Por eso se recorren recursivamente.
    # Además, las claves de JSON deben ser strings.
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    # Las listas, tuplas y sets pueden contener objetos complejos.
    # JSON solo tiene arrays, así que todo se convierte a lista.
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]

    # Un DataFrame completo puede ser muy pesado para un manifest.
    # Aquí se guarda solo metadata suficiente para identificarlo.
    if isinstance(obj, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "shape": list(obj.shape),
            "columns": obj.columns.tolist(),
        }

    # Series de pandas tampoco son serializables directamente.
    # Se convierten a lista para conservar sus valores.
    if isinstance(obj, pd.Series):
        return obj.tolist()

    # Arrays de NumPy no son serializables por json.dumps.
    # Se convierten a listas estándar de Python.
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Escalares de NumPy, como np.int64 o np.float32, tampoco son tipos JSON nativos.
    # .item() los transforma en int/float/bool estándar de Python.
    if isinstance(obj, np.generic):
        return obj.item()

    # Los paths no son serializables directamente.
    # Se guardan como strings para que el manifest conserve la ruta.
    if isinstance(obj, Path):
        return str(obj)

    # Los dataclass, como LinkCatalogueItem si fue definido así,
    # se convierten a diccionario y luego se limpian recursivamente.
    if is_dataclass(obj):
        return to_serializable(asdict(obj))

    # Soporte para modelos Pydantic v2.
    # model_dump() devuelve un dict, pero puede contener objetos complejos.
    if hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())

    # Soporte para modelos Pydantic v1.
    if hasattr(obj, "dict"):
        return to_serializable(obj.dict())

    # Objetos propios simples suelen guardar sus atributos en __dict__.
    # Esto cubre clases como LinkCatalogueItem si no son dataclass/Pydantic.
    if hasattr(obj, "__dict__"):
        return to_serializable(vars(obj))

    # Tipos ya compatibles con JSON: str, int, float, bool, None.
    return obj


def save_manifest(master_artifact: dict[str, Any], manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = to_serializable(master_artifact)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
