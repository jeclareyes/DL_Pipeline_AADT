import hydra
from sklearn.model_selection import KFold

import pickle
from typing import Optional, Any, Dict
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
import numpy as np
import torch
import pandas as pd

def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    # try project root (two levels above this file)
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path_str
    if candidate.exists():
        return candidate
    # fallback to cwd
    cwd_candidate = Path.cwd() / path_str
    if cwd_candidate.exists():
        return cwd_candidate
    # return original Path (caller will raise if it doesn't exist)
    return p

def run_pipeline(cfg, config_path: Optional[str] = None, config: Optional[dict] = None):

    # 0. Configuration Setup

    # Convert to resolved container so interpolations like ${...} are expanded
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    """# Example: load graph using the YAML key `input_routes.graph`
    graph_path_str = cfg_dict.get("input_routes", {}).get("graph")
    if not graph_path_str:
        raise RuntimeError("Missing `input_routes.graph` in config")

    graph_path = _resolve_path(graph_path_str)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")"""

    # Example: use sampling params from YAML
    graph_path = cfg["input_routes"]["graph"]
    route_path = cfg["input_routes"]["route_cache"]
    od_path = cfg["input_routes"]["od_matrix"]
    link_path = cfg["input_routes"]["link_df"]

    sampling_cfg = cfg_dict.get("sampling", {})
    flow_rate = sampling_cfg.get("flow_rate")
    od_rate = sampling_cfg.get("od_rate")

    # 1. Data Loading

    # Load the graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    with open(route_path, "rb") as f:
        route_cache = pickle.load(f)

    od_matrix = np.load(od_path)

    link_df = pd.read_parquet(link_path)

    # 2. Sampling
    # locate sampling YAML relative to project root
    project_root = Path(__file__).resolve().parents[2]
    sampling_yaml = project_root / "configs" / "sampling" / "sampling.yaml"

    if not sampling_yaml.exists():
        raise FileNotFoundError(f"Sampling config not found: {sampling_yaml}")

    # Load sampling YAML as DictConfig
    loaded_sampling_cfg = OmegaConf.load(str(sampling_yaml))

    # Merge: create a mutable DictConfig from resolved cfg to allow sampling YAML to provide defaults
    # We use OmegaConf.create on the resolved container so that new keys (e.g. input_routes.general_route)
    # can be added by the sampling config without hitting a structured-config error.
    base_cfg: DictConfig = OmegaConf.create(cfg_dict)
    merged_sampling_cfg: DictConfig = OmegaConf.merge(base_cfg, loaded_sampling_cfg)

    from src.components.sampling.engine import SamplingEngine
    # Provide the merged DictConfig to the engine so it has all required settings
    engine = SamplingEngine(config=merged_sampling_cfg)
    sampled_flow_mask = engine.run(save=False, overwrite=False)

    # 3. Cross-Validation Setup

    # 4. Model Training - Validation Loop

    pass

"""def train_one_epoch(model, vdf_function, batch, train_params):

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    pass"""


# Hydra entrypoint: adjust config_path if your `configs` folder is in a different location.
@hydra.main(version_base=None, config_path="../../configs/training", config_name="training")
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)

if __name__ == "__main__":
    main()