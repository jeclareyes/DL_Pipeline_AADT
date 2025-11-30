import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import numpy as np
import networkx as nx
import sys

from omegaconf import DictConfig, OmegaConf
import hydra

from src.components.sampling.base import BaseSampler
from src.components.sampling import sampling as core_sampling

logger = logging.getLogger(__name__)


class SamplingEngine:
    """
    Generic engine that delegates sampling to a strategy.
    strategy: either an instance of BaseStrategy, or a string 'module.path:ClassName'.
    strategy_params: kwargs passed to the strategy constructor.
    """
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Engine que orquesta el proceso de sampling.

        Args:
            config_path: Ruta a un YAML de configuración (si se proporciona, prevalece).
            config: Diccionario de configuración (si se proporciona, usado cuando config_path es None).
                    Puede ser un omegaconf.DictConfig (Hydra) o un dict.
        """
        if config_path is None and config is None:
            raise ValueError("Debe proporcionar config_path o config")

        if config_path is not None:
            self.config_path = Path(config_path)
            # Use OmegaConf to load + resolve interpolations (Hydra/OmegaConf style)
            # This replaces the previous yaml.safe_load which left ${...} unresolved.
            cfg_om = OmegaConf.load(str(self.config_path))
            # convert to plain dict and resolve interpolations
            self.cfg = OmegaConf.to_container(cfg_om, resolve=True)
            # Expand any environment vars in configured paths (defensive)
            import os
            gr = self.cfg.get('input_routes', {}).get('graph_route')
            if isinstance(gr, str):
                self.cfg.setdefault('input_routes', {})['graph_route'] = os.path.expanduser(os.path.expandvars(gr))
        else:
            # Accept DictConfig from Hydra directly
            if isinstance(config, DictConfig):
                self.cfg = OmegaConf.to_container(config, resolve=True)
            else:
                self.cfg = config
            self.config_path = None

        # Normalize some fields and defaults
        self.dataset = self.cfg.get('dataset')
        self.data_root = Path(self.cfg.get('data', {}).get('root', 'data/processed'))
        self.input_routes = self.cfg.get('input_routes', {})
        self.graph_route = Path(self.input_routes.get('graph_route')) if self.input_routes.get('graph_route') else None

        self.link_train_ratio = float(self.cfg.get('data_split', {}).get('link_train_ratio', 0.8))
        self.strategy_name = self.cfg.get('sampling', {}).get('strategy', self.cfg.get('strategy'))

        # mapping of strategy keys -> spec (module:Class or path to .py)
        self.strategy_map = self.cfg.get('sampling_strategies_routes', {})

        # sampling params
        self.flow_rate = float(self.cfg.get('sampling', {}).get('flow_rate', self.cfg.get('flow_rate', 0.1)))
        self.od_rate = float(self.cfg.get('sampling', {}).get('od_rate', self.cfg.get('od_rate', 0.0)))
        self.sampling_basis = self.cfg.get('sampling', {}).get('sampling_basis', 'link_wise_based')

        # seed and reproducibility
        self.seed = int(self.cfg.get('data', {}).get('random_seed', self.cfg.get('random_seed', 42)))
        np.random.seed(self.seed)

        # output settings
        self.output_dir = Path(self.cfg.get('sampling', {}).get('output_dir', 'outputs/masks'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_filename = Path(self.cfg.get('sampling', {}).get('output_filename', 'sampled_flow_mask.npy'))
        self.output_path = self.output_dir / self.output_filename

        # strategy params passthrough
        self.strategy_params = self.cfg.get('strategy_params', {}) or {}
        # ensure seed in params
        self.strategy_params.setdefault('random_seed', self.seed)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _load_graph(self) -> nx.DiGraph:
        if self.graph_route is None:
            raise FileNotFoundError("graph_route no definido en la configuración.")

        graph_path = Path(self.graph_route)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        logger.info(f"Cargando grafo desde: {graph_path}")
        import pickle
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        if not isinstance(G, (nx.Graph, nx.DiGraph)):
            raise TypeError("El archivo de grafo no contiene un NetworkX Graph/Digraph")

        return G

    def _prepare_train_flow_mask(self, graph: nx.DiGraph, year: Optional[int] = None):
        """Crea train_flow_mask alineada con list(graph.edges()) verificando existencia de Volume_<year> o 'flow'.

        - Si no encuentra una columna Volume_<year> ni 'flow' lanza error.
        - Devuelve train_flow_mask (np.ndarray float32) y all_flows (np.ndarray float32)
        """
        edges = list(graph.edges(data=True))
        n_links = len(edges)

        # determine flow key
        flow_key = None
        if year is not None:
            #candidate = f'volume'
            candidate = f'Volume_{year}' # TODO: implementar después, para esto se necesita ajustar TODO de carga de las columnas Volume
            # check on first few edges
            for _, _, attrs in edges[:min(20, n_links)]:
                if candidate in attrs:
                    flow_key = candidate
                    break
        if flow_key is None:
            # check for 'flow' attr
            for _, _, attrs in edges[:min(20, n_links)]:
                if 'flow' in attrs:
                    flow_key = 'flow'
                    break

        if flow_key is None:
            raise RuntimeError(f"No se encontró atributo de flujo 'Volume_<year>' ni 'flow' en las aristas del grafo. Necesario para sampling.")

        all_flows = np.zeros(n_links, dtype=np.float32)
        valid_mask = np.zeros(n_links, dtype=np.float32)

        for i, (_, _, attrs) in enumerate(edges):
            val = attrs.get(flow_key, np.nan)
            try:
                fv = float(val)
            except Exception:
                fv = np.nan
            all_flows[i] = 0.0 if np.isnan(fv) else fv
            valid_mask[i] = 0.0 if np.isnan(fv) else 1.0

        # split train/test on valid indices using seed
        valid_indices = np.where(valid_mask > 0)[0]
        if len(valid_indices) == 0:
            raise RuntimeError("No hay enlaces con valores de flujo válidos en el grafo.")

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(valid_indices))
        split_idx = int(len(valid_indices) * self.link_train_ratio)
        train_idx = valid_indices[perm[:split_idx]]

        train_mask = np.zeros(n_links, dtype=np.float32)
        train_mask[train_idx] = 1.0

        return all_flows, train_mask

    def _load_strategy_from_spec(self, spec: str):
        """Carga estrategia desde 'module.path:ClassName' o desde ruta a .py.

        Retorna una instancia de BaseSampler.
        """
        # If mapping exists in strategy_map, resolve
        if spec in self.strategy_map:
            spec_val = self.strategy_map[spec]
        else:
            spec_val = spec

        # If colon present -> module:Class
        if ':' in spec_val:
            module_path, class_name = spec_val.split(':', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            inst = cls(**self.strategy_params)
            if not isinstance(inst, BaseSampler):
                raise TypeError(f"Loaded strategy {spec_val} is not an instance of BaseSampler")
            return inst

        # If endswith .py -> load from file
        p = Path(spec_val)
        if p.suffix == '.py' and p.exists():
            # import from file
            spec_mod = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec_mod)
            sys.modules[p.stem] = mod
            spec_mod.loader.exec_module(mod)  # type: ignore
            # find class that is subclass of BaseSampler
            for attr in dir(mod):
                obj = getattr(mod, attr)
                try:
                    if isinstance(obj, type) and issubclass(obj, BaseSampler) and obj is not BaseSampler:
                        inst = obj(**self.strategy_params)
                        return inst
                except Exception:
                    continue
            raise RuntimeError(f"No se encontró una clase BaseSampler en {p}")

        # Otherwise try module path without class (module must expose 'Strategy' or 'Sampler')
        try:
            module = importlib.import_module(spec_val)
            # try to find class
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, BaseSampler) and obj is not BaseSampler:
                    inst = obj(**self.strategy_params)
                    return inst
        except Exception:
            pass

        raise RuntimeError(f"No se pudo cargar la estrategia desde spec: {spec}")

    # ----------------------------
    # Run
    # ----------------------------
    def run(self, save: bool = True, overwrite: bool = True) -> np.ndarray:
        # TODO overwrite debe colocarse en el yaml
        """Ejecuta el sampling y opcionalmente guarda la máscara de flujo muestreada.

        Args:
            save: si True intentará guardar la máscara en disco (outputs/masks/)
            overwrite: si False y el archivo existe, no sobreescribirá.

        Returns:
            sampled_flow_mask: np.ndarray (float32)
        """
        # 1. Load graph
        G = self._load_graph()

        # 2. Prepare masks
        # year = self.cfg.get('data', {}).get('volume_year', None) # TODO
        year = 2022  # placeholder until volume_year handling is implemented
        if year is not None:
            try:
                year = int(year)
            except Exception:
                year = None

        all_flows, train_mask = self._prepare_train_flow_mask(G, year=year)

        # 3. Load strategy
        strategy_spec = self.strategy_name
        if strategy_spec is None:
            raise RuntimeError("No se definió 'strategy' en el config (ej: 'random_sampling' o module:Class)")

        strategy_inst = self._load_strategy_from_spec(strategy_spec)

        # 4. Run sampling via strategy instance
        sampled_flow_mask, sampled_od_mask = strategy_inst.create_partial_data_masks(
            train_flow_mask=train_mask,
            od_mask=np.zeros(1, dtype=np.float32),  # placeholder, od sampling not implemented
            flow_rate=self.flow_rate,
            od_rate=self.od_rate,
            graph=G,
            volume_year=year
        )

        # 5. Save if requested
        if save:
            if self.output_path.exists() and not overwrite:
                logger.info(f"Output exists and overwrite=False -> skipping save: {self.output_path}")
            else:
                np.save(self.output_path, sampled_flow_mask)
                logger.info(f"Saved sampled_flow_mask to: {self.output_path}")

        return sampled_flow_mask


# Hydra entrypoint
@hydra.main(version_base=None, config_path="../../../configs", config_name="sampling")
def hydra_entry(cfg: DictConfig):
    """Convenience entrypoint to run the SamplingEngine from Hydra.

    Usage from project root:
      python -m src.components.sampling.engine
    or
      python -m src.components.sampling.engine --config configs/sampling/sampling.yaml

    When running with Hydra write:
      python -m src.components.sampling.engine hydra.run.dir=. -m
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    engine = SamplingEngine(config=cfg)
    engine.run()


# -----------------------------
# CLI Entrypoint
# -----------------------------

def main(argv: Optional[list] = None):
    import argparse
    parser = argparse.ArgumentParser(description='SamplingEngine CLI')
    parser.add_argument('--config', type=str, default='configs/sampling/sampling.yaml', help='Path to sampling YAML')
    parser.add_argument('--no-save', action='store_true', help='Do not save outputs to disk')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mask files')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    engine = SamplingEngine(config_path=args.config)
    engine.run(save=not args.no_save, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
