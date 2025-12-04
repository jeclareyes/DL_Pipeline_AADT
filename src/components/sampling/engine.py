import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import networkx as nx
import sys
import pickle

from omegaconf import DictConfig, OmegaConf
import hydra

# Importamos la interfaz y las utilidades compartidas
from src.components.sampling.base import BaseSampler
from src.components.sampling import sampling as utils

logger = logging.getLogger(__name__)


class SamplingEngine:
    """
    Orquestador de Sampling.
    Su única responsabilidad es:
    1. Cargar configuración y recursos (Grafo).
    2. Preparar el estado inicial (Máscaras observadas).
    3. Delegar la estrategia de muestreo a la clase configurada.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # --- 1. Carga de Configuración (Hydra/OmegaConf) ---
        if config_path is None and config is None:
            raise ValueError("Debe proporcionar config_path o config")

        if config_path is not None:
            self.config_path = Path(config_path)
            cfg_om = OmegaConf.load(str(self.config_path))
            self.cfg = OmegaConf.to_container(cfg_om, resolve=True)
            # Expandir variables de entorno si es necesario
            import os
            gr = self.cfg.get('input_routes', {}).get('graph_route')
            if isinstance(gr, str):
                self.cfg.setdefault('input_routes', {})['graph_route'] = os.path.expanduser(os.path.expandvars(gr))
        else:
            self.cfg = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
            self.config_path = None

        # --- 2. Parámetros Principales ---
        self.dataset = self.cfg.get('dataset')

        # Rutas
        self.input_routes = self.cfg.get('input_routes', {})
        self.graph_route = Path(self.input_routes.get('graph_route')) if self.input_routes.get('graph_route') else None

        # Estrategia
        self.strategy_name = self.cfg.get('sampling', {}).get('strategy', 'random')
        self.strategy_map = self.cfg.get('sampling_strategies_routes', {})
        self.strategy_params = self.cfg.get('strategy_params', {}) or {}

        # Rates y Config
        self.flow_rate = float(self.cfg.get('sampling', {}).get('flow_rate', 0.1))
        self.od_rate = float(self.cfg.get('sampling', {}).get('od_rate', 0.0))
        self.volume_year = self.cfg.get('data', {}).get('volume_year', 2022)

        # Reproducibilidad
        self.seed = int(self.cfg.get('data', {}).get('random_seed', 42))
        # Inyectar seed en params de estrategia si no existe
        self.strategy_params.setdefault('random_seed', self.seed)
        np.random.seed(self.seed)

        # Output
        self.output_dir = Path(self.cfg.get('sampling', {}).get('output_dir', 'outputs/masks'))
        self.output_filename = self.cfg.get('sampling', {}).get('output_filename', 'sampled_flow_mask.npy')

    # ----------------------------
    # Internal Helpers
    # ----------------------------

    def _load_graph_from_disk(self) -> nx.DiGraph:
        """Carga el grafo desde disco usando pickle (para modo standalone)."""
        if not self.graph_route or not self.graph_route.exists():
            raise FileNotFoundError(f"Graph file not found at: {self.graph_route}")

        logger.info(f"Cargando grafo desde disco: {self.graph_route}")
        with open(self.graph_route, 'rb') as f:
            G = pickle.load(f)

        if not isinstance(G, (nx.Graph, nx.DiGraph)):
            raise TypeError("El archivo cargado no es un NetworkX Graph/DiGraph")
        return G

    def _compute_observed_mask(self, graph: nx.DiGraph) -> np.ndarray:
        """
        Calcula qué links tienen datos reales usando las utilidades centrales.
        Reemplaza la lógica antigua de 'adivinar columnas'.
        """
        logger.info(f"Calculando máscara observada para año: {self.volume_year}")

        # Usamos la utilidad centralizada para extraer datos
        df = utils.extract_graph_data(graph, self.volume_year)

        # Validar si tenemos datos
        if df['flow'].isnull().all():
            logger.warning(f"¡ATENCIÓN! No se encontraron flujos válidos para el año {self.volume_year}.")
            logger.warning("Verifique 'volume_year' en config o los atributos del grafo.")
            # Retornamos máscara vacía o error según preferencia. Aquí warn y ceros.
            return np.zeros(len(df), dtype=np.float32)

        # Máscara: 1.0 donde hay dato (no es NaN), 0.0 donde es NaN
        mask = (~df['flow'].isna()).astype(np.float32).values
        return mask

    def _load_strategy_instance(self) -> BaseSampler:
        """Instancia dinámica de la estrategia (Reflection)."""
        spec = self.strategy_name

        # Resolver alias desde config (ej: 'spatial_LHS' -> 'src...:SpatialLHSStrategy')
        if spec in self.strategy_map:
            spec = self.strategy_map[spec]

        logger.info(f"Instanciando estrategia: {spec}")

        # Caso A: 'module.path:ClassName'
        if ':' in spec:
            module_path, class_name = spec.split(':', 1)
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"No se pudo cargar la estrategia {spec}: {e}")

            return cls(**self.strategy_params)

        # Caso B: Ruta a archivo .py
        p = Path(spec)
        if p.suffix == '.py' and p.exists():
            spec_mod = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec_mod)
            sys.modules[p.stem] = mod  # Registrar en sys.modules
            spec_mod.loader.exec_module(mod)

            # Buscar subclase de BaseSampler
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, BaseSampler) and obj is not BaseSampler:
                    return obj(**self.strategy_params)

            raise RuntimeError(f"No se encontró subclase de BaseSampler en {p}")

        # Fallback: Intentar importar como módulo directo
        try:
            module = importlib.import_module(spec)
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, BaseSampler) and obj is not BaseSampler:
                    return obj(**self.strategy_params)
        except ImportError:
            pass

        raise RuntimeError(f"No se pudo resolver la estrategia: {self.strategy_name}")

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self,
            override_graph: Optional[nx.DiGraph] = None,
            override_observed_flow_mask: Optional[np.ndarray] = None,
            override_observed_od_mask: Optional[np.ndarray] = None,
            save_output: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta el proceso de sampling.

        Args:
            override_graph: Grafo inyectado desde el pipeline. Si es None, carga desde disco.
            override_observed_flow_mask: Máscara de datos disponibles. Si es None, se calcula del grafo.
            save_output: Si True, guarda los resultados en self.output_dir.
        """

        # 1. Obtener Grafo
        if override_graph is not None:
            G = override_graph
        else:
            G = self._load_graph_from_disk()

        # 2. Obtener Máscara de Observados (Universo Válido)
        if override_observed_flow_mask is not None:
            observed_flow_mask = override_observed_flow_mask
        else:
            # Ahora calculamos esto dinámicamente en lugar de devolver np.ones() ciegamente
            observed_flow_mask = self._compute_observed_mask(G)

        # OD Mask default
        if override_observed_od_mask is not None:
            observed_od_mask = override_observed_od_mask
        else:
            observed_od_mask = np.zeros(1, dtype=np.float32)  # Placeholder si no hay OD

        # 3. Ejecutar Estrategia
        strategy = self._load_strategy_instance()

        logger.info(f"--- Sampling Engine Start ---")
        logger.info(f"Strategy: {strategy.__class__.__name__}")
        logger.info(f"Universe (Observed Links): {int(observed_flow_mask.sum())}")
        logger.info(f"Target Flow Rate: {self.flow_rate}")

        train_flow_mask, train_od_mask = strategy.create_partial_data_masks(
            train_flow_mask=observed_flow_mask,
            od_mask=observed_od_mask,
            flow_rate=self.flow_rate,
            od_rate=self.od_rate,
            graph=G,
            volume_year=self.volume_year
        )

        logger.info(f"Sampling Complete. Train Links: {int(train_flow_mask.sum())}")

        # 4. Guardado Opcional (Útil para CLI / Debugging)
        if save_output:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / self.output_filename
            np.save(out_path, train_flow_mask)
            logger.info(f"Máscara guardada en: {out_path}")

        return train_flow_mask, train_od_mask


# -----------------------------
# Entrypoints (CLI & Hydra)
# -----------------------------

@hydra.main(version_base=None, config_path="../../../configs", config_name="sampling")
def hydra_entry(cfg: DictConfig):
    """Entrypoint para ejecución con Hydra."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    engine = SamplingEngine(config=cfg)
    # En modo CLI/Hydra standalone, generalmente queremos guardar el resultado
    engine.run(save_output=True)


def main():
    """Entrypoint CLI simple (argparse)"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sampling/sampling.yaml')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    engine = SamplingEngine(config_path=args.config)
    engine.run(save_output=True)


if __name__ == '__main__':
    main()