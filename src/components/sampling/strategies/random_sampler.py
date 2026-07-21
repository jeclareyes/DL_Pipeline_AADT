import numpy as np
import networkx as nx
import logging
from typing import Optional

from src.components.sampling.base import BaseSampler
from src.components.sampling import sampling as utils

# Logger específico para esta estrategia
logger = logging.getLogger(__name__)


class RandomStrategy(BaseSampler):
    def create_partial_data_masks(
            self,
            train_flow_mask: np.ndarray,
            od_mask: np.ndarray,
            flow_rate: float,
            od_rate: float,
            graph: nx.Graph,
            volume_year: Optional[int] = None
    ):
        seed = self.params.get("random_seed", 42)
        basis = self.params.get("sampling_basis", "link_wise_based")

        # 1. Preparar datos
        internal_df = utils.extract_graph_data(graph, volume_year)

        # 2. Obtener grupos (universo de muestreo)
        # Aquí ocurre la reducción de "Links Individuales" a "Sensores Lógicos"
        groups = utils.get_observation_groups(train_flow_mask, basis, internal_df)

        # --- LOGGING DE AGRUPACIÓN ---
        n_raw_links = int(train_flow_mask.sum())
        n_groups = len(groups)

        if basis == "traffic_counts_based":
            logger.info(f"[Grouping Analysis] Basis: {basis}")
            logger.info(f"   - Lecturas originales (Links): {n_raw_links}")
            logger.info(f"   - Agrupadas en (Sensores/Grupos): {n_groups}")
            if n_groups < n_raw_links:
                logger.info(f"   - Reducción de redundancia: {100 * (1 - n_groups / n_raw_links):.1f}%")
        else:
            logger.info(f"[Grouping Analysis] Basis: {basis} (No grouping applied)")

        # 3. Sampling Logic
        num_samples = int(n_groups * flow_rate)
        rng = np.random.default_rng(seed)

        if n_groups > 0 and num_samples > 0:
            selected_indices = rng.choice(len(groups), size=num_samples, replace=False)
            selected_groups = [groups[i] for i in selected_indices]
        else:
            selected_groups = []

        # 4. Construir máscara final
        sampled_flow_mask = utils.build_mask_from_groups(train_flow_mask, selected_groups)

        # --- LOGGING FINAL ---
        n_final_links = int(sampled_flow_mask.sum())
        logger.info(f"✅ [Sampling Result]")
        logger.info(f"   - Grupos seleccionados para Train: {len(selected_groups)} / {n_groups}")
        logger.info(f"   - Links finales en Train Mask: {n_final_links} / {n_raw_links}")

        # 5. Sampling OD
        sampled_od_mask = utils.sample_od_simple(od_mask, od_rate, seed)

        return sampled_flow_mask, sampled_od_mask