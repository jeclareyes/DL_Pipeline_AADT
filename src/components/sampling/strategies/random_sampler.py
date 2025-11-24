from typing import Optional
import numpy as np
import networkx as nx

from src.components.sampling.base import BaseSampler
from src.components.sampling import sampling as _core_sampling


class RandomStrategy(BaseSampler):
    def create_partial_data_masks(self, train_flow_mask, od_mask, flow_rate, od_rate, graph, volume_year=None):
        # pass strategy-specific defaults via self.params if needed
        seed = self.params.get("random_seed", 42)
        return _core_sampling.create_partial_data_masks(
            train_flow_mask=train_flow_mask,
            od_mask=od_mask,
            flow_rate=flow_rate,
            od_rate=od_rate,
            graph=graph,
            volume_year=volume_year,
            random_seed=seed,
            strategy="random",
            sampling_basis=self.params.get("sampling_basis", "link_wise_based"),
        )