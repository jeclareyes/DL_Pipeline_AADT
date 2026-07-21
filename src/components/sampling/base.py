# File: src/components/sampling/base.py
import importlib
from typing import Any, Dict, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx

# -------------------------
# Strategy interface
# -------------------------
class BaseSampler(ABC):
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def create_partial_data_masks(
        self,
        train_flow_mask: np.ndarray,
        od_mask: np.ndarray,
        flow_rate: float,
        od_rate: float,
        graph: nx.Graph,
        volume_year: Optional[int] = None
    ) -> tuple:
        """Return (sampled_flow_mask, sampled_od_mask)"""

