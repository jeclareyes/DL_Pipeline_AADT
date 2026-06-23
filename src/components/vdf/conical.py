import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .base import BaseCostFunction


class ConicalCostFunction(BaseCostFunction):
    """
    Función Cónica de Spiess.
    Evita el crecimiento exponencial infinito de la BPR.
    """

    def __init__(self, t0, capacity, num_link_groups, link_group, learnable_params=True):
        super().__init__()
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))
        self.learnable_params = learnable_params

        # Beta suele ser un factor de estabilidad pequeño o 1/(2*alpha - 1)
        self.beta_val = 0.01

        if learnable_params:
            # Alpha > 1. Típicamente entre 4 y 10.
            self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), 4.0))
        else:
            self.register_buffer('alpha_raw', torch.full((num_link_groups,), 4.0))

    def get_alpha(self):
        if self.learnable_params:
            return torch.clamp(F.softplus(self.alpha_raw), min=1.01, max=15.0)
        return self.alpha_raw

    def forward(self, link_flows):
        alpha = self.get_alpha()
        alpha_links = alpha[self.link_group]

        x = link_flows / (self.capacity + 1e-9)

        # t = t0 * (2 + sqrt(alpha^2 * (1-x)^2 + beta^2) - alpha * (1-x) - beta)
        # Nota: Existen variantes de esta fórmula. Esta es la clásica de Spiess normalizada.

        term1 = (alpha_links ** 2) * ((1 - x) ** 2) + self.beta_val ** 2
        sqrt_term = torch.sqrt(torch.clamp(term1, min=1e-9))

        factor = 2 + sqrt_term - alpha_links * (1 - x) - self.beta_val

        # Aseguramos que el factor sea al menos 1.0 (no viajar más rápido que t0)
        return self.t0 * torch.clamp(factor, min=1.0)

    @classmethod
    def get_required_columns(cls) -> list[str]:
        return ["free_flow_time_col", "capacity_col", "alpha_col"]

    @classmethod
    def evaluate_costs_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> np.ndarray:
        fft_col = kwargs.get("free_flow_time_col", "free_flow_time")
        cap_col = kwargs.get("capacity_col", "capacity")
        alpha_col = kwargs.get("alpha_col", "alpha")
        toll_col = kwargs.get("toll_col", None)

        free_flow_time = link_table[fft_col].to_numpy(dtype=float)
        capacity = link_table[cap_col].to_numpy(dtype=float)
        alpha = link_table[alpha_col].to_numpy(dtype=float)
        
        if toll_col and toll_col in link_table.columns:
            toll = link_table[toll_col].to_numpy(dtype=float)
        else:
            toll = np.zeros(len(link_table), dtype=float)

        beta_val = 0.01
        x = link_flows / capacity

        term1 = (alpha ** 2) * ((1 - x) ** 2) + beta_val ** 2
        sqrt_term = np.sqrt(np.maximum(term1, 1e-9))
        factor = 2 + sqrt_term - alpha * (1 - x) - beta_val

        costs = free_flow_time * np.maximum(factor, 1.0) + toll
        return costs

    @classmethod
    def evaluate_beckmann_integral_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> np.ndarray:
        raise NotImplementedError("La integral de Beckmann para la función Cónica no ha sido implementada.")