import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .base import BaseCostFunction


class AkcelikCostFunction(BaseCostFunction):
    """
    Función de costo de Akçelik basada en teoría de colas.
    Fórmula: t = t0 + 0.25*T [ (x-1) + sqrt( (x-1)^2 + 8*J*x / (C*T) ) ]
    """

    def __init__(self, t0, capacity, num_link_groups, link_group,
                 T=1.0, learnable_params=True):
        super().__init__()
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))
        self.register_buffer('T', torch.tensor(float(T)))  # Periodo de análisis (horas)

        self.learnable_params = learnable_params

        # J es el parámetro de demora. Valores típicos 0.1 - 1.5
        if learnable_params:
            self.J_raw = nn.Parameter(torch.full((num_link_groups,), 0.4))
        else:
            self.register_buffer('J_raw', torch.full((num_link_groups,), 0.4))

    def get_J(self):
        if self.learnable_params:
            # Softplus para asegurar J > 0
            return torch.clamp(F.softplus(self.J_raw), min=1e-4, max=2.0)
        return self.J_raw

    def forward(self, link_flows):
        J = self.get_J()
        J_links = J[self.link_group]

        # x = V/C
        x = link_flows / (self.capacity + 1e-9)

        # Término dentro de la raíz: (x-1)^2 + (8*J*x)/(C*T)
        term_sqrt = (x - 1) ** 2 + (8 * J_links * x) / (self.capacity * self.T + 1e-9)

        # Demora por sobrecapacidad (Overflow delay)
        # Usamos clamp min=0 en la raíz por estabilidad numérica
        d_overflow = 0.25 * self.T * ((x - 1) + torch.sqrt(torch.clamp(term_sqrt, min=1e-9)))

        # Akcelik es t0 + demora. Aseguramos que no sea menor que t0.
        return self.t0 + torch.clamp(d_overflow, min=0.0)

    @classmethod
    def get_required_columns(cls) -> list[str]:
        return ["free_flow_time_col", "capacity_col", "J_col"]

    @classmethod
    def evaluate_costs_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> np.ndarray:
        fft_col = kwargs.get("free_flow_time_col", "free_flow_time")
        cap_col = kwargs.get("capacity_col", "capacity")
        J_col = kwargs.get("J_col", "J")
        toll_col = kwargs.get("toll_col", None)
        T = float(kwargs.get("T", 1.0))

        free_flow_time = link_table[fft_col].to_numpy(dtype=float)
        capacity = link_table[cap_col].to_numpy(dtype=float)
        
        if J_col in link_table.columns:
            J = link_table[J_col].to_numpy(dtype=float)
        else:
            # Fallback a un valor por defecto si no existe la columna
            J = np.full(len(link_table), 0.4, dtype=float)
            
        if toll_col and toll_col in link_table.columns:
            toll = link_table[toll_col].to_numpy(dtype=float)
        else:
            toll = np.zeros(len(link_table), dtype=float)

        x = link_flows / capacity
        term_sqrt = (x - 1) ** 2 + (8 * J * x) / (capacity * T + 1e-9)
        d_overflow = 0.25 * T * ((x - 1) + np.sqrt(np.maximum(term_sqrt, 1e-9)))

        costs = free_flow_time + np.maximum(d_overflow, 0.0) + toll
        return costs

    @classmethod
    def evaluate_beckmann_integral_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> np.ndarray:
        raise NotImplementedError("La integral de Beckmann para la función de Akcelik no ha sido implementada.")