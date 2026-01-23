import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms.flow import capacity_scaling

from .base import BaseCostFunction  # Asegúrate de tener esta base

class BPRCostFunction(BaseCostFunction):
    """
    Función de costo BPR modularizada.
    """
    def __init__(self, t0, capacity, num_link_groups, link_group, lanes, learnable_params=True):
        super().__init__()
        # Registramos buffers (no se entrenan, son datos)
        self.register_buffer('lanes', lanes)
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))
        self.learnable_params = learnable_params

        if learnable_params:
            # Parámetros aprendibles por grupo
            self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), 0.15))
            self.beta_raw = nn.Parameter(torch.full((num_link_groups,), 4.0))
        else:
            self.register_buffer('alpha_raw', torch.full((num_link_groups,), 0.15))
            self.register_buffer('beta_raw', torch.full((num_link_groups,), 4.0))

    def get_alpha(self):
        if self.learnable_params:
            return torch.clamp(F.softplus(self.alpha_raw), min=0.01, max=2.0)
        return self.alpha_raw

    def get_beta(self):
        if self.learnable_params:
            return torch.clamp(1.0 + F.softplus(self.beta_raw), min=1.1, max=10.0)
        return self.beta_raw

    def forward(self, link_flows):
        alpha = self.get_alpha()
        beta = self.get_beta()

        alpha_links = alpha[self.link_group]
        beta_links = beta[self.link_group]

        capacity_factor = 16 # Factor de escala para la capacidad según número de horas efectivas

        # Evitar divisiones por cero y explosiones numéricas
        flow_ratio = torch.clamp(link_flows / ((self.capacity * capacity_factor) * self.lanes  + 1e-9), max=5.0)

        # Verificamos si existe al menos un valor mayor a 1.0 en el tensor
        # TODO uncomment
        """if (flow_ratio > 1.0).any():
            logging.warning(
                f"flow_ratio excede el límite en algunos enlaces. Máximo actual: {flow_ratio.max().item():.4f}")"""

        cost = self.t0 * (1 + alpha_links * flow_ratio ** beta_links)
        return cost