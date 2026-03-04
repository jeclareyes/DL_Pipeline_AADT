import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms.flow import capacity_scaling

from .base import BaseCostFunction  # Asegúrate de tener esta base

class BPRCostFunction_old(BaseCostFunction):
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

        capacity_factor = 1 # Factor de escala para la capacidad según número de horas efectivas

        # Evitar divisiones por cero y explosiones numéricas
        flow_ratio = torch.clamp(link_flows / ((self.capacity * capacity_factor) * self.lanes  + 1e-9), max=5.0)

        # Verificamos si existe al menos un valor mayor a 1.0 en el tensor
        # TODO uncomment
        """if (flow_ratio > 1.0).any():
            logging.warning(
                f"flow_ratio excede el límite en algunos enlaces. Máximo actual: {flow_ratio.max().item():.4f}")"""

        cost = self.t0 * (1 + alpha_links * flow_ratio ** beta_links)
        return cost
    

class BPRCostFunction(nn.Module):
    """
    Differentiable BPR Function.
    Pragmatic modifications:
    1. Safe capacity clamping to prevent division by zero (Logical Connectors).
    2. V/C clamping to prevent gradient explosion early in training.
    """
    def __init__(self, t0: torch.Tensor, capacity: torch.Tensor, lanes: torch.Tensor, 
                 num_link_groups: int, link_group: torch.Tensor, 
                 learn_alpha: bool = True, learn_beta: bool = True, 
                 alpha_init: float = 0.15, beta_init: float = 4.0, **kwargs):
        super().__init__()
        
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        
        # Pragmatic initialization: using global parameters. 
        # (Could be expanded to group-specific parameters using num_link_groups)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)), requires_grad=learn_alpha)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=learn_beta)

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        """
        Calculates dynamic link costs.
        link_flows: [Batch, Num_Links]
        """
        # 1. Handle Logical Connectors (capacity == 0)
        safe_capacity = torch.where(
            self.capacity > 0, 
            self.capacity, 
            torch.tensor(1e9, device=self.capacity.device)
        )
        
        # 2. Compute Volume/Capacity ratio
        v_c = link_flows / safe_capacity
        
        # 3. Prevent Gradient Explosion (Crucial for early epochs)
        v_c = torch.clamp(v_c, max=5.0)
        
        # 4. Mask connectors out of congestion calculation
        is_regular_link = (self.capacity > 0).float()
        
        # 5. BPR Formula
        delay = self.alpha * (v_c ** self.beta) * is_regular_link
        link_costs = self.t0 * (1.0 + delay)
        
        return link_costs