import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseCostFunction


class BPRModifiedCostFunction(BaseCostFunction):
    """
    BPR Modificada (Linealizada).
    Se comporta como BPR normal si V/C <= 1.
    Si V/C > 1, proyecta una tangente lineal para evitar explosión exponencial.
    """

    def __init__(self, t0, capacity, num_link_groups, link_group, learnable_params=True):
        super().__init__()
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))
        self.learnable_params = learnable_params

        if learnable_params:
            self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), 0.15))
            self.beta_raw = nn.Parameter(torch.full((num_link_groups,), 4.0))
        else:
            self.register_buffer('alpha_raw', torch.full((num_link_groups,), 0.15))
            self.register_buffer('beta_raw', torch.full((num_link_groups,), 4.0))

    def get_alpha_beta(self):
        if self.learnable_params:
            a = torch.clamp(F.softplus(self.alpha_raw), min=0.01, max=5.0)
            b = torch.clamp(1.0 + F.softplus(self.beta_raw), min=1.1, max=15.0)
            return a, b
        return self.alpha_raw, self.beta_raw

    def forward(self, link_flows):
        alpha, beta = self.get_alpha_beta()
        alpha_l = alpha[self.link_group]
        beta_l = beta[self.link_group]

        x = link_flows / (self.capacity + 1e-9)

        # Parte 1: Régimen normal (x <= 1)
        # t = t0 * (1 + alpha * x^beta)
        cost_normal = self.t0 * (1 + alpha_l * torch.pow(x, beta_l))

        # Parte 2: Régimen sobresaturado (x > 1) -> Linealización
        # Pendiente en x=1 es t0 * alpha * beta
        # Valor en x=1 es t0 * (1 + alpha)
        # t_linear = t(1) + slope * (x - 1)
        cost_congested = self.t0 * (1 + alpha_l) + (self.t0 * alpha_l * beta_l) * (x - 1)

        # Combinar usando torch.where (diferenciable)
        return torch.where(x <= 1.0, cost_normal, cost_congested)