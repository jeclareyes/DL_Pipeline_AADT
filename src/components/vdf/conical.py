import torch
import torch.nn as nn
import torch.nn.functional as F
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