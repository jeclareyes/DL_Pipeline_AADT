import torch
import torch.nn as nn
import torch.nn.functional as F
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