import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.components.vdf.base import BaseCostFunction

class BPRCostFunction(BaseCostFunction):
    """
    Differentiable BPR Function.
    Pragmatic modifications:
    1. Safe capacity clamping to prevent division by zero (Logical Connectors).
    2. V/C clamping to prevent gradient explosion early in training.
    """
    def __init__(self, t0: torch.Tensor, capacity: torch.Tensor, lanes: torch.Tensor, 
                 num_link_groups: int, link_group: torch.Tensor, 
                 learn_alpha: bool = True, learn_beta: bool = True, 
                 alpha_init: float = 0.15, beta_init: float = 4.0,
                 effetive_capacity_multiplier: float = 16.0, **kwargs
                 ):
        super().__init__()
        
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('lanes', lanes)
        self.effetive_capacity_multiplier = effetive_capacity_multiplier
        
        # Pragmatic initialization: using global parameters. 
        # (Could be expanded to group-specific parameters using num_link_groups)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)), requires_grad=learn_alpha)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=learn_beta)

        # Keep calibrated VDF parameters inside physically meaningful ranges.
        self.alpha_min = float(kwargs.get("alpha_min", 0.1))
        self.alpha_max = float(kwargs.get("alpha_max", 0.5))
        self.beta_min = float(kwargs.get("beta_min", 1.0))
        self.beta_max = float(kwargs.get("beta_max", 6.0))

        # Optional per-parameter gradient clipping for stable calibration.
        self.alpha_grad_clip = float(kwargs.get("alpha_grad_clip", 1e3))
        self.beta_grad_clip = float(kwargs.get("beta_grad_clip", 1e3))

        if learn_alpha and self.alpha_grad_clip > 0:
            self.alpha.register_hook(lambda g: torch.clamp(g, -self.alpha_grad_clip, self.alpha_grad_clip))
        if learn_beta and self.beta_grad_clip > 0:
            self.beta.register_hook(lambda g: torch.clamp(g, -self.beta_grad_clip, self.beta_grad_clip))

    def _bounded_alpha(self) -> torch.Tensor:
        return torch.clamp(self.alpha, min=self.alpha_min, max=self.alpha_max)

    def _bounded_beta(self) -> torch.Tensor:
        return torch.clamp(self.beta, min=self.beta_min, max=self.beta_max)

    def get_alpha(self) -> torch.Tensor:
        return self._bounded_alpha().detach()

    def get_beta(self) -> torch.Tensor:
        return self._bounded_beta().detach()

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
        ) * self.lanes * self.effetive_capacity_multiplier
        
        # 2. Compute Volume/Capacity ratio
        v_c = link_flows / safe_capacity
        
        # 3. Prevent Gradient Explosion (Crucial for early epochs)
        v_c = torch.clamp(v_c, max=5.0)
        
        # 4. Mask connectors out of congestion calculation
        is_regular_link = (self.capacity > 0).float()

        alpha_eff = self._bounded_alpha()
        beta_eff = self._bounded_beta()
        
        # 5. BPR Formula
        delay = alpha_eff * (v_c ** beta_eff) * is_regular_link
        link_costs = self.t0 * (1.0 + delay)
        
        return link_costs

    @classmethod
    def get_required_columns(cls) -> list[str]:
        """
        Retorna las columnas requeridas (nombres genéricos) para BPR.
        """
        return ["free_flow_time_col", "capacity_col", "alpha_col", "beta_col"]

    @classmethod
    def evaluate_costs_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Calcula BPR costs estáticos en NumPy.
        Mapeo de columnas a través de kwargs o nombres por defecto.
        """
        # Extraer nombres de columna (con defaults por si acaso)
        free_flow_time_col = kwargs.get("free_flow_time_col", "free_flow_time")
        capacity_col = kwargs.get("capacity_col", "capacity")
        alpha_col = kwargs.get("alpha_col", "alpha")
        beta_col = kwargs.get("beta_col", "beta")
        toll_col = kwargs.get("toll_col", None)

        free_flow_time = link_table[free_flow_time_col].to_numpy(dtype=float)
        capacity = link_table[capacity_col].to_numpy(dtype=float)
        alpha = link_table[alpha_col].to_numpy(dtype=float)
        beta = link_table[beta_col].to_numpy(dtype=float)

        toll = np.zeros_like(free_flow_time)
        if toll_col is not None and toll_col in link_table.columns:
            toll = link_table[toll_col].to_numpy(dtype=float)

        costs = free_flow_time * (1.0 + alpha * (link_flows / capacity) ** beta) + toll
        return costs

    @classmethod
    def evaluate_beckmann_integral_numpy(cls, link_flows: np.ndarray, link_table: pd.DataFrame, **kwargs) -> float:
        """
        Calcula la integral de Beckmann para BPR en NumPy.
        Integral de BPR: t0 * w + t0 * alpha * (C / (beta + 1)) * (w / C)^(beta + 1)
        """
        free_flow_time_col = kwargs.get("free_flow_time_col", "free_flow_time")
        capacity_col = kwargs.get("capacity_col", "capacity")
        alpha_col = kwargs.get("alpha_col", "alpha")
        beta_col = kwargs.get("beta_col", "beta")
        toll_col = kwargs.get("toll_col", None)

        free_flow_time = link_table[free_flow_time_col].to_numpy(dtype=float)
        capacity = link_table[capacity_col].to_numpy(dtype=float)
        alpha = link_table[alpha_col].to_numpy(dtype=float)
        beta = link_table[beta_col].to_numpy(dtype=float)

        toll = np.zeros_like(free_flow_time)
        if toll_col is not None and toll_col in link_table.columns:
            toll = link_table[toll_col].to_numpy(dtype=float)

        v_c = link_flows / capacity
        
        # Integral = sum_a ( toll_a * x_a + t0_a * x_a + t0_a * alpha_a * (capacity_a / (beta_a + 1)) * (x_a / capacity_a)^(beta_a + 1) )
        integral_link = toll * link_flows + free_flow_time * link_flows + \
                        free_flow_time * alpha * (capacity / (beta + 1.0)) * (v_c ** (beta + 1.0))

        return float(np.sum(integral_link))