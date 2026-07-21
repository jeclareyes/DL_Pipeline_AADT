import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class BaseVDF(nn.Module):
    """
    Abstract base class for all Volume-Delay Functions (VDFs).
    Ensures all VDFs have the same interface for the Assignments Layer.
    """

    def __init__(self, t0: torch.Tensor, capacity: torch.Tensor,
                 num_link_groups: int, link_group: torch.Tensor):
        super().__init__()
        # Buffers are not learnable parameters
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))
        self.num_link_groups = num_link_groups

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Each VDF must implement its own forward.")

    def get_regularization_loss(self) -> torch.Tensor:
        """Returns the specific regularization loss for this VDF's parameters."""
        return torch.tensor(0.0, device=self.t0.device)


# =============================================================================
# Concrete Implementations
# =============================================================================

class BPRVDF(BaseVDF):
    """Standard Bureau of Public Roads (BPR) function."""

    def __init__(self, t0, capacity, num_link_groups, link_group,
                 alpha_init=0.15, beta_init=4.0):
        super().__init__(t0, capacity, num_link_groups, link_group)
        # Initialization with standard BPR values
        self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), float(alpha_init)))
        self.beta_raw = nn.Parameter(torch.full((num_link_groups,), float(beta_init)))
        self.alpha_ref = alpha_init
        self.beta_ref = beta_init

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        # Softplus to ensure positivity, clamping for numerical stability
        alpha = torch.clamp(F.softplus(self.alpha_raw), min=0.01, max=5.0)
        # Beta is usually >= 1.0
        beta = torch.clamp(1.0 + F.softplus(self.beta_raw), min=1.0, max=15.0)
        alpha_links = alpha[self.link_group]
        beta_links = beta[self.link_group]

        # Flow ratio with clipping to prevent gradient explosion in extreme congestion
        flow_ratio = torch.clamp(link_flows / (self.capacity + 1e-9), max=10.0)

        return self.t0 * (1 + alpha_links * torch.pow(flow_ratio, beta_links))

    def get_regularization_loss(self) -> torch.Tensor:
        # Regularizes towards the reference values (physical priors)
        alpha = F.softplus(self.alpha_raw)
        beta = 1.0 + F.softplus(self.beta_raw)
        return (torch.norm(alpha - self.alpha_ref, p=2) +
                torch.norm(beta - self.beta_ref, p=2))


class ConicalVDF(BaseVDF):
    """
    t = t0 * (2 + sqrt(alpha^2 * (1-x)^2 + beta) - alpha*(1-x) - beta).
    """

    def __init__(self, t0, capacity, num_link_groups, link_group, alpha_init=4.0):
        super().__init__(t0, capacity, num_link_groups, link_group)
        self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), float(alpha_init)))
        # Beta is sometimes used as a smoothing factor, we leave it fixed or small learnable
        self.beta = 0.01  # Stability factor

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(F.softplus(self.alpha_raw), min=1.0, max=10.0)
        alpha_links = alpha[self.link_group]

        x = link_flows / (self.capacity + 1e-9)

        term1 = alpha_links ** 2 * (1 - x) ** 2 + self.beta
        # We use a small epsilon inside sqrt to avoid NaN gradients if term1 approaches 0
        sqrt_term = torch.sqrt(torch.clamp(term1, min=1e-9))

        cost_factor = 2 + sqrt_term - alpha_links * (1 - x) - self.beta
        # Usual conical normalization so that when x=0, t=t0
        # The standard formula sometimes requires adjustment. This is a common variant.
        # Ensure the factor is at least 1.0
        cost_factor = torch.clamp(cost_factor, min=1.0)

        return self.t0 * cost_factor

    def get_regularization_loss(self) -> torch.Tensor:
        return torch.norm(F.softplus(self.alpha_raw) - 4.0, p=2) * 0.1


class AkcelikVDF(BaseVDF):
    """
    Akçelik function (based on time-dependent queuing theory).
    t = t0 + 0.25*T [ (x-1) + sqrt( (x-1)^2 + 8*J*(x)/(C*T) ) ]
    T: Analysis period (e.g., 1 hour).
    J: Delay parameter (learnable).
    """

    def __init__(self, t0, capacity, num_link_groups, link_group,
                 T_hours=1.0, J_init=0.1):
        super().__init__(t0, capacity, num_link_groups, link_group)
        self.T = T_hours
        self.J_raw = nn.Parameter(torch.full((num_link_groups,), float(J_init)))

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        J = torch.clamp(F.softplus(self.J_raw), min=1e-4, max=1.0)
        J_links = J[self.link_group]

        x = link_flows / (self.capacity + 1e-9)

        term_inside_sqrt = (x - 1) ** 2 + (8 * J_links * x) / (self.capacity * self.T + 1e-9)
        sqrt_term = torch.sqrt(torch.clamp(term_inside_sqrt, min=1e-9))

        overflow_term = 0.25 * self.T * ((x - 1) + sqrt_term)

        # Akcelik is t0 + delay due to overcapacity.
        # Ensure it is not less than t0.
        return self.t0 + torch.clamp(overflow_term, min=0.0)

    def get_regularization_loss(self) -> torch.Tensor:
        return torch.norm(F.softplus(self.J_raw) - 0.1, p=2)


class LogisticVDF(BaseVDF):
    """
    Logistic function. Useful because it has a natural upper bound,
    preventing infinite travel times that break gradients.
    t = t0 * (1 + alpha * sigmoid(beta * (v/c - gamma)))
    """

    def __init__(self, t0, capacity, num_link_groups, link_group,
                 alpha_init=10.0, beta_init=6.0, gamma_init=1.0):
        super().__init__(t0, capacity, num_link_groups, link_group)
        self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), float(alpha_init)))
        self.beta_raw = nn.Parameter(torch.full((num_link_groups,), float(beta_init)))
        # Gamma is usually 1.0 (the inflection point is capacity)
        self.register_buffer('gamma', torch.tensor(float(gamma_init)))

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.alpha_raw)  # Extra maximum relative delay
        beta = F.softplus(self.beta_raw)  # Slope of the S-curve

        alpha_l = alpha[self.link_group]
        beta_l = beta[self.link_group]

        x = link_flows / (self.capacity + 1e-9)

        sigmoid_term = torch.sigmoid(beta_l * (x - self.gamma))
        return self.t0 * (1 + alpha_l * sigmoid_term)


# =============================================================================
# Factory for ease of use
# =============================================================================
VDF_REGISTRY = {
    "bpr": BPRVDF,
    "conical": ConicalVDF,
    # "akcelik": AkcelikVDF,
    "logistic": LogisticVDF
    # "davidson": DavidsonVDF (Can be added if a numerically stable form is found)
}


def get_vdf(name: str, t0, capacity, num_link_groups, link_group, **kwargs) -> BaseVDF:
    """Factory function to instantiate a VDF by name."""
    name = name.lower()
    if name not in VDF_REGISTRY:
        raise ValueError(f"VDF '{name}' not implemented. Options: {list(VDF_REGISTRY.keys())}")
    return VDF_REGISTRY[name](t0, capacity, num_link_groups, link_group, **kwargs)