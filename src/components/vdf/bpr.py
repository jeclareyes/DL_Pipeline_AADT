import torch
import torch.nn as nn

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
                 alpha_init: float = 0.15, beta_init: float = 4.0,
                 effetive_capacity_multiplier: float = 16.0, **kwargs
                 ):
        super().__init__()
        
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        
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