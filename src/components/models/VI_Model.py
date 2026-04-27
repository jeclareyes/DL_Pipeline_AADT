"""
Variational-Inference traffic assignment model with physics-aware congestion dynamics.

This module contains the full Stage-1 / IMD-ready pipeline used in the project:

1) `PhysicsInformedBPRNet` learns bounded BPR parameters from static link attributes.
2) `ImplicitEquilibriumLayer` solves a user-equilibrium fixed point with entropy mirror descent.
3) `VariationalInferenceModel` couples learned OD demand with the equilibrium solver.
4) `Loss` combines flow fitting, OD supervision, and unknown-OD regularization.
5) `VIDiagnostician` collects optimization and physics diagnostics during training.

The implementation emphasizes physically plausible congestion behavior (positive
capacity, bounded alpha/beta), stable gradients, and clear telemetry for solver
convergence and forward/backward consistency in IMD mode.
"""

import json
import os
import csv
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.runtime_contracts import ArtifactSchemaError, require_keys

class PhysicsInformedBPRNet(nn.Module):
    """
    Supply module that predicts physically bounded BPR parameters.

    Instead of directly regressing travel times from flows, this network predicts
    BPR controls and keeps the analytic BPR structure in the solver.

    Inputs:
        link_features: [num_links, num_features] static normalized attributes.

    Learned outputs:
        - group-level alpha in [alpha_min, alpha_max]
        - group-level beta in [beta_min, beta_max]
        - link-level capacity multiplier in [cap_mult_min, cap_mult_max]

    Why this design:
        - keeps monotone congestion response
        - improves interpretability (alpha/beta/capacity multiplier)
        - avoids unstable unconstrained parameters
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_link_groups: int = 1,
        link_group: Optional[torch.Tensor] = None,
        alpha_min: float = 0.1,
        alpha_max: float = 0.5,
        beta_min: float = 1.0,
        beta_max: float = 6.0,
        cap_mode: str = "global",
        cap_min: float = 12.0,
        cap_max: float = 16.0,
        mlp_hidden_dim: int = 64,
    ):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.cap_mode = str(cap_mode)
        self.cap_min = float(cap_min)
        self.cap_max = float(cap_max)
        self.num_link_groups = max(int(num_link_groups), 1)

        if link_group is None:
            link_group = torch.zeros(1, dtype=torch.long)
        self.register_buffer("link_group", link_group.to(dtype=torch.long))

        # One alpha/beta pair per link group.
        self.alpha_group_raw = nn.Parameter(torch.zeros(self.num_link_groups, dtype=torch.float32))
        self.beta_group_raw = nn.Parameter(torch.zeros(self.num_link_groups, dtype=torch.float32))

        # Shallow network with smooth bounded mappings for physical parameters.
        self.param_estimator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

        # Capacity Multiplier Logic based on Mode
        if self.cap_mode == "spatial":
            # MLP for link-by-link inference
            self.capacity_estimator = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 1),
            )
        elif self.cap_mode == "group":
            # Learnable vector: one scalar per link group
            self.cap_group_raw = nn.Parameter(torch.zeros(num_link_groups))
        elif self.cap_mode == "global":
            # Single learnable scalar for the entire network
            self.cap_global_raw = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Invalid capacity mode: {cap_mode}. Choose 'spatial', 'group', or 'global'.")

    def forward(self, link_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map static link features to bounded BPR parameters per link."""
        # Predict per-link capacity multiplier with a smooth bounded map.
        if self.cap_mode == "spatial":
            cap_mult_raw = self.capacity_estimator(link_features).squeeze(-1)
        elif self.cap_mode == "group":
            cap_mult_raw = self.cap_group_raw[self.link_group]
        else:  # global
            cap_mult_raw = self.cap_global_raw.expand(link_features.size(0))

        capacity_multiplier = self.cap_min + (self.cap_max - self.cap_min) * torch.sigmoid(cap_mult_raw)

        # Learn one alpha/beta pair per link group and broadcast to links via indexing.
        alpha_group = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_group_raw)
        beta_group = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_group_raw)

        link_group = self.link_group
        if link_group.numel() != link_features.shape[0]:
            # Fallback for inconsistent metadata: map all links to group 0.
            link_group = torch.zeros(link_features.shape[0], dtype=torch.long, device=link_features.device)
        else:
            link_group = link_group.to(device=link_features.device)

        alpha = alpha_group[link_group]
        beta = beta_group[link_group]

        return {
            'alpha': alpha,
            'beta': beta,
            'capacity_multiplier': capacity_multiplier,
        }


class PathCostAggregator(nn.Module):
    """
    Deterministic topology operator from link costs to path costs.

    Given incidence matrix Delta, computes path costs as Delta * link_costs.
    Supports both a single vector [L] and batched input [B, L].
    """
    def forward(self, link_costs: torch.Tensor, route_link_matrix: torch.Tensor) -> torch.Tensor:
        """Aggregate additive link costs into route/path costs."""
        if link_costs.dim() == 1:
            return torch.sparse.mm(route_link_matrix, link_costs.unsqueeze(1)).squeeze(1)

        # Batched version: [B, L] -> [B, R]
        path_costs_t = torch.sparse.mm(route_link_matrix, link_costs.t())
        return path_costs_t.t()


class IMDEquilibriumFunction(torch.autograd.Function):
    """
    Custom autograd bridge for IMD over the mirror-descent fixed-point map.

    Forward pass:
        runs the equilibrium iterator to a terminal state and stores the
        linearization point.

    Backward pass:
        solves the implicit linear fixed-point equation for the adjoint vector
        with Jacobian-free fixed-point iterations (JFB).
    """

    @staticmethod
    def forward(ctx, layer, route_init, od_demands, alpha, beta, cap_mult):
        """Run equilibrium iterations without graph unrolling and cache IMD state."""
        with torch.no_grad():
            route_flows = route_init
            prev_link_flows = None
            used_iters = layer.max_iterations
            final_gap = 0.0
            converged = False

            params = {
                "alpha": alpha,
                "beta": beta,
                "capacity_multiplier": cap_mult,
            }

            for it in range(1, layer.max_iterations + 1):
                # Evaluate one fixed-point update at current iterate.
                link_flows = layer._route_to_link_flows(route_flows)
                next_route_flows = layer._fixed_point_step(
                    route_flows=route_flows,
                    od_demands=od_demands,
                    bpr_params=params,
                    iter_idx=it,
                )

                if prev_link_flows is not None:
                    rel = torch.norm(link_flows - prev_link_flows, p=2, dim=1) / (
                        torch.norm(prev_link_flows, p=2, dim=1) + layer.eps
                    )
                    final_gap = float(rel.mean().item())
                    if layer.use_early_stop and it >= layer.min_iterations and bool(torch.all(rel < layer.tol_rel_flow)):
                        route_flows = next_route_flows
                        used_iters = it
                        converged = True
                        break

                route_flows = next_route_flows
                prev_link_flows = link_flows
                used_iters = it

            # Freeze the local operator for a few extra steps so IMD backward linearizes
            # around a stationary map consistent with the final forward state.
            linearization_iter = max(1, int(used_iters))
            tail_steps = max(0, int(layer.imd_stationary_tail_steps))
            for _ in range(tail_steps):
                route_flows = layer._fixed_point_step(
                    route_flows=route_flows,
                    od_demands=od_demands,
                    bpr_params=params,
                    iter_idx=linearization_iter,
                )

        layer._imd_last_info = {
            "iterations": float(used_iters),
            "final_gap": float(final_gap),
            "converged": float(1.0 if converged else 0.0),
            "implicit_grad": True,
            "linearization_iter": float(linearization_iter),
        }

        ctx.layer = layer
        ctx.linearization_iter = linearization_iter
        ctx.save_for_backward(route_flows, od_demands, alpha, beta, cap_mult)
        return route_flows

    @staticmethod
    def backward(ctx, grad_output):
        """Compute implicit gradients with JFB, optionally damped via IMD config."""
        layer = ctx.layer
        route_star, od_demands, alpha, beta, cap_mult = ctx.saved_tensors

        with torch.enable_grad():
            route_star = route_star.detach().requires_grad_(True)
            od_demands = od_demands.detach().requires_grad_(True)
            alpha = alpha.detach().requires_grad_(True)
            beta = beta.detach().requires_grad_(True)
            cap_mult = cap_mult.detach().requires_grad_(True)

            params = {
                "alpha": alpha,
                "beta": beta,
                "capacity_multiplier": cap_mult,
            }
            backward_iter_idx = max(1, int(getattr(ctx, "linearization_iter", layer.max_iterations)))
            route_next = layer._fixed_point_step(
                route_flows=route_star,
                od_demands=od_demands,
                bpr_params=params,
                iter_idx=backward_iter_idx,
            )

            # JFB iteration: z_{k+1} = g + J^T z_k.
            # If configured, use damped update for extra stability.
            z = grad_output.clone()
            max_backward_iters = max(int(layer.imd_backward_iters), 1)
            residual_history = []
            iter_count = max_backward_iters
            use_damping = bool(getattr(layer, "imd_use_damping", False))
            damping_factor = 1.0
            if use_damping:
                damping_factor = float(np.clip(layer.imd_damping, 0.0, 1.0))

            for _ in range(max_backward_iters):
                vjp_route = torch.autograd.grad(
                    route_next,
                    route_star,
                    grad_outputs=z,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                rhs = grad_output + vjp_route

                residual_num = torch.norm(z - rhs, p=2)
                residual_den = torch.norm(rhs, p=2) + layer.eps
                residual_val = float((residual_num / residual_den).detach().item())
                residual_history.append(residual_val)

                if use_damping:
                    z = ((1.0 - damping_factor) * z) + (damping_factor * rhs)
                else:
                    z = rhs

            # Explicit forward/backward consistency telemetry for diagnostics.
            vjp_route_final = torch.autograd.grad(
                route_next,
                route_star,
                grad_outputs=z,
                retain_graph=True,
                allow_unused=False,
            )[0]
            rhs = grad_output + vjp_route_final
            residual_num = torch.norm(z - rhs, p=2)
            residual_den = torch.norm(rhs, p=2) + layer.eps
            fixed_point_residual = float((residual_num / residual_den).detach().item())
            layer._imd_last_backward_info = {
                "available": 1.0,
                "iter_idx": float(backward_iter_idx),
                "fixed_point_residual": fixed_point_residual,
                "residual_peak": float(max(residual_history) if len(residual_history) > 0 else 0.0),
                "z_norm": float(torch.norm(z, p=2).detach().item()),
                "grad_output_norm": float(torch.norm(grad_output, p=2).detach().item()),
                "backward_iters": float(iter_count),
                "damping": float(damping_factor),
                "pure_jfb": float(0.0 if use_damping else 1.0),
                "fallback_used": float(1.0 if use_damping else 0.0),
            }

            grads = torch.autograd.grad(
                route_next,
                (od_demands, alpha, beta, cap_mult),
                grad_outputs=z,
                allow_unused=True,
            )

        grad_od, grad_alpha, grad_beta, grad_cap = grads
        return None, None, grad_od, grad_alpha, grad_beta, grad_cap


class ImplicitEquilibriumLayer(nn.Module):
    """
    Variational-inequality solver using entropy mirror descent steps on route flows.

    This layer enforces feasibility and redistributes route flows per OD pair
    until an approximate Wardrop equilibrium is reached.

    It supports two gradient regimes:
        - standard truncated backprop through explicit iterations
        - IMD mode (custom implicit backward via `IMDEquilibriumFunction`)

    Inputs:
        od_demands: [B, OD] or [OD]
        bpr_params: dictionary with alpha, beta, capacity_multiplier

    Outputs:
        final_link_flows: [B, L]
        final_route_flows: [B, R]
        info: convergence metadata used by diagnostics/logging
    """
    def __init__(
        self,
        delta_matrix: torch.Tensor,
        route_validity_mask: torch.Tensor,
        t0: torch.Tensor,
        capacity: torch.Tensor,
        max_iterations: int = 50,
        eta: float = 1.0,
        min_iterations: int = 5,
        tol_rel_flow: float = 1e-4,
        use_early_stop: bool = True,
        imd_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.max_iterations = int(max_iterations)
        self.eta = float(eta)
        self.min_iterations = int(min_iterations)
        self.tol_rel_flow = float(tol_rel_flow)
        self.use_early_stop = bool(use_early_stop)
        self.eps = 1e-9

        imd_cfg = dict(imd_cfg or {})
        self.imd_enabled = bool(imd_cfg.get("enabled", False))
        self.imd_backward_iters = int(imd_cfg.get("backward_iters", 10))
        self.imd_use_damping = bool(imd_cfg.get("use_damping", False))
        self.imd_damping = float(imd_cfg.get("damping", 0.5))
        self.imd_residual_explode_threshold = float(imd_cfg.get("residual_explode_threshold", 1e3))
        self.imd_residual_growth_threshold = float(imd_cfg.get("residual_growth_threshold", 20.0))
        self.imd_residual_noise_floor = float(imd_cfg.get("residual_noise_floor", 1e-3))
        self.imd_stationary_tail_steps = int(imd_cfg.get("stationary_tail_steps", 0))
        self.imd_warm_start = bool(imd_cfg.get("warm_start", True))
        self._imd_last_info = {
            "iterations": 0.0,
            "final_gap": 0.0,
            "converged": 0.0,
            "implicit_grad": True,
        }
        self._imd_last_backward_info = {
            "available": 0.0,
            "iter_idx": 0.0,
            "fixed_point_residual": 0.0,
            "residual_peak": 0.0,
            "z_norm": 0.0,
            "grad_output_norm": 0.0,
            "backward_iters": float(self.imd_backward_iters),
            "damping": float(self.imd_damping if self.imd_use_damping else 1.0),
            "pure_jfb": float(0.0 if self.imd_use_damping else 1.0),
            "fallback_used": float(1.0 if self.imd_use_damping else 0.0),
        }

        self.register_buffer("delta_matrix", delta_matrix.coalesce())  # [L, R]
        self.register_buffer("route_validity_mask", route_validity_mask.bool())  # [OD, K]
        self.register_buffer("t0", t0.float())
        self.register_buffer("capacity", capacity.float())

        self.num_od, self.k_paths = self.route_validity_mask.shape
        self.num_routes = self.num_od * self.k_paths
        self.register_buffer("cached_route_flows", torch.zeros(1, self.num_routes, dtype=torch.float32))
        self.register_buffer("cache_valid", torch.tensor(False, dtype=torch.bool))

    def _route_to_link_flows(self, route_flows: torch.Tensor) -> torch.Tensor:
        """Project route flows [B, R] to link flows [B, L] through sparse incidence."""
        return torch.sparse.mm(self.delta_matrix, route_flows.t()).t()

    def _route_costs(self, route_flows: torch.Tensor, bpr_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute route costs [B, OD, K] from current route flows and BPR parameters."""
        link_flows = self._route_to_link_flows(route_flows)
        link_costs = self._compute_bpr_cost(link_flows, bpr_params)
        route_costs_flat = torch.sparse.mm(self.delta_matrix.t(), link_costs.t()).t()
        return route_costs_flat.view(-1, self.num_od, self.k_paths)

    def _fixed_point_step(
        self,
        route_flows: torch.Tensor,
        od_demands: torch.Tensor,
        bpr_params: Dict[str, torch.Tensor],
        iter_idx: int,
    ) -> torch.Tensor:
        route_costs = self._route_costs(route_flows, bpr_params)
        return self._mirror_descent_step(
            route_flows_flat=route_flows,
            route_costs=route_costs,
            od_demands=od_demands,
            iter_idx=iter_idx,
        )

    def forward(
        self,
        od_demands: torch.Tensor,
        bpr_params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Solve equilibrium and return link flows, route flows, and convergence stats."""
        # Reset backward audit information each forward call.
        self._imd_last_backward_info = {
            "available": 0.0,
            "iter_idx": 0.0,
            "fixed_point_residual": 0.0,
            "residual_peak": 0.0,
            "z_norm": 0.0,
            "grad_output_norm": 0.0,
            "backward_iters": float(self.imd_backward_iters),
            "damping": float(self.imd_damping if self.imd_use_damping else 1.0),
            "pure_jfb": float(0.0 if self.imd_use_damping else 1.0),
            "fallback_used": float(1.0 if self.imd_use_damping else 0.0),
        }

        od_demands = self._ensure_2d(od_demands)  # [B, OD]
        route_init = self._initialize_flows(
            od_demands,
            warm_start=self.imd_warm_start and self.training,
        )  # [B, R]

        # IMD mode uses a custom autograd function to avoid storing all iterations.
        if self.imd_enabled and self.training:
            current_route_flows = IMDEquilibriumFunction.apply(
                self,
                route_init,
                od_demands,
                bpr_params["alpha"],
                bpr_params["beta"],
                bpr_params["capacity_multiplier"],
            )
            final_link_flows = self._route_to_link_flows(current_route_flows)
            info = {
                "iterations": float(self._imd_last_info["iterations"]),
                "final_gap": float(self._imd_last_info["final_gap"]),
                "converged": float(self._imd_last_info["converged"]),
                "linearization_iter": float(self._imd_last_info.get("linearization_iter", self._imd_last_info["iterations"])),
                "implicit_grad": True,
            }
            self.cached_route_flows.copy_(current_route_flows.detach().mean(dim=0, keepdim=True))
            self.cache_valid.fill_(True)
            return final_link_flows, current_route_flows, info

        current_route_flows = route_init

        prev_link_flows = None
        used_iters = self.max_iterations
        final_gap = 0.0
        converged = False

        for it in range(1, self.max_iterations + 1):
            link_flows = self._route_to_link_flows(current_route_flows)  # [B, L]
            next_route_flows = self._fixed_point_step(
                route_flows=current_route_flows,
                od_demands=od_demands,
                bpr_params=bpr_params,
                iter_idx=it,
            )

            if prev_link_flows is not None:
                # Relative link-flow change is the stopping criterion.
                rel = torch.norm(link_flows - prev_link_flows, p=2, dim=1) / (
                    torch.norm(prev_link_flows, p=2, dim=1) + self.eps
                )
                final_gap = float(rel.mean().item())
                if self.use_early_stop and it >= self.min_iterations and bool(torch.all(rel < self.tol_rel_flow)):
                    current_route_flows = next_route_flows
                    used_iters = it
                    converged = True
                    break

            current_route_flows = next_route_flows
            prev_link_flows = link_flows
            used_iters = it

        final_link_flows = self._route_to_link_flows(current_route_flows)
        info = {
            "iterations": float(used_iters),
            "final_gap": float(final_gap),
            "converged": float(1.0 if converged else 0.0),
            "implicit_grad": False,
        }
        self.cached_route_flows.copy_(current_route_flows.detach().mean(dim=0, keepdim=True))
        self.cache_valid.fill_(True)
        return final_link_flows, current_route_flows, info

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _initialize_flows(self, od_demands: torch.Tensor, warm_start: bool = False) -> torch.Tensor:
        """Initialize feasible route flows using cache or uniform valid-route split."""
        if warm_start and bool(self.cache_valid.item()) and self.cached_route_flows.shape[1] == self.num_routes:
            cached = self.cached_route_flows.to(device=od_demands.device, dtype=od_demands.dtype)
            route_flows = cached.expand(od_demands.shape[0], -1)
            return self._project_to_feasible_route_flows(route_flows, od_demands)

        # Uniform route split per OD over valid routes only.
        valid = self.route_validity_mask.float()  # [OD, K]
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        probs = valid / denom
        route_flows = od_demands.unsqueeze(-1) * probs.unsqueeze(0)  # [B, OD, K]
        return route_flows.reshape(od_demands.shape[0], self.num_routes)

    def _project_to_feasible_route_flows(self, route_flows_flat: torch.Tensor, od_demands: torch.Tensor) -> torch.Tensor:
        """Project arbitrary route flows into nonnegative OD-wise simplex constraints."""
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        valid = self.route_validity_mask.unsqueeze(0).expand(route_flows.shape[0], -1, -1)
        next_flows = self._project_to_masked_demand_simplex(
            values=route_flows,
            od_demands=od_demands,
            valid=valid,
        )
        return next_flows.reshape(route_flows.shape[0], self.num_routes)

    def _project_to_masked_demand_simplex(
        self,
        values: torch.Tensor,
        od_demands: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euclidean projection onto masked OD-wise simplices.

        For each [batch, od], solves:
            min_x 0.5 ||x - values||^2
            s.t.   x_i >= 0 on valid routes, x_i = 0 on invalid routes,
                   sum_i x_i = od_demands[batch, od].

        Implemented fully vectorized over [B, OD, K] using sorting.
        """
        demands = torch.clamp(od_demands.unsqueeze(-1), min=0.0)
        valid_f = valid.to(dtype=values.dtype)
        has_valid = valid.any(dim=2, keepdim=True)

        # Push invalid entries to the tail of the descending sort.
        neg_large = torch.full_like(values, -1e9)
        values_masked = torch.where(valid, values, neg_large)

        sorted_values, sorted_idx = torch.sort(values_masked, dim=2, descending=True)
        sorted_valid = torch.gather(valid_f, dim=2, index=sorted_idx)

        sorted_values_valid = sorted_values * sorted_valid
        cumsum_values = torch.cumsum(sorted_values_valid, dim=2)
        cumsum_counts = torch.cumsum(sorted_valid, dim=2)
        denom = torch.clamp(cumsum_counts, min=1.0)

        tau_candidates = (cumsum_values - demands) / denom
        active = (sorted_valid > 0.0) & ((sorted_values - tau_candidates) > 0.0)
        rho = active.sum(dim=2, keepdim=True).clamp(min=1)
        rho_idx = (rho - 1).long()

        tau_num = torch.gather(cumsum_values, dim=2, index=rho_idx)
        tau_den = torch.gather(cumsum_counts, dim=2, index=rho_idx).clamp(min=1.0)
        tau = (tau_num - demands) / tau_den

        projected = torch.clamp(values - tau, min=0.0) * valid_f
        projected = torch.where(has_valid, projected, torch.zeros_like(projected))
        return projected

    def _compute_bpr_cost(self, link_flows: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute link travel time using bounded BPR parameters and adjusted capacity."""
        alpha = params["alpha"]
        beta = params["beta"]
        cap_mult = params["capacity_multiplier"]

        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)
        if cap_mult.dim() == 1:
            cap_mult = cap_mult.unsqueeze(0)

        adj_capacity = (self.capacity.unsqueeze(0) * cap_mult).clamp(min=self.eps)
        v_over_c = torch.clamp(link_flows / adj_capacity, min=0.0, max=5.0)

        return self.t0.unsqueeze(0) * (1.0 + alpha * torch.pow(v_over_c, beta))

    def _mirror_descent_step(
        self,
        route_flows_flat: torch.Tensor,
        route_costs: torch.Tensor,
        od_demands: torch.Tensor,
        iter_idx: int,
    ) -> torch.Tensor:
        """
        One entropy mirror-descent step over valid routes.

        Update:
            sigma_next is proportional to sigma * exp(-eta_t * cost)
            x_next = demand * sigma_next

        where sigma are OD-wise route choice probabilities over valid routes.
        This is the path-based multiplicative update described in the report,
        implemented with a masked log-softmax for numerical stability.
        """
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        valid = self.route_validity_mask.unsqueeze(0).expand(route_flows.shape[0], -1, -1)
        valid_f = valid.to(dtype=route_flows.dtype)
        has_valid = valid.any(dim=2, keepdim=True)

        demands = torch.clamp(od_demands.unsqueeze(-1), min=0.0)
        denom = torch.clamp(demands, min=self.eps)

        # Convert flows to OD-wise route choice probabilities on valid routes.
        sigma = (route_flows * valid_f) / denom
        sigma = torch.clamp(sigma, min=self.eps) * valid_f
        sigma = sigma / torch.clamp(sigma.sum(dim=2, keepdim=True), min=self.eps)

        eta_t = self.eta / (float(iter_idx) ** 0.5)

        neg_large = torch.full_like(route_costs, -1e9)
        logits = torch.log(sigma + self.eps) - (eta_t * route_costs)
        logits = torch.where(valid, logits, neg_large)

        sigma_next = torch.softmax(logits, dim=2) * valid_f
        sigma_next = sigma_next / torch.clamp(sigma_next.sum(dim=2, keepdim=True), min=self.eps)

        next_flows = demands * sigma_next
        next_flows = torch.where(has_valid, next_flows, torch.zeros_like(next_flows))
        return next_flows.reshape(route_flows.shape[0], self.num_routes)


class VariationalInferenceModel(nn.Module):
    """
    End-to-end VI model: supply net + equilibrium solver + OD inference.

    High-level flow for each batch:
        1) Build normalized static link features.
        2) Predict BPR parameters with `PhysicsInformedBPRNet`.
        3) Produce positive OD estimates from trainable logits.
        4) Solve equilibrium with entropy mirror descent (or IMD path when enabled).
        5) Compute supervised loss terms when OD targets are available.

    The class also exposes optimizer parameter groups to decouple learning
    rates between supply parameters and OD logits.
    """
    def __init__(
        self,
        num_links: int,
        num_od_pairs: int,
        architecture,
        delta_matrix: torch.Tensor,
        route_validity_mask: torch.Tensor,
        od_pair_indices: torch.Tensor,
        t0: torch.Tensor,
        capacity: torch.Tensor,
        length: torch.Tensor,
        lanes: torch.Tensor,
        speed: torch.Tensor,
        link_group: torch.Tensor,
        num_link_groups: int,
        initial_mean: float = 1.0,
        link_scale: float = 1.0,
        od_scale: float = 1.0,
        solver: Optional[dict] = None,
        loss: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_links = int(num_links)
        self.num_od_pairs = int(num_od_pairs)
        self.num_link_groups = int(num_link_groups)
        self.link_scale = float(link_scale)
        self.od_scale = float(od_scale)
        self.initial_mean = float(initial_mean)
        self.supply_lr_multiplier = float(kwargs.get("supply_lr_multiplier", 0.05))
        self.demand_lr_multiplier = float(kwargs.get("demand_lr_multiplier", 2.5))
        self.anchor_known_od_in_solver = bool(kwargs.get("anchor_known_od_in_solver", True))
        self.init_unknown_od_low = bool(kwargs.get("init_unknown_od_low", False))
        self.unknown_od_init_value = float(kwargs.get("unknown_od_init_value", 0.01))
        self._unknown_od_init_applied = False

        self.register_buffer("t0", t0.float())
        self.register_buffer("capacity", capacity.float())
        self.register_buffer("length", length.float())
        self.register_buffer("lanes", lanes.float())
        self.register_buffer("speed", speed.float())
        self.register_buffer("link_group", link_group.long())
        self.register_buffer("od_pair_indices", od_pair_indices.long())
        self.od_pair_node_labels = kwargs.get("od_pair_node_labels", None)

        hidden_dim = int(getattr(architecture, "hidden_dim", 64))
        if hasattr(architecture, "hidden_dim_from_link"):
            hidden_dim = int(getattr(architecture, "hidden_dim_from_link"))

        self.supply_net = PhysicsInformedBPRNet(
            input_dim=6,
            hidden_dim=hidden_dim,
            num_link_groups=self.num_link_groups,
            link_group=self.link_group,
            alpha_min=float(kwargs.get("alpha_min", 0.1)),
            alpha_max=float(kwargs.get("alpha_max", 0.5)),
            beta_min=float(kwargs.get("beta_min", 1.0)),
            beta_max=float(kwargs.get("beta_max", 6.0)),
            cap_mode=str(kwargs.get("cap_mode", "group")),
            cap_min=float(kwargs.get("min_value", 12.0)),
            cap_max=float(kwargs.get("max_value", 16.0)),
            mlp_hidden_dim=int(kwargs.get("mlp_hidden_dim", 64))
            )
        

        solver_cfg = dict(solver or {})
        self.equilibrium_solver = ImplicitEquilibriumLayer(
            delta_matrix=delta_matrix,
            route_validity_mask=route_validity_mask,
            t0=t0,
            capacity=capacity,
            max_iterations=int(solver_cfg.get("max_iterations", 50)),
            eta=float(solver_cfg.get("eta", 1.0)),
            min_iterations=int(solver_cfg.get("min_iterations", 5)),
            tol_rel_flow=float(solver_cfg.get("tol_rel_flow", 1e-4)),
            use_early_stop=bool(solver_cfg.get("use_early_stop", True)),
            imd_cfg=dict(solver_cfg.get("imd", {})),
        )

        init_target_norm = max(float(initial_mean) / max(self.od_scale, 1e-6), 1e-6)
        init_raw = self._inverse_softplus(init_target_norm)
        self.od_logits = nn.Parameter(torch.full((self.num_od_pairs,), init_raw, dtype=torch.float32))

        loss_cfg = dict(loss or {})
        loss_cfg.pop("_target_", None)
        loss_cfg.pop("link_scale", None)
        loss_cfg.pop("od_scale", None)
        self.loss_fn = Loss(
            link_scale=self.link_scale,
            od_scale=self.od_scale,
            **loss_cfg,
        )

    def get_optimizer_param_groups(self, base_lr: float, weight_decay: float = 0.0):
        """Return optimizer param groups with decoupled LR for supply and demand."""
        grouped = []
        used_param_ids = set()

        def _collect(module: nn.Module):
            params = [p for p in module.parameters() if p.requires_grad]
            for p in params:
                used_param_ids.add(id(p))
            return params

        supply_params = _collect(self.supply_net)
        if len(supply_params) > 0:
            grouped.append(
                {
                    "params": supply_params,
                    "lr": float(base_lr) * self.supply_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )

        grouped.append(
            {
                "params": [self.od_logits],
                "lr": float(base_lr) * self.demand_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )
        used_param_ids.add(id(self.od_logits))

        remaining = [p for p in self.parameters() if p.requires_grad and id(p) not in used_param_ids]
        if len(remaining) > 0:
            grouped.append(
                {
                    "params": remaining,
                    "lr": float(base_lr),
                    "weight_decay": weight_decay,
                }
            )

        return grouped

    def _build_link_features(self) -> torch.Tensor:
        """Assemble normalized per-link features used by the supply network."""
        eps = 1e-6
        t0_n = self.t0 / (self.t0.mean() + eps)
        cap_n = self.capacity / (self.capacity.mean() + eps)
        len_n = self.length / (self.length.mean() + eps)
        lanes_n = self.lanes / (self.lanes.mean() + eps)
        speed_n = self.speed / (self.speed.mean() + eps)

        if self.num_link_groups > 1:
            group_n = self.link_group.float() / float(self.num_link_groups - 1)
        else:
            group_n = torch.zeros_like(self.link_group, dtype=torch.float32)

        return torch.stack([len_n, t0_n, speed_n, lanes_n, group_n, cap_n], dim=1)

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        x = max(float(value), 1e-12)
        if x > 20.0:
            return x
        return float(np.log(np.expm1(x) + 1e-12))

    @staticmethod
    def _inverse_softplus_tensor(x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, min=1e-12)
        return torch.where(
            x_safe > 20.0,
            x_safe,
            torch.log(torch.expm1(x_safe) + 1e-12),
        )

    def _maybe_apply_unknown_od_init(
        self,
        true_od_sparse: Optional[torch.Tensor],
        od_mask_sparse: Optional[torch.Tensor],
    ) -> None:
        """
        Optional one-time OD initialization policy for partially observed OD labels.

        Known OD entries start from supervised values; unknown entries can start
        from a conservative low value to reduce early flow-dumping artifacts.
        """
        if self._unknown_od_init_applied or (not self.init_unknown_od_low):
            return
        if true_od_sparse is None or od_mask_sparse is None:
            return

        if od_mask_sparse.dim() == 2:
            known_mask = (od_mask_sparse > 0.5).any(dim=0)
            known_target = (true_od_sparse * od_mask_sparse).sum(dim=0) / torch.clamp(od_mask_sparse.sum(dim=0), min=1.0)
        else:
            known_mask = od_mask_sparse > 0.5
            known_target = true_od_sparse

        known_target = known_target.to(device=self.od_logits.device, dtype=self.od_logits.dtype)
        known_target = torch.clamp(known_target, min=0.0)

        unknown_target = torch.full_like(self.od_logits, float(max(self.unknown_od_init_value, 0.0)))
        init_target = torch.where(known_mask.to(device=self.od_logits.device), known_target, unknown_target)

        init_norm = torch.clamp(init_target / max(self.od_scale, 1e-6), min=1e-12)

        with torch.no_grad():
            self.od_logits.copy_(self._inverse_softplus_tensor(init_norm))
        self._unknown_od_init_applied = True

    def _learned_od(self, batch_size: int) -> torch.Tensor:
        """Decode positive OD demand from logits and broadcast to batch size."""
        od_pos = F.softplus(self.od_logits) * self.od_scale
        return od_pos.unsqueeze(0).expand(batch_size, -1)

    def _align_od_targets(
        self,
        true_od: torch.Tensor,
        od_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align dense OD supervision tensors to the model's sparse OD indexing.

        Preferred order:
            1) label-space mapping using `od_pair_node_labels`
            2) index-space mapping using `od_pair_indices`
            3) safe fallback by truncation/padding
        """
        if true_od.dim() == 1:
            true_od = true_od.unsqueeze(0)
        if od_mask is None:
            od_mask = torch.ones_like(true_od)
        elif od_mask.dim() == 1:
            od_mask = od_mask.unsqueeze(0)

        if true_od.shape[1] == self.num_od_pairs:
            return true_od, od_mask

        dense_dim = true_od.shape[1]
        n_nodes = int(round(dense_dim ** 0.5))

        # Preferred mapping: label-space alignment when OD matrix indexing does not
        # match graph node indexing.
        if self.od_pair_node_labels is not None and n_nodes * n_nodes == dense_dim:
            labels_flat = [str(x) for pair in self.od_pair_node_labels for x in pair]

            def _sort_key(label: str):
                return (0, int(label)) if label.isdigit() else (1, label)

            unique_labels = sorted(set(labels_flat), key=_sort_key)
            if len(unique_labels) == n_nodes:
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

                pair_idx = []
                pair_valid = []
                for u, v in self.od_pair_node_labels:
                    su, sv = str(u), str(v)
                    if su in label_to_idx and sv in label_to_idx:
                        i = label_to_idx[su]
                        j = label_to_idx[sv]
                        pair_idx.append(i * n_nodes + j)
                        pair_valid.append(True)
                    else:
                        pair_idx.append(0)
                        pair_valid.append(False)

                flat_idx = torch.as_tensor(pair_idx, dtype=torch.long, device=true_od.device)
                valid = torch.as_tensor(pair_valid, dtype=torch.bool, device=true_od.device)

                sparse_true = true_od[:, flat_idx]
                sparse_mask = od_mask[:, flat_idx]

                if torch.any(~valid):
                    sparse_true[:, ~valid] = 0.0
                    sparse_mask[:, ~valid] = 0.0

                return sparse_true, sparse_mask

        if n_nodes * n_nodes == dense_dim:
            flat_idx = self.od_pair_indices[:, 0] * n_nodes + self.od_pair_indices[:, 1]
            valid = (flat_idx >= 0) & (flat_idx < dense_dim)
            flat_idx = torch.clamp(flat_idx, min=0, max=dense_dim - 1)

            sparse_true = true_od[:, flat_idx]
            sparse_mask = od_mask[:, flat_idx]

            if torch.any(~valid):
                sparse_true[:, ~valid] = 0.0
                sparse_mask[:, ~valid] = 0.0

            return sparse_true, sparse_mask

        # Fallback: truncate to model OD space.
        cut = min(self.num_od_pairs, true_od.shape[1])
        padded_true = torch.zeros(true_od.shape[0], self.num_od_pairs, device=true_od.device)
        padded_mask = torch.zeros(od_mask.shape[0], self.num_od_pairs, device=od_mask.device)
        padded_true[:, :cut] = true_od[:, :cut]
        padded_mask[:, :cut] = od_mask[:, :cut]
        return padded_true, padded_mask

    def forward(
        self,
        observed_flows: torch.Tensor,
        flow_mask: Optional[torch.Tensor] = None,
        true_od_demand: Optional[torch.Tensor] = None,
        od_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run one end-to-end VI pass and optionally return supervised loss terms."""
        if observed_flows.dim() == 1:
            observed_flows = observed_flows.unsqueeze(0)
        if flow_mask is None:
            flow_mask = torch.ones_like(observed_flows)
        elif flow_mask.dim() == 1:
            flow_mask = flow_mask.unsqueeze(0)

        batch_size = observed_flows.shape[0]

        link_features = self._build_link_features()
        bpr_params = self.supply_net(link_features)

        true_od_sparse = None
        od_mask_sparse = None
        if true_od_demand is not None:
            true_od_sparse, od_mask_sparse = self._align_od_targets(true_od_demand, od_mask)

        if self.training:
            self._maybe_apply_unknown_od_init(true_od_sparse=true_od_sparse, od_mask_sparse=od_mask_sparse)

        estimated_od = self._learned_od(batch_size)

        if self.training and self.anchor_known_od_in_solver and true_od_sparse is not None and od_mask_sparse is not None:
            od_mask_solver = od_mask_sparse.float()
            hard_anchored_od = (od_mask_solver * true_od_sparse) + ((1.0 - od_mask_solver) * estimated_od)
            # Straight-through anchor: forward uses known OD values, backward keeps gradients through logits.
            od_input = estimated_od + (hard_anchored_od - estimated_od).detach()
        else:
            od_input = estimated_od

        pred_link_flows, route_flows, conv_info = self.equilibrium_solver(od_input, bpr_params)

        outputs = {
            "estimated_demand": estimated_od.squeeze(0) if batch_size == 1 else estimated_od,
            "reconstructed_flows": pred_link_flows.squeeze(0) if batch_size == 1 else pred_link_flows,
            "route_flows": route_flows.squeeze(0) if batch_size == 1 else route_flows,
            "convergence_info": {
                **conv_info,
                "mode": "physics_imd" if bool(conv_info.get("implicit_grad", False)) else "physics_stage1",
                "implicit_grad": bool(conv_info.get("implicit_grad", False)),
            },
            "learned_alpha": bpr_params["alpha"],
            "learned_beta": bpr_params["beta"],
        }

        if true_od_demand is not None:
            loss_dict = self.loss_fn(
                predicted_flows=pred_link_flows,
                true_flows=observed_flows,
                flow_mask=flow_mask,
                predicted_od=estimated_od,
                true_od=true_od_sparse,
                od_mask=od_mask_sparse,
            )
            outputs["loss"] = loss_dict

        return outputs

    def get_evaluation_artifacts(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs_map = require_keys(
            outputs,
            ["reconstructed_flows", "estimated_demand"],
            context="VI_Model.get_evaluation_artifacts outputs",
            exc_type=ArtifactSchemaError,
        )
        return {
            "pred_flows": outputs_map["reconstructed_flows"].detach().cpu(),
            "pred_od": outputs_map["estimated_demand"].detach().cpu(),
            "route_flows": outputs_map.get("route_flows"),
            "convergence": outputs_map.get("convergence_info", {}),
            "learned_alpha": outputs_map.get("learned_alpha"),
            "learned_beta": outputs_map.get("learned_beta"),
        }


class Loss(nn.Module):
    """
    Multi-objective training loss for flow reconstruction and OD consistency.

    Terms:
        - `l_flow`: masked MSE on link flows
        - `l_od`: masked Huber loss for supervised OD entries
        - `l_demand_reg`: RMS regularization on unsupervised OD entries

    All flow/OD values are normalized by configurable scales before loss
    computation to keep gradient magnitudes comparable across datasets.
    """
    def __init__(
        self,
        link_scale: float = 1.0,
        od_scale: float = 1.0,
        w_link: float = 1.0,
        w_prior: float = 0.1,
        w_reg: float = 0.01,
        huber_delta: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.link_scale = float(link_scale)
        self.od_scale = float(od_scale)
        self.w_link = float(w_link)
        self.w_prior = float(w_prior)
        self.w_reg = float(w_reg)
        self.huber_delta = float(huber_delta)
        self.eps = 1e-9

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def forward(
        self,
        predicted_flows: torch.Tensor,
        true_flows: torch.Tensor,
        flow_mask: torch.Tensor,
        predicted_od: torch.Tensor,
        true_od: torch.Tensor,
        od_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted loss components and return a logging-friendly dictionary."""
        predicted_flows = self._ensure_2d(predicted_flows)
        true_flows = self._ensure_2d(true_flows)
        flow_mask = self._ensure_2d(flow_mask)
        predicted_od = self._ensure_2d(predicted_od)
        true_od = self._ensure_2d(true_od)
        od_mask = self._ensure_2d(od_mask)

        flow_mask = flow_mask.float()
        od_mask = od_mask.float()

        scaled_pred_flows = predicted_flows / self.link_scale
        scaled_true_flows = true_flows / self.link_scale
        flow_sq_error = (scaled_pred_flows - scaled_true_flows) ** 2
        loss_links = (flow_sq_error * flow_mask).sum() / (flow_mask.sum() + self.eps)

        scaled_pred_od = predicted_od / self.od_scale
        scaled_true_od = true_od / self.od_scale
        od_huber = F.smooth_l1_loss(
            scaled_pred_od,
            scaled_true_od,
            reduction="none",
            beta=self.huber_delta,
        )
        loss_prior_od = (od_huber * od_mask).sum() / (od_mask.sum() + self.eps)

        unknown_od_mask = (1.0 - od_mask).clamp(min=0.0, max=1.0)
        if unknown_od_mask.sum() > 0:
            loss_unknown_regularization = torch.sqrt(
                ((scaled_pred_od ** 2) * unknown_od_mask).sum() / (unknown_od_mask.sum() + self.eps) + self.eps
            )
        else:
            loss_unknown_regularization = torch.zeros((), device=predicted_od.device)

        total_loss = (
            (self.w_link * loss_links)
            + (self.w_prior * loss_prior_od)
            + (self.w_reg * loss_unknown_regularization)
        )

        return {
            "total_loss": total_loss,
            "l_flow": loss_links,
            "l_od": loss_prior_od,
            "l_demand_reg": loss_unknown_regularization,
        }


class VIDiagnostician:
    """
    Training diagnostics companion for `VariationalInferenceModel`.

    Responsibilities:
        - track flow fit metrics (R2, MAE) and loss components across epochs
        - monitor solver convergence signals (`iterations`, `final_gap`, mode)
        - monitor gradient health for key modules and OD logits
        - audit forward/backward consistency when IMD is enabled
        - export summary JSON, diagnostic CSV tables, and plotting artifacts
    """

    def __init__(
        self,
        history_window: int = 100,
        gap_alert_threshold: float = 5e-3,
        od_grad_low_threshold: float = 1e-8,
        r2_stall_window: int = 10,
        r2_stall_delta: float = 1e-4,
    ):
        self.window = int(history_window)
        self.gap_alert_threshold = float(gap_alert_threshold)
        self.od_grad_low_threshold = float(od_grad_low_threshold)
        self.r2_stall_window = int(r2_stall_window)
        self.r2_stall_delta = float(r2_stall_delta)

        self.full_history = {
            "r2_flow": [],
            "mae_flow": [],
            "total_loss": [],
            "l_flow": [],
            "l_od": [],
            "l_demand_reg": [],
            "iterations": [],
            "final_gap": [],
            "converged": [],
            "mode": [],
            "max_grad": [],
            "od_logits_grad": [],
            "grad_norms": {},
            "alerts": [],
        }
        self.window_history = {
            "r2_flow": [],
            "mae_flow": [],
            "max_grad": [],
        }

        self.gradient_detailed_rows = []
        self.link_type_epoch_rows = []
        self.flow_comparison_rows = []
        self.demand_comparison_rows = []
        self.forward_backward_audit_rows = []

    def _to_float(self, value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().mean().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _append_scalar(self, key: str, value, default: float = 0.0):
        self.full_history[key].append(self._to_float(value, default=default))

    def _push_window(self, key: str, value: float):
        self.window_history[key].append(value)
        if len(self.window_history[key]) > self.window:
            self.window_history[key].pop(0)

    def update(self, outputs: Dict, targets: Dict, model=None, **kwargs):
        """Ingest one training/eval step and append all tracked diagnostics."""
        with torch.no_grad():
            epoch = int(kwargs.get("epoch", len(self.full_history["r2_flow"]) + 1))
            pred_flow = outputs.get("reconstructed_flows")
            true_flow = targets.get("flows")
            mask_flow = targets.get("flow_mask")
            if mask_flow is None:
                mask_flow = targets.get("mask")

            if pred_flow is None or true_flow is None:
                return
            if mask_flow is None:
                mask_flow = torch.ones_like(true_flow)

            if pred_flow.dim() == 2 and pred_flow.shape[0] == 1:
                pred_flow = pred_flow.squeeze(0)
            if true_flow.dim() == 2 and true_flow.shape[0] == 1:
                true_flow = true_flow.squeeze(0)
            if mask_flow.dim() == 2 and mask_flow.shape[0] == 1:
                mask_flow = mask_flow.squeeze(0)

            mask_bool = mask_flow > 0
            if mask_bool.sum() <= 0:
                return

            y_pred = pred_flow[mask_bool].detach().cpu().numpy()
            y_true = true_flow[mask_bool].detach().cpu().numpy()
            mae = float(np.mean(np.abs(y_true - y_pred)))

            if len(y_true) > 1 and np.var(y_true) > 0:
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            else:
                r2 = 0.0

            self.full_history["r2_flow"].append(r2)
            self.full_history["mae_flow"].append(mae)
            self._push_window("r2_flow", r2)
            self._push_window("mae_flow", mae)

            loss_dict = kwargs.get("loss_dict") or outputs.get("loss") or {}
            if isinstance(loss_dict, dict):
                self._append_scalar("total_loss", loss_dict.get("total_loss"), default=0.0)
                self._append_scalar("l_flow", loss_dict.get("l_flow"), default=0.0)
                self._append_scalar("l_od", loss_dict.get("l_od"), default=0.0)
                self._append_scalar("l_demand_reg", loss_dict.get("l_demand_reg"), default=0.0)

            conv_info = outputs.get("convergence_info", {}) or {}
            self._append_scalar("iterations", conv_info.get("iterations"), default=0.0)
            self._append_scalar("final_gap", conv_info.get("final_gap"), default=0.0)
            self._append_scalar("converged", conv_info.get("converged"), default=0.0)
            self.full_history["mode"].append(str(conv_info.get("mode", "physics_stage1")))

            static_info = kwargs.get("static_info", {}) or {}
            self._collect_link_type_rows(
                epoch=epoch,
                outputs=outputs,
                static_info=static_info,
            )
            self._collect_flow_demand_rows(
                epoch=epoch,
                outputs=outputs,
                targets=targets,
            )
            self._collect_forward_backward_audit_row(
                epoch=epoch,
                outputs=outputs,
                model=model,
            )

            self._record_alerts(epoch=len(self.full_history["r2_flow"]) - 1)

    def _collect_forward_backward_audit_row(self, epoch: int, outputs: Dict, model: Optional[nn.Module]):
        """Record IMD forward/backward iteration consistency and residual indicators."""
        conv_info = outputs.get("convergence_info", {}) if isinstance(outputs, dict) else {}
        if not isinstance(conv_info, dict):
            conv_info = {}

        mode = str(conv_info.get("mode", "unknown"))
        implicit_grad = bool(conv_info.get("implicit_grad", False))

        fwd_iterations = self._to_float(conv_info.get("iterations"), default=0.0)
        fwd_final_gap = self._to_float(conv_info.get("final_gap"), default=0.0)
        fwd_linearization_iter = self._to_float(conv_info.get("linearization_iter"), default=0.0)

        bwd_available = False
        bwd_iter_idx = 0.0
        bwd_fixed_point_residual = 0.0
        bwd_z_norm = 0.0
        bwd_grad_output_norm = 0.0
        bwd_iters = 0.0
        bwd_damping = 0.0

        if model is not None and hasattr(model, "equilibrium_solver"):
            solver = getattr(model, "equilibrium_solver")
            if hasattr(solver, "_imd_last_info") and isinstance(solver._imd_last_info, dict):
                if fwd_linearization_iter <= 0.0:
                    fwd_linearization_iter = self._to_float(
                        solver._imd_last_info.get("linearization_iter"),
                        default=fwd_iterations,
                    )
            if hasattr(solver, "_imd_last_backward_info") and isinstance(solver._imd_last_backward_info, dict):
                bwd_info = solver._imd_last_backward_info
                bwd_available = bool(self._to_float(bwd_info.get("available"), default=0.0) > 0.5)
                bwd_iter_idx = self._to_float(bwd_info.get("iter_idx"), default=0.0)
                bwd_fixed_point_residual = self._to_float(bwd_info.get("fixed_point_residual"), default=0.0)
                bwd_z_norm = self._to_float(bwd_info.get("z_norm"), default=0.0)
                bwd_grad_output_norm = self._to_float(bwd_info.get("grad_output_norm"), default=0.0)
                bwd_iters = self._to_float(bwd_info.get("backward_iters"), default=0.0)
                bwd_damping = self._to_float(bwd_info.get("damping"), default=0.0)

        iter_match = False
        if implicit_grad and bwd_available:
            iter_match = int(round(fwd_linearization_iter)) == int(round(bwd_iter_idx))
            if not iter_match:
                self.full_history["alerts"].append((int(epoch), "imd_iter_mismatch", float(abs(fwd_linearization_iter - bwd_iter_idx))))

        status = "non_imd"
        if implicit_grad and bwd_available:
            status = "ok" if iter_match else "iter_mismatch"
        elif implicit_grad and (not bwd_available):
            status = "missing_backward_info"

        self.forward_backward_audit_rows.append(
            {
                "epoch": int(epoch),
                "mode": mode,
                "implicit_grad": int(implicit_grad),
                "forward_iterations": float(fwd_iterations),
                "forward_final_gap": float(fwd_final_gap),
                "forward_linearization_iter": float(fwd_linearization_iter),
                "backward_info_available": int(bwd_available),
                "backward_iter_idx": float(bwd_iter_idx),
                "iter_match": int(iter_match) if implicit_grad else 1,
                "backward_fixed_point_residual": float(bwd_fixed_point_residual),
                "backward_z_norm": float(bwd_z_norm),
                "backward_grad_output_norm": float(bwd_grad_output_norm),
                "backward_iters": float(bwd_iters),
                "backward_damping": float(bwd_damping),
                "status": status,
            }
        )

    def capture_gradient_history(self, model: nn.Module, epoch: int):
        """Capture module-wise and parameter-wise gradient norms for health checks."""
        key_modules = {
            "supply_net": getattr(model, "supply_net", None),
            "equilibrium_solver": getattr(model, "equilibrium_solver", None),
        }

        max_grad = 0.0
        for name, module in key_modules.items():
            if module is None:
                continue
            total_norm_sq = 0.0
            named_params = list(module.named_parameters())
            if len(named_params) == 0:
                self.gradient_detailed_rows.append(
                    {
                        "epoch": int(epoch) + 1,
                        "module": str(name),
                        "param_name": "__no_trainable_params__",
                        "norm2": 0.0,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "finite_ratio": 0.0,
                        "non_finite_count": 0,
                    }
                )
            for param_name, p in named_params:
                row = self._build_gradient_row(
                    epoch=int(epoch) + 1,
                    module=name,
                    param_name=f"{name}.{param_name}",
                    grad=(p.grad.detach() if p.grad is not None else None),
                )
                self.gradient_detailed_rows.append(row)

                if p.grad is not None:
                    g = p.grad.data.norm(2).item()
                    total_norm_sq += g * g
            total_norm = total_norm_sq ** 0.5
            self.full_history["grad_norms"].setdefault(name, []).append(total_norm)
            max_grad = max(max_grad, total_norm)

        od_logits_grad = 0.0
        if hasattr(model, "od_logits") and getattr(model, "od_logits") is not None:
            grad = model.od_logits.grad
            if grad is not None:
                od_logits_grad = float(grad.detach().norm(2).item())
            self.gradient_detailed_rows.append(
                self._build_gradient_row(
                    epoch=int(epoch) + 1,
                    module="od_logits",
                    param_name="od_logits",
                    grad=(grad.detach() if grad is not None else None),
                )
            )

        self.full_history["max_grad"].append(float(max_grad))
        self.full_history["od_logits_grad"].append(float(od_logits_grad))
        self._push_window("max_grad", float(max_grad))

        if epoch >= 2 and od_logits_grad < self.od_grad_low_threshold:
            self.full_history["alerts"].append((int(epoch), "od_logits_low_grad", float(od_logits_grad)))

    def _record_alerts(self, epoch: int):
        """Emit heuristic alerts for poor convergence or stalled validation dynamics."""
        final_gap = self.full_history["final_gap"][-1] if self.full_history["final_gap"] else 0.0
        if final_gap > self.gap_alert_threshold:
            self.full_history["alerts"].append((int(epoch), "high_final_gap", float(final_gap)))

        if len(self.full_history["r2_flow"]) >= self.r2_stall_window:
            tail = self.full_history["r2_flow"][-self.r2_stall_window:]
            drift = float(max(tail) - min(tail))
            if drift < self.r2_stall_delta:
                self.full_history["alerts"].append((int(epoch), "r2_stall", drift))

    def _build_gradient_row(self, epoch: int, module: str, param_name: str, grad: Optional[torch.Tensor]) -> Dict[str, float]:
        if grad is None:
            return {
                "epoch": int(epoch),
                "module": str(module),
                "param_name": str(param_name),
                "norm2": 0.0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "finite_ratio": 0.0,
                "non_finite_count": 0,
            }

        g = grad.detach().reshape(-1)
        finite_mask = torch.isfinite(g)
        finite_count = int(finite_mask.sum().item())
        total_count = int(g.numel())
        non_finite_count = int(total_count - finite_count)
        finite_ratio = float(finite_count / max(total_count, 1))

        if finite_count > 0:
            g_finite = g[finite_mask]
            norm2 = float(torch.norm(g_finite, p=2).item())
            max_abs = float(torch.max(torch.abs(g_finite)).item())
            mean_abs = float(torch.mean(torch.abs(g_finite)).item())
        else:
            norm2 = 0.0
            max_abs = 0.0
            mean_abs = 0.0

        return {
            "epoch": int(epoch),
            "module": str(module),
            "param_name": str(param_name),
            "norm2": norm2,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "finite_ratio": finite_ratio,
            "non_finite_count": non_finite_count,
        }

    def _collect_link_type_rows(self, epoch: int, outputs: Dict, static_info: Dict):
        """Aggregate alpha/beta/volume-capacity stats grouped by link type."""
        alpha = outputs.get("learned_alpha")
        beta = outputs.get("learned_beta")
        pred_flow = outputs.get("reconstructed_flows")
        capacity = static_info.get("capacity", None)
        link_types = static_info.get("link_types", None)

        if alpha is None or beta is None or pred_flow is None or capacity is None or link_types is None:
            return

        alpha_np = alpha.detach().cpu().numpy().reshape(-1)
        beta_np = beta.detach().cpu().numpy().reshape(-1)

        pred_flow_np = pred_flow.detach().cpu().numpy()
        if pred_flow_np.ndim > 1:
            pred_flow_np = pred_flow_np.reshape(pred_flow_np.shape[0], -1)[0]
        else:
            pred_flow_np = pred_flow_np.reshape(-1)

        cap_np = np.asarray(capacity).reshape(-1)
        types_np = np.asarray(link_types).reshape(-1).astype(str)

        n = min(len(alpha_np), len(beta_np), len(pred_flow_np), len(cap_np), len(types_np))
        if n <= 0:
            return

        alpha_np = alpha_np[:n]
        beta_np = beta_np[:n]
        pred_flow_np = pred_flow_np[:n]
        cap_np = np.clip(cap_np[:n], a_min=1e-9, a_max=None)
        types_np = types_np[:n]

        vc_np = pred_flow_np / cap_np

        for link_type in sorted(set(types_np.tolist())):
            mask = types_np == link_type
            if not np.any(mask):
                continue
            self.link_type_epoch_rows.append(
                {
                    "epoch": int(epoch),
                    "link_type": str(link_type),
                    "count_links": int(mask.sum()),
                    "alpha_mean": float(np.mean(alpha_np[mask])),
                    "beta_mean": float(np.mean(beta_np[mask])),
                    "estimated_volume_mean": float(np.mean(pred_flow_np[mask])),
                    "capacity_mean": float(np.mean(cap_np[mask])),
                    "vc_mean": float(np.mean(vc_np[mask])),
                    "vc_p50": float(np.percentile(vc_np[mask], 50)),
                    "vc_p95": float(np.percentile(vc_np[mask], 95)),
                }
            )

    def _collect_flow_demand_rows(self, epoch: int, outputs: Dict, targets: Dict):
        """Store row-wise comparison tables for estimated vs target flow and demand."""
        pred_flow = outputs.get("reconstructed_flows")
        true_flow = targets.get("flows")
        flow_mask = targets.get("mask", targets.get("flow_mask", None))

        if pred_flow is not None and true_flow is not None:
            pred_flow_np = pred_flow.detach().cpu().numpy().reshape(-1)
            true_flow_np = true_flow.detach().cpu().numpy().reshape(-1)
            n = min(len(pred_flow_np), len(true_flow_np))
            if n > 0:
                if flow_mask is not None:
                    flow_mask_np = flow_mask.detach().cpu().numpy().reshape(-1)
                else:
                    flow_mask_np = np.ones(n, dtype=np.float32)
                flow_mask_np = np.asarray(flow_mask_np).reshape(-1)
                m = min(n, len(flow_mask_np))
                for i in range(m):
                    self.flow_comparison_rows.append(
                        {
                            "epoch": int(epoch),
                            "link_index": int(i),
                            "true_flow": float(true_flow_np[i]),
                            "estimated_flow": float(pred_flow_np[i]),
                            "flow_mask": float(flow_mask_np[i]),
                        }
                    )

        pred_od = outputs.get("estimated_demand")
        true_od = targets.get("od")
        od_mask = targets.get("od_mask")
        if pred_od is not None and true_od is not None:
            pred_od_np = pred_od.detach().cpu().numpy().reshape(-1)
            true_od_np = true_od.detach().cpu().numpy().reshape(-1)
            n = min(len(pred_od_np), len(true_od_np))
            if n > 0:
                if od_mask is not None:
                    od_mask_np = od_mask.detach().cpu().numpy().reshape(-1)
                else:
                    od_mask_np = np.ones(n, dtype=np.float32)
                od_mask_np = np.asarray(od_mask_np).reshape(-1)
                m = min(n, len(od_mask_np))
                for i in range(m):
                    self.demand_comparison_rows.append(
                        {
                            "epoch": int(epoch),
                            "od_index": int(i),
                            "true_demand": float(true_od_np[i]),
                            "estimated_demand": float(pred_od_np[i]),
                            "od_mask": float(od_mask_np[i]),
                        }
                    )

    def _write_csv(self, path: str, fieldnames, rows):
        """Write CSV safely, creating parent directories when necessary."""
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def plot_evolution(self, filename: str):
        """Plot R2 and MAE trends across epochs."""
        if len(self.full_history["r2_flow"]) == 0:
            return
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.full_history["r2_flow"], label="R2 Flow", color="tab:blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("R2", color="tab:blue")
        ax1.grid(alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(self.full_history["mae_flow"], label="MAE Flow", color="tab:orange")
        ax2.set_ylabel("MAE", color="tab:orange")

        plt.title("VI Training Evolution")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_physics(self, filename: str):
        """Plot loss decomposition and solver convergence trajectories."""
        if len(self.full_history["iterations"]) == 0 and len(self.full_history["l_flow"]) == 0:
            return
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        ax1.plot(self.full_history["l_flow"], label="l_flow", color="tab:blue", alpha=0.85)
        ax1.plot(self.full_history["l_od"], label="l_od", color="tab:green", alpha=0.85)
        ax1.plot(self.full_history["l_demand_reg"], label="l_demand_reg", color="tab:orange", alpha=0.85)
        ax1.plot(self.full_history["total_loss"], label="total_loss", color="tab:red", alpha=0.65)
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.2)
        ax1.legend(loc="upper right")

        ax2.plot(self.full_history["iterations"], label="iterations", color="tab:purple", alpha=0.85)
        ax2.plot(self.full_history["final_gap"], label="final_gap", color="tab:brown", alpha=0.85)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Solver")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="upper right")

        plt.suptitle("VI Physics Diagnostics")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_gradient_health(self, filename: str):
        """Plot gradient norms over epochs (log scale when values are positive)."""
        grads = self.full_history.get("grad_norms", {})
        if (not grads) and len(self.full_history.get("od_logits_grad", [])) == 0:
            return

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        for name, values in grads.items():
            if values:
                plt.plot(values, label=name)

        if len(self.full_history.get("od_logits_grad", [])) > 0:
            plt.plot(self.full_history["od_logits_grad"], label="od_logits_grad", linewidth=1.6)

        if len(self.full_history.get("max_grad", [])) > 0:
            plt.plot(self.full_history["max_grad"], label="max_grad", linewidth=1.2, alpha=0.8)

        all_vals = []
        for values in grads.values():
            all_vals.extend([float(v) for v in values])
        all_vals.extend([float(v) for v in self.full_history.get("od_logits_grad", [])])
        all_vals.extend([float(v) for v in self.full_history.get("max_grad", [])])
        if any(v > 0.0 for v in all_vals):
            plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title("VI Gradient Health")
        plt.grid(alpha=0.2)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_summary(self, filename: str):
        """Export summary JSON and detailed diagnostic CSV artifacts."""
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        alerts = self.full_history.get("alerts", [])
        alert_counts = {}
        for _, kind, _ in alerts:
            alert_counts[kind] = alert_counts.get(kind, 0) + 1

        summary = {
            "steps": len(self.full_history.get("r2_flow", [])),
            "metrics": {
                "r2_flow_mean": float(np.mean(self.full_history.get("r2_flow", [0.0]))),
                "mae_flow_mean": float(np.mean(self.full_history.get("mae_flow", [0.0]))),
                "total_loss_mean": float(np.mean(self.full_history.get("total_loss", [0.0]))),
                "l_flow_mean": float(np.mean(self.full_history.get("l_flow", [0.0]))),
                "l_od_mean": float(np.mean(self.full_history.get("l_od", [0.0]))),
                "iterations_mean": float(np.mean(self.full_history.get("iterations", [0.0]))),
                "final_gap_mean": float(np.mean(self.full_history.get("final_gap", [0.0]))),
                "od_logits_grad_mean": float(np.mean(self.full_history.get("od_logits_grad", [0.0]))),
            },
            "alerts": {
                "total": len(alerts),
                "counts": alert_counts,
                "examples": [
                    {"epoch": int(epoch), "type": str(kind), "value": float(value)}
                    for epoch, kind, value in alerts[-10:]
                ],
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Export detailed gradient history in CGAME_PhysicsMirrorDescent style.
        grad_csv = os.path.join(out_dir, "gradient_history_detailed.csv")
        self._write_csv(
            grad_csv,
            fieldnames=[
                "epoch",
                "module",
                "param_name",
                "norm2",
                "max_abs",
                "mean_abs",
                "finite_ratio",
                "non_finite_count",
            ],
            rows=self.gradient_detailed_rows,
        )

        # Export a single forward/backward consistency audit file across all epochs.
        fb_audit_csv = os.path.join(out_dir, "forward_backward_consistency_audit.csv")
        self._write_csv(
            fb_audit_csv,
            fieldnames=[
                "epoch",
                "mode",
                "implicit_grad",
                "forward_iterations",
                "forward_final_gap",
                "forward_linearization_iter",
                "backward_info_available",
                "backward_iter_idx",
                "iter_match",
                "backward_fixed_point_residual",
                "backward_z_norm",
                "backward_grad_output_norm",
                "backward_iters",
                "backward_damping",
                "status",
            ],
            rows=self.forward_backward_audit_rows,
        )

        # Export link-type alpha/beta/vc summaries in a single historical CSV.
        diagnostics_root = os.path.dirname(out_dir)
        physics_dir = os.path.join(diagnostics_root, "physics")
        link_type_all_csv = os.path.join(physics_dir, "link_type_alpha_beta_vc_history.csv")
        self._write_csv(
            link_type_all_csv,
            fieldnames=[
                "epoch",
                "link_type",
                "count_links",
                "alpha_mean",
                "beta_mean",
                "estimated_volume_mean",
                "capacity_mean",
                "vc_mean",
                "vc_p50",
                "vc_p95",
            ],
            rows=self.link_type_epoch_rows,
        )

        # Export comparison tables for flows and demands.
        flow_csv = os.path.join(out_dir, "true_vs_estimated_flow_history.csv")
        self._write_csv(
            flow_csv,
            fieldnames=["epoch", "link_index", "true_flow", "estimated_flow", "flow_mask"],
            rows=self.flow_comparison_rows,
        )

        demand_csv = os.path.join(out_dir, "true_vs_estimated_demand_history.csv")
        self._write_csv(
            demand_csv,
            fieldnames=["epoch", "od_index", "true_demand", "estimated_demand", "od_mask"],
            rows=self.demand_comparison_rows,
        )


# Backward compatible names used in early sketches.
EndToEndSemiParametricModel = VariationalInferenceModel
ODEstimationLoss = Loss