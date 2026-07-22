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
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts.runtime_contracts import ArtifactSchemaError, require_keys

#%%

class PhysicsInformedBPRNet(nn.Module):
    """
    Supply module that predicts physically bounded BPR (Bureau of Public Roads) parameters.
    
    Instead of directly regressing travel times, this network estimates the parameters 
    of the BPR function, ensuring the congestion response remains monotonic and physically plausible.
    """
    def __init__(
        self,
        input_dim: int = 6,           # Number of static link features (length, lanes, speed, etc.)
        hidden_dim: int = 64,          # Neurons in the hidden layers of the parameter MLP
        num_link_groups: int = 1,      # Number of unique categories of links (e.g., motorway, primary)
        link_group: Optional[torch.Tensor] = None, # Tensor mapping each link to its group index
        alpha_min: float = 0.1,        # Lower bound for the alpha scaling parameter
        alpha_max: float = 0.5,        # Upper bound for the alpha scaling parameter
        beta_min: float = 1.0,         # Lower bound for the beta exponent (non-linearity)
        beta_max: float = 6.0,         # Upper bound for the beta exponent
        cap_mode: str = "global",      # Granularity of capacity correction: 'spatial', 'group', or 'global'
        cap_min: float = 12.0,         # Lower bound for the capacity multiplier
        cap_max: float = 16.0,         # Upper bound for the capacity multiplier
        mlp_hidden_dim: int = 64,      # Hidden dimension specifically for the capacity MLP in 'spatial' mode
    ):
        super().__init__()
        # Store physical bounds as floats to ensure consistent tensor operations
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.cap_mode = str(cap_mode)
        self.cap_min = float(cap_min)
        self.cap_max = float(cap_max)
        self.num_link_groups = max(int(num_link_groups), 1)

        # Buffer for link classification; buffers are part of the state_dict but not learnable
        if link_group is None:
            link_group = torch.zeros(1, dtype=torch.long)
        self.register_buffer("link_group", link_group.to(dtype=torch.long))

        # Alpha and Beta initialization:
        # Initializing as zeros means sigmoid(0) = 0.5, placing the initial 
        # parameters exactly at the midpoint of the [min, max] range.
        self.alpha_group_raw = nn.Parameter(torch.zeros(self.num_link_groups, dtype=torch.float32))
        self.beta_group_raw = nn.Parameter(torch.zeros(self.num_link_groups, dtype=torch.float32))

        # Generic parameter estimator (not used directly in current BPR structure but kept for future extensions)
        self.param_estimator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

        # Capacity Multiplier Logic Selection:
        if self.cap_mode == "spatial":
            # Link-specific inference: A full MLP maps static link attributes to a capacity multiplier.
            self.capacity_estimator = nn.Sequential(
                nn.Linear(input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, 1),
            )
        elif self.cap_mode == "group":
            # Category-specific inference: Learns one multiplier for each link group (e.g., all 'motorways').
            self.cap_group_raw = nn.Parameter(torch.zeros(num_link_groups))
        elif self.cap_mode == "global":
            # Unified inference: A single scalar multiplier applied to the entire network.
            self.cap_global_raw = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Invalid capacity mode: {cap_mode}. Choose 'spatial', 'group', or 'global'.")

    def forward(self, link_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps raw learnable parameters and static link features to bounded BPR parameters.
        
        Args:
            link_features: [num_links, input_dim] tensor containing normalized 
                           static attributes (length, capacity, etc.).
        
        Returns:
            A dictionary containing the active alpha, beta, and capacity_multiplier 
            for every link in the network.
        """
        
        # --- 1. CAPACITY MULTIPLIER DERIVATION ---
        # Depending on the selected 'cap_mode', we calculate a raw value (cap_mult_raw)
        # which is then mapped to the [cap_min, cap_max] range.
        
        if self.cap_mode == "spatial":
            # Pass link features through the MLP to get a unique multiplier per link.
            cap_mult_raw = self.capacity_estimator(link_features).squeeze(-1)
        elif self.cap_mode == "group":
            # Use the link_group buffer to index the group-specific learnable parameter.
            cap_mult_raw = self.cap_group_raw[self.link_group]
        else:  # global mode
            # Expand a single learnable scalar to match the number of links in the batch.
            cap_mult_raw = self.cap_global_raw.expand(link_features.size(0))

        # Apply Sigmoid to bound the multiplier:
        # result = min + (max - min) * sigmoid(raw).
        capacity_multiplier = self.cap_min + (self.cap_max - self.cap_min) * torch.sigmoid(cap_mult_raw)

        # --- 2. ALPHA AND BETA PARAMETER DERIVATION ---
        # We transform the raw group parameters using the same sigmoid-bounding logic.
        
        # alpha_group: Sensitivity of travel time to the volume/capacity ratio.
        alpha_group = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_group_raw)
        
        # beta_group: The exponent that determines the 'sharpness' of the congestion curve.
        beta_group = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_group_raw)

        # --- 3. BROADCASTING TO LINK LEVEL ---
        # Since alpha and beta are defined per group, we must map them back to individual links.
        
        link_group = self.link_group
        # Safety check: ensure the group mapping matches the input link count.
        if link_group.numel() != link_features.shape[0]:
            # Fallback: if metadata is inconsistent, map all links to group 0 to avoid crashes.
            link_group = torch.zeros(link_features.shape[0], dtype=torch.long, device=link_features.device)
        else:
            link_group = link_group.to(device=link_features.device)

        # Indexing: transform [num_groups] -> [num_links].
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
    def forward(ctx, layer, route_init, od_demands, alpha, beta, cap_mult, theta):
        """
        Runs the traffic assignment solver to reach equilibrium and saves the 
        final state for the implicit gradient calculation later.
        
        Args:
            ctx: Context object used to store information for the backward pass.
            layer: The ImplicitEquilibriumLayer instance (contains solver settings).
            route_init: Starting point for route flows.
            od_demands: The traffic demand between O-D pairs.
            alpha, beta, cap_mult: BPR physical parameters from the supply net.
            theta: Dispersion parameter (only used if in SUE mode).
        """
        # We disable gradient tracking during the solver iterations to save memory.
        # We only care about the final converged state.
        with torch.no_grad():
            route_flows = route_init
            used_iters = layer.max_iterations
            final_gap = 0.0
            converged = False
            relative_gap_history: List[float] = []

            # Pack parameters for the BPR cost function
            params = {
                "alpha": alpha,
                "beta": beta,
                "capacity_multiplier": cap_mult,
                "theta": theta,
            }

            # MAIN SOLVER LOOP: Iteratively approach Wardrop Equilibrium
            for it in range(1, layer.max_iterations + 1):
                # 1. Evaluate current costs and theoretical convergence gap BEFORE moving
                route_costs = layer._route_costs(route_flows, params)
                final_gap = layer._compute_active_cost_gap(route_flows, route_costs)
                relative_gap_history.append(final_gap)
                
                # 2. CONVERGENCE CHECK (Wardrop Principle)
                if layer.use_early_stop and it >= layer.min_iterations and final_gap < layer.tol_rel_flow:
                    used_iters = it
                    converged = True
                    break

                # 3. Perform a Fixed-Point step (Mirror Descent update)
                next_route_flows = layer._mirror_descent_step(
                    route_flows_flat=route_flows,
                    route_costs=route_costs,
                    od_demands=od_demands,
                    bpr_params=params,
                    iter_idx=it,
                )

                route_flows = next_route_flows
                used_iters = it

            # --- IMD STATIONARY TAIL ---
            # To improve gradient stability, we perform a few extra 'frozen' steps.
            # This ensures the point we linearize around is a very stable fixed point.
            linearization_iter = max(1, int(used_iters))
            tail_steps = max(0, int(layer.imd_stationary_tail_steps))
            for _ in range(tail_steps):
                route_flows = layer._fixed_point_step(
                    route_flows=route_flows,
                    od_demands=od_demands,
                    bpr_params=params,
                    iter_idx=linearization_iter,
                )

        # Telemetry: Store solver performance data in the layer for the Diagnostician.
        layer._imd_last_info = {
            "iterations": float(used_iters),
            "final_gap": float(final_gap),
            "converged": float(1.0 if converged else 0.0),
            "implicit_grad": True,
            "linearization_iter": float(linearization_iter),
            "relative_gap_history": relative_gap_history,
        }

        # CONTEXT STORAGE: Save necessary tensors for the backward(ctx, grad_output) call.
        ctx.layer = layer
        ctx.linearization_iter = linearization_iter
        # We must save the converged flows and all inputs that require gradients.

        tensors_to_save = [route_flows, od_demands, alpha, beta, cap_mult]
        if theta is not None:
            tensors_to_save.append(theta)
        ctx.save_for_backward(*tensors_to_save)
        
        return route_flows


    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes gradients for all inputs (OD demand and BPR parameters) using the 
        Implicit Function Theorem and Jacobian-free iterations.
        
        Args:
            ctx: The context object containing saved tensors from the forward pass.
            grad_output: The gradient of the loss with respect to the output route flows.
        """
        layer = ctx.layer
        # Retrieve the converged equilibrium state and inputs saved in forward()
        saved = ctx.saved_tensors

        route_star = saved[0]
        od_demands = saved[1]
        alpha = saved[2]
        beta = saved[3]
        cap_mult = saved[4]
        
        # Asignamos theta solo si sabemos que se guardó
        theta = saved[5] if getattr(ctx, 'has_theta', False) else None

        # We decouple and isolate parameters which WILL NOT vary across the differentiation process with respect to route_star.
        od_const = od_demands.detach()
        alpha_const = alpha.detach()
        beta_const = beta.detach()
        cap_mult_const = cap_mult.detach()
        theta_const = theta.detach() if theta is not None else None

        # Reconstruct the parameter dictionary used in the forward pass
        params_const = {
            "alpha": alpha_const,
            "beta": beta_const,
            "capacity_multiplier": cap_mult_const,
        }

        if theta_const is not None:
            params_const["theta"] = theta_const

        # Point of Equilibrium in which we will iterate WILL DO require gradient
        route_var = route_star.detach().requires_grad_(True)

        # Re-evaluate ONE single step of the solver at the equilibrium point.
        # This is the 'g(x, theta)' mapping used to linearize the system.

        # Explanation:
        # We reached equilibrium in the forward pass without tracking gradients (to save memory). 
        # However, to compute the implicit gradient, we need to know the 'slope' of the system 
        # at that exact point. By re-executing one single step with gradients enabled, we create 
        # a local differentiable bridge. This allows us to compute Vector-Jacobian Products (VJP) 
        # as if we had tracked the entire solver, but with the memory cost of just one iteration.

        # Building the "pure" function which unique argument is route
        def fixed_point_step_fn(route_flows):
            # This functions only looks at the route_flows variable, treating all other inputs as constants.
            return layer._fixed_point_step(
                route_flows=route_flows,
                od_demands=od_const,
                bpr_params=params_const,
                iter_idx=ctx.linearization_iter, # also fixed
            )
        
        # JFB Loop using vjp instead of autograd.grad with retain_graph

        # --- JACOBIAN-FREE FIXED POINT ITERATION (JFB) ---
        # We solve the adjoint equation: z = grad_output + J^T * z
        # where J is the Jacobian of the fixed-point mapping.

        # Explanation:
        # The full gradient in a fixed-point system depends on the term (I - J)^(-1). 
        # Since the Jacobian (J) of a large-scale transport network is too massive to 
        # store or invert, we use the Neumann Series expansion (an iterative method).
        # 
        # Starting with z = grad_output, we repeatedly apply the V-J-P (Vector-Jacobian Product). 
        # In each iteration, the error signal 'z' travels further through the network's 
        # dependencies. We iterate until 'z' converges, effectively reconstructing the 
        # global gradient and capturing how a change in one link affects the entire 
        # equilibrium state.

        # z = grad_output.clone() / (grad_output.norm() + layer.eps) # Normalize to improve numerical stability
        z = grad_output.clone()

        max_backward_iters = max(int(layer.imd_backward_iters), 1)
        residual_history = []
        
        # Damping logic: improves stability if the mapping is not a perfect contraction.
        use_damping = bool(getattr(layer, "imd_use_damping", False))
        damping_factor = float(np.clip(layer.imd_damping, 0.0, 1.0)) if use_damping else 1.0


        for _ in range(max_backward_iters):
            # Compute the Vector-Jacobian Product (VJP): vjp = z^T * (d_route_next / d_route_star)
            _, vjp_route = torch.autograd.functional.vjp(
                func=fixed_point_step_fn,
                inputs=route_var,
                v=z
            )

            # The new estimate for the adjoint vector (Fixed point equation)
            rhs = grad_output + vjp_route

            # NUMERICAL SHIELD: Prevent diverging Neumann series from injecting NaNs
            # If the forward pass oscillated, the spectral radius of the Jacobian might be > 1.
            rhs = torch.nan_to_num(rhs, nan=0.0, posinf=1e5, neginf=-1e5)
                        
            # Convergence tracking for the backward solver
            residual_num = torch.norm(z - rhs, p=2)
            residual_den = torch.norm(rhs, p=2) + layer.eps
            residual_history.append(float((residual_num / residual_den).detach().item()))

            # Apply update (with optional damping for numerical stability)
            if use_damping:
                z = ((1.0 - damping_factor) * z) + (damping_factor * rhs)
            else:
                z = rhs

        # --- FINAL GRADIENT EXTRACTION ---
        # Now that we have the converged adjoint vector 'z', we compute the 
        # final gradients with respect to the actual model parameters.
        with torch.enable_grad():
            od_var = od_demands.detach().requires_grad_(True)
            alpha_var = alpha.detach().requires_grad_(True)
            beta_var = beta.detach().requires_grad_(True)
            cap_var = cap_mult.detach().requires_grad_(True)
            theta_var = theta.detach().requires_grad_(True) if theta is not None else None

            params_var = {
                "alpha": alpha_var,
                "beta": beta_var,
                "capacity_multiplier": cap_var,
            }
            if theta_var is not None:
                params_var["theta"] = theta_var

            # Construimos el subgrafo final que une inputs reales con la salida
            route_next_final = layer._fixed_point_step(
                route_flows=route_star, # Punto fijo como input constante
                od_demands=od_var,
                bpr_params=params_var,
                iter_idx=ctx.linearization_iter,
            )

            inputs = [od_var, alpha_var, beta_var, cap_var]
            if theta_var is not None:
                inputs.append(theta_var)

            # Extraemos el gradiente proyectando el z convergido
            grads = torch.autograd.grad(
                outputs=route_next_final,
                inputs=tuple(inputs),
                grad_outputs=z,
                allow_unused=True,
            )

        fixed_point_residual = float(residual_history[-1] if residual_history else 0.0)
        layer._imd_last_backward_info = {
            "available": 1.0,
            "iter_idx": float(ctx.linearization_iter),
            "fixed_point_residual": fixed_point_residual,
            "backward_iters": float(max_backward_iters),
            "z_norm": float(torch.norm(z, p=2).detach().item()),
            "grad_output_norm": float(torch.norm(grad_output, p=2).detach().item()),
            "damping": damping_factor if use_damping else 1.0,
        }

        return None, None, grads[0], grads[1], grads[2], grads[3], (grads[4] if theta is not None else None)


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
        delta_matrix: torch.Tensor,        # Sparse matrix [Links x Routes] mapping paths to links
        route_validity_mask: torch.Tensor, # Boolean mask [OD x K] for valid pre-computed paths
        t0: torch.Tensor,                  # Free-flow travel times for each link
        capacity: torch.Tensor,            # Nominal capacity of each link
        solver_cfg: Optional[dict] = None, # Configuration dictionary for solver settings
        imd_cfg: Optional[dict] = None,    # Configuration for Implicit Model Differentiation (IMD)
    ):
        super().__init__()

        self.eps = 1e-9 # Small constant to prevent division by zero

        # --- Solver hyperparameters ---
        self.solver_cfg = dict(solver_cfg or {})
       
        self.eta = float(self.solver_cfg.get("eta", 1.0)) # Step size for the entropy update
        self.max_iterations = int(self.solver_cfg.get("max_iterations", 50)) # maximum steps for the Mirror Descent solver
        self.min_iterations = int(self.solver_cfg.get("min_iterations", 5))  # Minimum steps to run before allowing early stop
        self.tol_rel_flow = float(self.solver_cfg.get("tol_rel_flow", 1e-4))  # Convergence tolerance for relative flow change
        self.use_early_stop = bool(self.solver_cfg.get("use_early_stop", True))  # Whether to stop before max_iterations if converged
        
        # sue_mode: If True, solves for Stochastic User Equilibrium (SUE) 
        # which accounts for driver perception errors via a logit-like distribution.
        self.sue_mode = bool(self.solver_cfg.get("sue_mode", False))
        self.initial_theta = float(self.solver_cfg.get("initial_theta", 0.85)) # Initial dispersion parameter for SUE
        self.min_theta = float(self.solver_cfg.get("min_theta", 0.1)) # Minimum theta for SUE (if enabled)
        self.max_theta = float(self.solver_cfg.get("max_theta", 10.0)) # Maximum theta for SUE (if enabled)


        # --- IMD (Implicit Model Differentiation) Configuration ---
        imd_cfg = dict(imd_cfg or {})
        self.imd_enabled = bool(imd_cfg.get("enabled", True))
        self.imd_backward_iters = int(imd_cfg.get("backward_iters", 10))
        self.imd_use_damping = bool(imd_cfg.get("use_damping", False))
        self.imd_damping = float(imd_cfg.get("damping", 0.5))
        self.imd_stationary_tail_steps = int(imd_cfg.get("stationary_tail_steps", 0))
        self.imd_warm_start = bool(imd_cfg.get("warm_start", True))
        
    
        # --- NETWORK TOPOLOGY REGISTRATION ---
        # We register these as buffers so they move with the model (CPU/GPU) 
        # but are not considered learnable parameters.
        self.register_buffer("delta_matrix", delta_matrix.coalesce()) 
        self.register_buffer("route_validity_mask", route_validity_mask.bool())
        self.register_buffer("t0", t0.float())
        self.register_buffer("capacity", capacity.float())

        # Metadata for indexing and reshaping
        self.num_od, self.k_paths = self.route_validity_mask.shape
        self.num_routes = self.num_od * self.k_paths
        
        # Flow caching for Warm Start: allows the solver to start from the 
        # last known solution, drastically reducing iterations in training.
        self.register_buffer("cached_route_flows", torch.zeros(1, self.num_routes))
        self.register_buffer("cache_valid", torch.tensor(False, dtype=torch.bool))


    def _compute_active_cost_gap(self, route_flows_flat: torch.Tensor, route_costs: torch.Tensor, flow_threshold: float = 1e-3) -> float:
        """
        Evaluates the Wardrop equilibrium condition: variance of costs among active routes.
        """
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        active_mask = (route_flows > flow_threshold) & self.route_validity_mask.unsqueeze(0)
        
        # Mask unused routes with a huge number so they don't affect the minimum
        masked_costs_min = torch.where(active_mask, route_costs, torch.full_like(route_costs, 1e9))
        min_active_cost, _ = masked_costs_min.min(dim=2, keepdim=True)
        
        # Mask unused routes with zero so they don't affect the maximum
        masked_costs_max = torch.where(active_mask, route_costs, torch.zeros_like(route_costs))
        max_active_cost, _ = masked_costs_max.max(dim=2, keepdim=True)
        
        cost_gap = max_active_cost - min_active_cost
        relative_cost_gap = cost_gap / torch.clamp(min_active_cost, min=self.eps)
        
        # Average over all OD pairs that have at least one active route
        has_active = active_mask.any(dim=2)
        if has_active.any():
            return float(relative_cost_gap[has_active].mean().item())
        return 0.0


    def _route_to_link_flows(self, route_flows: torch.Tensor) -> torch.Tensor:
        """
        Projects route-level flows [Batch, Routes] to link-level flows [Batch, Links].
        
        Logic: 
        Uses the sparse incidence matrix (Delta) to perform: link_flows = Delta * route_flows.
        This is a purely topological operation that satisfies flow conservation.
        """
        # Multiplication in the dual space: [L, R] * [R, B] -> [L, B]. 
        # Then we transpose back to [B, L].
        return torch.sparse.mm(self.delta_matrix, route_flows.t()).t()


    def _route_costs(self, route_flows: torch.Tensor, bpr_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the total travel cost for every path [Batch, OD, K] based on current congestion.
        
        Logic:
        1. Convert route flows to link flows.
        2. Apply the BPR physical function to get travel times for each link.
        3. Use the transpose of Delta to sum link costs into path costs.
        """
        # Step 1: Get the link-level volumes
        link_flows = self._route_to_link_flows(route_flows)
        
        # Step 2: Compute link travel times using the Physics-Informed BPR parameters
        link_costs = self._compute_bpr_cost(link_flows, bpr_params)
        
        # Step 3: Aggregate link costs into route costs: [R, L] * [L, B] -> [R, B]
        route_costs_flat = torch.sparse.mm(self.delta_matrix.t(), link_costs.t()).t()
        
        # Reshape to a structured view [Batch, Origins-Destinations, K-Paths]
        return route_costs_flat.view(-1, self.num_od, self.k_paths)

    
    def _fixed_point_step(
        self,
        route_flows: torch.Tensor,
        od_demands: torch.Tensor,
        bpr_params: Dict[str, torch.Tensor],
        iter_idx: int,
    ) -> torch.Tensor:
        """
        Executes one complete iteration of the equilibrium solver.
        
        Logic:
        This is the g(x, theta) mapping we discussed in the IMD section. It takes the 
        current flow state, evaluates the 'pain' (costs), and returns the next flow 
        state that is slightly closer to equilibrium.
        """
        # 1. Evaluate the costs of the current flow distribution
        route_costs = self._route_costs(route_flows, bpr_params)
        
        # 2. Perform the Mirror Descent update (Entropy-based)
        # This step redistributes flows from high-cost paths to low-cost paths.
        return self._mirror_descent_step(
            route_flows_flat=route_flows,
            route_costs=route_costs,
            od_demands=od_demands,
            bpr_params=bpr_params,
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

        # 1. PREPARATION
        od_demands = self._ensure_2d(od_demands)  # [B, OD]
        # Initialize route flows, potentially using the cache from previous batches.
        route_init = self._initialize_flows(
            od_demands,
            warm_start=self.imd_warm_start and self.training,
        )

        # 2. SOLVER EXECUTION (IMD vs. Explicit)
        if self.imd_enabled and self.training:
            # --- IMPLICIT GRADIENT PATH ---
            # We call the custom autograd function. This is a "black box" 
            # for PyTorch's memory manager.
            current_route_flows = IMDEquilibriumFunction.apply(
                self,
                route_init,
                od_demands,
                bpr_params["alpha"],
                bpr_params["beta"],
                bpr_params["capacity_multiplier"],
                bpr_params.get("theta", None),
            )
            
            # Map the resulting route flows to physical link flows (counts).
            final_link_flows = self._route_to_link_flows(current_route_flows)
            
            # Pack convergence info gathered during the solver's execution.
            info = {
                "iterations": float(self._imd_last_info["iterations"]),
                "final_gap": float(self._imd_last_info["final_gap"]),
                "converged": float(self._imd_last_info["converged"]),
                "linearization_iter": float(self._imd_last_info.get("linearization_iter", self._imd_last_info["iterations"])),
                "relative_gap_history": list(self._imd_last_info.get("relative_gap_history", [])),
                "implicit_grad": True,
            }

            # Update cache for the next forward pass.
            self.cached_route_flows.copy_(current_route_flows.detach().mean(dim=0, keepdim=True))
            self.cache_valid.fill_(True)
            return final_link_flows, current_route_flows, info

        # --- EXPLICIT SOLVER PATH (For Inference or non-IMD training) ---
        current_route_flows = route_init
        used_iters = self.max_iterations
        final_gap = 0.0
        converged = False
        relative_gap_history: List[float] = []

        for it in range(1, self.max_iterations + 1):
            # 1. Evaluate costs and gap
            route_costs = self._route_costs(current_route_flows, bpr_params)
            final_gap = self._compute_active_cost_gap(current_route_flows, route_costs)
            relative_gap_history.append(final_gap)
            
            # 2. Check early stop
            if self.use_early_stop and it >= self.min_iterations and final_gap < self.tol_rel_flow:
                used_iters = it
                converged = True
                break

            # 3. Take one step closer to equilibrium
            next_route_flows = self._mirror_descent_step(
                route_flows_flat=current_route_flows,
                route_costs=route_costs,
                od_demands=od_demands,
                bpr_params=bpr_params,
                iter_idx=it,
            )

            current_route_flows = next_route_flows
            used_iters = it

        final_link_flows = self._route_to_link_flows(current_route_flows)

        info = {
            "iterations": float(used_iters),
            "final_gap": float(final_gap),
            "converged": float(1.0 if converged else 0.0),
            "relative_gap_history": relative_gap_history,
            "implicit_grad": False,
        }

        # Update cache.
        self.cached_route_flows.copy_(current_route_flows.detach().mean(dim=0, keepdim=True))
        self.cache_valid.fill_(True)
        return final_link_flows, current_route_flows, info

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x


    def _initialize_flows(self, od_demands: torch.Tensor, warm_start: bool = False) -> torch.Tensor:
        """
        Initializes feasible route flows using either a Warm Start (cache) 
        or a uniform distribution across valid routes.
        """
        # WARM START LOGIC: If we have a solution from a previous batch, we reuse it.
        # This is the "high-value friction" mentioned in your rules: it leverages 
        # previous analytical work to reach the current goal faster.
        if warm_start and bool(self.cache_valid.item()) and self.cached_route_flows.shape[1] == self.num_routes:
            cached = self.cached_route_flows.to(device=od_demands.device, dtype=od_demands.dtype)
            route_flows = cached.expand(od_demands.shape[0], -1)
            # We must project the cache to ensure it matches the CURRENT demand (which might have changed).
            return self._project_to_feasible_route_flows(route_flows, od_demands)

        # COLD START LOGIC: If no cache exists, we split the demand equally 
        # among all valid (pre-calculated) paths for each OD pair.
        valid = self.route_validity_mask.float()  # [OD, K]
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        probs = valid / denom  # Uniform probability across valid paths
        route_flows = od_demands.unsqueeze(-1) * probs.unsqueeze(0)  # [B, OD, K]
        return route_flows.reshape(od_demands.shape[0], self.num_routes)


    def _project_to_feasible_route_flows(self, route_flows_flat: torch.Tensor, od_demands: torch.Tensor) -> torch.Tensor:
        """
        Projects arbitrary route flows into nonnegative OD-wise simplex constraints.
        Ensures flows sum to demand and no path has negative flow.
        """
        # Reshape to 3D structure [Batch, OD-Pairs, K-Paths] for vectorized math
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        valid = self.route_validity_mask.unsqueeze(0).expand(route_flows.shape[0], -1, -1)
        
        # Call the core mathematical projection
        next_flows = self._project_to_masked_demand_simplex(
            values=route_flows,
            od_demands=od_demands,
            valid=valid,
        )
        return next_flows.reshape(route_flows.shape[0], self.num_routes)


    def _project_to_masked_demand_simplex(
        self,
        values: torch.Tensor,    # Current flows (possibly infeasible)
        od_demands: torch.Tensor, # Target total demand per OD
        valid: torch.Tensor,      # Mask of which paths are physically possible
    ) -> torch.Tensor:
        """
        Vectorized Euclidean projection onto masked OD-wise simplices.
        Algorithm: Solves the sorted-Lagrangian dual problem for the simplex constraint.
        """
        demands = torch.clamp(od_demands.unsqueeze(-1), min=0.0)
        valid_f = valid.to(dtype=values.dtype)
        has_valid = valid.any(dim=2, keepdim=True)

        # 1. MASKING: Push invalid entries to a very low value so they end up 
        # at the end of the sort and don't affect the threshold calculation.
        neg_large = torch.full_like(values, -1e9)
        values_masked = torch.where(valid, values, neg_large)

        # 2. SORTING: Sort path flows in descending order.
        sorted_values, sorted_idx = torch.sort(values_masked, dim=2, descending=True)
        sorted_valid = torch.gather(valid_f, dim=2, index=sorted_idx)

        # 3. THRESHOLD CALCULATION (The Lagrangian multiplier 'tau'):
        # We find the 'cut-off' value (tau) such that (values - tau) sums to demand.
        sorted_values_valid = sorted_values * sorted_valid
        cumsum_values = torch.cumsum(sorted_values_valid, dim=2)
        cumsum_counts = torch.cumsum(sorted_valid, dim=2)
        denom = torch.clamp(cumsum_counts, min=1.0)

        tau_candidates = (cumsum_values - demands) / denom
        
        # Identify which elements stay positive after subtracting the threshold.
        active = (sorted_valid > 0.0) & ((sorted_values - tau_candidates) > 0.0)
        rho = active.sum(dim=2, keepdim=True).clamp(min=1)
        rho_idx = (rho - 1).long()

        # Gather the final threshold 'tau' based on the last active element (rho).
        tau_num = torch.gather(cumsum_values, dim=2, index=rho_idx)
        tau_den = torch.gather(cumsum_counts, dim=2, index=rho_idx).clamp(min=1.0)
        tau = (tau_num - demands) / tau_den

        # 4. FINAL ASSIGNMENT: Apply the threshold and clamp at zero.
        projected = torch.clamp(values - tau, min=0.0) * valid_f
        
        # Safety: If an OD pair has no valid paths, its flow remains zero.
        projected = torch.where(has_valid, projected, torch.zeros_like(projected))
        return projected
    

    def _compute_bpr_cost(self, link_flows: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes link travel time using bounded BPR parameters and adjusted capacity.
        
        Formula: t = t0 * (1 + alpha * (flow / (capacity * multiplier))^beta)
        """
        alpha = params["alpha"]
        beta = params["beta"]
        cap_mult = params["capacity_multiplier"]

        # Ensure parameters are broadcastable across the batch
        if alpha.dim() == 1: alpha = alpha.unsqueeze(0)
        if beta.dim() == 1: beta = beta.unsqueeze(0)
        if cap_mult.dim() == 1: cap_mult = cap_mult.unsqueeze(0)

        # Apply the capacity multiplier to the static capacity. 
        # We clamp at eps to avoid division by zero if capacity is 0.
        adj_capacity = (self.capacity.unsqueeze(0) * cap_mult).clamp(min=self.eps)
        
        # v_over_c: The degree of saturation. 
        # We clamp at 10.0 to prevent numerical overflow in the power (beta) function.
        v_over_c = link_flows / adj_capacity

        if v_over_c.max() > 10.0:
            clamping = 10.0
            logging.warning("v/c ratio: %f", v_over_c.max(), "applying clamping at ", clamping, " to prevent numerical issues.")
            v_over_c = torch.clamp(link_flows / adj_capacity, min=0.0, max=10.0)

        return self.t0.unsqueeze(0) * (1.0 + alpha * torch.pow(v_over_c, beta))

    def _mirror_descent_step(
        self,
        route_flows_flat: torch.Tensor,
        route_costs: torch.Tensor,
        od_demands: torch.Tensor,
        bpr_params: Dict[str, torch.Tensor],
        iter_idx: int,
    ) -> torch.Tensor:
        """
        Performs one entropy-based mirror descent update to redistribute route flows.
        This step is equivalent to a Logit-based update in discrete choice modeling.
        """
        # Reshape to separate OD pairs and their K paths: [Batch, OD_Pairs, K_Paths]
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        valid = self.route_validity_mask.unsqueeze(0).expand(route_flows.shape[0], -1, -1)
        valid_f = valid.to(dtype=route_flows.dtype)
        has_valid = valid.any(dim=2, keepdim=True)

        demands = torch.clamp(od_demands.unsqueeze(-1), min=0.0)
        denom = torch.clamp(demands, min=self.eps)

        # 1. Convert absolute flows to route choice probabilities (sigma)
        sigma = (route_flows * valid_f) / denom
        # Clamp to avoid log(0) while maintaining valid path selection
        sigma = torch.clamp(sigma, min=self.eps) * valid_f
        sigma = sigma / torch.clamp(sigma.sum(dim=2, keepdim=True), min=self.eps)

        # 2. Adaptive learning rate calculation
        # Constant step size avoids artificial stagnation, allowing the true Wardrop gap to close.
        eta_t = self.eta

        # 3. SUE MODIFICATION (Stochastic Behavior)
        # If enabled, we add a term that represents the variance in driver perception.
        if getattr(self, "sue_mode", False) and bpr_params.get("theta", None) is not None:
            theta = bpr_params["theta"]
            
            # PRAGMATIC SAFETY: To keep the mapping as a contraction, 
            # we ensure the step size eta_t doesn't exceed the dispersion theta.
            safe_eta_t = torch.clamp(torch.tensor(eta_t, device=theta.device), max=theta * 0.95)
            eta_t_val = float(safe_eta_t.item())

            # The entropy penalty adds 'noise' to the costs, making choice less deterministic.
            entropy_penalty = (1.0 / theta) * torch.log(sigma + self.eps)
            route_costs = route_costs + entropy_penalty
            eta_t = eta_t_val

        # 4. LOGIT UPDATE (Numerically Stable)
        neg_large = torch.full_like(route_costs, -1e9)
        raw_logits = torch.log(sigma + self.eps) - (eta_t * route_costs)
        
        # Pragmatic shift to prevent e^(large negative) underflow to exactly 0.0
        # By shifting the maximum logit per OD pair to 0, we ensure the denominator 
        # of the softmax is always >= 1.0, making division by zero impossible.
        max_logits, _ = raw_logits.max(dim=2, keepdim=True)
        shifted_logits = raw_logits - max_logits
        
        logits = torch.where(valid, shifted_logits, neg_large)

        # 5. RE-NORMALIZATION
        # Apply exponential and manually normalize to maintain physical interpretability
        exp_logits = torch.exp(logits) * valid_f
        sigma_next = exp_logits / torch.clamp(exp_logits.sum(dim=2, keepdim=True), min=self.eps)

        # 6. RETURN TO FLOW SPACE
        # Convert probabilities back to vehicle counts (flows).
        next_flows = demands * sigma_next
        next_flows = torch.where(has_valid, next_flows, torch.zeros_like(next_flows))
        
        return next_flows.reshape(route_flows.shape[0], self.num_routes)


class VariationalInferenceModel(nn.Module):
    """
    End-to-end Variational Inference model for Traffic Assignment.
    
    Coordinates the PhysicsInformedBPRNet (Supply) and the learnable OD demand (Demand)
    to minimize the discrepancy between observed and simulated traffic flows.
    """
    def __init__(
        self,
        num_links: int,               # Total number of links (edges) in the network
        num_od_pairs: int,            # Total number of Origin-Destination pairs
        delta_matrix: torch.Tensor,    # Sparse incidence matrix
        route_validity_mask: torch.Tensor, # Mask for pre-computed valid paths
        od_pair_indices: torch.Tensor, # Indices mapping OD pairs to nodes
        t0: torch.Tensor,             # Free-flow travel times
        capacity: torch.Tensor,       # Link capacities
        length: torch.Tensor,         # Link lengths
        lanes: torch.Tensor,          # Number of lanes per link
        speed: torch.Tensor,          # Speed limits
        link_group: torch.Tensor,     # Mapping of links to functional groups
        num_link_groups: int,         # Number of unique link groups
        # Configuration blocks yaml
        solver: Dict,
        imd: Dict,
        loss: Dict,
        capacity_correction: Dict,
        od_estimation_policy: Dict,
        architecture: Dict,           # Config object for the supply network architecture
        # General Parameters
        initial_mean: float = 1.0,    # Starting average value for OD demand
        link_scale: float = 1.0,      # Normalization factor for link flows
        od_scale: float = 1.0,        # Normalization factor for OD demand
        **kwargs,
    ):
        super().__init__()

        # 0. CONFIGURATION PARSING
        self.solver_cfg = dict(solver)
        self.imd_cfg = dict(imd)
        self.loss_cfg = dict(loss)
        self.capacity_correction_cfg = dict(capacity_correction)
        self.od_estimation_policy_cfg = dict(od_estimation_policy)
        self.architecture_cfg = dict(architecture)

        # 1. PARAMETER AND SCALE REGISTRATION
        self.num_links = int(num_links)
        self.num_od_pairs = int(num_od_pairs)
        self.num_link_groups = int(num_link_groups)
        self.link_scale = float(link_scale)
        self.od_scale = float(od_scale)
        self.initial_mean = float(initial_mean)
        
        # Hyperparameters for learning rate decoupling
        self.supply_lr_multiplier = float(kwargs.get("supply_lr_multiplier"))
        self.demand_lr_multiplier = float(kwargs.get("demand_lr_multiplier"))
        
        # Strategy flags for OD initialization
        self.apply_od_init = kwargs.get("apply_od_init", bool(self.od_estimation_policy_cfg.get("apply_od_init", False)))
        self.anchor_known_od_in_solver = bool(self.od_estimation_policy_cfg.get("anchor_known_od_in_solver", False))
        self.unknown_od_init_value = float(kwargs.get("unknown_od_init_value", 0.0))
        self._unknown_od_init_applied = False # Toggle to ensure one-time OD initialization

        # Register static network attributes as buffers (persistent but not learnable)
        self.register_buffer("t0", t0.float())
        self.register_buffer("capacity", capacity.float())
        self.register_buffer("length", length.float())
        self.register_buffer("lanes", lanes.float())
        self.register_buffer("speed", speed.float())
        self.register_buffer("link_group", link_group.long())
        self.register_buffer("od_pair_indices", od_pair_indices.long())

        # 2. SUPPLY NETWORK INSTANTIATION (PhysicsInformedBPRNet)
        self.supply_net = PhysicsInformedBPRNet(
            input_dim=6, # Standard link features stack
            hidden_dim=int(getattr(architecture, "hidden_dim", 64)),
            num_link_groups=self.num_link_groups,
            link_group=self.link_group,
            alpha_min=float(kwargs.get("alpha_min", 0.1)),
            alpha_max=float(kwargs.get("alpha_max", 0.5)),
            beta_min=float(kwargs.get("beta_min", 1.0)),
            beta_max=float(kwargs.get("beta_max", 6.0)),
            cap_mode=str(kwargs.get("cap_mode", "group")),
            cap_min=float(self.capacity_correction_cfg.get("min_value")),
            cap_max=float(self.capacity_correction_cfg.get("max_value")),
            mlp_hidden_dim=int(kwargs.get("mlp_hidden_dim", 64))
        )

        # 3. STOCHASTIC USER EQUILIBRIUM (SUE) SETUP
        self.sue_mode = bool(self.solver_cfg.get("sue_mode"))
        
        if self.sue_mode:
            # Theta bounds to ensure numerical stability
            theta_min = float(self.solver_cfg.get("min_theta"))
            theta_max = float(self.solver_cfg.get("max_theta"))
            
            self.register_buffer("theta_min", torch.tensor(theta_min))
            self.register_buffer("theta_max", torch.tensor(theta_max))

            init_theta = float(self.solver_cfg.get("initial_theta"))
            
            # Map target theta back to raw logit space via inverse sigmoid
            init_norm = (init_theta - self.theta_min) / (self.theta_max - self.theta_min + 1e-9)
            init_norm = np.clip(init_norm, 0.01, 0.99)
            inv_sig = float(np.log(init_norm / (1.0 - init_norm)))
            
            
            # Parameter which optimizer will update.
            self.theta_raw = nn.Parameter(torch.tensor(inv_sig, dtype=torch.float32))

        # 4. EQUILIBRIUM SOLVER LAYER
        self.equilibrium_solver = ImplicitEquilibriumLayer(
            delta_matrix=delta_matrix,
            route_validity_mask=route_validity_mask,
            t0=t0,
            capacity=capacity,
            solver_cfg=self.solver_cfg,
            imd_cfg=self.imd_cfg,
        )

        # 5. OD LOGITS INITIALIZATION
        # We start with a target normalization of (mean / scale)
        init_target_norm = max(float(initial_mean) / max(self.od_scale, 1e-6), 1e-6)
        # Use inverse softplus so that softplus(init_raw) * scale = initial_mean
        init_raw = self._inverse_softplus(init_target_norm)
        self.od_logits = nn.Parameter(torch.full((self.num_od_pairs,), init_raw, dtype=torch.float32))

        # 6. LOSS FUNCTION
        self.loss_fn = Loss(
            link_scale=self.link_scale,
            od_scale=self.od_scale,
            **dict(loss or {})
        )

        # 7. INSTANTIATE DELEGATOR
        self.delegator = VIModelDelegator(self)

    def get_optimizer_param_groups(self, base_lr: float):
            """
            Assigns different learning rates to supply and demand parameters.
            
            Logic:
            Demand parameters (OD) often need a higher LR to move the matrix 
            significantly, while supply parameters (alpha, beta) need a lower, 
            more stable LR to avoid physically unrealistic oscillations.
            """
            # 1. Capture Demand parameters (the OD matrix logits)
            demand_params = [self.od_logits]
            
            # 2. Capture ALL active Supply parameters 
            # This includes the BPR parameters (alpha, beta) and capacity parameters.
            supply_params = [
                p for name, p in self.supply_net.named_parameters() 
                if p.requires_grad
            ]

            # 3. Add SUE parameter (theta) to the supply group if active
            if self.sue_mode and hasattr(self, "theta_raw") and self.theta_raw.requires_grad:
                supply_params.append(self.theta_raw)

            # 4. Apply multipliers defined in the YAML config
            supply_lr = base_lr * self.supply_lr_multiplier
            demand_lr = base_lr * self.demand_lr_multiplier

            return [
                {"params": supply_params, "lr": supply_lr},
                {"params": demand_params, "lr": demand_lr},
            ]


    def _build_link_features(self) -> torch.Tensor:
        """
        Assembles and normalizes static link attributes into a feature tensor.
        
        Logic:
        Neural networks converge faster when inputs are zero-centered or 
        normalized to a similar scale. We divide each attribute by its mean.
        """
        eps = 1e-6
        # Normalizing by the mean ensures that features are centered around 1.0
        t0_n = self.t0 / (self.t0.mean() + eps)
        cap_n = self.capacity / (self.capacity.mean() + eps)
        len_n = self.length / (self.length.mean() + eps)
        lanes_n = self.lanes / (self.lanes.mean() + eps)
        speed_n = self.speed / (self.speed.mean() + eps)

        # Group normalization: represents road hierarchy in a [0, 1] range
        if self.num_link_groups > 1:
            group_n = self.link_group.float() / float(self.num_link_groups - 1)
        else:
            group_n = torch.zeros_like(self.link_group, dtype=torch.float32)

        # Stack features into a [Num_Links, 6] tensor
        return torch.stack([len_n, t0_n, speed_n, lanes_n, group_n, cap_n], dim=1)


    @staticmethod
    def _inverse_softplus(value: float) -> float:
        x = max(float(value), 1e-12)
        if x > 20.0:
            return x
        return float(np.log(np.expm1(x) + 1e-12))
    

    @staticmethod
    def _inverse_softplus_tensor(x: torch.Tensor) -> torch.Tensor:
        # Usamos la implementación nativa de PyTorch que ya es estable numéricamente
        # y maneja el umbral de 20.0 internamente para evitar infinitos.
        return F.softplus(x)


    def _maybe_apply_unknown_od_init(
            self,
            true_od_sparse: Optional[torch.Tensor],
            od_mask_sparse: Optional[torch.Tensor],
        ) -> None:
            """
            One-time initialization policy for the OD demand matrix.
            
            Ensures that known OD entries start at their supervised targets, 
            while unknown entries start from a controlled baseline.
            """
            # Not apply unknown od initialization or already applied
            if not self.apply_od_init or self._unknown_od_init_applied:
                return
            if true_od_sparse is None or od_mask_sparse is None:
                return

            # 1. Identify known cells and their target values
            # od_mask_sparse: 1.0 = known, 0.0 = unknown
            if od_mask_sparse.dim() == 2:
                # Handle batched masks by checking if any batch element is supervised
                known_mask = (od_mask_sparse > 0.5).any(dim=0)
                # Compute the mean target across batches for initialization
                known_target = (true_od_sparse * od_mask_sparse).sum(dim=0) / torch.clamp(od_mask_sparse.sum(dim=0), min=1.0)
            else:
                known_mask = od_mask_sparse > 0.5
                known_target = true_od_sparse

            known_target = known_target.to(device=self.od_logits.device, dtype=self.od_logits.dtype)
            known_target = torch.clamp(known_target, min=0.0)

            # 2. Assign values to unknown cells
            # We use the strategy value defined in the YAML config
            unknown_target = torch.full_like(self.od_logits, float(max(self.unknown_od_init_value, 0.0)))

            # 3. Combine: Real targets for known, Baseline for unknown
            init_target = torch.where(known_mask.to(device=self.od_logits.device), known_target, unknown_target)

            # 4. Normalize by scale before converting to logit space
            # Neural networks optimize normalized values for better gradient flow
            init_norm = torch.clamp(init_target / max(self.od_scale, 1e-6), min=1e-12)

            # 5. Apply the Inverse Softplus transformation
            # Formula: x = ln(exp(y) - 1). This ensures that softplus(logits) = init_norm
            with torch.no_grad():
                # For values > 20, softplus(x) is almost exactly x, so we skip the exp()
                # to prevent numerical overflow (Infinity)
                inv_softplus = torch.where(
                    init_norm > 20.0,
                    init_norm,
                    torch.log(torch.expm1(init_norm) + 1e-12)
                )
                # Copy values into the learnable parameter
                self.od_logits.copy_(inv_softplus)

            self._unknown_od_init_applied = True


    def _learned_od(self, batch_size: int) -> torch.Tensor:
        """
        Decodes learnable logits into positive OD demand and broadcasts to batch size.
        
        Logic:
        1. Apply Softplus: Ensures demand is always >= 0.
        2. Multiply by od_scale: Returns the value to its original physical magnitude.
        3. Expand: Broadcasts the static vector to match the current training batch size.
        """
        # Softplus(x) = ln(1 + exp(x)). It is a smooth version of ReLU.
        od_pos = F.softplus(self.od_logits) * self.od_scale
        
        # Returns [Batch, Num_OD_Pairs]
        return od_pos.unsqueeze(0).expand(batch_size, -1)


    def _align_od_targets(
        self,
        true_od: torch.Tensor,
        od_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aligns external OD supervision data to the internal model indexing.
        
        Logic:
        The function attempts three strategies to find the correct mapping:
        1. Label-space: Uses string node IDs (e.g., 'Node_101') if provided.
        2. Index-space: Uses raw integer node IDs from the graph.
        3. Fallback: Direct truncation if dimensions happen to match.
        """
        if true_od.dim() == 1:
            true_od = true_od.unsqueeze(0)
        if od_mask is None:
            od_mask = torch.ones_like(true_od)
        elif od_mask.dim() == 1:
            od_mask = od_mask.unsqueeze(0)

        # Quick check: If dimensions already match, no alignment needed.
        if true_od.shape[1] == self.num_od_pairs:
            return true_od, od_mask

        dense_dim = true_od.shape[1]
        n_nodes = int(round(dense_dim ** 0.5))

        # STRATEGY 1: Label-space alignment (Highest precision)
        # Used when nodes have specific names/labels in the dataset.
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

                # Invalidate entries where no node mapping was found
                if torch.any(~valid):
                    sparse_true[:, ~valid] = 0.0
                    sparse_mask[:, ~valid] = 0.0

                return sparse_true, sparse_mask

        # STRATEGY 2: Index-space mapping
        # Maps (Origin_Node_ID, Destination_Node_ID) to flat indices.
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

        # STRATEGY 3: Fallback (Truncate/Pad)
        # Final safety measure to prevent code crashes, though potentially inaccurate.
        cut = min(self.num_od_pairs, true_od.shape[1])
        padded_true = torch.zeros(true_od.shape[0], self.num_od_pairs, device=true_od.device)
        padded_mask = torch.zeros(od_mask.shape[0], self.num_od_pairs, device=od_mask.device)
        padded_true[:, :cut] = true_od[:, :cut]
        padded_mask[:, :cut] = od_mask[:, :cut]
        return padded_true, padded_mask
    

    def forward(
        self,
        observed_flows: torch.Tensor,      # Traffic counts observed on links [Batch, Links]
        flow_mask: Optional[torch.Tensor] = None, # Mask for links with available sensors
        true_od_demand: Optional[torch.Tensor] = None, # Real OD matrix (supervision)
        od_mask: Optional[torch.Tensor] = None, # Mask for known vs. unknown OD pairs
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs one end-to-end Variational Inference pass.
        """
        # Ensure inputs are at least 2D for batch consistency
        if observed_flows.dim() == 1:
            observed_flows = observed_flows.unsqueeze(0)
        batch_size = observed_flows.shape[0]

        is_pure_inference = kwargs.get("is_pure_inference", False)

        # 1. SUPPLY: Map static features to bounded physical BPR parameters
        link_features = self._build_link_features()
        bpr_params = self.supply_net(link_features)

        # Handle learnable theta for Stochastic User Equilibrium (SUE)
        if self.sue_mode and hasattr(self, "theta_raw"):
            # Mapping: raw -> sigmoid -> [theta_min, theta_max]
            # This ensures that theta stays within the range specified in the YAML config.
            theta_val = self.theta_min + (self.theta_max - self.theta_min) * torch.sigmoid(self.theta_raw)
            bpr_params["theta"] = theta_val

        # 2. ALIGNMENT: Map external OD targets to the model's sparse structure
        true_od_sparse = None
        od_mask_sparse = None
        if true_od_demand is not None:
            true_od_sparse, od_mask_sparse = self._align_od_targets(true_od_demand, od_mask)

        # 3. INITIALIZATION: Apply one-time initialization policy if in training
        if self.training:
            self._maybe_apply_unknown_od_init(true_od_sparse=true_od_sparse, od_mask_sparse=od_mask_sparse)

        # 4. DEMAND: Decode learnable logits into physical OD demand
        estimated_od = self._learned_od(batch_size)

        # 5. ANCHORING: Decision to trust ground truth over learned demand for solver stability
        if (self.training and not is_pure_inference and 
                self.anchor_known_od_in_solver and 
                true_od_sparse is not None and 
                od_mask_sparse is not None):
            # Mask cells where we have ground truth data

            is_known = od_mask_sparse > 0.5
            # Inject ground truth into the solver input while keeping gradients for unknown cells
            od_input = torch.where(is_known, true_od_sparse, estimated_od)
        else:
            od_input = estimated_od

        # 6. EQUILIBRIUM: Solve for traffic flows (counts) and gather convergence telemetry
        pred_link_flows, route_flows, conv_info = self.equilibrium_solver(od_input, bpr_params)

        if is_pure_inference:
        # Podrías inyectar temporalmente una mayor precisión aquí si el solver lo permite
            current_mode = "final_pure_assignment"
        else:
            current_mode = "physics_imd" if bool(conv_info.get("implicit_grad")) else "physics_stage1"

        # Assemble results dictionary
        outputs = {
            "estimated_demand": estimated_od.squeeze(0) if batch_size == 1 else estimated_od,
            "reconstructed_flows": pred_link_flows.squeeze(0) if batch_size == 1 else pred_link_flows,
            "route_flows": route_flows.squeeze(0) if batch_size == 1 else route_flows,
            "convergence_info": {
                **conv_info,
                "mode": current_mode,
            },
            "learned_alpha": bpr_params["alpha"],
            "learned_beta": bpr_params["beta"],
            "learned_capacity_multiplier": bpr_params["capacity_multiplier"]
        }

        if self.sue_mode and hasattr(self, "theta_raw"):
            outputs["learned_theta"] = bpr_params["theta"]

        # 7. LOSS CALCULATION
        # Loss is only meaningful when supervision targets are available
        should_compute_loss = (
            true_od_sparse is not None
            and od_mask_sparse is not None
            and not is_pure_inference
        )

        if should_compute_loss:
            loss_dict = self.loss_fn(
                predicted_flows=pred_link_flows,
                true_flows=observed_flows,
                flow_mask=flow_mask if flow_mask is not None else torch.ones_like(observed_flows),
                predicted_od=estimated_od,
                true_od=true_od_sparse,
                od_mask=od_mask_sparse,
            )
            outputs["loss"] = loss_dict

        return outputs


    def get_evaluation_artifacts(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extracts and formats model outputs into a clean dictionary for logging and analysis.
        
        Logic:
        1. Validates that the necessary keys exist in the output.
        2. Moves tensors from GPU to CPU.
        3. Removes gradient information to save memory and allow for NumPy conversion.
        """
        # 1. CONTRACT VALIDATION
        # Uses a helper to ensure the output dictionary has the minimum required information.
        outputs_map = require_keys(
            outputs,
            ["reconstructed_flows", "estimated_demand"],
            context="VI_Model.get_evaluation_artifacts outputs",
            exc_type=ArtifactSchemaError,
        )

        # 2. ARTIFACT ASSEMBLY
        # We move only the 'final' values needed for reports, maps, and CSVs.
        artifacts = {
            "pred_flows": outputs_map["reconstructed_flows"].detach().cpu(),
            "pred_od": outputs_map["estimated_demand"].detach().cpu(),
            "route_flows": outputs_map.get("route_flows"), # Full path distribution
            "convergence": outputs_map.get("convergence_info", {}),
            
            # Physics inspection: how did the supply network adjust the links?
            "learned_alpha": outputs_map.get("learned_alpha"),
            "learned_beta": outputs_map.get("learned_beta"),
        }

        # Handle optional learnable theta for Stochastic models
        if "learned_theta" in outputs_map:
            artifacts["learned_theta"] = outputs_map["learned_theta"].detach().cpu()

        return artifacts


class Loss(nn.Module):
    """
    Multi-objective training loss for flow reconstruction and OD consistency.
    
    This class combines link flow errors, supervised OD errors, and 
    unsupervised regularization to guide the Variational Inference process.
    """
    def __init__(
        self,
        link_scale: float = 1.0,   # Normalization factor for traffic counts
        od_scale: float = 1.0,     # Normalization factor for OD demand
        w_link: float = 1.0,       # Importance weight for link flow matching
        w_prior: float = 0.1,      # Importance weight for supervised OD entries
        w_reg: float = 0.01,       # Weight for L2 regularization on unknown OD cells
        huber_delta: float = 1.0,  # Transition point for the Huber Loss
        **kwargs,
    ):
        super().__init__()
        self.link_scale = float(link_scale)
        self.od_scale = float(od_scale)
        self.use_feature_scaling_in_loss = bool(kwargs.get("use_feature_scaling_in_loss"))
        self.w_link = float(w_link)
        self.w_prior = float(w_prior)
        self.w_reg = float(w_reg)
        self.huber_delta = float(huber_delta)
        self.eps = 1e-9 # Stability constant

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def forward(
        self,
        predicted_flows: torch.Tensor, # Simulated link flows from the solver
        true_flows: torch.Tensor,      # Observed traffic counts (sensors)
        flow_mask: torch.Tensor,      # Mask identifying links with active sensors
        predicted_od: torch.Tensor,     # Demand estimated by the model
        true_od: torch.Tensor,         # Ground truth demand (if available)
        od_mask: torch.Tensor,         # Mask for known vs. unknown OD pairs
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        # Ensure all inputs are 2D for consistent vectorized operations
        predicted_flows = self._ensure_2d(predicted_flows)
        true_flows = self._ensure_2d(true_flows)
        flow_mask = self._ensure_2d(flow_mask).float()
        predicted_od = self._ensure_2d(predicted_od)
        true_od = self._ensure_2d(true_od)
        od_mask = self._ensure_2d(od_mask).float()

        # 1. LINK FLOW LOSS (Observed counts)
        # if use_feature_scaling_in loss, we apply scaling to the flows and OD Demand before computing the loss.
        # This might help stabilize training when the magnitudes of flows and demands vary widely across datasets.
        if self.use_feature_scaling_in_loss:
            scaled_pred_flows = predicted_flows / self.link_scale
            scaled_true_flows = true_flows / self.link_scale
        else:
            scaled_pred_flows = predicted_flows
            scaled_true_flows = true_flows

        flow_mask_bool = flow_mask > 0.5
        if flow_mask_bool.any():
            loss_links = F.mse_loss(scaled_pred_flows[flow_mask_bool], scaled_true_flows[flow_mask_bool])
        else:
            loss_links = torch.tensor(0.0, device=predicted_flows.device)

        # 2. KNOWN OD LOSS (Prior Protection)
        # Huber loss provides robustness against outliers in the prior survey data.
        if self.use_feature_scaling_in_loss:
            scaled_pred_od = predicted_od / self.od_scale
            scaled_true_od = true_od / self.od_scale
        else:
            scaled_pred_od = predicted_od
            scaled_true_od = true_od

        od_mask_bool = od_mask > 0.5
        if od_mask_bool.any():
            loss_prior_od = F.huber_loss(
                scaled_pred_od[od_mask_bool], 
                scaled_true_od[od_mask_bool], 
                delta=self.huber_delta
            )
        else:
            loss_prior_od = torch.tensor(0.0, device=predicted_od.device)

        # 3. UNKNOWN OD REGULARIZATION (Complexity Penalty)
        # Pure L2 penalty on scaled values prevents unknown demand from exploding.
        # This acts as a 'simplicity' prior for the unobserved parts of the matrix.
        unknown_mask_bool = ~od_mask_bool
        if unknown_mask_bool.any():
            loss_unknown_regularization = torch.mean(scaled_pred_od[unknown_mask_bool] ** 2)
        else:
            loss_unknown_regularization = torch.tensor(0.0, device=predicted_od.device)

        # 4. TOTAL GRADIENT COMPOSITION
        # The weights act as Lagrange multipliers to balance the multi-objective problem.
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


class VIModelDelegator:
    """
    Orchestrates model-specific logistics: epoch logging, metrics calculation, 
    and final artifact (CSV) generation for the Variational Inference framework.
    """
    def __init__(self, model_instance):
        self.model = model_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("VIModelDelegator initialized for model: %s", type(model_instance).__name__)

    def format_epoch_log(self, epoch, current_loss, loss_dict, outputs, grad_stats, has_val, val_loss):
        """
        Formats the training log string with a focus on transport physics.
        """
        def _as_float(x):
            if torch.is_tensor(x): return float(x.detach().item())
            try: return float(x)
            except Exception: return 0.0

        flow_loss_val = _as_float(loss_dict.get('l_flow', 0.0))
        od_loss_val = _as_float(loss_dict.get('l_od', 0.0))
        reg_od_loss_val = _as_float(loss_dict.get('l_demand_reg', 0.0))

        # Basic Loss Info
        log_msg = (
            f"Epoch {epoch + 1}: Train Loss {current_loss:.6g} | "
            f"Flow Loss {flow_loss_val:.6g} | "
            f"OD Loss {od_loss_val:.6g} | "
            f"REG OD Loss {reg_od_loss_val:.6g}"
        )

        # Equilibrium Engine Diagnostics
        conv_info = outputs.get('convergence_info', {})
        iters = conv_info.get('iterations')
        gap = conv_info.get('final_gap')
        mode = conv_info.get('mode')
        
        if iters is not None or gap is not None:
            iter_txt = f"{float(iters):.1f}" if iters is not None else "n/a"
            gap_txt = f"{float(gap):.2e}" if gap is not None else "n/a"
            mode_txt = f", mode={mode}" if mode is not None else ""
            log_msg += f" | Eq iters={iter_txt}, gap={gap_txt}{mode_txt}"

        # Gradient Health (Passed from Trainer)
        log_msg += (
            f" | GradNorm pre={grad_stats['pre_clip_norm']:.2e}, post={grad_stats['post_clip_norm']:.2e}, "
            f"clip_ratio={grad_stats['clip_ratio']:.3f}, clip_freq10={grad_stats['clip_freq10']:.0%}"
        )

        if has_val:
            log_msg += f" | Val MSE {val_loss:.4f}"

        return log_msg

    def compute_metrics(self, pred_tensor, target_tensor, mask_tensor):
        """
        Calculates R2, MAE, RMSE, and MAPE for traffic flows.
        """
        y_pred = pred_tensor.detach().cpu().numpy().flatten()
        y_true = target_tensor.detach().cpu().numpy().flatten()
        mask = mask_tensor.detach().cpu().numpy().flatten().astype(bool)

        y_pred = y_pred[mask]
        y_true = y_true[mask]

        if len(y_true) == 0:
            return {"R2": 0.0, "MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}

        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        non_zero = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else 0.0

        return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    def generate_final_csvs(self, output_dir, outputs, targets, network_params, epoch):
        """
        Exports standardized CSVs for flows and O-D demand.
        """
        if outputs is None or targets is None:
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Link Flow Export
        pred_flow = outputs.get('reconstructed_flows')
        true_flow = targets.get('flows')
        flow_mask_t = targets.get('mask', targets.get('flow_mask'))

        if pred_flow is not None and true_flow is not None:
            pred_flow_np = pred_flow.detach().cpu().numpy().reshape(-1)
            true_flow_np = true_flow.detach().cpu().numpy().reshape(-1)
            flow_mask_np = flow_mask_t.detach().cpu().numpy().reshape(-1) if flow_mask_t is not None else np.ones_like(true_flow_np)

            flow_rows = []
            for i in range(len(pred_flow_np)):
                flow_rows.append({
                    'epoch': int(epoch),
                    'link_id': int(i),
                    'real_flow': float(true_flow_np[i]),
                    'estimated_flow': float(pred_flow_np[i]),
                    'is_observed_link': bool(flow_mask_np[i] > 0.5),
                })
            
            self._write_csv(os.path.join(output_dir, 'estimated_vs_real_flows.csv'), 
                            ['epoch', 'link_id', 'real_flow', 'estimated_flow', 'is_observed_link'], 
                            flow_rows)

        # 2. OD Demand Export
        pred_od = outputs.get('estimated_demand')
        true_od = targets.get('od')
        od_mask_t = targets.get('od_mask')
        
        if pred_od is not None and true_od is not None:
            pred_od_np = pred_od.detach().cpu().numpy().reshape(-1)
            true_od_np = true_od.detach().cpu().numpy().reshape(-1)
            od_mask_np = od_mask_t.detach().cpu().numpy().reshape(-1) if od_mask_t is not None else np.ones_like(true_od_np)

            # Map O-D indices to node IDs if possible
            od_pair_labels = self._resolve_od_labels(network_params, len(pred_od_np))

            demand_rows = []
            for i in range(len(pred_od_np)):
                demand_rows.append({
                    'epoch': int(epoch),
                    'od_index': int(i),
                    'od_pair_id': od_pair_labels[i],
                    'real_demand': float(true_od_np[i]),
                    'estimated_demand': float(pred_od_np[i]),
                    'is_known_demand': bool(od_mask_np[i] > 0.5),
                })
            
            self._write_csv(os.path.join(output_dir, 'estimated_vs_real_demand.csv'),
                            ['epoch', 'od_index', 'od_pair_id', 'real_demand', 'estimated_demand', 'is_known_demand'],
                            demand_rows)

    def _write_csv(self, path, fieldnames, rows):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _resolve_od_labels(self, network_params, n_od):
        od_pair_indices = network_params.get('od_pair_indices')
        if od_pair_indices is not None:
            try:
                pairs = od_pair_indices.detach().cpu().numpy()
                return [f"{int(o)}-{int(d)}" for o, d in pairs]
            except:
                pass
        return [str(i) for i in range(n_od)]


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
        gap_alert_threshold: float = 1e-6,
        od_grad_low_threshold: float = 1e-8,
        r2_stall_window: int = 10,
        r2_stall_delta: float = 1e-4,
    ):
        self.window = int(history_window)
        self.gap_alert_threshold = float(gap_alert_threshold)
        self.od_grad_low_threshold = float(od_grad_low_threshold)
        self.r2_stall_window = int(r2_stall_window)
        self.r2_stall_delta = float(r2_stall_delta)
        self.tb_logger = None
        self.val_freq = 10

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
            "learned_theta": [],
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
        self.imd_relative_gap_rows = []
        self.od_matrix_rows = []

        # Initialize route flow tracking if the model provides route-level outputs
        self.route_flow_rows = []
        self.flow_conservation_rows = []
        self.flow_conservation_tolerance = 1e-2 # Umbral paramétrico

        # Initialize mass balance tracking for OD demand
        self.mass_conservation_rows = []
        self.mass_conservation_tolerance = 1e-2 # Umbral paramétrico

    def attach_tensorboard(self, tb_logger, val_freq: int):
        """Hooks the external TensorBoard logger to this diagnostician."""
        self.tb_logger = tb_logger
        self.val_freq = val_freq

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
            is_final = bool(kwargs.get("is_final", False)) # <--- Recuperar flag
            is_heavy_log_epoch = epoch % self.val_freq == 0

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

            # --- TensorBoard Scalar Logging (Lightweight, every epoch) ---
            if self.tb_logger is not None:
                self.tb_logger.log_scalar("Metrics_Flow/R2", r2, epoch)
                self.tb_logger.log_scalar("Metrics_Flow/MAE", mae, epoch)
                
                loss_dict = kwargs.get("loss_dict") or outputs.get("loss") or {}
                if isinstance(loss_dict, dict):
                    self.tb_logger.log_scalar("Loss/Total", loss_dict.get("total_loss", 0), epoch)
                    self.tb_logger.log_scalar("Loss/Flow", loss_dict.get("l_flow", 0), epoch)
                    self.tb_logger.log_scalar("Loss/OD_Prior", loss_dict.get("l_od", 0), epoch)
                    self.tb_logger.log_scalar("Loss/Demand_Reg", loss_dict.get("l_demand_reg", 0), epoch)

                conv_info = outputs.get("convergence_info", {}) or {}
                self.tb_logger.log_scalar("Equilibrium/Iterations", conv_info.get("iterations", 0), epoch)
                self.tb_logger.log_scalar("Equilibrium/Final_Gap", conv_info.get("final_gap", 0), epoch)
                
                # Check for IMD consistency tracking
                if conv_info.get("implicit_grad", False) and model is not None:
                    try:
                        bwd_info = model.equilibrium_solver._imd_last_backward_info
                        self.tb_logger.log_scalar("IMD/Fixed_Point_Residual", bwd_info.get("fixed_point_residual", 0), epoch)
                    except AttributeError:
                        pass
                
                # --- TensorBoard Heavy Logging (Gated) ---
                if is_heavy_log_epoch:
                    if outputs.get("learned_alpha") is not None:
                        self.tb_logger.log_histogram("Physics/Alpha", outputs["learned_alpha"], epoch)
                    if outputs.get("learned_beta") is not None:
                        self.tb_logger.log_histogram("Physics/Beta", outputs["learned_beta"], epoch)
                    if outputs.get("learned_capacity_multiplier") is not None:
                        self.tb_logger.log_histogram("Physics/Capacity_Multiplier", outputs["learned_capacity_multiplier"], epoch)

            # -----------------------------------------------------

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
            self._append_scalar("learned_theta", outputs.get("learned_theta"), default=0.0)
            self.full_history["mode"].append(str(conv_info.get("mode", "physics_stage1")))
            self._collect_imd_relative_gap_rows(epoch=epoch, conv_info=conv_info)

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
                static_info=static_info,
            )
            self._collect_forward_backward_audit_row(
                epoch=epoch,
                outputs=outputs,
                model=model,
            )

            self._record_alerts(epoch=len(self.full_history["r2_flow"]) - 1)

        # --- LÓGICA DE CAPTURA DE RUTAS ---
        # Por defecto solo capturamos la última epoch (is_final)
        # Para cambiar esto, modifica la condición: if is_final or (epoch % 50 == 0):
        if is_final and model is not None:
                route_flows = outputs.get("route_flows")
                pred_link_flows = outputs.get("reconstructed_flows")
                solver = getattr(model, "equilibrium_solver", None)
                
                if route_flows is not None and pred_link_flows is not None and solver is not None:
                    # 1. CAPTURA DE ROUTE FLOWS (El paso anterior)
                    mask = solver.route_validity_mask 
                    num_od, k_paths = mask.shape
                    
                    rf_np = route_flows.detach().cpu().numpy().reshape(-1)
                    mask_np = mask.detach().cpu().numpy().reshape(-1)

                    for i in range(len(rf_np)):
                        if mask_np[i]:
                            self.route_flow_rows.append({
                                "epoch": epoch,
                                "od_index": i // k_paths,
                                "route_index": i % k_paths,
                                "route_flow": float(rf_np[i])
                            })

                    # 2. AUDITORÍA DE CONSERVACIÓN DE FLUJO (f = Delta * h)
                    delta_matrix = solver.delta_matrix
                    
                    # Asegurar dimensiones [Routes] y [Links]
                    rf_t = route_flows.detach().squeeze()
                    pf_t = pred_link_flows.detach().squeeze()
                    
                    if rf_t.dim() == 1 and pf_t.dim() == 1:
                        # Multiplicación de matriz dispersa en PyTorch: [Links, Routes] x [Routes, 1]
                        rf_col = rf_t.unsqueeze(1) 
                        f_recon_t = torch.sparse.mm(delta_matrix, rf_col).squeeze(1) 
                        
                        pf_np = pf_t.cpu().numpy()
                        f_recon_np = f_recon_t.cpu().numpy()
                        
                        abs_err = np.abs(pf_np - f_recon_np)
                        # Prevenir división por cero en el error relativo
                        rel_err = abs_err / np.maximum(pf_np, 1e-9)
                        
                        tolerance = self.flow_conservation_tolerance
                        
                        for link_idx in range(len(pf_np)):
                            self.flow_conservation_rows.append({
                                "epoch": epoch,
                                "link_id": link_idx,
                                "estimated_flow": float(pf_np[link_idx]),
                                "reconstructed_flow": float(f_recon_np[link_idx]),
                                "abs_error": float(abs_err[link_idx]),
                                "rel_error": float(rel_err[link_idx]),
                                "status": "OK" if abs_err[link_idx] <= tolerance else "FAIL"
                            })

        # --- AUDITORÍA DE CONSERVACIÓN DE MASA EN NODOS (A * f = E) ---
        if is_final and model is not None:
            link_nodes = static_info.get("link_pair_indices") 
            
            # Use raw string labels if available to avoid tensor index mapping mismatches
            od_nodes = static_info.get("od_pair_node_labels")
            if od_nodes is None:
                od_nodes = static_info.get("od_pair_indices") # Fallback
            
            if link_nodes is not None and od_nodes is not None:
                net_demand = {}
                pred_od_np = outputs.get("estimated_demand").detach().cpu().numpy().reshape(-1)
                
                # 1. Calculate Vector E (Net Demand per Node) using STRING labels
                for idx, pair in enumerate(od_nodes):
                    # Force string casting to unify the ID space
                    o, d = str(pair[0]), str(pair[1]) 
                    dem = float(pred_od_np[idx])
                    net_demand[o] = net_demand.get(o, 0.0) + dem
                    net_demand[d] = net_demand.get(d, 0.0) - dem

                net_flows = {}
                pred_flow_np = outputs.get("reconstructed_flows").detach().cpu().numpy().reshape(-1)
                
                # 2. Calculate A * f (Net Flow per Node) using STRING labels
                for idx, pair in enumerate(link_nodes):
                    u, v = str(pair[0]), str(pair[1])
                    f = float(pred_flow_np[idx])
                    net_flows[u] = net_flows.get(u, 0.0) + f
                    net_flows[v] = net_flows.get(v, 0.0) - f

                # 3. Vectorial Comparison
                all_nodes = set(net_demand.keys()).union(set(net_flows.keys()))
                
                for node in all_nodes:
                    d_net = net_demand.get(node, 0.0)
                    f_net = net_flows.get(node, 0.0)
                    abs_err = abs(d_net - f_net)
                    
                    self.mass_conservation_rows.append({
                        "epoch": epoch,
                        "node_id": node, 
                        "net_demand_generated": d_net,
                        "net_flow_routed": f_net,
                        "abs_error": abs_err,
                        "status": "OK" if abs_err <= self.mass_conservation_tolerance else "FAIL"
                    })


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

    def _collect_imd_relative_gap_rows(self, epoch: int, conv_info: Dict):
        """Store per-iteration relative-gap traces for IMD epochs."""
        if not isinstance(conv_info, dict):
            return
        raw_history = conv_info.get("relative_gap_history", None)
        if raw_history is None:
            return
        if isinstance(raw_history, np.ndarray):
            history = raw_history.tolist()
        elif isinstance(raw_history, (list, tuple)):
            history = list(raw_history)
        else:
            return

        mode = str(conv_info.get("mode", "unknown"))
        implicit_grad = int(bool(conv_info.get("implicit_grad", False)))
        for iter_idx, gap_val in enumerate(history, start=1):
            try:
                gap_float = float(gap_val)
            except (TypeError, ValueError):
                continue
            self.imd_relative_gap_rows.append(
                {
                    "epoch": int(epoch),
                    "imd_iteration": int(iter_idx),
                    "relative_gap": float(gap_float),
                    "mode": mode,
                    "implicit_grad": int(implicit_grad),
                }
            )

    def capture_gradient_history(self, model: nn.Module, epoch: int):
        """Capture module-wise and parameter-wise gradient norms for health checks."""
        key_modules = {
            "supply_net": getattr(model, "supply_net", None),
            "equilibrium_solver": getattr(model, "equilibrium_solver", None),
        }

        max_grad = 0.0
        is_heavy_log_epoch = (epoch + 1) % self.val_freq == 0  
              
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

            if self.tb_logger is not None and is_heavy_log_epoch and grad is not None:
                self.tb_logger.log_histogram("Gradients/od_logits", grad, epoch + 1)

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

        cap_mult = outputs.get("learned_capacity_multiplier", None)

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

        cap_np = np.clip(cap_np[:n], a_min=1e-9, a_max=None)
        # Ajustar la capacidad base con el multiplicador aprendido
        if cap_mult is not None:
            cap_mult_np = cap_mult.detach().cpu().numpy().reshape(-1)[:n]
            adj_cap_np = cap_np * cap_mult_np
        else:
            adj_cap_np = cap_np

        alpha_np = alpha_np[:n]
        beta_np = beta_np[:n]
        pred_flow_np = pred_flow_np[:n]
        cap_np = np.clip(cap_np[:n], a_min=1e-9, a_max=None)
        types_np = types_np[:n]

        vc_np = pred_flow_np / np.clip(adj_cap_np, a_min=1e-9, a_max=None)

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

    def _collect_flow_demand_rows(self, epoch: int, outputs: Dict, targets: Dict, static_info: Optional[Dict] = None):
        """Store row-wise comparison tables for estimated vs target flow and demand."""
        static_info = static_info or {}
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
                od_pair_indices = static_info.get("od_pair_indices", None)
                if torch.is_tensor(od_pair_indices):
                    od_pair_indices = od_pair_indices.detach().cpu().numpy()
                if od_pair_indices is not None:
                    od_pair_indices = np.asarray(od_pair_indices)
                known_rank = 0
                unknown_rank = 0
                for i in range(m):
                    mask_val = float(od_mask_np[i])
                    is_known = int(mask_val > 0.5)
                    if is_known:
                        known_rank += 1
                        known_od_index = int(known_rank)
                        unknown_od_index = -1
                        od_status = "known"
                    else:
                        unknown_rank += 1
                        known_od_index = -1
                        unknown_od_index = int(unknown_rank)
                        od_status = "unknown"

                    origin_index = -1
                    destination_index = -1
                    if od_pair_indices is not None and od_pair_indices.ndim >= 2 and i < od_pair_indices.shape[0] and od_pair_indices.shape[1] >= 2:
                        origin_index = int(od_pair_indices[i, 0])
                        destination_index = int(od_pair_indices[i, 1])

                    self.demand_comparison_rows.append(
                        {
                            "epoch": int(epoch),
                            "od_index": int(i),
                            "true_demand": float(true_od_np[i]),
                            "estimated_demand": float(pred_od_np[i]),
                            "od_mask": mask_val,
                        }
                    )
                    self.od_matrix_rows.append(
                        {
                            "epoch": int(epoch),
                            "od_index": int(i),
                            "origin_index": int(origin_index),
                            "destination_index": int(destination_index),
                            "od_status": od_status,
                            "known_od_index": int(known_od_index),
                            "unknown_od_index": int(unknown_od_index),
                            "od_mask": mask_val,
                            "true_demand": float(true_od_np[i]),
                            "estimated_demand": float(pred_od_np[i]),
                        }
                    )

    def _write_csv(self, path: str, fieldnames, rows):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        def format_value(v):
            if isinstance(v, float):
                return str(v).replace(".", ",")
            return v

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=";"
            )
            writer.writeheader()

            for row in rows:
                formatted_row = {k: format_value(v) for k, v in row.items()}
                writer.writerow(formatted_row)

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

        # Export diagnostics requested at run-level diagnostics root.
        diagnostics_root = os.path.dirname(out_dir)
        imd_gap_csv = os.path.join(diagnostics_root, "imd_relative_gap_history_by_epoch.csv")
        self._write_csv(
            imd_gap_csv,
            fieldnames=["epoch", "imd_iteration", "relative_gap", "mode", "implicit_grad"],
            rows=self.imd_relative_gap_rows,
        )

        epoch_loss_rows = []
        n_epochs = max(
            len(self.full_history.get("l_flow", [])),
            len(self.full_history.get("l_od", [])),
            len(self.full_history.get("total_loss", [])),
            len(self.full_history.get("l_demand_reg", [])),
        )
        for idx in range(n_epochs):
            l_flow = float(self.full_history.get("l_flow", [0.0])[idx]) if idx < len(self.full_history.get("l_flow", [])) else 0.0
            l_od = float(self.full_history.get("l_od", [0.0])[idx]) if idx < len(self.full_history.get("l_od", [])) else 0.0
            total_loss = float(self.full_history.get("total_loss", [0.0])[idx]) if idx < len(self.full_history.get("total_loss", [])) else 0.0
            l_demand_reg = float(self.full_history.get("l_demand_reg", [0.0])[idx]) if idx < len(self.full_history.get("l_demand_reg", [])) else 0.0
            epoch_loss_rows.append(
                {
                    "epoch": int(idx + 1),
                    "l_flow": l_flow,
                    "l_od": l_od,
                    "total_loss": total_loss,
                    "l_demand_reg": l_demand_reg,
                }
            )

        loss_epoch_csv = os.path.join(diagnostics_root, "epoch_loss_evolution.csv")
        self._write_csv(
            loss_epoch_csv,
            fieldnames=["epoch", "l_flow", "l_od", "total_loss", "l_demand_reg"],
            rows=epoch_loss_rows,
        )

        od_matrix_csv = os.path.join(diagnostics_root, "od_matrix_history_by_epoch.csv")
        self._write_csv(
            od_matrix_csv,
            fieldnames=[
                "epoch",
                "od_index",
                "origin_index",
                "destination_index",
                "od_status",
                "known_od_index",
                "unknown_od_index",
                "od_mask",
                "true_demand",
                "estimated_demand",
            ],
            rows=self.od_matrix_rows,
        )

        # Export link-type alpha/beta/vc summaries in a single historical CSV.
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

        # Exportar el historial de flujos por ruta
        if self.route_flow_rows:
            route_csv = os.path.join(out_dir, "route_flow_history.csv")
            self._write_csv(
                route_csv,
                fieldnames=["epoch", "od_index", "route_index", "route_flow"],
                rows=self.route_flow_rows
            )

        # 1. Exportar reporte de auditoría de conservación de flujo (Links vs Rutas)
        if hasattr(self, "flow_conservation_rows") and self.flow_conservation_rows:
            cons_csv = os.path.join(out_dir, "flow_conservation_audit.csv")
            self._write_csv(
                cons_csv,
                fieldnames=["epoch", "link_id", "estimated_flow", "reconstructed_flow", "abs_error", "rel_error", "status"],
                rows=self.flow_conservation_rows
            )

        # 2. Exportar reporte de auditoría de conservación de masa (Nodos)
        if hasattr(self, "mass_conservation_rows") and self.mass_conservation_rows:
            mass_csv = os.path.join(out_dir, "node_mass_conservation_audit.csv")
            self._write_csv(
                mass_csv,
                fieldnames=["epoch", "node_id", "net_demand_generated", "net_flow_routed", "abs_error", "status"],
                rows=self.mass_conservation_rows
            )


# Backward compatible names used in early sketches.
EndToEndSemiParametricModel = VariationalInferenceModel
ODEstimationLoss = Loss