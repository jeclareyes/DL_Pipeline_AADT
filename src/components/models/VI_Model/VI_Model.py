# src/components/models/VI_Model/VI_Model.py
"""
Variational-Inequality traffic assignment model with physics-aware congestion dynamics.

This module contains the full Stage-1 / IMD-ready pipeline used in the project:

1) `PhysicsInformedBPRNet` learns bounded BPR parameters from static link attributes.
2) `ImplicitEquilibriumLayer` solves a user-equilibrium fixed point with entropy mirror descent.
3) `VariationalInequalityModel` couples learned OD demand with the equilibrium solver.
4) `Loss` combines flow fitting, OD supervision, and unknown-OD regularization.
5) `VIDiagnostician` collects optimization and physics diagnostics during training.

The implementation emphasizes physically plausible congestion behavior (positive
capacity, bounded alpha/beta), stable gradients, and clear telemetry for solver
convergence and forward/backward consistency in IMD mode.
"""


from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.models.VI_Model.VI_Model_Auditer import (
    VIDiagnostician,
    run_vi_assignment_audit,
)
from src.components.models.VI_Model.VI_Model_Logistician import VIModelDelegator
from src.contracts.runtime_contracts import ArtifactSchemaError, require_keys


#%% Demand Estmation Module

# OPTION A. 
# Esto es más correcto que dejar que todas las OD, incluso las conocidas, 
# sean simplemente aprendidas. En tu código ya existe una idea similar con 
# anchor_known_od_in_solver, pero todavía no hay un DemandNet; lo que hay es 
# un vector libre od_logits.

# La ventaja es que el solver siempre recibe una matriz OD completa e inelástica, 
# pero esa matriz fue generada por una función diferenciable.

class ODDemandCompletionNet(nn.Module):
    """
    Differentiable OD demand completion module.

    This module receives a partially observed OD vector and its observation mask,
    then returns a complete, nonnegative OD demand vector.

    Methodological role:
        - Demand remains inelastic inside the assignment solver.
        - This network only completes the OD vector before the equilibrium problem.
        - The solver still receives a fixed OD demand and enforces:
              sum_p h_p^{od} = q_od

    Two anchoring modes are supported:
        1. Hard-anchor mode:
            Known OD entries are copied exactly from the observed OD vector.
            Unknown OD entries are predicted by the network.

        2. Flexible-anchor mode:
            Known OD entries are softly blended between the observed value and
            the network prediction. This is useful when the observed OD values
            are considered noisy or uncertain.
    """

    def __init__(
        self,
        num_od_pairs: int,
        od_scale: float = 1.0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        hard_anchor_known_od: bool = True,
        known_anchor_weight: float = 0.95,
    ):
        super().__init__()

        self.num_od_pairs = int(num_od_pairs)
        self.od_scale = float(od_scale)
        self.hard_anchor_known_od = bool(hard_anchor_known_od)

        # Used only in flexible-anchor mode.
        # 1.0 means fully trust known OD values.
        # 0.0 means fully trust the network prediction.
        self.known_anchor_weight = float(known_anchor_weight)

        if not 0.0 <= self.known_anchor_weight <= 1.0:
            raise ValueError(
                "known_anchor_weight must be within [0.0, 1.0]. "
                f"Received: {known_anchor_weight}"
            )

        input_dim = 2 * self.num_od_pairs
        output_dim = self.num_od_pairs

        layers = []
        current_dim = input_dim

        for _ in range(max(int(num_layers), 1)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure a tensor has shape [B, OD]."""
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def forward(
        self,
        known_od: torch.Tensor,
        od_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complete a partially observed OD vector.

        Args:
            known_od:
                Tensor with shape [B, OD] or [OD].
                Known entries contain observed OD demand.
                Unknown entries can be zero or any placeholder.

            od_mask:
                Tensor with shape [B, OD] or [OD].
                1.0 means the OD entry is observed.
                0.0 means the OD entry is unknown.

        Returns:
            completed_od:
                Tensor with shape [B, OD], nonnegative and in physical demand units.
        """
        known_od = self._ensure_2d(known_od).float()
        od_mask = self._ensure_2d(od_mask).float()

        if known_od.shape != od_mask.shape:
            raise ValueError(
                "known_od and od_mask must have the same shape. "
                f"Received known_od={tuple(known_od.shape)}, "
                f"od_mask={tuple(od_mask.shape)}."
            )

        if known_od.shape[1] != self.num_od_pairs:
            raise ValueError(
                "known_od second dimension must match num_od_pairs. "
                f"Expected {self.num_od_pairs}, received {known_od.shape[1]}."
            )

        # Normalize only for the neural network input.
        # The mask is concatenated so the model knows which entries are reliable.
        known_od_norm = known_od / max(self.od_scale, 1e-6)
        network_input = torch.cat([known_od_norm * od_mask, od_mask], dim=1)

        # The network predicts a complete OD vector in normalized units.
        # Softplus guarantees nonnegative demand.
        predicted_od_norm = F.softplus(self.net(network_input))
        predicted_od = predicted_od_norm * self.od_scale

        if self.hard_anchor_known_od:
            # Known entries are preserved exactly.
            completed_od = (od_mask * known_od) + ((1.0 - od_mask) * predicted_od)
        else:
            # Known entries are softly anchored.
            # This allows the model to correct noisy OD observations while still
            # keeping them close to their provided values through the OD loss.
            blended_known_od = (
                self.known_anchor_weight * known_od
                + (1.0 - self.known_anchor_weight) * predicted_od
            )
            completed_od = (od_mask * blended_known_od) + ((1.0 - od_mask) * predicted_od)

        return torch.clamp(completed_od, min=0.0)

#%% Volume-Delay Function Implementations with Learnable Parameters

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

#%% Deterministic Topology Operator: Link Costs to Path Costs

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

#%% Implicit Model Differentiation (IMD) Layer for Traffic Equilibrium

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
        # The implicit backward pass will reconstruct the local fixed-point derivative.
        with torch.no_grad():
            # Pack parameters for the BPR cost function.
            params = {
                "alpha": alpha,
                "beta": beta,
                "capacity_multiplier": cap_mult,
            }
            if theta is not None:
                params["theta"] = theta

            # Use the same convergence contract as the explicit solver path.
            route_flows, solver_info = layer._run_equilibrium_iterations(
                route_init=route_init,
                od_demands=od_demands,
                bpr_params=params,
            )

            used_iters = int(solver_info["iterations"])
            linearization_iter = max(1, used_iters)

            # --- IMD STATIONARY TAIL ---
            # To improve gradient stability, perform a few additional frozen steps
            # around the terminal fixed-point state.
            tail_steps = max(0, int(layer.imd_stationary_tail_steps))
            for _ in range(tail_steps):
                route_flows = layer._fixed_point_step(
                    route_flows=route_flows,
                    od_demands=od_demands,
                    bpr_params=params,
                    iter_idx=linearization_iter,
                )

        # Telemetry: Store solver performance data in the layer for diagnostics.
        layer._imd_last_info = {
            **solver_info,
            "implicit_grad": True,
            "linearization_iter": float(linearization_iter),
        }


        # CONTEXT STORAGE: Save necessary tensors for the backward(ctx, grad_output) call.
        ctx.layer = layer
        ctx.linearization_iter = linearization_iter
        ctx.has_theta = theta is not None
        # We must save the converged flows and all inputs that require gradients.

        tensors_to_save = [route_flows, od_demands, alpha, beta, cap_mult]
        if ctx.has_theta:
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
        theta = saved[5] if ctx.has_theta else None

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

#%% Implicit Equilibrium Layer with Mirror Descent and IMD Support

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
        lanes: torch.Tensor,               # Number of lanes for each link (used in BPR)
        solver_cfg: Optional[dict] = None, # Configuration dictionary for solver settings
        imd_cfg: Optional[dict] = None,    # Configuration for Implicit Model Differentiation (IMD)
    ):
        super().__init__()

        self.eps = 1e-9 # Small constant to prevent division by zero

        # --- Solver hyperparameters ---
        self.solver_cfg = dict(solver_cfg or {})
       
        self.eta = float(self.solver_cfg.get("eta", 1.0))
        self.max_iterations = int(self.solver_cfg.get("max_iterations", 50))
        self.min_iterations = int(self.solver_cfg.get("min_iterations", 5))
        self.use_early_stop = bool(self.solver_cfg.get("use_early_stop", True))

        # sue_mode: If True, the solver uses an entropy/logit-like stochastic update.
        # In that case, deterministic Wardrop gaps are still useful diagnostics, but
        # they should not necessarily be enforced as hard stopping criteria.
        self.sue_mode = bool(self.solver_cfg.get("sue_mode", False))

        # -------------------------------------------------------------------------
        # Convergence tolerances
        # -------------------------------------------------------------------------
        # tol_rel_flow is kept as a single legacy/default tolerance source.
        # The actual stopping logic below uses explicit, responsibility-specific
        # tolerances to avoid hiding different numerical concepts behind one value.
        self.tol_rel_flow = float(self.solver_cfg.get("tol_rel_flow", 1e-4))

        # Relative change between two consecutive route-flow states.
        self.tol_flow_change = float(
            self.solver_cfg.get("tol_flow_change", self.tol_rel_flow)
        )

        # Deterministic Wardrop relative gap over all valid paths.
        self.tol_wardrop_gap = float(
            self.solver_cfg.get("tol_wardrop_gap", self.tol_rel_flow)
        )

        # OD-wise feasibility errors: route flows must sum to OD demand.
        self.tol_feasibility_abs = float(
            self.solver_cfg.get("tol_feasibility_abs", self.solver_cfg.get("tol_feasibility", 1e-6))
        )
        self.tol_feasibility_rel = float(
            self.solver_cfg.get("tol_feasibility_rel", 1e-6)
        )

        # Invalid routes should carry no flow.
        self.tol_invalid_route_flow = float(
            self.solver_cfg.get("tol_invalid_route_flow", 1e-8)
        )

        # By default, enforce deterministic Wardrop only in deterministic UE mode.
        # In SUE mode, Wardrop gap is computed as a diagnostic but not required for
        # early stopping until a dedicated SUE residual is added in a later phase.
        self.enforce_wardrop_gap = bool(
            self.solver_cfg.get("enforce_wardrop_gap", not self.sue_mode)
        )



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
        self.register_buffer("lanes", lanes.float())
        # Metadata for indexing and reshaping
        self.num_od, self.k_paths = self.route_validity_mask.shape
        self.num_routes = self.num_od * self.k_paths
        
        # Flow caching for Warm Start: allows the solver to start from the 
        # last known solution, drastically reducing iterations in training.
        self.register_buffer("cached_route_flows", torch.zeros(1, self.num_routes))
        self.register_buffer("cache_valid", torch.tensor(False, dtype=torch.bool))


    def _compute_wardrop_relative_gap(
        self,
        route_flows_flat: torch.Tensor,
        route_costs: torch.Tensor,
        od_demands: torch.Tensor,
    ) -> float:
        """
        Compute a deterministic Wardrop relative gap over all valid paths.

        This diagnostic avoids the false-positive behavior of checking only
        active routes. For each OD pair, the minimum path cost is computed over
        all valid routes, including unused ones. If a used path is more expensive
        than an available cheaper path, the gap becomes positive.

        Formula:
            gap = sum_p h_p * (c_p - c_min_od)
                --------------------------------
                sum_od q_od * c_min_od

        Notes
        -----
        - This is a deterministic UE diagnostic.
        - In SUE mode, it should usually be logged but not enforced as the only
        stopping condition, because stochastic equilibria may assign positive
        probability to non-minimum-cost paths.
        """
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        od_demands = self._ensure_2d(od_demands)

        valid = self.route_validity_mask.unsqueeze(0).expand_as(route_flows)
        valid_f = valid.to(dtype=route_flows.dtype)

        # Compute the minimum cost over all valid paths, not only active paths.
        inf_cost = torch.full_like(route_costs, 1e12)
        valid_costs = torch.where(valid, route_costs, inf_cost)
        min_valid_cost, _ = valid_costs.min(dim=2)  # [B, OD]

        # Only OD pairs with demand and at least one valid path are meaningful.
        has_valid_path = valid.any(dim=2)
        has_positive_demand = od_demands > self.eps
        active_od = has_valid_path & has_positive_demand

        if not bool(active_od.any()):
            return 0.0

        # Excess cost is positive when assigned flow uses paths above the minimum.
        excess_cost = torch.clamp(route_costs - min_valid_cost.unsqueeze(-1), min=0.0)
        assigned_valid_flow = route_flows * valid_f

        numerator_by_od = (assigned_valid_flow * excess_cost).sum(dim=2)
        denominator_by_od = od_demands * torch.clamp(min_valid_cost, min=self.eps)

        numerator = numerator_by_od[active_od].sum()
        denominator = denominator_by_od[active_od].sum().clamp(min=self.eps)

        gap = numerator / denominator
        return float(gap.detach().item())


    def _compute_relative_flow_change(
        self,
        current_route_flows: torch.Tensor,
        next_route_flows: torch.Tensor,
    ) -> float:
        """
        Compute the relative change between consecutive route-flow states.

        This metric checks whether the mirror-descent iterator is still moving
        the assignment. It is not a full equilibrium certificate by itself, but it
        prevents stopping only because costs look similar on active paths.
        """
        current_route_flows = self._ensure_2d(current_route_flows)
        next_route_flows = self._ensure_2d(next_route_flows)

        diff_norm = torch.norm(next_route_flows - current_route_flows, p=2, dim=1)
        base_norm = torch.norm(current_route_flows, p=2, dim=1).clamp(min=self.eps)

        # Use the maximum over the batch to avoid declaring convergence when only
        # some samples in the batch have stabilized.
        relative_change = diff_norm / base_norm
        return float(relative_change.max().detach().item())


    def _compute_feasibility_errors(
        self,
        route_flows_flat: torch.Tensor,
        od_demands: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Check OD-wise feasibility of route flows.

        For inelastic path-based assignment, each OD demand must be fully assigned
        across valid paths:

            sum_{p in P_od} h_p = q_od

        This function also reports the maximum absolute flow assigned to invalid
        padded paths, which should remain numerically zero.
        """
        route_flows = route_flows_flat.view(-1, self.num_od, self.k_paths)
        od_demands = self._ensure_2d(od_demands)

        valid = self.route_validity_mask.unsqueeze(0).expand_as(route_flows)
        valid_f = valid.to(dtype=route_flows.dtype)

        assigned_by_od = (route_flows * valid_f).sum(dim=2)
        abs_error = torch.abs(assigned_by_od - od_demands)

        rel_error = abs_error / torch.clamp(torch.abs(od_demands), min=1.0)

        has_valid_path = valid.any(dim=2)
        if bool(has_valid_path.any()):
            max_abs_error = abs_error[has_valid_path].max()
            max_rel_error = rel_error[has_valid_path].max()
        else:
            max_abs_error = torch.tensor(0.0, device=route_flows.device)
            max_rel_error = torch.tensor(0.0, device=route_flows.device)

        invalid_flow = torch.where(
            valid,
            torch.zeros_like(route_flows),
            torch.abs(route_flows),
        )
        max_invalid_flow = invalid_flow.max()

        return {
            "feasibility_abs_error": float(max_abs_error.detach().item()),
            "feasibility_rel_error": float(max_rel_error.detach().item()),
            "invalid_route_flow": float(max_invalid_flow.detach().item()),
        }


    def _compute_convergence_diagnostics(
        self,
        current_route_flows: torch.Tensor,
        next_route_flows: torch.Tensor,
        route_costs: torch.Tensor,
        od_demands: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Build a complete convergence diagnostic dictionary for one solver step.

        Responsibilities:
        - Wardrop gap checks deterministic route optimality over all valid paths.
        - Flow change checks whether the fixed-point iterator is still moving.
        - Feasibility checks whether the path-flow state respects OD conservation.
        """
        wardrop_gap = self._compute_wardrop_relative_gap(
            route_flows_flat=current_route_flows,
            route_costs=route_costs,
            od_demands=od_demands,
        )

        relative_flow_change = self._compute_relative_flow_change(
            current_route_flows=current_route_flows,
            next_route_flows=next_route_flows,
        )

        feasibility = self._compute_feasibility_errors(
            route_flows_flat=next_route_flows,
            od_demands=od_demands,
        )

        return {
            "wardrop_gap": wardrop_gap,
            "relative_flow_change": relative_flow_change,
            **feasibility,
        }


    def _check_convergence(self, diagnostics: Dict[str, float]) -> bool:
        """
        Decide whether the current solver state satisfies the Phase-1 convergence contract.

        In deterministic UE mode, the Wardrop gap is enforced.
        In SUE mode, the Wardrop gap is only diagnostic by default; the dedicated
        SUE/logit residual should be added in a later phase.
        """
        flow_has_stabilized = diagnostics["relative_flow_change"] <= self.tol_flow_change

        feasibility_is_ok = (
            diagnostics["feasibility_abs_error"] <= self.tol_feasibility_abs
            and diagnostics["feasibility_rel_error"] <= self.tol_feasibility_rel
            and diagnostics["invalid_route_flow"] <= self.tol_invalid_route_flow
        )

        wardrop_is_ok = (
            diagnostics["wardrop_gap"] <= self.tol_wardrop_gap
            if self.enforce_wardrop_gap
            else True
        )

        return bool(flow_has_stabilized and feasibility_is_ok and wardrop_is_ok)


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


    def _run_equilibrium_iterations(
    self,
    route_init: torch.Tensor,
    od_demands: torch.Tensor,
    bpr_params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """
        Run mirror-descent equilibrium iterations and return the terminal route flows.

        This method centralizes solver convergence logic so that explicit training,
        inference, and IMD forward passes use the same stopping contract.

        The caller decides whether gradients are tracked:
        - IMD forward calls this method under torch.no_grad().
        - Explicit mode may call it with gradient tracking enabled.
        """
        current_route_flows = route_init
        used_iters = self.max_iterations
        converged = False

        # Main scalar histories.
        wardrop_gap_history: List[float] = []
        relative_flow_change_history: List[float] = []
        feasibility_abs_history: List[float] = []
        feasibility_rel_history: List[float] = []
        invalid_route_flow_history: List[float] = []

        # Final diagnostics default to a safe non-converged state.
        final_diagnostics = {
            "wardrop_gap": float("inf"),
            "relative_flow_change": float("inf"),
            "feasibility_abs_error": float("inf"),
            "feasibility_rel_error": float("inf"),
            "invalid_route_flow": float("inf"),
        }

        for it in range(1, self.max_iterations + 1):
            # 1. Evaluate route costs at the current route-flow state.
            route_costs = self._route_costs(current_route_flows, bpr_params)

            # 2. Compute the next mirror-descent route-flow state.
            next_route_flows = self._mirror_descent_step(
                route_flows_flat=current_route_flows,
                route_costs=route_costs,
                od_demands=od_demands,
                bpr_params=bpr_params,
                iter_idx=it,
            )

            # 3. Compute convergence diagnostics after the candidate update.
            diagnostics = self._compute_convergence_diagnostics(
                current_route_flows=current_route_flows,
                next_route_flows=next_route_flows,
                route_costs=route_costs,
                od_demands=od_demands,
            )
            final_diagnostics = diagnostics

            wardrop_gap_history.append(diagnostics["wardrop_gap"])
            relative_flow_change_history.append(diagnostics["relative_flow_change"])
            feasibility_abs_history.append(diagnostics["feasibility_abs_error"])
            feasibility_rel_history.append(diagnostics["feasibility_rel_error"])
            invalid_route_flow_history.append(diagnostics["invalid_route_flow"])

            # 4. Accept the new state before stopping so the returned solution is the
            # most recent feasible mirror-descent update.
            current_route_flows = next_route_flows
            used_iters = it

            # 5. Early stop only after the minimum number of iterations.
            if (
                self.use_early_stop
                and it >= self.min_iterations
                and self._check_convergence(diagnostics)
            ):
                converged = True
                break

        info = {
            "iterations": float(used_iters),
            "converged": float(1.0 if converged else 0.0),

            # Keep final_gap as the main scalar used by existing diagnostics.
            # It now refers to the all-valid-path Wardrop gap, not the old
            # active-route-only gap.
            "final_gap": float(final_diagnostics["wardrop_gap"]),
            "wardrop_gap": float(final_diagnostics["wardrop_gap"]),
            "relative_flow_change": float(final_diagnostics["relative_flow_change"]),
            "feasibility_abs_error": float(final_diagnostics["feasibility_abs_error"]),
            "feasibility_rel_error": float(final_diagnostics["feasibility_rel_error"]),
            "invalid_route_flow": float(final_diagnostics["invalid_route_flow"]),

            # Histories for diagnostics and debugging.
            "wardrop_gap_history": wardrop_gap_history,
            "relative_flow_change_history": relative_flow_change_history,
            "feasibility_abs_history": feasibility_abs_history,
            "feasibility_rel_history": feasibility_rel_history,
            "invalid_route_flow_history": invalid_route_flow_history,

            # Backward-compatible alias: this now stores the Wardrop gap history.
            "relative_gap_history": wardrop_gap_history,
        }

        return current_route_flows, info
    

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
                **self._imd_last_info,
                "iterations": float(self._imd_last_info["iterations"]),
                "final_gap": float(self._imd_last_info["final_gap"]),
                "converged": float(self._imd_last_info["converged"]),
                "linearization_iter": float(
                    self._imd_last_info.get(
                        "linearization_iter",
                        self._imd_last_info["iterations"],
                    )
                ),
                "implicit_grad": True,
            }

            # Update cache for the next forward pass.
            self.cached_route_flows.copy_(current_route_flows.detach().mean(dim=0, keepdim=True))
            self.cache_valid.fill_(True)
            return final_link_flows, current_route_flows, info

        # --- EXPLICIT SOLVER PATH (For Inference or non-IMD training) ---
        current_route_flows, solver_info = self._run_equilibrium_iterations(
            route_init=route_init,
            od_demands=od_demands,
            bpr_params=bpr_params,
        )

        final_link_flows = self._route_to_link_flows(current_route_flows)

        info = {
            **solver_info,
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
        
        Formula: t = t0 * (1 + alpha * (flow / (capacity * lanes * multiplier))^beta)
        """
        alpha = params["alpha"]
        beta = params["beta"]
        cap_mult = params["capacity_multiplier"]

        # Ensure parameters are broadcastable across the batch
        if alpha.dim() == 1: 
            alpha = alpha.unsqueeze(0)
        if beta.dim() == 1: 
            beta = beta.unsqueeze(0)
        if cap_mult.dim() == 1: 
            cap_mult = cap_mult.unsqueeze(0)

        # Capacity in the network table is interpreted as per-lane capacity.
        # Therefore, the physical link capacity must be expanded by the number
        # of lanes before applying the learnable correction multiplier.
        base_capacity = self.capacity.unsqueeze(0)  # [1, Links]
        lane_count = self.lanes.unsqueeze(0).clamp(min=1.0)  # [1, Links]

        effective_capacity = base_capacity * lane_count

        # Apply the learnable correction multiplier after the physical lane expansion.
        # This preserves the distinction between fixed infrastructure attributes
        # and model-calibrated capacity correction.
        adj_capacity = (effective_capacity * cap_mult).clamp(min=self.eps)
        
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

#%% Main Model Class (Orchestator)

class VariationalInequalityModel(nn.Module):
    """
    End-to-end Variational Inequality model for Traffic Assignment.
    
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
        diagnostics: Optional[Dict] = None,
        # General Parameters
        initial_mean: float = 1.0,    # Starting average value for OD demand
        link_scale: float = 1.0,      # Normalization factor for link flows
        od_scale: float = 1.0,        # Normalization factor for OD demand
        oracle_alpha: Optional[torch.Tensor] = None, # Optional oracle alpha for diagnostics
        oracle_beta: Optional[torch.Tensor] = None,  # Optional oracle beta for diagnostics
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

        # TODO: TEMPORAL PARA ARREGLAR EL TEMA DE LA ALINEACION DE DATOS
        # TEMPORARY DEBUG FLAG: force the solver to use the true OD demand directly.
        # Remove after the oracle-demand audit.
        self.debug_force_true_od = bool(kwargs.get("debug_force_true_od", False))
        ##########################################################

        # TODO: TEMPORAL PARA ARREGLAR LO DEL DELTA/ROUTE PROJECTION AUDIT
        self.debug_delta_route_projection = bool(kwargs.get("debug_delta_route_projection", False))
        ##########################################################

        # TEMPORAL #################################
        # Supply mode controls how BPR parameters are produced.
        # - "learned": use PhysicsInformedBPRNet.
        # - "oracle": bypass PhysicsInformedBPRNet and use true scenario parameters.
        self.supply_mode = str(kwargs.get("supply_mode", "learned")).lower()


        if self.supply_mode not in {"learned", "oracle"}:
            raise ValueError(
                "Invalid supply_mode. Expected one of {'learned', 'oracle'}, "
                f"received: {self.supply_mode}"
            )
        ###########################################

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
        # self.apply_od_init = kwargs.get("apply_od_init", bool(self.od_estimation_policy_cfg.get("apply_od_init", False)))
        # self.anchor_known_od_in_solver = bool(self.od_estimation_policy_cfg.get("anchor_known_od_in_solver", False))
        # self.unknown_od_init_value = float(kwargs.get("unknown_od_init_value", 0.0))
        # self._unknown_od_init_applied = False # Toggle to ensure one-time OD initialization

        # Register static network attributes as buffers (persistent but not learnable)
        self.register_buffer("t0", t0.float())
        self.register_buffer("capacity", capacity.float())
        self.register_buffer("length", length.float())
        self.register_buffer("lanes", lanes.float())
        self.register_buffer("speed", speed.float())
        self.register_buffer("link_group", link_group.long())
        self.register_buffer("od_pair_indices", od_pair_indices.long())

        # TEMPORAL
        # ------------------------------------------------------------------
        # Optional oracle BPR parameters
        # ------------------------------------------------------------------
        # These buffers are used only when model.supply_mode == "oracle".
        # They should come directly from the generated network file.
        if oracle_alpha is not None:
            if oracle_alpha.numel() != self.num_links:
                raise ValueError(
                    "oracle_alpha must have one value per link. "
                    f"Expected {self.num_links}, received {oracle_alpha.numel()}."
                )

            self.register_buffer(
                "oracle_alpha",
                oracle_alpha.float().reshape(-1),
            )
        else:
            self.register_buffer(
                "oracle_alpha",
                torch.empty(0, dtype=torch.float32),
            )

        if oracle_beta is not None:
            if oracle_beta.numel() != self.num_links:
                raise ValueError(
                    "oracle_beta must have one value per link. "
                    f"Expected {self.num_links}, received {oracle_beta.numel()}."
                )

            self.register_buffer(
                "oracle_beta",
                oracle_beta.float().reshape(-1),
            )
        else:
            self.register_buffer(
                "oracle_beta",
                torch.empty(0, dtype=torch.float32),
            )

        ##############################################

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
            lanes=lanes,
            solver_cfg=self.solver_cfg,
            imd_cfg=self.imd_cfg,
        )


        # 5. OD DEMAND COMPLETION MODULE
        # Demand is no longer represented as a free parameter vector.
        # Instead, it is produced by a differentiable completion function:
        #     completed_od = D_phi(known_od, od_mask)
        #
        # The completed OD is then treated as inelastic demand by the equilibrium solver.
        demand_completion_cfg = dict(self.od_estimation_policy_cfg.get("demand_completion", {}))

        self.demand_net = ODDemandCompletionNet(
            num_od_pairs=self.num_od_pairs,
            od_scale=self.od_scale,
            hidden_dim=int(demand_completion_cfg.get("hidden_dim", 128)),
            num_layers=int(demand_completion_cfg.get("num_layers", 2)),
            dropout=float(demand_completion_cfg.get("dropout", 0.0)),
            hard_anchor_known_od=bool(demand_completion_cfg.get("hard_anchor_known_od", True)),
            known_anchor_weight=float(demand_completion_cfg.get("known_anchor_weight", 0.95)),
        )

        # 6. LOSS FUNCTION
        loss_cfg = dict(loss or {})
        loss_cfg.pop("link_scale", None)
        loss_cfg.pop("od_scale", None)
        self.loss_fn = Loss(
            link_scale=self.link_scale,
            od_scale=self.od_scale,
            **loss_cfg
        )

        # 7. INSTANTIATE DELEGATOR
        self.delegator = VIModelDelegator(self)

        # 8. INSTANTIATE DIAGNOSTICIAN (optional)
        # The YAML can disable diagnostics entirely via diagnostics.enabled.
        self.diagnostics_cfg = dict(diagnostics or {})
        self.diagnostician = None
        if self.diagnostics_cfg:
            diag_cfg = {
                key: value
                for key, value in self.diagnostics_cfg.items()
                if key not in {"_target_"}
            }
            self.diagnostician = VIDiagnostician(**diag_cfg)

    # TEMPORAL ##############################################################

    def _get_bpr_params(self) -> Dict[str, torch.Tensor]:
        """
        Return the BPR parameters used by the equilibrium solver.

        Modes
        -----
        learned:
            Uses PhysicsInformedBPRNet to estimate bounded alpha, beta and
            capacity_multiplier.

        oracle:
            Bypasses PhysicsInformedBPRNet and uses the true BPR parameters stored
            in the synthetic scenario artifact. This mode is intended only for
            diagnostic/oracle experiments.
        """

        if self.supply_mode == "learned":
            link_features = self._build_link_features()
            return self.supply_net(link_features)

        if self.supply_mode == "oracle":
            if self.oracle_alpha.numel() != self.num_links:
                raise RuntimeError(
                    "supply_mode='oracle' requires oracle_alpha with one value per link."
                )

            if self.oracle_beta.numel() != self.num_links:
                raise RuntimeError(
                    "supply_mode='oracle' requires oracle_beta with one value per link."
                )

            device = self.t0.device
            dtype = self.t0.dtype

            # In Synthetic04, effective capacity is scaled to daily values
            # through the configured hourly-to-daily multiplier.
            capacity_multiplier = torch.full(
                size=(self.num_links,),
                fill_value=float(self.capacity_correction_cfg.get("oracle_value", 12.0)),
                device=device,
                dtype=dtype,
            )

            return {
                "alpha": self.oracle_alpha.to(device=device, dtype=dtype),
                "beta": self.oracle_beta.to(device=device, dtype=dtype),
                "capacity_multiplier": capacity_multiplier,
            }

        raise RuntimeError(f"Unsupported supply_mode: {self.supply_mode}")

    #########################################################################

    def get_optimizer_param_groups(self, base_lr: float):
        """
        Assign different learning rates to supply-side and demand-side parameters.

        Demand-side parameters:
            Parameters of ODDemandCompletionNet.

        Supply-side parameters:
            Parameters of the physics-informed BPR module and, if active, the
            SUE dispersion parameter theta.
        """
        demand_params = [
            p for p in self.demand_net.parameters()
            if p.requires_grad
        ]

        supply_params = [
            p for _, p in self.supply_net.named_parameters()
            if p.requires_grad
        ]

        if self.sue_mode and hasattr(self, "theta_raw") and self.theta_raw.requires_grad:
            supply_params.append(self.theta_raw)

        supply_lr = float(base_lr) * self.supply_lr_multiplier
        demand_lr = float(base_lr) * self.demand_lr_multiplier

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


    # TODO: TEMPORAL BORRAR
    def _debug_od_alignment_once(
        self,
        true_od_demand,
        od_mask,
        true_od_sparse,
        od_mask_sparse,
        current_epoch=None,
    ):
        """
        Temporary OD alignment audit.
        Remove after debugging.
        """
        if getattr(self, "_debug_od_alignment_printed", False):
            return

        self._debug_od_alignment_printed = True

        def _stats(name, x):
            if x is None:
                logging.warning("[OD AUDIT] %s = None", name)
                return

            x_det = x.detach().float().cpu()
            logging.warning(
                "[OD AUDIT] %s | shape=%s | sum=%.6f | mean=%.6f | min=%.6f | max=%.6f | nonzero=%d",
                name,
                tuple(x_det.shape),
                float(x_det.sum().item()),
                float(x_det.mean().item()),
                float(x_det.min().item()),
                float(x_det.max().item()),
                int((x_det != 0).sum().item()),
            )

        logging.warning("=" * 90)
        logging.warning("[OD AUDIT] current_epoch=%s", str(current_epoch))
        logging.warning("[OD AUDIT] self.num_od_pairs=%s", str(self.num_od_pairs))

        _stats("true_od_demand BEFORE _align_od_targets", true_od_demand)
        _stats("od_mask BEFORE _align_od_targets", od_mask)
        _stats("true_od_sparse AFTER _align_od_targets", true_od_sparse)
        _stats("od_mask_sparse AFTER _align_od_targets", od_mask_sparse)

        if hasattr(self, "od_pair_indices"):
            logging.warning(
                "[OD AUDIT] od_pair_indices shape=%s | first_20=%s",
                tuple(self.od_pair_indices.shape),
                self.od_pair_indices[:20].detach().cpu().tolist(),
            )

        if hasattr(self, "od_pair_node_labels") and self.od_pair_node_labels is not None:
            logging.warning(
                "[OD AUDIT] od_pair_node_labels first_20=%s",
                self.od_pair_node_labels[:20],
            )

        if true_od_sparse is not None and od_mask_sparse is not None:
            active = od_mask_sparse.detach().cpu().reshape(-1) > 0
            sparse_flat = true_od_sparse.detach().cpu().reshape(-1)

            logging.warning(
                "[OD AUDIT] supervised sparse OD count=%d | supervised sparse OD sum=%.6f",
                int(active.sum().item()),
                float(sparse_flat[active].sum().item()) if active.any() else 0.0,
            )

            logging.warning(
                "[OD AUDIT] first_20 sparse true OD=%s",
                sparse_flat[:20].tolist(),
            )

            logging.warning(
                "[OD AUDIT] first_20 sparse mask=%s",
                od_mask_sparse.detach().cpu().reshape(-1)[:20].tolist(),
            )

        logging.warning("=" * 90)

    def _align_od_targets(self, true_od, od_mask):
        if true_od.dim() == 1:
            true_od = true_od.unsqueeze(0)

        if od_mask is None:
            od_mask = torch.ones_like(true_od)
        elif od_mask.dim() == 1:
            od_mask = od_mask.unsqueeze(0)

        if true_od.shape[1] != self.num_od_pairs:
            raise ArtifactSchemaError(
                "VI_Model expects OD targets already aligned with model OD order. "
                f"Expected {self.num_od_pairs}, received {true_od.shape[1]}. "
                "Do not perform OD alignment inside the model. Fix the data-processing artifact."
            )

        if od_mask.shape != true_od.shape:
            raise ArtifactSchemaError(
                "od_mask must have the same shape as true_od_demand. "
                f"true_od={tuple(true_od.shape)}, od_mask={tuple(od_mask.shape)}."
            )

        return true_od, od_mask

    def _align_od_targets_old(
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


    def _estimate_od(
        self,
        batch_size: int,
        true_od_sparse: Optional[torch.Tensor],
        od_mask_sparse: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Estimate a complete inelastic OD demand vector.

        This method prepares the partial OD information required by
        ODDemandCompletionNet.

        If OD supervision is available:
            - true_od_sparse provides observed OD entries.
            - od_mask_sparse indicates which entries are known.

        If OD supervision is missing:
            - the model receives an all-zero OD vector and an all-zero mask.
            - this corresponds to fully unsupervised OD completion from the learned
            prior encoded in the network parameters.

        Returns:
            estimated_od:
                Complete nonnegative OD demand tensor with shape [B, OD].
        """
        if true_od_sparse is None:
            known_od = torch.zeros(
                batch_size,
                self.num_od_pairs,
                device=device,
                dtype=dtype,
            )
        else:
            known_od = true_od_sparse.to(device=device, dtype=dtype)
            if known_od.dim() == 1:
                known_od = known_od.unsqueeze(0)
            if known_od.shape[0] == 1 and batch_size > 1:
                known_od = known_od.expand(batch_size, -1)

        if od_mask_sparse is None:
            od_mask = torch.zeros(
                batch_size,
                self.num_od_pairs,
                device=device,
                dtype=dtype,
            )
        else:
            od_mask = od_mask_sparse.to(device=device, dtype=dtype)
            if od_mask.dim() == 1:
                od_mask = od_mask.unsqueeze(0)
            if od_mask.shape[0] == 1 and batch_size > 1:
                od_mask = od_mask.expand(batch_size, -1)

        return self.demand_net(
            known_od=known_od,
            od_mask=od_mask,
        )
        
    # TODO: TEMPORAL PARA ARREGLAR LA CUESTIÓN DE LA DEMANDA ORACLE Y POTENCAL DESALINEAMIENTO 
    def _debug_oracle_demand_pass(
        self,
        estimated_od: torch.Tensor,
        true_od_sparse: torch.Tensor | None,
        od_mask_sparse: torch.Tensor | None,
        current_epoch=None,
    ) -> torch.Tensor:
        """
        Temporary diagnostic: bypass ODDemandCompletionNet and use the true OD demand
        directly as the inelastic demand passed to the equilibrium solver.

        This checks whether the assignment layer + route incidence + flow targets
        are mutually consistent.
        """
        if true_od_sparse is None:
            raise RuntimeError(
                "[ORACLE OD AUDIT] Cannot force true OD because true_od_sparse is None. "
                "Pass true_od_demand and od_mask during training."
            )

        oracle_od = true_od_sparse.to(
            device=estimated_od.device,
            dtype=estimated_od.dtype,
        )

        if oracle_od.dim() == 1:
            oracle_od = oracle_od.unsqueeze(0)

        if oracle_od.shape != estimated_od.shape:
            raise RuntimeError(
                "[ORACLE OD AUDIT] Shape mismatch. "
                f"oracle_od={tuple(oracle_od.shape)}, "
                f"estimated_od={tuple(estimated_od.shape)}."
            )

        # Optional but useful: if the mask is not complete, warn loudly.
        if od_mask_sparse is not None:
            mask_sum = float(od_mask_sparse.detach().float().sum().cpu().item())
            expected = float(od_mask_sparse.numel())
            if mask_sum < expected:
                logging.warning(
                    "[ORACLE OD AUDIT] OD mask is not fully observed: %.0f / %.0f.",
                    mask_sum,
                    expected,
                )

        if not getattr(self, "_debug_oracle_demand_printed", False):
            self._debug_oracle_demand_printed = True

            est = estimated_od.detach().float().cpu()
            tru = oracle_od.detach().float().cpu()
            abs_diff = torch.abs(est - tru)

            logging.warning("=" * 90)
            logging.warning("[ORACLE OD AUDIT] current_epoch=%s", str(current_epoch))
            logging.warning("[ORACLE OD AUDIT] FORCING estimated_od := true_od_sparse")
            logging.warning(
                "[ORACLE OD AUDIT] estimated_od BEFORE | shape=%s | sum=%.6f | mean=%.6f | min=%.6f | max=%.6f",
                tuple(est.shape),
                float(est.sum().item()),
                float(est.mean().item()),
                float(est.min().item()),
                float(est.max().item()),
            )
            logging.warning(
                "[ORACLE OD AUDIT] oracle_od TRUE | shape=%s | sum=%.6f | mean=%.6f | min=%.6f | max=%.6f",
                tuple(tru.shape),
                float(tru.sum().item()),
                float(tru.mean().item()),
                float(tru.min().item()),
                float(tru.max().item()),
            )
            logging.warning(
                "[ORACLE OD AUDIT] abs(est-true) BEFORE forcing | mean=%.6f | max=%.6f | sum=%.6f",
                float(abs_diff.mean().item()),
                float(abs_diff.max().item()),
                float(abs_diff.sum().item()),
            )
            logging.warning("[ORACLE OD AUDIT] first_20 estimated_od=%s", est.reshape(-1)[:20].tolist())
            logging.warning("[ORACLE OD AUDIT] first_20 oracle_od=%s", tru.reshape(-1)[:20].tolist())
            logging.warning("=" * 90)

        return oracle_od

    # TODO: TEMPORAL Delta/Route Projection Audit.

    def _debug_delta_route_projection_once(
        self,
        true_od_sparse: torch.Tensor | None,
        observed_flows: torch.Tensor,
        flow_mask: torch.Tensor,
        bpr_params: dict | None = None,
        current_epoch=None,
    ):
        """
        Temporary Delta/Route Projection Audit.

        This audit checks whether route_validity_mask + delta_matrix can project the
        true OD demand into link flows that are at least structurally compatible with
        the target link flows.

        It builds two non-trained diagnostic assignments:

        1. Equal split:
        Each OD demand is divided uniformly among its valid routes.

        2. Free-flow AON proxy:
        Each OD demand is assigned to the valid route with minimum free-flow path cost.

        Remove after debugging.
        """
        if getattr(self, "_debug_delta_projection_printed", False):
            return

        self._debug_delta_projection_printed = True

        if true_od_sparse is None:
            logging.warning(
                "[DELTA AUDIT] Skipped because true_od_sparse is None."
            )
            return

        device = observed_flows.device
        dtype = observed_flows.dtype

        true_od = true_od_sparse.to(device=device, dtype=dtype)
        if true_od.dim() == 1:
            true_od = true_od.unsqueeze(0)

        target = observed_flows.to(device=device, dtype=dtype)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        mask = flow_mask.to(device=device, dtype=dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if true_od.shape[1] != self.num_od_pairs:
            raise RuntimeError(
                "[DELTA AUDIT] true_od_sparse is not aligned with model OD space. "
                f"Expected {self.num_od_pairs}, got {true_od.shape[1]}."
            )

        valid = self.equilibrium_solver.route_validity_mask.to(device=device)
        valid_f = valid.to(dtype=dtype)

        batch_size = true_od.shape[0]
        num_od = self.num_od_pairs
        k_paths = self.equilibrium_solver.k_paths
        num_routes = num_od * k_paths

        # ------------------------------------------------------------------
        # 1. Equal split route flows
        # ------------------------------------------------------------------
        valid_counts = valid_f.sum(dim=1).clamp(min=1.0)  # [OD]
        equal_probs = valid_f / valid_counts.unsqueeze(1)  # [OD, K]

        equal_route_flows = (
            true_od.unsqueeze(-1) * equal_probs.unsqueeze(0)
        ).reshape(batch_size, num_routes)

        equal_link_flows = self.equilibrium_solver._route_to_link_flows(
            equal_route_flows
        )

        # ------------------------------------------------------------------
        # 2. Free-flow shortest valid route / AON proxy
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Route cost at free-flow: Delta.T @ t0
            t0 = self.equilibrium_solver.t0.to(device=device, dtype=dtype)

            ff_route_costs_flat = torch.sparse.mm(
                self.equilibrium_solver.delta_matrix.t(),
                t0.unsqueeze(1),
            ).squeeze(1)

            ff_route_costs = ff_route_costs_flat.view(num_od, k_paths)

            inf = torch.full_like(ff_route_costs, 1.0e12)
            ff_route_costs_valid = torch.where(valid, ff_route_costs, inf)

            best_k = torch.argmin(ff_route_costs_valid, dim=1)  # [OD]

            aon_probs = torch.zeros(
                (num_od, k_paths),
                dtype=dtype,
                device=device,
            )
            aon_probs[
                torch.arange(num_od, device=device),
                best_k,
            ] = 1.0

            # Safety: OD pairs with no valid routes should receive zero flow.
            has_valid = valid.any(dim=1)
            aon_probs = aon_probs * has_valid.unsqueeze(1).to(dtype=dtype)

            aon_route_flows = (
                true_od.unsqueeze(-1) * aon_probs.unsqueeze(0)
            ).reshape(batch_size, num_routes)

            aon_link_flows = self.equilibrium_solver._route_to_link_flows(
                aon_route_flows
            )

        # ------------------------------------------------------------------
        # 3. Optional: current solver projection if bpr_params is provided
        # ------------------------------------------------------------------
        solver_link_flows = None
        if bpr_params is not None:
            with torch.no_grad():
                solver_link_flows, _, solver_info = self.equilibrium_solver(
                    od_demands=true_od,
                    bpr_params=bpr_params,
                )
        else:
            solver_info = {}

        # ------------------------------------------------------------------
        # 4. Metrics helpers
        # ------------------------------------------------------------------
        def _masked_metrics(name: str, pred: torch.Tensor):
            pred = pred.detach().float()
            y_true = target.detach().float()
            m = mask.detach().bool()

            if m.shape != y_true.shape:
                raise RuntimeError(
                    f"[DELTA AUDIT] mask shape mismatch for {name}: "
                    f"mask={tuple(m.shape)}, target={tuple(y_true.shape)}."
                )

            yp = pred[m]
            yt = y_true[m]

            if yt.numel() == 0:
                logging.warning("[DELTA AUDIT] %s has no masked target data.", name)
                return

            ss_res = torch.sum((yt - yp) ** 2)
            ss_tot = torch.sum((yt - yt.mean()) ** 2).clamp(min=1.0e-8)
            r2 = 1.0 - ss_res / ss_tot

            mae = torch.mean(torch.abs(yt - yp))
            rmse = torch.sqrt(torch.mean((yt - yp) ** 2))

            true_total = yt.sum()
            pred_total = yp.sum()
            total_gap = torch.abs(pred_total - true_total)
            rel_total_gap = total_gap / torch.clamp(torch.abs(true_total), min=1.0e-8)

            corr = torch.corrcoef(
                torch.stack([yt.reshape(-1), yp.reshape(-1)])
            )[0, 1] if yt.numel() > 1 else torch.tensor(float("nan"))

            logging.warning(
                "[DELTA AUDIT] %s | count=%d | R2=%.6f | corr=%.6f | MAE=%.6f | RMSE=%.6f | true_total=%.6f | pred_total=%.6f | rel_total_gap=%.6f",
                name,
                int(yt.numel()),
                float(r2.detach().cpu().item()),
                float(corr.detach().cpu().item()) if torch.isfinite(corr) else float("nan"),
                float(mae.detach().cpu().item()),
                float(rmse.detach().cpu().item()),
                float(true_total.detach().cpu().item()),
                float(pred_total.detach().cpu().item()),
                float(rel_total_gap.detach().cpu().item()),
            )

        def _tensor_stats(name: str, x: torch.Tensor):
            x = x.detach().float().reshape(-1).cpu()
            logging.warning(
                "[DELTA AUDIT] %s | shape=%s | sum=%.6f | mean=%.6f | min=%.6f | max=%.6f | nonzero=%d",
                name,
                tuple(x.shape),
                float(x.sum().item()),
                float(x.mean().item()),
                float(x.min().item()),
                float(x.max().item()),
                int((x != 0).sum().item()),
            )

        # ------------------------------------------------------------------
        # 5. Log structural diagnostics
        # ------------------------------------------------------------------
        logging.warning("=" * 100)
        logging.warning("[DELTA AUDIT] current_epoch=%s", str(current_epoch))
        logging.warning("[DELTA AUDIT] num_od=%d | k_paths=%d | num_routes=%d | num_links=%d",
                        num_od, k_paths, num_routes, int(self.equilibrium_solver.delta_matrix.shape[0]))

        logging.warning(
            "[DELTA AUDIT] delta_matrix shape=%s | nnz=%d | route_validity valid=%d/%d",
            tuple(self.equilibrium_solver.delta_matrix.shape),
            int(self.equilibrium_solver.delta_matrix._nnz()),
            int(valid.sum().detach().cpu().item()),
            int(valid.numel()),
        )

        logging.warning(
            "[DELTA AUDIT] valid route count per OD | min=%.0f | mean=%.3f | max=%.0f | od_without_routes=%d",
            float(valid_f.sum(dim=1).min().detach().cpu().item()),
            float(valid_f.sum(dim=1).mean().detach().cpu().item()),
            float(valid_f.sum(dim=1).max().detach().cpu().item()),
            int((valid_f.sum(dim=1) == 0).sum().detach().cpu().item()),
        )

        _tensor_stats("true_od", true_od)
        _tensor_stats("target_flows", target)
        _tensor_stats("flow_mask", mask)
        _tensor_stats("equal_link_flows", equal_link_flows)
        _tensor_stats("aon_link_flows", aon_link_flows)

        _masked_metrics("EQUAL_SPLIT_vs_TARGET", equal_link_flows)
        _masked_metrics("FREEFLOW_AON_vs_TARGET", aon_link_flows)

        if solver_link_flows is not None:
            _tensor_stats("solver_link_flows", solver_link_flows)
            _masked_metrics("SOLVER_vs_TARGET", solver_link_flows)

            logging.warning(
                "[DELTA AUDIT] solver_info=%s",
                {
                    key: solver_info.get(key)
                    for key in [
                        "iterations",
                        "converged",
                        "final_gap",
                        "wardrop_gap",
                        "relative_flow_change",
                        "feasibility_abs_error",
                        "feasibility_rel_error",
                        "invalid_route_flow",
                        "mode",
                    ]
                    if key in solver_info
                },
            )

        # ------------------------------------------------------------------
        # 6. First links sample for manual inspection
        # ------------------------------------------------------------------
        target_flat = target.detach().cpu().reshape(-1)
        mask_flat = mask.detach().cpu().reshape(-1)
        equal_flat = equal_link_flows.detach().cpu().reshape(-1)
        aon_flat = aon_link_flows.detach().cpu().reshape(-1)

        logging.warning("[DELTA AUDIT] first_20 target_flows=%s", target_flat[:20].tolist())
        logging.warning("[DELTA AUDIT] first_20 mask=%s", mask_flat[:20].tolist())
        logging.warning("[DELTA AUDIT] first_20 equal_link_flows=%s", equal_flat[:20].tolist())
        logging.warning("[DELTA AUDIT] first_20 aon_link_flows=%s", aon_flat[:20].tolist())

        # ------------------------------------------------------------------
        # 7. Error ranking audit
        # ------------------------------------------------------------------
        def _top_error_report(name: str, pred_flat: torch.Tensor, top_n: int = 30):
            """
            Report links with largest absolute errors under the active flow mask.
            This helps distinguish between:
            - a global link-order permutation;
            - a few extreme outlier links;
            - target/prediction support mismatch.
            """
            target_cpu = target.detach().float().cpu().reshape(-1)
            mask_cpu = mask.detach().float().cpu().reshape(-1) > 0.5
            pred_cpu = pred_flat.detach().float().cpu().reshape(-1)

            if target_cpu.shape != pred_cpu.shape:
                raise RuntimeError(
                    f"[DELTA AUDIT] {name} shape mismatch: "
                    f"target={tuple(target_cpu.shape)}, pred={tuple(pred_cpu.shape)}."
                )

            active_idx = torch.where(mask_cpu)[0]

            if active_idx.numel() == 0:
                logging.warning("[DELTA AUDIT] %s top-error report skipped: empty mask.", name)
                return

            yt = target_cpu[active_idx]
            yp = pred_cpu[active_idx]

            abs_err = torch.abs(yt - yp)
            signed_err = yp - yt

            k = min(int(top_n), int(active_idx.numel()))
            top_vals, order = torch.topk(abs_err, k=k, largest=True)

            top_global_idx = active_idx[order]

            rows = []
            for rank, local_pos in enumerate(order.tolist(), start=1):
                global_idx = int(active_idx[local_pos].item())
                true_val = float(target_cpu[global_idx].item())
                pred_val = float(pred_cpu[global_idx].item())
                err_val = float(abs(pred_val - true_val))
                signed_val = float(pred_val - true_val)

                rel_err = (
                    err_val / max(abs(true_val), 1.0e-8)
                    if abs(true_val) > 1.0e-8
                    else float("inf")
                )

                rows.append(
                    {
                        "rank": rank,
                        "link_idx": global_idx,
                        "target": true_val,
                        "pred": pred_val,
                        "abs_err": err_val,
                        "signed_err": signed_val,
                        "rel_err": rel_err,
                    }
                )

            logging.warning("[DELTA AUDIT] %s TOP_%d_ABS_ERRORS=%s", name, k, rows)

            # Support mismatch diagnostics
            target_nonzero = target_cpu.abs() > 1.0e-8
            pred_nonzero = pred_cpu.abs() > 1.0e-8

            active_target_nonzero = target_nonzero & mask_cpu
            active_pred_nonzero = pred_nonzero & mask_cpu

            target_zero_pred_positive = mask_cpu & (~target_nonzero) & pred_nonzero
            target_positive_pred_zero = mask_cpu & target_nonzero & (~pred_nonzero)

            logging.warning(
                "[DELTA AUDIT] %s SUPPORT | active=%d | target_nonzero_active=%d | pred_nonzero_active=%d | target_zero_pred_positive=%d | target_positive_pred_zero=%d",
                name,
                int(mask_cpu.sum().item()),
                int(active_target_nonzero.sum().item()),
                int(active_pred_nonzero.sum().item()),
                int(target_zero_pred_positive.sum().item()),
                int(target_positive_pred_zero.sum().item()),
            )

            if target_zero_pred_positive.any():
                idxs = torch.where(target_zero_pred_positive)[0][:20].tolist()
                sample = [
                    {
                        "link_idx": int(i),
                        "target": float(target_cpu[i].item()),
                        "pred": float(pred_cpu[i].item()),
                    }
                    for i in idxs
                ]
                logging.warning(
                    "[DELTA AUDIT] %s SAMPLE target_zero_pred_positive=%s",
                    name,
                    sample,
                )

            if target_positive_pred_zero.any():
                idxs = torch.where(target_positive_pred_zero)[0][:20].tolist()
                sample = [
                    {
                        "link_idx": int(i),
                        "target": float(target_cpu[i].item()),
                        "pred": float(pred_cpu[i].item()),
                    }
                    for i in idxs
                ]
                logging.warning(
                    "[DELTA AUDIT] %s SAMPLE target_positive_pred_zero=%s",
                    name,
                    sample,
                )


        def _rank_overlap_report(name: str, pred_flat: torch.Tensor, top_n: int = 30):
            """
            Compare top-N most loaded target links against top-N predicted links.
            Low overlap with good total balance indicates spatial redistribution mismatch.
            """
            target_cpu = target.detach().float().cpu().reshape(-1)
            mask_cpu = mask.detach().float().cpu().reshape(-1) > 0.5
            pred_cpu = pred_flat.detach().float().cpu().reshape(-1)

            active_idx = torch.where(mask_cpu)[0]

            if active_idx.numel() == 0:
                logging.warning("[DELTA AUDIT] %s rank-overlap skipped: empty mask.", name)
                return

            yt = target_cpu[active_idx]
            yp = pred_cpu[active_idx]

            k = min(int(top_n), int(active_idx.numel()))

            target_top_local = torch.topk(yt, k=k, largest=True).indices
            pred_top_local = torch.topk(yp, k=k, largest=True).indices

            target_top_global = set(active_idx[target_top_local].tolist())
            pred_top_global = set(active_idx[pred_top_local].tolist())

            overlap = target_top_global.intersection(pred_top_global)

            logging.warning(
                "[DELTA AUDIT] %s TOP_%d_RANK_OVERLAP | overlap=%d/%d | jaccard=%.6f | target_only_sample=%s | pred_only_sample=%s",
                name,
                k,
                len(overlap),
                k,
                len(overlap) / max(len(target_top_global.union(pred_top_global)), 1),
                sorted(list(target_top_global - pred_top_global))[:20],
                sorted(list(pred_top_global - target_top_global))[:20],
            )


        _top_error_report("EQUAL_SPLIT_vs_TARGET", equal_flat, top_n=30)
        _top_error_report("FREEFLOW_AON_vs_TARGET", aon_flat, top_n=30)

        _rank_overlap_report("EQUAL_SPLIT_vs_TARGET", equal_flat, top_n=30)
        _rank_overlap_report("FREEFLOW_AON_vs_TARGET", aon_flat, top_n=30)

        if solver_link_flows is not None:
            solver_flat = solver_link_flows.detach().cpu().reshape(-1)
            _top_error_report("SOLVER_vs_TARGET", solver_flat, top_n=30)
            _rank_overlap_report("SOLVER_vs_TARGET", solver_flat, top_n=30)
            
        logging.warning("=" * 100)


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

        # TEMPORAL ###############################################
        # 1. SUPPLY: Get BPR parameters.
        # In normal training this comes from supply_net.
        # In oracle mode this bypasses supply_net and uses true scenario parameters.
        bpr_params = self._get_bpr_params()
        ######################################################

        # Handle learnable theta for Stochastic User Equilibrium (SUE)
        """if self.sue_mode and hasattr(self, "theta_raw"):
            # Mapping: raw -> sigmoid -> [theta_min, theta_max]
            # This ensures that theta stays within the range specified in the YAML config.
            theta_val = self.theta_min + (self.theta_max - self.theta_min) * torch.sigmoid(self.theta_raw)
            bpr_params["theta"] = theta_val"""

        # TEMPORAL ##############################################################
        # Handle theta for Stochastic User Equilibrium (SUE).
        if self.sue_mode:
            if self.supply_mode == "oracle":
                theta_val = torch.tensor(
                    float(self.solver_cfg.get("oracle_theta", self.solver_cfg.get("initial_theta", 2.0))),
                    device=observed_flows.device,
                    dtype=observed_flows.dtype,
                )
            elif hasattr(self, "theta_raw"):
                theta_val = self.theta_min + (
                    self.theta_max - self.theta_min
                ) * torch.sigmoid(self.theta_raw)
            else:
                theta_val = None

            if theta_val is not None:
                bpr_params["theta"] = theta_val

        #########################################################################

        # 2. ALIGNMENT: Map external OD targets to the model's sparse structure
        true_od_sparse = None
        od_mask_sparse = None
        if true_od_demand is not None:
            true_od_sparse, od_mask_sparse = self._align_od_targets(true_od_demand, od_mask)

            # TEMPORAL BORRAR
            self._debug_od_alignment_once(
                true_od_demand=true_od_demand,
                od_mask=od_mask,
                true_od_sparse=true_od_sparse,
                od_mask_sparse=od_mask_sparse,
                current_epoch=None,
            )

        # 3. DEMAND COMPLETION
        # The OD vector is completed by a differentiable function D_phi.
        # Once completed, it is treated as fixed inelastic demand by the equilibrium solver.
        estimated_od = self._estimate_od(
            batch_size=batch_size,
            true_od_sparse=true_od_sparse,
            od_mask_sparse=od_mask_sparse,
            device=observed_flows.device,
            dtype=observed_flows.dtype,
        )

        # TODO: TEMPORAL PARA ARREGLAR LO DE LA DEMANDA ORACLE EN EL ENTRENAMIENTO
        if getattr(self, "debug_force_true_od", False):
            estimated_od = self._debug_oracle_demand_pass(
                estimated_od=estimated_od,
                true_od_sparse=true_od_sparse,
                od_mask_sparse=od_mask_sparse,
                current_epoch=kwargs.get("current_epoch", None),
            )

        # 4. INELASTIC DEMAND INPUT TO THE EQUILIBRIUM SOLVER
        # No additional anchoring is needed here because ODDemandCompletionNet already
        # applies either hard anchoring or flexible anchoring internally.
        od_input = estimated_od

        # TODO: TEMPORAL PARA ARREGLAR LO DE DELTA/ROUTE PROJECTION AUDIT
        if getattr(self, "debug_delta_route_projection", False):
            self._debug_delta_route_projection_once(
                true_od_sparse=true_od_sparse,
                observed_flows=observed_flows,
                flow_mask=flow_mask,
                bpr_params=bpr_params,
                current_epoch=None,
            )

        ##################################################

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
            "learned_capacity_multiplier": bpr_params["capacity_multiplier"],
            "effective_capacity": (
                self.capacity * self.lanes.clamp(min=1.0)),
            "adjusted_capacity": (
                self.capacity
                * self.lanes.clamp(min=1.0)
                * bpr_params["capacity_multiplier"].detach()
            )
        }

        if self.sue_mode and hasattr(self, "theta_raw"):
            outputs["learned_theta"] = bpr_params["theta"]

        # 7. LOSS CALCULATION
        # Loss is only meaningful when supervision targets are available
        should_compute_loss = not is_pure_inference

        if should_compute_loss:
            loss_dict = self.loss_fn(
                predicted_flows=pred_link_flows,
                true_flows=observed_flows,
                flow_mask=flow_mask if flow_mask is not None else torch.ones_like(observed_flows),
                predicted_od=estimated_od,
                true_od=(
                    true_od_sparse
                    if true_od_sparse is not None
                    else torch.zeros_like(estimated_od)
                ),
                od_mask=(
                    od_mask_sparse
                    if od_mask_sparse is not None
                    else torch.zeros_like(estimated_od)
                ),
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

#%% Tailored Loss Function for VI Training

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
            loss_links = torch.tensor(
                0.0,
                device=predicted_flows.device,
                dtype=predicted_flows.dtype,
            )

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
            loss_prior_od = torch.tensor(
                0.0,
                device=predicted_od.device,
                dtype=predicted_od.dtype,
            )

        # Unknown OD regularization.
        # Only entries with od_mask == 0 are regularized.
        # This prevents the completion module from creating unrealistically large
        # unobserved demands when link-count supervision is sparse.
        unknown_mask_bool = od_mask <= 0.5

        if unknown_mask_bool.any():
            loss_unknown_regularization = torch.mean(
                scaled_pred_od[unknown_mask_bool] ** 2
            )
        else:
            loss_unknown_regularization = torch.tensor(
                0.0,
                device=predicted_od.device,
                dtype=predicted_od.dtype,
            )


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
