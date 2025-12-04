"""
=============================================================================
MODEL NAME: STATIC_CYCLIC_MLP_SUE (CGME_MLP_SUE)
BASED ON:   Cyclic Graph Attentive Match Encoder (Li et al., 2022)
TYPE:       Cyclic Graph Matcher MLP-Encoder-Decoder with Stochastic User Equilibrium
DOMAIN:     Static OD Estimation + Static Flow Estimation (No time dimension)
=============================================================================

ARCHITECTURE OVERVIEW:
----------------------
Este modelo es una adaptación estática del framework CGAME. Reemplaza la inferencia
temporal y la red neuronal inversa (backward) por un validador físico basado en
Asignación de Tráfico (SUE - Stochastic User Equilibrium).

      [Observed Counts] --(Encoder MLP)--> [h_x] --(Graph Matcher)--> [g_x]
                                             ^            |
                                             | (Matching) v
      [True OD Demand] --(Encoder MLP)--- [h_y] <--(Decoder MLP)-- [Pred OD]
            |
            +----(Assignment Validator / SUE)----> [Reconstructed Flows]

KEY COMPONENTS:
1. ODEncoder (MLP):
   - Red neuronal simple (Linear -> LeakyReLU -> Linear).
   - Procesa vectores de conteos estáticos (sin dimensión temporal T).

2. GraphMatcher (The Core - from CGAME):
   - Alinea los espacios latentes Forward (h_x) y Backward (h_y).
   - Utiliza matrices de estructura (M) y valor (V) con mecanismo de atención
     para filtrar coincidencias incorrectas entre estimación y asignación[cite: 10].

3. AssignmentValidator (The Physics - SUE):
   - Reemplaza la "Backward Network" del paper original.
   - Implementa un simulador de tráfico diferenciable (MSA Loop).
   - Calcula probabilidades de ruta (Logit) y costos de congestión (BPR)
     para garantizar que los flujos reconstruidos respeten la física del tráfico.

INPUTS/OUTPUTS:
---------------
- Input:  Tensor de conteos observados [Batch, Num_Links].
- Output: Tensor de demanda OD estimada [Batch, Num_OD_Pairs].
- Loss:   Mezcla de error de reconstrucción de flujos y consistencia latente.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from typing import Optional, Dict, Any
import logging


class ODEncoder(nn.Module):
    """Takes the vehicle counts vector (how many cars passed each sensor)
    and compresses it into a feature (embedding) vector.
    """

    def __init__(self, num_links: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        # Paper: LeakyReLU(W2(LeakyReLU(W1x+b1))+b2)
        self.network = nn.Sequential(
            nn.Linear(num_links, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, counts_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.
            Input: [counts_vector]: counts tensor [batch_size, num_links]
            Output: latent feature tensor [batch_size, feature_dim]
        """
        return self.network(counts_vector)


class ODDecoder(nn.Module):
    """Performs the inverse of the Encoder. Takes the processed latent vector
    and predicts how many trips exist between each Origin-Destination pair.
    """

    def __init__(self, num_od_pairs: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_od_pairs),
            nn.Softplus()
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
            Input: [g_x]: latent features tensor [batch_size, feature_dim]
            Output: predicted OD demands tensor [batch_size, num_od_pairs]
        """
        return self.network(g_x)


class GraphMatcher(nn.Module):
    """
    Implementation of the Graph Matcher following the CGAME paper.
    References: Equations 7, 9, 10, 11 and Algorithm 1.
    """

    def __init__(self, feature_dim: int, num_structures: int,
                 lambda_m: float = 0.1, lambda_v: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_structures = num_structures

        # Decay parameters for the update (momentum)
        self.lambda_m = lambda_m
        self.lambda_v = lambda_v

        # --- FIX 1: Initialization to ones (Paper) ---
        # M: [feature_dim, num_structures]
        self.register_buffer('M', torch.ones(feature_dim, num_structures))
        # V: [1, num_structures]
        self.register_buffer('V', torch.ones(1, num_structures))

    def update_matrices(self, h_x: torch.Tensor, h_y: torch.Tensor):
        """
        Performs the M and V update based on similarity (Algorithm 1 of the paper).
        """
        # Avoid gradients during memory update
        with torch.no_grad():
            batch_size = h_x.size(0)

            # --- Equation 9: Update of M ---
            # Paper: M_j = (1 - lambda)*M + lambda * similarity(h_x, h_y)
            # Compute element-wise cosine-like similarity summed over the batch
            dot_xy = torch.sum(h_x * h_y, dim=0, keepdim=True).T  # [feature_dim, 1]
            norm_x = torch.sqrt(torch.sum(h_x * h_x, dim=0, keepdim=True)).T + 1e-8
            norm_y = torch.sqrt(torch.sum(h_y * h_y, dim=0, keepdim=True)).T + 1e-8

            similarity_term_M = dot_xy / (norm_x * norm_y)  # [feature_dim, 1]

            # Expand to all structures
            # captures different subsets, but mathematically the base update is the same
            # if there are no external masks. Apply the same update to all columns).
            similarity_M_expanded = similarity_term_M.expand(-1, self.num_structures).clone()

            # Adding random noise to each structure for divergence TODO: not sure if this works
            noise = torch.randn_like(similarity_M_expanded) * 0.01
            similarity_M_expanded += noise

            self.M.data = (1 - self.lambda_m) * self.M.data + \
                          self.lambda_m * similarity_M_expanded

            # --- Equation 10: Intermediate decay of V ---
            self.V.data = (1 - self.lambda_m) * self.V.data

            # --- Equation 11: Update of V ---
            # V measures similarity between (h_x transformed by M) and h_y

            # 1. Transform h_x with current M: (h_x @ 1) * M -> Element-wise with broadcasting
            # h_x: [Batch, Feat] -> [Batch, Feat, 1]
            # M: [Feat, Struct]
            h_x_trans = h_x.unsqueeze(2) * self.M.unsqueeze(0)  # [Batch, Feat, Struct]

            # 2. Prepare h_y for comparison
            h_y_exp = h_y.unsqueeze(2)  # [Batch, Feat, 1]

            # 3. Compute cosine similarity over the 'feature' dimension (dim 1)
            # Numerator: sum_nf( h_x_trans * h_y )
            num = torch.sum(h_x_trans * h_y_exp, dim=1)  # [Batch, Struct]

            # Denominators
            den_x = torch.sqrt(torch.sum(h_x_trans ** 2, dim=1)) + 1e-8  # [Batch, Struct]
            den_y = torch.sqrt(torch.sum(h_y_exp ** 2, dim=1)) + 1e-8  # [Batch, 1] (broadcastable)

            # Average similarity over the batch for each structure
            similarity_V = torch.mean(num / (den_x * den_y), dim=0, keepdim=True)  # [1, Struct]

            # Apply final update to V
            self.V.data = self.V.data + self.lambda_v * similarity_V

    def forward(self, h_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Equation 7 of the paper).
        There are no neural networks here, just matrix operations with M and V.
        """
        # h_x: [Batch, Feature_dim]
        # M:   [Feature_dim, Num_Structures]
        # V:   [1, Num_Structures]

        # Broadcasting for element-wise operation:
        # h_x -> [Batch, Feat, 1]
        # M   -> [1,     Feat, Struct]
        # V   -> [1,     1,    Struct]

        h_x_exp = h_x.unsqueeze(2)
        M_exp = self.M.unsqueeze(0)
        V_exp = self.V.unsqueeze(0)

        # Equation 7: h_x * M * V
        weighted_features = h_x_exp * M_exp * V_exp  # [Batch, Feat, Struct]

        # Equation 7: Mean over structures dimension (n_s)
        g_x = torch.mean(weighted_features, dim=2)  # [Batch, Feat]

        return g_x


class AssignmentValidator(nn.Module):
    """Acts as a differentiable traffic simulator.
    Takes the predicted OD demand and computes which links would be used,
    respecting congestion (if a link fills up, drivers change routes).
    """

    def __init__(self, num_links: int, t0: torch.Tensor, capacity: torch.Tensor,
                 route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: DictConfig,  # MODIFICATION: receives config
                 max_iters: int = 100,
                 convergence_threshold: float = 1e-2,
                 max_trips_scaler: float = 1.0): # Scaling factor to denormalize
        super().__init__()

        self.max_iters = max_iters # Maximum number of SUE iterations
        self.convergence_threshold = convergence_threshold
        self.max_trips_scaler = max_trips_scaler

        self.register_buffer('t0', t0)

        # Cost function with improved initialization
        self.cost_function = hydra.utils.instantiate(
            vdf_config,
            t0=t0,
            capacity=capacity,
            num_link_groups=num_link_groups,
            link_group=link_group,
            _recursive_=False
        )

        self.assignment_layer = StaticAssignmentLayer(route_masks, od_pair_indices, num_od_pairs)

        # Convergence tracking
        self.register_buffer('last_iterations', torch.tensor(0.0))

    def forward(self, normalized_demand: torch.Tensor, warmup: bool = False) -> tuple:
        """
        Args:
            normalized_demand: Tensor [Batch, OD] approximately in [0, 1] (neural network output).
            warmup: If True, does only 1 iteration.
        """
        batch_size = normalized_demand.shape[0]

        # -----------------------------------------------------------
        # 1. DENORMALIZATION (Neural Scale -> Physical Scale)
        # -----------------------------------------------------------
        # Convert demand to vehicles/hour so the VDF has physical meaning.
        real_demand = normalized_demand * self.max_trips_scaler

        # Initial costs (Free-flow)
        # Use detach() here for safety, although zeros has no gradient.
        freeflow_costs = self.cost_function(torch.zeros_like(self.t0).expand(batch_size, -1)).detach()

        # -----------------------------------------------------------
        # 2. EQUILIBRIUM SEARCH PHASE (No Gradients)
        # -----------------------------------------------------------
        # We run MSA inside 'torch.no_grad()'.
        # This avoids storing many graph copies and prevents exploding gradients.
        # VDF parameters are NOT updated based on what happens here.

        with torch.no_grad():
            # Initial assignment (All-or-Nothing or initial Logit)
            flows, _ = self.assignment_layer(freeflow_costs, real_demand)

            converged = False
            actual_iters = 1

            if not warmup:
                for it in range(1, self.max_iters + 1):
                    prev_flows = flows.clone()

                    # a. Compute Costs (Physics)
                    costs = self.cost_function(flows)

                    # b. New auxiliary assignment
                    new_flows, _ = self.assignment_layer(costs, real_demand)

                    # c. MSA averaging (Method of Successive Averages)
                    alpha_msa = 1.0 / (it + 1)
                    flows = flows + alpha_msa * (new_flows - flows)

                    # d. Check convergence (Relative Gap)
                    # Avoid division by zero with 1e-9
                    flow_change = torch.norm(flows - prev_flows, dim=1) / (torch.norm(prev_flows, dim=1) + 1e-9)
                    max_change = torch.max(flow_change).item()  # Scalar for checking

                    actual_iters = it
                    if max_change < self.convergence_threshold:
                        converged = True
                        # logging.info("MSA converged after {} iters. Error: {:.6f}".format(it, max_change))
                        break

            # Save statistic for monitoring
            self.last_iterations.data = torch.tensor(float(actual_iters))

            # Conditional logging (useful for debug, be careful if printed in training loop)
            if not converged and not warmup:
                logging.debug(f"MSA did not converge after {self.max_iters} iters. Error: {max_change:.6f}") # TODO: important

        # -----------------------------------------------------------
        # 3. GRADIENT PHASE (One-Step Unrolling)
        # -----------------------------------------------------------
        # This is where the magic happens. We take the equilibrium flow 'flows' computed above,
        # BUT treat it as a fixed constant (detached).
        # We recompute Costs and Assignment ONCE allowing gradients.

        # A. Final Costs:
        # By passing 'flows.detach()', we cut the MSA history.
        # However, 'self.cost_function' uses its internal parameters (alpha, beta).
        # Therefore, gradients for alpha/beta are produced based on this final state.
        final_costs = self.cost_function(flows.detach())

        # B. Reconstructed Flows Final:
        # 'real_demand' DOES have gradient (comes from the Decoder).
        # 'final_costs' DOES have gradient (from VDF parameters).
        # The result 'reconstructed_flows' connects everything for backprop.
        reconstructed_flows_real, route_probs = self.assignment_layer(final_costs, real_demand)

        # C. Renormalize for the DL pipeline
        # We return the flows in [0, 1] scale so Loss and Encoder backward
        # won't get crazy large numbers.
        reconstructed_flows_norm = reconstructed_flows_real / self.max_trips_scaler

        # -----------------------------------------------------------
        # 4. EXTRACT PARAMETERS (For visualization/logs)
        # -----------------------------------------------------------
        learned_alpha = getattr(self.cost_function, 'get_alpha', lambda: None)()
        learned_beta = getattr(self.cost_function, 'get_beta', lambda: None)()

        if learned_alpha is None: learned_alpha = torch.tensor(0.15, device=self.t0.device)
        if learned_beta is None: learned_beta = torch.tensor(4.0, device=self.t0.device)

        convergence_info = {
            "converged": converged,
            "iterations": actual_iters,
            "final_gap": max_change if not warmup and 'max_change' in locals() else 0.0
        }

        # IMPORTANT: reconstructed_flows is in REAL scale (Veh/h).
        # Whoever computes the Loss function must decide if they want to normalize it again or not.
        return reconstructed_flows_norm, learned_alpha, learned_beta, convergence_info, route_probs


class StaticAssignmentLayer(nn.Module):
    """
    Assignment layer optimized for sparse tensors (no einsum).
    Replaces dense logic to avoid memory/runtime errors.

    Performs the heavy math: map path-level trip assignments
    to the physical links that compose those paths.
    """

    def __init__(self, route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, mu: float = 1.0):
        super().__init__()

        # Store original dimensions
        self.num_od, self.k_paths, self.num_links = route_masks.shape
        self.register_buffer('od_pair_indices', od_pair_indices)
        self.num_od_pairs = num_od_pairs

        # --- LOGIC COPIED FROM CYCLIC_MODEL (Flattening 3D -> 2D) ---
        # Convert mask [OD, K, Links] to [OD*K, Links] to use sparse.mm
        if route_masks.is_sparse:
            route_masks = route_masks.coalesce()
            indices = route_masks.indices()
            values = route_masks.values()

            # Compute new row indices: row = od_idx * K + k_idx
            new_rows = indices[0] * self.k_paths + indices[1]
            new_cols = indices[2]

            new_indices = torch.stack([new_rows, new_cols])

            # 2D sparse matrix: [Rows=TotalRoutes, Cols=Links]
            self.register_buffer(
                'sparse_mask_2d',
                torch.sparse_coo_tensor(
                    new_indices,
                    values,
                    size=(self.num_od * self.k_paths, self.num_links)
                )
            )
        else:
            # Dense fallback converted to sparse
            self.register_buffer('sparse_mask_2d',
                                 route_masks.reshape(-1, self.num_links).to_sparse())

        # Learnable parameter mu
        self.mu_raw = nn.Parameter(torch.tensor(mu))

    @property
    def mu(self):
        return torch.clamp(F.softplus(self.mu_raw), min=0.1, max=10.0)

    def forward(self, link_costs: torch.Tensor, demands: torch.Tensor) -> tuple:
        """
        Computes flows using sparse matrix multiplication (sparse.mm).
        """
        batch_size = link_costs.shape[0]

        # =====================================================================
        # STEP 1: Compute Path Costs (Link -> Path)
        # Replacement for: torch.einsum('bl,okl->bok', link_costs, self.route_masks)
        # Logic: RouteCosts = (Mask @ LinkCosts.T).T
        # =====================================================================

        # LinkCosts.T [OD*K, Links] (Sparse) -> [Links, Batch]
        costs_t = torch.transpose(link_costs, 0, 1)

        # Sparse MM: [OD*K, Links] @ [Links, Batch] -> [OD*K, Batch]
        route_costs_flat_t = torch.sparse.mm(self.sparse_mask_2d, costs_t)

        # Reshape: [Batch, OD, K]
        route_costs = route_costs_flat_t.transpose(0, 1).view(batch_size, self.num_od, self.k_paths)

        # =====================================================================
        # STEP 2: Logit Probabilities (Stabilization)
        # =====================================================================
        min_costs, _ = torch.min(route_costs, dim=2, keepdim=True)
        stable_costs = route_costs - min_costs.detach()
        stable_costs = torch.clamp(stable_costs, max=50.0)

        exp_utility = torch.exp(-self.mu * stable_costs)
        sum_utility = torch.sum(exp_utility, dim=2, keepdim=True)
        route_probs = exp_utility / (sum_utility + 1e-9)

        # Assign Demand: [Batch, OD, K]
        route_flows = route_probs * demands.unsqueeze(2)

        # =====================================================================
        # STEP 3: Project to Links (Path -> Link)
        # Replacement for: torch.einsum('bok,okl->bl', route_flows, self.route_masks)
        # Logic: LinkFlows = (Mask.T @ RouteFlowsFlat.T).T
        # =====================================================================

        # Flatten route flows: [Batch, OD*K]
        route_flows_flat = route_flows.view(batch_size, -1)

        # Transpose to multiply: [OD*K, Batch]
        rf_t = torch.transpose(route_flows_flat, 0, 1)

        # PyTorch trick: mask.t() on sparse is fast (just inverts indices)
        mask_t = self.sparse_mask_2d.t()  # [Links, OD*K]

        # Sparse MM: [Links, OD*K] (Sparse) @ [OD*K, Batch] -> [Links, Batch]
        link_flows_t = torch.sparse.mm(mask_t, rf_t)

        # Back to batch format: [Batch, Links]
        link_flows = link_flows_t.transpose(0, 1)

        # Return TUPLE (Flows, Probabilities) for pipeline compatibility
        return link_flows, route_probs


class LossCalculator(nn.Module):
    """Adaptive and balanced loss function."""

    def __init__(self, w_counts: float = 1.0, w_od: float = 1.0, w_reg: float = 0.01,
                 adaptive_weights: bool = True):
        super().__init__()
        self.register_buffer('w_counts', torch.tensor(w_counts))
        self.register_buffer('w_od', torch.tensor(w_od))
        self.w_reg = w_reg
        self.adaptive_weights = adaptive_weights
        self.mse_loss = nn.MSELoss()

        # For tracking historical losses
        self.register_buffer('loss_history_counts', torch.tensor(0.0))
        self.register_buffer('loss_history_od', torch.tensor(0.0))
        self.register_buffer('update_count', torch.tensor(0.0))

    def forward(self,
                predicted_flows: torch.Tensor,  # Sent by pipeline
                true_flows: torch.Tensor,  # Sent by pipeline
                flow_mask: torch.Tensor,  # Sent by pipeline
                predicted_od: torch.Tensor,  # Sent by pipeline
                true_od: torch.Tensor,  # Sent by pipeline
                od_mask: torch.Tensor,  # Sent by pipeline
                learned_alpha: Optional[torch.Tensor] = None,
                learned_beta: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        # Internal adapted logic

        # 1. Flow loss
        if flow_mask.any():
            masked_pred = predicted_flows * flow_mask
            masked_true = true_flows * flow_mask
            l_counts = self.mse_loss(masked_pred, masked_true)
        else:
            l_counts = torch.tensor(0.0, device=predicted_flows.device)

        # 2. OD loss
        l_od = torch.tensor(0.0, device=predicted_flows.device)
        od_ratio = 0.0
        if od_mask is not None and od_mask.any():
            masked_pred_od = predicted_od * od_mask
            masked_true_od = true_od * od_mask
            l_od = self.mse_loss(masked_pred_od, masked_true_od)
            od_ratio = od_mask.float().mean().item()

        # 3. Regularization
        l_reg = torch.tensor(0.0, device=predicted_flows.device)
        if learned_alpha is not None and learned_beta is not None:
            l_reg = (torch.norm(learned_alpha - 0.15, p=2) +
                     torch.norm(learned_beta - 4.0, p=2))

        # 4. Weight adaptation (Your original logic)
        if self.adaptive_weights and self.training:
            self._adapt_weights(l_counts.detach(), l_od.detach(), od_ratio)

        total_loss = (self.w_counts * l_counts +
                      self.w_od * l_od +
                      self.w_reg * l_reg)

        return {
            "total_loss": total_loss,
            "l_flow": l_counts,  # Renamed for compatibility with pipeline logs (was l_counts)
            "l_od": l_od,
            "l_reg": l_reg,
            "w_flow": self.w_counts.item(),  # Renamed for logs
            "w_od": self.w_od.item()
        }

    def _adapt_weights(self, l_counts: torch.Tensor, l_od: torch.Tensor, od_ratio: float):
        """Automatic weight adaptation based on historical losses."""
        alpha = 0.9  # Smoothing factor

        # Update moving averages
        upd = self.update_count.item() if isinstance(self.update_count, torch.Tensor) else float(self.update_count)
        if upd == 0.0:
            self.loss_history_counts.data = l_counts
            self.loss_history_od.data = l_od if l_od > 0 else torch.tensor(1.0)
        else:
            self.loss_history_counts.data = alpha * self.loss_history_counts + (1 - alpha) * l_counts
            if l_od > 0:
                self.loss_history_od.data = alpha * self.loss_history_od + (1 - alpha) * l_od

        # Balance weights based on relative magnitudes
        if self.loss_history_od > 1e-6:
            ratio = self.loss_history_counts / self.loss_history_od
            # Adjust w_od inversely proportional to ratio and to OD data availability
            self.w_od.data = torch.clamp(ratio * (0.1 + od_ratio), min=0.1, max=5.0)


        # Increment update counter
        if isinstance(self.update_count, torch.Tensor):
            self.update_count.data = self.update_count.data + 1.0
        else:
            self.update_count = upd + 1.0


class CyclicODModel(nn.Module):
    """Improved model with a more robust architecture."""

    def __init__(self, num_links, num_od_pairs, hidden_dim, feature_dim, num_structures,
                 t0, capacity, route_masks, od_pair_indices, num_link_groups, link_group,
                 vdf_config: DictConfig,
                 dropout: float = 0.1,
                 max_trips_scaler: float = 1.0,
                 **kwargs):
        super().__init__()

        # --- 1. Dual Encoders (As in the paper) ---
        self.forward_encoder = ODEncoder(num_links, hidden_dim, feature_dim, dropout)
        self.backward_encoder = ODEncoder(num_links, hidden_dim, feature_dim, dropout)

        self.decoder = ODDecoder(num_od_pairs, hidden_dim, feature_dim, dropout)

        # --- 2. Strict Graph Matcher (No attention_net) ---
        self.graph_matcher = GraphMatcher(feature_dim, num_structures)

        # --- 3. Physical Backward Network (Your design) ---
        self.validator = AssignmentValidator(
            num_links, t0, capacity, route_masks, od_pair_indices,
            num_od_pairs, num_link_groups, link_group,
            vdf_config=vdf_config,
            max_trips_scaler=max_trips_scaler
        )

    def forward(self, observed_counts: torch.Tensor, true_od_demand: torch.Tensor = None,
                warmup: bool = False) -> dict:

        # Batch handling
        is_batched = observed_counts.dim() == 2
        if not is_batched:
            observed_counts = observed_counts.unsqueeze(0)
            if true_od_demand is not None:
                true_od_demand = true_od_demand.unsqueeze(0)

        # 1. Forward Encoder: obtain h_x
        h_x = self.forward_encoder(observed_counts)

        # 2. Training and Matching logic
        if self.training and true_od_demand is not None:
            # A. Generate h_y using the Backward Network (Your validator + Backward Encoder)
            #    Note: The paper uses a backward neural network; you use the physical validator.
            #    This is correct for your hybrid design.
            with torch.no_grad():
                true_flows_norm, _, _, _, _ = self.validator(true_od_demand, warmup=True)
                h_y = self.backward_encoder(true_flows_norm)  # Separate encoder

            # B. Update matrices M and V (Only during training)
            self.graph_matcher.update_matrices(h_x.detach(), h_y.detach())

        # 3. Apply Graph Matcher (Equation 7) to obtain g_x
        g_x = self.graph_matcher(h_x)

        # 4. Decoder
        estimated_demand = self.decoder(g_x)

        # 5. SUE Validation
        reconstructed_flows, learned_alpha, learned_beta, convergence_info, route_probs = self.validator(
            estimated_demand, warmup=warmup)

        if not is_batched:
            estimated_demand = estimated_demand.squeeze(0)
            reconstructed_flows = reconstructed_flows.squeeze(0)
            if route_probs is not None:
                route_probs = route_probs.squeeze(0)

        return {
            "estimated_demand": estimated_demand,
            "reconstructed_flows": reconstructed_flows,
            "learned_alpha": learned_alpha,
            "learned_beta": learned_beta,
            "convergence_info": convergence_info,
            "route_probs": route_probs
        }

    def validate_latent_space_consistency(self, observed_counts, true_od_demand):
        """Diagnostic to check if the two encoders are converging."""
        with torch.no_grad():
            h_x = self.forward_encoder(observed_counts)

            if true_od_demand is not None:
                true_flows, _, _, _, _ = self.validator(true_od_demand, warmup=True)
                h_y = self.backward_encoder(true_flows)

                cosine_sim = F.cosine_similarity(h_x, h_y, dim=1).mean()
                return {"cosine_similarity": cosine_sim.item()}
        return None

# -----------------------------------------------------------------------------
# Compatibility with `train_cyclic_model_deprecated.py`
# -----------------------------------------------------------------------------
# `train_cyclic_model_deprecated.py` expects to be able to do:
# from src.models.Cyclic_Model.cyclic_model_ultra import CyclicODModelUltra, PartialDataLoss
# and then instantiate `CyclicODModelUltra(..., cost_function_type=..., dropout=...)`
# and use `PartialDataLoss(w_flow=..., w_od=..., w_reg=...)`.
# To maintain compatibility, we add small "shims" (wrappers) that
# expose the same classes/signatures the training script expects.
# -----------------------------------------------------------------------------


class PartialDataLoss(nn.Module):
    """
    Compatibility wrapper that replicates the API of `PartialDataLoss` used
    by the original training. Implementation based on the version from
    `CGAME_MLP_SUE.py` to ensure consistent behavior.
    """

    def __init__(self, w_flow: float = 1.0, w_od: float = 1.0, w_reg: float = 0.01):
        super().__init__()
        self.register_buffer('w_flow', torch.tensor(w_flow))
        self.register_buffer('w_od', torch.tensor(w_od))
        self.w_reg = w_reg
        self.mse_loss = nn.MSELoss()

    def forward(self,
                predicted_flows: torch.Tensor,
                true_flows: torch.Tensor,
                flow_mask: torch.Tensor,
                predicted_od: torch.Tensor,
                true_od: torch.Tensor,
                od_mask: torch.Tensor,
                learned_alpha: Optional[torch.Tensor] = None,
                learned_beta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        # Flow loss (only observed links)
        if flow_mask.any():
            masked_pred_flows = predicted_flows * flow_mask
            masked_true_flows = true_flows * flow_mask
            l_flow = self.mse_loss(masked_pred_flows, masked_true_flows)
            flow_coverage = float(flow_mask.float().mean().item())
        else:
            l_flow = torch.tensor(0.0, device=predicted_flows.device)
            flow_coverage = 0.0

        # OD loss (only known demands)
        if od_mask is not None and od_mask.any():
            masked_pred_od = predicted_od * od_mask
            masked_true_od = true_od * od_mask
            l_od = self.mse_loss(masked_pred_od, masked_true_od)
            od_coverage = float(od_mask.float().mean().item())
        else:
            l_od = torch.tensor(0.0, device=predicted_od.device)
            od_coverage = 0.0

        # BPR parameter regularization
        l_reg = torch.tensor(0.0, device=predicted_flows.device)
        if learned_alpha is not None and learned_beta is not None:
            l_reg = (torch.norm(learned_alpha - 0.15, p=2) +
                     torch.norm(learned_beta - 4.0, p=2))

        # Total loss
        total_loss = (self.w_flow * l_flow +
                      self.w_od * l_od +
                      self.w_reg * l_reg)

        return {
            "total_loss": total_loss,
            "l_flow": l_flow,
            "l_od": l_od,
            "l_reg": l_reg,
            "flow_coverage": flow_coverage,
            "od_coverage": od_coverage,
            "w_flow": float(self.w_flow.item()),
            "w_od": float(self.w_od.item())
        }


class CyclicODModelUltra(CyclicODModel):
    """
    Compatibility wrapper that accepts the argument `cost_function_type`
    (which is passed from `train_cyclic_model_deprecated.py`) but ignores or stores it
    for possible future use. Keeps the same signature as `CyclicODModel`.
    """

    def __init__(self, *args, cost_function_type: str = 'bpr', **kwargs):
        # Consume cost_function_type for compatibility; UltraCyclicODModel does not require it
        super().__init__(*args, **kwargs)
        self.cost_function_type = cost_function_type

    def forward(self, observed_flows: torch.Tensor, flow_mask: torch.Tensor,
                true_od_demand: torch.Tensor = None, warmup: bool = False) -> dict:
        """
        Compatibility wrapper matching the signature of `CyclicODModel.forward`.

        - Applies the mask to observed flows (like the original model does).
        - Calls the Ultra model forward which expects `observed_counts`.
        """
        is_batched = observed_flows.dim() == 2
        if not is_batched:
            observed_flows_proc = observed_flows.unsqueeze(0)
            flow_mask_proc = flow_mask.unsqueeze(0)
            if true_od_demand is not None:
                true_od_proc = true_od_demand.unsqueeze(0)
            else:
                true_od_proc = None
        else:
            observed_flows_proc = observed_flows
            flow_mask_proc = flow_mask
            true_od_proc = true_od_demand

        # Apply mask (set 0 where there is no observation)
        observed_counts = observed_flows_proc * flow_mask_proc

        # Call the Ultra forward (expects observed_counts)
        outputs = super().forward(observed_counts=observed_counts,
                                  true_od_demand=true_od_proc,
                                  warmup=warmup)

        # If it wasn't batched, undo the added dimension
        if not is_batched:
            outputs['estimated_demand'] = outputs['estimated_demand'].squeeze(0)
            outputs['reconstructed_flows'] = outputs['reconstructed_flows'].squeeze(0)
            # learned_alpha/beta can be None or tensors; if they are tensors with batch dim, handle it
            if isinstance(outputs.get('learned_alpha', None), torch.Tensor) and outputs['learned_alpha'].dim() == 2:
                outputs['learned_alpha'] = outputs['learned_alpha'].squeeze(0)
            if isinstance(outputs.get('learned_beta', None), torch.Tensor) and outputs['learned_beta'].dim() == 2:
                outputs['learned_beta'] = outputs['learned_beta'].squeeze(0)

        return outputs

# Facilitate direct import without changes in the training script
__all__ = [
    'CyclicODModelUltra',
    'PartialDataLoss',
    'CyclicODModel',
    'LossCalculator'
]
