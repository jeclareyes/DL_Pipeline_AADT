import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from omegaconf import DictConfig
from typing import Dict, Optional
import hydra
import os

logger = logging.getLogger(__name__)

# =============================================================================
# 1. NEURAL NETWORK COMPONENTS (Encoder/Decoder/Matcher)
# =============================================================================

class ODEncoder(nn.Module):
    """
    Encodes input vectors (link flows or ODs) into latent features.
    Deep architecture: [Input -> H -> H/2 -> Feature]
    """
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.network(x)
        return hx


class ODDecoder(nn.Module):
    """
    Decodes the latent vector into physical outputs (ODs or link flows).
    Deep architecture: [Feature -> H -> H/2 -> Output]
    """
    def __init__(self, output_dim: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus() # Ensures positive values (physical realism)
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        return self.network(g_x)

class GraphMatcher(nn.Module):
    """
    Enhanced Graph Matcher (replaces ImprovedGraphMatcher).
    - Includes a 'forward' method (required by PyTorch).
    - Separates matrix updates (update) from the forward pass.
    """
    def __init__(self, feature_dim: int, num_structures: int,
                 lambda_m: float = 0.01, lambda_v: float = 0.01,
                 reg_strength: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_structures = num_structures
        self.lambda_m = lambda_m
        self.lambda_v = lambda_v
        self.reg_strength = reg_strength

        # M and V matrices with soft initialization
        self.register_buffer('M', torch.randn(feature_dim, num_structures) * 0.1)
        self.register_buffer('V', torch.ones(1, num_structures))
        self.register_buffer('update_count', torch.tensor(0.0))

        # Red de atención aprendible
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, num_structures),
            nn.Softmax(dim=-1)
        )

    def _ensure_batch_dim(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            return h.unsqueeze(0)
        return h

    @torch.no_grad()
    def update(self, h_x: torch.Tensor, h_y: torch.Tensor):
        """Explicit update of matrices M and V."""
        h_x = self._ensure_batch_dim(h_x)
        h_y = self._ensure_batch_dim(h_y)

        # Normalization
        h_x_norm = F.normalize(h_x, p=2, dim=1)
        h_y_norm = F.normalize(h_y, p=2, dim=1)

        # 1. Update M (Structural Projection)
        similarity_vector = (h_x_norm * h_y_norm).mean(dim=0)
        target_M = similarity_vector.unsqueeze(1).expand(-1, self.num_structures)

        # Regularization
        noise = torch.randn_like(self.M) * 0.01
        regularized_target = target_M + self.reg_strength * noise

        # Adaptive momentum
        momentum = min(self.lambda_m * (1 + self.update_count.item() * 0.001), 0.1)
        self.M.data = (1 - momentum) * self.M.data + momentum * regularized_target

        # 2. Update V (Global Attention)
        h_x_transformed = h_x.unsqueeze(2) * self.M 
        h_y_expanded = h_y.unsqueeze(2)             

        num = (h_x_transformed * h_y_expanded).sum(dim=1) 
        den = torch.norm(h_x_transformed, dim=1) * torch.norm(h_y_expanded, dim=1) + 1e-8
        cosine_sim = (num / den).mean(dim=0) 

        target_V = torch.clamp(cosine_sim, min=0.1, max=2.0).unsqueeze(0) 

        momentum_v = min(self.lambda_v * (1 + self.update_count.item() * 0.001), 0.1)
        self.V.data = (1 - momentum_v) * self.V.data + momentum_v * target_V

        self.update_count += 1

    def forward(self, h_x: torch.Tensor) -> torch.Tensor:
        """Apply the learned transformation."""
        h_x = self._ensure_batch_dim(h_x)
        
        # Learned attention
        attn_weights = self.attention_net(h_x) # [B, S]
        
        # Projection
        h_exp = h_x.unsqueeze(2) # [B, F, 1]
        h_struct = h_exp * self.M.unsqueeze(0) # [B, F, S]
        h_weighted = h_struct * self.V.unsqueeze(0) # [B, F, S]
        
        # Weighted combination
        g_x = (h_weighted * attn_weights.unsqueeze(1)).sum(dim=2) # [B, F]
        
        return g_x


class PhysicsAwareAssignment(nn.Module):
    def __init__(self, num_links, delta_matrix, vdf_config, 
                 t0, capacity, route_masks, lanes,
                 od_pair_indices, num_link_groups, link_group, route_validity_mask,
                 init_alpha=0.15, init_beta=4.0, num_iterations=100):
        
        super().__init__()

        # 2. Instantiate VDF with learnable parameters
        self.vdf = hydra.utils.instantiate(
            vdf_config,
            t0=t0, 
            capacity=capacity, 
            lanes=lanes,
            num_link_groups=num_link_groups, 
            link_group=link_group
        )

        # 3. Register structural tensors as buffers (so they travel to the GPU)
        self.register_buffer("delta", delta_matrix) 
        self.register_buffer("validity_mask", route_validity_mask) # [OD, K]
        self.num_iterations = num_iterations

    def initialize_routes(self, od_demand: torch.Tensor, K: int) -> torch.Tensor:
        """
        Initialize route proportions uniformly, assigning 0 to physically non-existent routes.
        """
        # batch_size = od_demand.shape[0]
        batch_size = 1

        # Count how many valid routes each OD pair has [OD, 1]
        valid_count = self.validity_mask.sum(dim=1, keepdim=True).float()
        valid_count = torch.clamp(valid_count, min=1.0) # Avoid div/0
        
        # Compute base proportion: e.g., if there are 4 routes, each gets 0.25
        # [OD, K]
        base_props = self.validity_mask.float() / valid_count
        
        # Expand to batch size and flatten to [Batch, OD * K]
        base_props = base_props.unsqueeze(0).expand(batch_size, -1, -1)
        return base_props.reshape(batch_size, -1)

    def simplex_projection(self, p_tilde: torch.Tensor, q: torch.Tensor, num_ods: int, K: int) -> torch.Tensor:
        """
        Proyecta proporciones no restringidas (p_tilde) al simplex donde suman q.
        """
        batch_size = p_tilde.shape[0]
        
        # 1. Reshape and strong masking
        # [Batch * OD, K]
        v = p_tilde.view(-1, K)
        
        # Expand the validity mask to cover the full batch
        # mask_expanded: [Batch * OD, K]
        mask_expanded = self.validity_mask.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, K)
        
        # Replace invalid routes with -inf so they always come last when sorting
        v = v.masked_fill(~mask_expanded, float('-inf'))
        
        q_flat = q.view(-1, 1)
        valid_demand_mask = (q_flat > 0).float()
        
        # 2. Sort v in descending order
        u, _ = torch.sort(v, descending=True, dim=1)
        
        # Temporarily remove -inf for the cumulative sum
        u_safe = torch.clamp(u, min=-1e9) 
        
        # 3. Cumulative sum of the sorted array
        cssv = torch.cumsum(u_safe, dim=1)
        
        # 4. Create an index vector [1, 2, ..., K]
        j = torch.arange(1, K + 1, device=v.device, dtype=v.dtype)
        
        # 5. Find the number of strictly positive components (rho)
        condition = (u_safe - (cssv - q_flat) / j) > 0
        rho = (condition * j).max(dim=1, keepdim=True)[0]
        rho = torch.clamp(rho, min=1.0) # Seguro contra fallos numéricos
        
        # 6. Compute the threshold (theta)
        rho_idx = (rho - 1).long()
        cssv_rho = torch.gather(cssv, 1, rho_idx)
        theta = (cssv_rho - q_flat) / rho
        
        # 7. Apply the projection
        w = torch.clamp(v - theta, min=0.0)
        
        # Double-check that invalid routes and ODs with no demand are zero
        w = w * mask_expanded.float()
        w = w * valid_demand_mask
        
        return w.view(batch_size, num_ods * K)

    def forward(self, estimated_od: torch.Tensor):
        batch_size, num_ods = estimated_od.shape
        K = self.validity_mask.shape[1]
        
        # 1. Initialize PROPORTIONS
        p = self.initialize_routes(estimated_od, K)
        
        theta = 5.0 
        
        # We set a very high safety limit to allow convergence,
        # but rely on early stopping to stop much earlier.
        max_safety_iters = 1000 
        
        # Expand the validity mask once
        mask_expanded = self.validity_mask.unsqueeze(0).expand(batch_size, -1, -1)

        for n in range(1, max_safety_iters + 1):
            gamma_n = 1.0 / n 
            
            # A. Link flows
            h_absolute = p * estimated_od.unsqueeze(-1).expand(-1, -1, K).reshape(batch_size, -1)
            v = torch.sparse.mm(self.delta, h_absolute.t()).t()
            
            # B. Compute costs
            t_links = self.vdf(v)
            c_routes = torch.sparse.mm(self.delta.t(), t_links.t()).t()
            
            # C. SEARCH DIRECTION
            c_routes_3d = c_routes.view(batch_size, num_ods, K)
            c_routes_3d_masked = c_routes_3d.masked_fill(~mask_expanded, float('inf'))
            
            c_min = c_routes_3d_masked.min(dim=2, keepdim=True)[0]
            c_shifted = c_routes_3d_masked - c_min
            
            p_target = torch.softmax(-theta * c_shifted, dim=2)
            p_target = p_target * mask_expanded.float()
            p_target = p_target.view(batch_size, -1)
            
            # D. MSA update
            p = (1 - gamma_n) * p + gamma_n * p_target

            # -----------------------------------------------------------------
            # E. EARLY STOPPING CRITERION AND LOGGING (1% tolerance)
            # -----------------------------------------------------------------
            with torch.no_grad():
                p_3d = p.view(batch_size, num_ods, K)
                
                # Define a route "in use" if it has more than 1% of the flow proportion
                used_mask = (p_3d > 0.01) & mask_expanded
                
                # Only evaluate ODs that have MORE THAN ONE route in use
                multi_route_mask = used_mask.sum(dim=2) > 1 
                
                if multi_route_mask.any():
                    # Find max and min cost only over USED routes
                    c_used_max = torch.where(used_mask, c_routes_3d, torch.tensor(-float('inf'), device=p.device)).max(dim=2)[0]
                    c_used_min = torch.where(used_mask, c_routes_3d, torch.tensor(float('inf'), device=p.device)).min(dim=2)[0]
                    
                    c_used_min_safe = torch.clamp(c_used_min, min=1e-6) 
                    
                    # Relative difference (1% tolerance = 0.01)
                    rel_diff = (c_used_max - c_used_min) / c_used_min_safe
                    
                    # --- LOGGING EVERY 50 ITERATIONS ---
                    if n % 50 == 0:
                        # Filter errors only for ODs with multiple used routes
                        valid_errors = torch.where(multi_route_mask, rel_diff, torch.zeros_like(rel_diff))
                        max_gap = valid_errors.max().item() * 100 # Convert to percent
                        print(f"  [MSA Assignment] Iter {n:4d} | Worst Relative Gap: {max_gap:.2f}%")
                    # -----------------------------------------------------

                    # There is a violation if the difference > 0.01 AND the OD has multiple used routes
                    violations = (rel_diff > 0.01) & multi_route_mask
                    
                    # If no violations and we've passed the first 5 iterations
                    if not violations.any() and n > 5:
                        print(f"\n⚡ Forced equilibrium reached. Tolerance < 1% satisfied at iteration {n}.")
                        break
                else:
                    # Edge case: all ODs use a single route
                    if n > 5:
                        print(f"\n⚡ Forced equilibrium reached. Single-route dominance at iteration {n}.")
                        break

        # If the loop finishes without breaking, print a warning
        if n == max_safety_iters:
            print(f"\n⚠️ Warning: Safety limit reached ({max_safety_iters} iters) without achieving 1% tolerance.")

        # Reconstrucción final
        h_final = p * estimated_od.unsqueeze(-1).expand(-1, -1, K).reshape(batch_size, -1)
        final_link_flows = torch.sparse.mm(self.delta, h_final.t()).t()
        final_t_links = self.vdf(final_link_flows)
        final_c_routes = torch.sparse.mm(self.delta.t(), final_t_links.t()).t()
        
        return final_link_flows, h_final, final_c_routes

# =============================================================================
# 2. MAIN MODEL (CRAME - Differential Layer)
# =============================================================================

class CyclicLoop(nn.Module):
    def __init__(
            self,
            num_links: int,
            num_od_pairs: int,
            feature_dim: int,
            h_enc: int,
            h_dec: int,
            delta_matrix: torch.Tensor,
            t0: torch.Tensor,
            capacity: torch.Tensor,
            route_masks: torch.Tensor,
            lanes: torch.Tensor,
            od_pair_indices: torch.Tensor,
            num_link_groups: int,
            link_group: torch.Tensor,
            vdf_config: DictConfig,
            route_validity_mask: torch.Tensor,
            max_routes_per_od: int = 10,
            dropout: float = 0.1,
            num_structures: int = 5,
            **kwargs
    ):
        super().__init__()
        logger.info("INIT: CGAME Hybrid with Differentiable Layer (Truncated Backprop)")

        self.link_scale = kwargs.get('link_scale', 1.0)
        self.od_scale = kwargs.get('od_scale', 1.0)

        # 1. Statistical Proposition (Encoder -> Matcher -> Decoder)
        self.f_encoder = ODEncoder(num_links, h_enc, feature_dim, dropout)
        self.matcher = GraphMatcher(feature_dim, num_structures)
        self.f_decoder = ODDecoder(num_od_pairs, h_dec, feature_dim, dropout)

        # 2. Physical Verification (Projected Gradient Descent Assignment)
        convergence_cfg = vdf_config.get('convergence', {})
        
        self.traffic_assignment = PhysicsAwareAssignment(
            num_links=num_links,
            delta_matrix=delta_matrix,
            vdf_config=vdf_config,
            t0=t0, capacity=capacity, route_masks=route_masks, lanes=lanes,
            od_pair_indices=od_pair_indices, num_link_groups=num_link_groups, link_group=link_group, route_validity_mask=route_validity_mask,
            init_alpha=vdf_config.get('init_alpha', 0.15),
            init_beta=vdf_config.get('init_beta', 4.0),
            num_iterations=convergence_cfg.get('max_iters', 20)
        )

        loss_cfg = kwargs.get("loss", {})
        if isinstance(loss_cfg, DictConfig):
            loss_cfg = dict(loss_cfg)
        else:
            loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
        loss_cfg.pop("_target_", None)

        # Priority: loss YAML -> model kwargs -> safe fallback (1.0)
        link_scale_cfg = loss_cfg.pop("link_scale", None)
        od_scale_cfg = loss_cfg.pop("od_scale", None)

        if link_scale_cfg is None:
            link_scale_cfg = kwargs.get("link_scale", None)
            if link_scale_cfg is None:
                link_scale_cfg = 1.0
                logger.warning("link_scale was not provided in model loss config or kwargs. Falling back to 1.0.")

        if od_scale_cfg is None:
            od_scale_cfg = kwargs.get("od_scale", None)
            if od_scale_cfg is None:
                od_scale_cfg = 1.0
                logger.warning("od_scale was not provided in model loss config or kwargs. Falling back to 1.0.")

        self.link_scale = float(link_scale_cfg)
        self.od_scale = float(od_scale_cfg)

        self.loss_fn = Loss(
            link_scale=float(self.link_scale),
            od_scale=float(self.od_scale),
            **loss_cfg,
        )

    def forward(
            self,
            observed_flows: torch.Tensor,
            flow_mask: Optional[torch.Tensor] = None,
            true_od_demand: Optional[torch.Tensor] = None,
            od_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        if flow_mask is None: 
            flow_mask = torch.ones_like(observed_flows)
            
        x_in = (observed_flows / self.link_scale) * flow_mask

        # Proposition
        hx = self.f_encoder(x_in)
        gx = self.matcher(hx)           
        raw_od_hat = self.f_decoder(gx)    
        od_hat = raw_od_hat * self.od_scale

        # Physical Verification
        # link_flows, alpha, beta, conv_info, route_flows = self.traffic_assignment(od_hat, kwargs.get('link_features', {}))
        link_flows, route_flows, route_costs = self.traffic_assignment(od_hat)



        output = {
            "estimated_demand": od_hat,
            #"route_flows": route_flows,
            "reconstructed_flows": link_flows,
            #"convergence_info": conv_info,
            #"learned_alpha": alpha,
            #"learned_beta": beta,
            "h_od": hx,
            "g_od": gx,
            "route_flows": route_flows,
            "route_costs": route_costs
        }

        if true_od_demand is not None:
            output["loss"] = self.loss_fn(
                predicted_flows=link_flows,
                true_flows=observed_flows,
                flow_mask=flow_mask,
                predicted_od=od_hat,
                true_od=true_od_demand,
                od_mask=od_mask,
            )

        return output

    def get_evaluation_artifacts(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(outputs, dict), "outputs must be a dict"
        assert "reconstructed_flows" in outputs, "Missing key 'reconstructed_flows'"
        assert "estimated_demand" in outputs, "Missing key 'estimated_demand'"
        return {
            "pred_flows": outputs["reconstructed_flows"].detach().cpu(),
            "pred_od": outputs["estimated_demand"].detach().cpu(),
            "route_flows": outputs.get("route_flows"),
            "route_costs": outputs.get("route_costs"),
            "h_od": outputs.get("h_od"),
            "g_od": outputs.get("g_od"),
        }

# =============================================================================
# 3. FUNCIÓN DE PÉRDIDA (LOSS - Adaptador Universal)
# =============================================================================

class Loss(nn.Module):
    def __init__(self, link_scale=1.0, od_scale=1.0, w_flow=1.0, w_od=1.0, **kwargs):
        super().__init__()
        self.register_buffer("link_scale", torch.tensor(float(link_scale)))
        self.register_buffer("od_scale", torch.tensor(float(od_scale)))
        self.w_flow = w_flow
        self.w_od = w_od
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, **kwargs) -> dict:
        """
        Universal Adapter for Loss Computation.
        Extracts specific tensors passed as kwargs by trainer.py
        """
        device = self.link_scale.device
        
        # 1. Physical Verification Loss (Flow)
        pred_flow = kwargs.get('predicted_flows')
        true_flow = kwargs.get('true_flows')
        flow_mask = kwargs.get('flow_mask')
        
        loss_flow = torch.tensor(0.0, device=device)
        
        if pred_flow is not None and true_flow is not None and flow_mask is not None:
            scaled_pred_f = pred_flow / self.link_scale
            scaled_true_f = true_flow / self.link_scale
            
            mse_flow = self.mse(scaled_pred_f, scaled_true_f) * flow_mask
            loss_flow = mse_flow.sum() / (flow_mask.sum() + 1e-6)

        # 2. Statistical Proposition Loss (OD Demand)
        pred_od = kwargs.get('predicted_od')
        true_od = kwargs.get('true_od')
        od_mask = kwargs.get('od_mask')
        
        loss_od = torch.tensor(0.0, device=device)

        if pred_od is not None and true_od is not None and od_mask is not None and od_mask.sum() > 0:
            scaled_pred_od = pred_od / self.od_scale
            scaled_true_od = true_od / self.od_scale
            
            mse_od = self.mse(scaled_pred_od, scaled_true_od) * od_mask
            loss_od = mse_od.sum() / (od_mask.sum() + 1e-6)

        # 3. Total Balanced Loss
        total_loss = (self.w_flow * loss_flow) + (self.w_od * loss_od)

        return {
            "total_loss": total_loss,
            "l_flow": loss_flow,
            "l_od": loss_od
        }

# =============================================================================
# 4. DIAGNOSTICADOR (TrainingDiagnostician - Full Plotting)
# =============================================================================

class TrainingDiagnostician:
    def __init__(self, history_window=100):
        self.window = history_window
        self.full_history = {
            'r2_flow': [], 'mae_flow': [], 'grad_norms': {}
        }
        self.window_history = {
            'r2_flow': [], 'mae_flow': []
        }

    def update(self, outputs: Dict, targets: Dict, model=None, is_final=False, **kwargs):
        """Update metrics while ignoring extra trainer parameters."""
        with torch.no_grad():
            pred_flow = outputs.get('reconstructed_flows')
            true_flow = targets.get('flows')
            mask_flow = targets.get('flow_mask')

            if pred_flow is None or true_flow is None: return
            if mask_flow is None: mask_flow = torch.ones_like(true_flow)
            
            mask_bool = mask_flow > 0
            
            if pred_flow.dim() == 2 and pred_flow.shape[0] == 1: pred_flow = pred_flow.squeeze(0)
            if true_flow.dim() == 2 and true_flow.shape[0] == 1: true_flow = true_flow.squeeze(0)
            if mask_bool.dim() == 2 and mask_bool.shape[0] == 1: mask_bool = mask_bool.squeeze(0)

            if mask_bool.sum() > 0:
                y_pred = pred_flow[mask_bool].cpu().numpy()
                y_true = true_flow[mask_bool].cpu().numpy()
                
                mae = np.mean(np.abs(y_true - y_pred))
                
                if len(y_true) > 1 and np.var(y_true) > 0:
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                else:
                    r2 = 0.0
                
                self.full_history['r2_flow'].append(r2)
                self.full_history['mae_flow'].append(mae)
                self._push_window('r2_flow', r2)
                self._push_window('mae_flow', mae)
        
        if is_final:
            self.audit_equilibrium(outputs)

    def check_gradients(self, model: nn.Module) -> str:
        max_grad = 0.0
        params_checked = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.data.norm(2).item())
                params_checked += 1
        if params_checked == 0: return "No Grads"
        
        status = []
        if max_grad > 1000: status.append("🔥 High Grads")
        if max_grad < 1e-6: status.append("⚠️ Low Grads")
        if not status: return f"Grads OK ({max_grad:.2e})"
        return " | ".join(status)

    def capture_gradient_history(self, model: nn.Module, epoch: int):
        if 'grad_norms' not in self.full_history:
            self.full_history['grad_norms'] = {}

        key_modules = {
            "Fwd Enc": getattr(model, 'f_encoder', None),
            "Fwd Dec": getattr(model, 'f_decoder', None),
            "Matcher": getattr(model, 'matcher', None),
            "Bwd Enc": getattr(model, 'b_encoder', None),
            "Bwd Dec": getattr(model, 'b_decoder', None)
        }

        for name, module in key_modules.items():
            if module is None: continue
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            if name not in self.full_history['grad_norms']:
                self.full_history['grad_norms'][name] = []
            self.full_history['grad_norms'][name].append(total_norm)

    def get_report(self) -> str:
        if not self.window_history['r2_flow']: return "Init..."
        avg_r2 = np.mean(self.window_history['r2_flow'])
        avg_mae = np.mean(self.window_history['mae_flow'])
        return f"R2: {avg_r2:.3f} | MAE: {avg_mae:.1f}"

    def _push_window(self, key, value):
        self.window_history[key].append(value)
        if len(self.window_history[key]) > self.window:
            self.window_history[key].pop(0)

    # --- PLOTTING METHODS ---

    def finalize_and_plot(self, filename_prefix="final_report"):
        plt.switch_backend('Agg') 
        self.plot_evolution(filename_prefix.replace(".png", "_evolution.png"))
        self.plot_physics(filename_prefix.replace(".png", "_physics.png"))
        self.plot_gradient_health(filename_prefix.replace(".png", "_gradients.png"))

    def plot_evolution(self, filename):
        r2 = self.full_history['r2_flow']
        mae = self.full_history['mae_flow']
        if not r2: return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('R2 Score', color='tab:blue')
        ax1.plot(r2, color='tab:blue', label='R2 Flow', alpha=0.7)
        ax1.set_ylim(-1, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel('MAE Flow', color='tab:orange')
        ax2.plot(mae, color='tab:orange', label='MAE Flow', alpha=0.7)

        plt.title('Training Evolution')
        plt.savefig(filename)
        plt.close()

    def plot_physics(self, filename):
        data = self.full_history['mae_flow']
        if not data: return
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='MAE Flow', color='gray', alpha=0.5)
        plt.title('Physics Consistency (Reconstruction Error)')
        plt.savefig(filename)
        plt.close()

    def plot_gradient_health(self, filename):
        grads = self.full_history.get('grad_norms', {})
        if not grads: return
        plt.figure(figsize=(12, 6))
        for name, values in grads.items():
            if values: plt.plot(values, label=name)
        plt.yscale('log')
        plt.legend()
        plt.title('Gradient Health')
        plt.savefig(filename)
        plt.close()

    def audit_equilibrium(self, outputs: Dict, filename_prefix="audit"):
            print("\n" + "="*60)
            print("🚀 RUNNING FINAL EPOCH EQUILIBRIUM AUDIT")
            print("="*60)
            
            est_demand = outputs.get("estimated_demand").detach().cpu().numpy()
            r_flows = outputs.get("route_flows").detach().cpu().numpy()
            r_costs = outputs.get("route_costs").detach().cpu().numpy()
            
            if est_demand is None or r_flows is None:
                return
                
            batch_size, num_ods = est_demand.shape
            K = r_flows.shape[1] // num_ods
            
            # Work assuming batch=1 for diagnostics
            demands = est_demand[0]
            flows_3d = r_flows[0].reshape(num_ods, K)
            costs_3d = r_costs[0].reshape(num_ods, K)
            
            # ---------------------------------------------------------
            # 1. DEMAND CONSERVATION AUDIT
            # ---------------------------------------------------------
            sum_flows = np.sum(flows_3d, axis=1)
            demand_diff = np.abs(sum_flows - demands)
            failed_demand_mask = demand_diff > 1.0 
            
            if np.any(failed_demand_mask):
                print(f"⚠️ {np.sum(failed_demand_mask)} OD pairs failed demand conservation:")
                for od_idx in np.where(failed_demand_mask)[0]:
                    print(f"  - OD {od_idx}: Estimated={demands[od_idx]:.2f}, Sum={sum_flows[od_idx]:.2f}")
            else:
                print("✅ All OD pairs strictly conserved estimated demand.")
                
            plt.figure(figsize=(8, 8))
            plt.scatter(demands, sum_flows, alpha=0.5, color='blue', edgecolor='k')
            max_val = max(np.max(demands), np.max(sum_flows))
            plt.plot([0, max_val], [0, max_val], 'r--', label='y = x (Perfect Conservation)')
            plt.xlabel("Estimated Demand (MLP)")
            plt.ylabel("Sum of Assigned Route Flows")
            plt.title("Demand Conservation Check")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(f"{filename_prefix}_demand_scatter.png")
            plt.close()

            # ---------------------------------------------------------
            # 2. TRAVEL TIME AUDIT (Wardrop 5% Tolerance)
            # ---------------------------------------------------------
            print("\nAuditing Travel Times (5% tolerance on USED routes)...")
            failed_cost_ods = 0
            plot_data = [] 
            
            for od_idx in range(num_ods):
                od_flows = flows_3d[od_idx]
                od_costs = costs_3d[od_idx]
                od_demand = demands[od_idx]
                
                # Dynamic threshold.
                # A route is considered "used" only if it carries more than 0.5 units OR more than 1% of the OD demand.
                # This removes numerical noise from the ML model.
                dynamic_threshold = max(0.5, od_demand * 0.01) 
                
                used_mask = od_flows > dynamic_threshold
                
                # If for some reason no route exceeds the threshold (very rare), take the highest-flow route
                if not np.any(used_mask):
                    best_route_idx = np.argmax(od_flows)
                    used_mask[best_route_idx] = True
                    
                used_costs = od_costs[used_mask]
                
                # Only audit if more than one route is used (if 1, it's trivially in equilibrium)
                if np.sum(used_mask) > 1:
                    mean_cost = np.mean(used_costs)
                    lower_bound = mean_cost * 0.95
                    upper_bound = mean_cost * 1.05
                    
                    violators = (used_costs < lower_bound) | (used_costs > upper_bound)
                    
                    if np.any(violators):
                        failed_cost_ods += 1
                        print(f"  - OD {od_idx} Violations. Mean Cost: {mean_cost:.2f}. Used Route Costs: {np.round(used_costs, 2)}")
                    
                    plot_data.append((od_idx, used_costs))
                    
            if failed_cost_ods == 0:
                print("✅ All OD pairs satisfy Wardrop's User Equilibrium (within 5% tolerance on effectively used routes).")
                
            if plot_data:
                # Limit to 50 OD pairs so the plot remains readable
                plot_data = plot_data[:50]
                plt.figure(figsize=(14, 6))
                
                for i, (od_idx, costs) in enumerate(plot_data):
                    x_vals = np.full_like(costs, i)
                    plt.scatter(x_vals, costs, alpha=0.7, color='purple', edgecolor='w')
                    
                    # Baseline line connecting the mean
                    plt.plot([i-0.2, i+0.2], [np.mean(costs), np.mean(costs)], color='black', alpha=0.5)
                
                plt.xticks(range(len(plot_data)), [str(d[0]) for d in plot_data], rotation=90)
                plt.xlabel("OD Pair Index (Sample of 50 multi-route pairs)")
                plt.ylabel("Travel Time (Used Routes Only)")
                plt.title("Travel Time Dispersion per OD Pair (Points should tightly cluster vertically)")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{filename_prefix}_cost_dispersion.png")
                plt.close()
                
            print("="*60 + "\n")