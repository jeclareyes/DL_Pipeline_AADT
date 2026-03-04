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
# 1. COMPONENTES DE RED NEURONAL (Encoder/Decoder/Matcher)
# =============================================================================

class ODEncoder(nn.Module):
    """
    Codifica vectores de entrada (Aforos u ODs) en características latentes.
    Arquitectura Profunda: [Input -> H -> H/2 -> Feature]
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
    Decodifica el vector latente a salida física (ODs o Aforos).
    Arquitectura Profunda: [Feature -> H -> H/2 -> Output]
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
            nn.Softplus() # Asegura valores positivos (física)
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        return self.network(g_x)

class GraphMatcher(nn.Module):
    """
    Graph Matcher Mejorado (Sustituye a ImprovedGraphMatcher).
    - Incluye método 'forward' (requerido por PyTorch).
    - Desacopla la actualización de matrices (update) del forward pass.
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

        # Matrices M y V con inicialización suave
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
        """Actualización explícita de matrices M y V."""
        h_x = self._ensure_batch_dim(h_x)
        h_y = self._ensure_batch_dim(h_y)
        
        # Normalización
        h_x_norm = F.normalize(h_x, p=2, dim=1)
        h_y_norm = F.normalize(h_y, p=2, dim=1)

        # 1. Actualizar M (Proyección Estructural)
        similarity_vector = (h_x_norm * h_y_norm).mean(dim=0)
        target_M = similarity_vector.unsqueeze(1).expand(-1, self.num_structures)
        
        # Regularización
        noise = torch.randn_like(self.M) * 0.01
        regularized_target = target_M + self.reg_strength * noise

        # Momento adaptativo
        momentum = min(self.lambda_m * (1 + self.update_count.item() * 0.001), 0.1)
        self.M.data = (1 - momentum) * self.M.data + momentum * regularized_target

        # 2. Actualizar V (Atención Global)
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
        """Aplica la transformación aprendida."""
        h_x = self._ensure_batch_dim(h_x)
        
        # Atención neuronal
        attn_weights = self.attention_net(h_x) # [B, S]
        
        # Proyección
        h_exp = h_x.unsqueeze(2) # [B, F, 1]
        h_struct = h_exp * self.M.unsqueeze(0) # [B, F, S]
        h_weighted = h_struct * self.V.unsqueeze(0) # [B, F, S]
        
        # Combinación ponderada
        g_x = (h_weighted * attn_weights.unsqueeze(1)).sum(dim=2) # [B, F]
        
        return g_x


class ProjectedAssignmentValidator(nn.Module):
    """
    Replaces the stochastic MSA loop with Projected Gradient Descent (PGD) 
    over the simplex of route flows, achieving physical User Equilibrium.
    """
    def __init__(self, num_links: int, t0: torch.Tensor, capacity: torch.Tensor,
                 lanes: torch.Tensor, route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: DictConfig, trips_scaler: float = 1.0,
                 max_iters: int = 50, grad_steps: int = 5, step_size: float = 0.01,
                 convergence_threshold: float = 1e-4, **kwargs):
        super().__init__()
        
        self.max_iters = max_iters
        self.grad_steps = grad_steps
        self.step_size = step_size
        self.tol = convergence_threshold
        self.trips_scaler = trips_scaler
        
        # Instantiate VDF
        self.cost_function = hydra.utils.instantiate(
            vdf_config, t0=t0, capacity=capacity, lanes=lanes,
            num_link_groups=num_link_groups, link_group=link_group, _recursive_=False
        )
        
        # Process Route Masks to create the Sparse Delta Matrix and Validity Mask
        self._initialize_topology(route_masks)
        
        # Warm Start Buffer
        self.register_buffer('running_route_flows', None)

    def _initialize_topology(self, route_masks: torch.Tensor):
        # route_masks is expected as [Num_OD, K_Paths, Num_Links]
        self.num_od, self.k_paths, self.num_links = route_masks.shape
        
        # 1. Create Route Validity Mask [Num_OD, K_Paths]
        # A route is valid if it has at least one link
        dense_masks = route_masks.to_dense() if route_masks.is_sparse else route_masks
        self.register_buffer('route_validity_mask', (dense_masks.sum(dim=-1) > 0).bool())
        
        # 2. Create 2D Sparse Delta Matrix [Num_OD * K_Paths, Num_Links]
        if route_masks.is_sparse:
            route_masks = route_masks.coalesce()
            indices = route_masks.indices()
            values = route_masks.values()
            new_rows = indices[0] * self.k_paths + indices[1]
            new_cols = indices[2]
            new_indices = torch.stack([new_rows, new_cols])
            sparse_2d = torch.sparse_coo_tensor(new_indices, values, size=(self.num_od * self.k_paths, self.num_links))
        else:
            sparse_2d = route_masks.reshape(-1, self.num_links).to_sparse()
            
        self.register_buffer('sparse_delta_2d', sparse_2d)

    def _simplex_projection(self, v: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Fast, vectorized projection onto the probability simplex {x | sum(x) = z, x >= 0}.
        v: [B, OD, K] - unprojected flows
        z: [B, OD] - target OD demand
        mask: [B, OD, K] - boolean mask of valid routes
        """
        v_masked = v.masked_fill(~mask, float('-inf'))
        u, _ = torch.sort(v_masked, dim=-1, descending=True)
        
        cssv = torch.cumsum(torch.where(mask, u, torch.zeros_like(u)), dim=-1)
        idx = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
        
        cond = (u - (cssv - z.unsqueeze(-1)) / idx > 0) & mask
        rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        
        theta = (torch.gather(cssv, 2, (rho - 1).clamp(min=0).long()) - z.unsqueeze(-1)) / rho.float()
        w = torch.clamp(v - theta, min=0.0)
        
        return w * mask

    def forward(self, estimated_demands: torch.Tensor, warmup: bool = False, override_max_iters: int = None):
        max_iters = override_max_iters if override_max_iters is not None else self.max_iters
        real_demands = estimated_demands * self.trips_scaler
        
        B = real_demands.shape[0]
        expanded_mask = self.route_validity_mask.unsqueeze(0).expand(B, -1, -1)
        
        # --- INITIALIZATION (Warm Start vs Cold Start) ---
        can_warm_start = (self.training and not warmup and 
                          self.running_route_flows is not None and 
                          self.running_route_flows.shape[0] == B)
        
        if can_warm_start:
            route_flows = self.running_route_flows.detach().clone()
        else:
            # Uniform initial distribution across valid routes
            valid_counts = expanded_mask.sum(dim=2, keepdim=True).clamp(min=1)
            route_flows = (real_demands.unsqueeze(-1) / valid_counts) * expanded_mask

        # --- DYNAMIC PROJECTED GRADIENT DESCENT LOOP ---
        grad_start_iter = max(0, max_iters - self.grad_steps)
        converged = False
        actual_iters = 0
        
        for it in range(1, max_iters + 1):
            # TRUNCATED BACKPROP: Only track gradients in the last steps to save memory
            requires_grad = self.training and (it > grad_start_iter)
            
            with torch.set_grad_enabled(requires_grad):
                # 1. Route Flows -> Link Flows (v = Delta * h)
                rf_flat = route_flows.view(B, -1)
                rf_t = torch.transpose(rf_flat, 0, 1) # [OD*K, B]
                link_flows_t = torch.sparse.mm(self.sparse_delta_2d.t(), rf_t) # [L, B]
                link_flows = link_flows_t.transpose(0, 1) # [B, L]
                
                # 2. Dynamic Costs
                link_costs = self.cost_function(link_flows)
                
                # 3. Link Costs -> Route Costs (c = Delta^T * t)
                costs_t = torch.transpose(link_costs, 0, 1) # [L, B]
                route_costs_flat_t = torch.sparse.mm(self.sparse_delta_2d, costs_t) # [OD*K, B]
                route_costs = route_costs_flat_t.transpose(0, 1).view(B, self.num_od, self.k_paths)
                
                # 4. Gradient Step
                unprojected_flows = route_flows - self.step_size * route_costs
                
                # 5. Projection
                new_route_flows = self._simplex_projection(unprojected_flows, real_demands, expanded_mask)
                
            # Convergence check (no grad needed)
            with torch.no_grad():
                shift = torch.max(torch.abs(new_route_flows - route_flows))
                if shift <= self.tol and it >= 5: # Force at least 5 iterations
                    converged = True
                    route_flows = new_route_flows
                    actual_iters = it
                    
                    # Ensure at least one grad step if it converged too early
                    if self.training and not requires_grad:
                        with torch.set_grad_enabled(True):
                            link_flows_t = torch.sparse.mm(self.sparse_delta_2d.t(), torch.transpose(route_flows.view(B, -1), 0, 1))
                            link_costs = self.cost_function(link_flows_t.transpose(0, 1))
                            route_costs_flat_t = torch.sparse.mm(self.sparse_delta_2d, torch.transpose(link_costs, 0, 1))
                            route_costs = route_costs_flat_t.transpose(0, 1).view(B, self.num_od, self.k_paths)
                            unprojected_flows = route_flows - self.step_size * route_costs
                            route_flows = self._simplex_projection(unprojected_flows, real_demands, expanded_mask)
                            # Re-compute link_flows for output
                            rf_flat = route_flows.view(B, -1)
                            link_flows = torch.sparse.mm(self.sparse_delta_2d.t(), torch.transpose(rf_flat, 0, 1)).transpose(0, 1)
                    break
                    
            route_flows = new_route_flows
            actual_iters = it
            
        if self.training:
            self.running_route_flows = route_flows.detach()

        conv_info = {"converged": converged, "iterations": actual_iters}
        learned_alpha = self.cost_function.alpha
        learned_beta = self.cost_function.beta
        
        return link_flows, learned_alpha, learned_beta, conv_info, route_flows

# =============================================================================
# 2. MODELO PRINCIPAL (CRAME - Differential Layer)
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
        
        self.validator = ProjectedAssignmentValidator(
            num_links=num_links,
            num_od_pairs=num_od_pairs,
            t0=t0,
            capacity=capacity,
            lanes=lanes,
            route_masks=route_masks,
            od_pair_indices=od_pair_indices,
            num_link_groups=num_link_groups,
            link_group=link_group,
            vdf_config=vdf_config,
            max_iters=convergence_cfg.get('max_iters', 50),
            grad_steps=convergence_cfg.get('grad_steps', 5),
            convergence_threshold=convergence_cfg.get('flow_tol', 1e-4)
        )

    def forward(
            self,
            observed_flows: torch.Tensor,
            flow_mask: Optional[torch.Tensor] = None,
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
        link_flows, alpha, beta, conv_info, route_flows = self.validator(
            estimated_demands=od_hat,
            warmup=kwargs.get('warmup', False),
            override_max_iters=kwargs.get('current_iter_count', None)
        )

        return {
            "estimated_demand": od_hat,
            "route_flows": route_flows,
            "reconstructed_flows": link_flows,
            "convergence_info": conv_info,
            "learned_alpha": alpha,
            "learned_beta": beta,
            "h_od": hx,
            "g_od": gx
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

    def update(self, outputs: Dict, targets: Dict, model=None, **kwargs):
        """Actualiza métricas ignorando parámetros extra del trainer."""
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