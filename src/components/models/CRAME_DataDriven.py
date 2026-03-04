import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from omegaconf import DictConfig
from typing import Dict, Optional
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
        return self.network(x)


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


class RouteAttention(nn.Module):
    """
    Distributes OD demand across available routes using masked attention.
    Theoretical Lens: Maps OD -> Routes using a learnable scoring function.
    Pragmatic Lens: Operates on a padded [OD, max_routes] tensor to avoid 
    massive dense [R, OD] matrices and leverage GPU parallelization.
    """
    def __init__(self, od_feature_dim: int, num_ods: int, max_routes_per_od: int = 10):
        super().__init__()
        self.num_ods = num_ods
        self.max_routes = max_routes_per_od
        
        # Neural network to score routes based on OD latent features
        self.route_scorer = nn.Sequential(
            nn.Linear(od_feature_dim, od_feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(od_feature_dim // 2, num_ods * max_routes_per_od)
        )

        # TODO: testear
        # Inicialización pragmática para arrancar con distribuciones uniformes
        self._initialize_scorer_weights()

    def _initialize_scorer_weights(self):
        """
        Reduce the variance of the final layer to output near-zero logits.
        This forces the Softmax to start with a near-uniform distribution,
        preventing erratic initial gradients and arbitrary route collapse.
        """
        final_layer = self.route_scorer[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
        nn.init.zeros_(final_layer.bias)
        
    def forward(self, h_od: torch.Tensor, od_demand: torch.Tensor, 
                route_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_od: Latent features of OD pairs. Shape: [B, num_ods, feature_dim]
            od_demand: Estimated physical OD demand. Shape: [B, num_ods]
            route_mask: Boolean mask indicating valid routes (False for padding). 
                        Shape: [num_ods, max_routes]
        Returns:
            route_flows: Physical flows assigned to each route. 
                         Shape: [B, num_ods * max_routes]
        """
        # Handle batch dimension safely
        B = h_od.size(0) if h_od.dim() > 1 else 1
        if h_od.dim() == 1:
            h_od = h_od.unsqueeze(0)
            
        # 1. Generate raw affinity scores (Global Context -> All Route Slots)
        raw_scores_flat = self.route_scorer(h_od) # [B, num_ods * max_routes]
        
        # 2. Reshape to separate ODs and Routes to match the route_mask
        raw_scores = raw_scores_flat.view(B, self.num_ods, self.max_routes) # [B, num_ods, max_routes]
        
        # 3. Topological Masking
        # expanded_mask shape: [1, num_ods, max_routes] (expands to B automatically during fill)
        expanded_mask = route_mask.unsqueeze(0).expand_as(raw_scores)
        masked_scores = raw_scores.masked_fill(~expanded_mask, float('-inf'))
        
        # 4. Behavioral Attention
        attention_weights = F.softmax(masked_scores, dim=-1) # [B, num_ods, max_routes]
        
        # 5. Flow Allocation
        # od_demand is [B, num_ods]. Add a dimension to broadcast against max_routes
        od_demand_expanded = od_demand.unsqueeze(-1) # [B, num_ods, 1]
        route_flows_grouped = attention_weights * od_demand_expanded # [B, num_ods, max_routes]
        
        # Flatten back to the 1D route structure for the Delta matrix multiplication
        route_flows = route_flows_grouped.view(B, -1) # [B, num_ods * max_routes]
        
        return route_flows
    
# =============================================================================
# 2. MODELO PRINCIPAL (CyclicCRAME - Data Driven)
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
            route_validity_mask: torch.Tensor,
            max_routes_per_od: int = 10,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__()
        logger.info("INIT: CyclicCRAME (Physics-Informed Route Attention)")

        self.register_buffer("delta_matrix", delta_matrix)
        self.register_buffer("route_mask", route_validity_mask)

        self.link_scale = kwargs.get('link_scale', 1.0)
        self.od_scale = kwargs.get('od_scale', 1.0)

        # --- A. FORWARD NETWORK (The Statistical Proposition) ---
        # Note: We skip the complex matcher here. It goes straight from Links to OD.
        self.f_encoder = ODEncoder(num_links, h_enc, feature_dim, dropout)
        self.f_decoder = ODDecoder(num_od_pairs, h_dec, feature_dim, dropout)

        # --- B. BACKWARD NETWORK (The Physical Verification) ---
        # Translates OD latent features to Route distribution probabilities
        self.route_attention = RouteAttention(feature_dim, num_od_pairs, max_routes_per_od)

    def forward(
            self,
            observed_flows: torch.Tensor,
            flow_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            observed_flows: Sensor readings [B, A]
            delta_matrix: Link-Route incidence matrix [A, R] (Ideally a Sparse Tensor)
            route_mask: Valid routes mask [OD, max_routes]
        """
        if flow_mask is None: 
            flow_mask = torch.ones_like(observed_flows)
            
        x_in = (observed_flows / self.link_scale) * flow_mask

        # 1. PROPOSITION: Infer OD Demand from Links
        h_od = self.f_encoder(x_in)           # [B, OD, F]
        raw_od_hat = self.f_decoder(h_od)     # [B, OD]
        od_hat = raw_od_hat * self.od_scale

        # 2. BEHAVIORAL ATTENTION: Distribute OD Demand to Routes
        # Returns flow for all R routes (R = num_ods * max_routes)
        f_r = self.route_attention(h_od, od_hat, self.route_mask) # [B, R]

        # 3. PHYSICAL VERIFICATION: Aggregate Route flows to Link flows
        # x_a = Delta * f_r
        # Matrix multiplication handles the aggregation automatically
        # reconstructed_flows = torch.matmul(f_r, delta_matrix.t()) # [B, A]
        # TODO: Verificar esta formulación. Si no, eliminarla

        # delta_matrix: [A, R] (Sparse)
        # f_r: [B, R] -> f_r.t(): [R, B] (Dense)
        # Resultado de sparse.mm: [A, B] -> Transpuesto final: [B, A]
        reconstructed_flows = torch.sparse.mm(self.delta_matrix, f_r.t()).t()

        return {
            "estimated_demand": od_hat,
            "route_flows": f_r,
            "reconstructed_flows": reconstructed_flows,
            "h_od": h_od
        }

# =============================================================================
# 3. FUNCIÓN DE PÉRDIDA (LOSS - Adaptador Universal)
# =============================================================================

class Loss(nn.Module):
    """
    Physics-informed Loss Function for Link-Route-OD topologies.
    Computes standard MSE for demand/flows and an Entropy regularizer 
    to prevent route-assignment collapse (equifinality).
    """
    def __init__(self, link_scale=1.0, od_scale=1.0, w_flow=1.0, w_od=1.0, w_entropy=0.01, **kwargs):
        super().__init__()
        self.register_buffer("link_scale", torch.tensor(float(link_scale)))
        self.register_buffer("od_scale", torch.tensor(float(od_scale)))
        
        self.w_flow = w_flow
        self.w_od = w_od
        self.w_entropy = w_entropy
        
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, outputs: dict, targets: dict, masks: dict) -> dict:
        """
        Args:
            outputs: Dict containing 'reconstructed_flows', 'estimated_demand', 'route_flows'.
            targets: Dict containing 'flows', 'od' (optional).
            masks: Dict containing 'flow_mask', 'od_mask', 'route_mask'.
        """
        device = self.link_scale.device
        
        # 1. Flow Loss (The Physical Verification: L_fisica)
        pred_flow = outputs.get('reconstructed_flows')
        true_flow = targets.get('flows')
        flow_mask = masks.get('flow_mask')
        
        scaled_pred_f = pred_flow / self.link_scale
        scaled_true_f = true_flow / self.link_scale
        
        mse_flow = self.mse(scaled_pred_f, scaled_true_f) * flow_mask
        loss_flow = mse_flow.sum() / (flow_mask.sum() + 1e-6)

        # 2. OD Demand Loss (The Statistical Proposition)
        loss_od = torch.tensor(0.0, device=device)
        pred_od = outputs.get('estimated_demand')
        true_od = targets.get('od')
        od_mask = masks.get('od_mask')

        if true_od is not None and od_mask is not None and od_mask.sum() > 0:
            scaled_pred_od = pred_od / self.od_scale
            scaled_true_od = true_od / self.od_scale
            
            mse_od = self.mse(scaled_pred_od, scaled_true_od) * od_mask
            loss_od = mse_od.sum() / (od_mask.sum() + 1e-6)

        # 3. Pragmatic Regularization: Route Entropy
        # Prevents the model from dumping 100% of demand into a single route arbitrarily.
        loss_entropy = torch.tensor(0.0, device=device)
        
        if self.w_entropy > 0:
            route_flows = outputs.get('route_flows') 
            route_mask = masks.get('route_mask')     
            
            B = route_flows.size(0)
            num_ods = route_mask.size(0)
            max_routes = route_mask.size(1)
            
            flows_grouped = route_flows.view(B, num_ods, max_routes)
            expanded_mask = route_mask.unsqueeze(0).expand_as(flows_grouped)
            
            # Probabilidades predichas (P)
            od_totals = flows_grouped.sum(dim=-1, keepdim=True) + 1e-8
            probs = flows_grouped / od_totals
            
            # Distribución Uniforme Ideal (U) sobre las rutas válidas
            valid_route_counts = expanded_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            uniform_probs = 1.0 / valid_route_counts
            
            # Enmascarar ceros para evitar log(0)
            valid_probs = torch.clamp(torch.where(expanded_mask, probs, torch.tensor(1.0, device=device)), min=1e-10)
            valid_uniforms = torch.where(expanded_mask, uniform_probs, torch.tensor(1.0, device=device))
            
            # KL(P || U)
            kl_div = valid_probs * torch.log(valid_probs / valid_uniforms)
            
            # Sumar solo los elementos válidos
            loss_entropy = (kl_div * expanded_mask).sum() / (expanded_mask.sum() * B + 1e-6)

        # Total Weighted Loss (Todo se SUMA, buscando el cero)
        total_loss = (self.w_flow * loss_flow) + (self.w_od * loss_od) + (self.w_entropy * loss_entropy)

        return {
            "total_loss": total_loss,
            "l_flow": loss_flow,
            "l_od": loss_od,
            "l_entropy": loss_entropy
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