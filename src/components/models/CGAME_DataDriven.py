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

# =============================================================================
# 2. MODELO PRINCIPAL (CyclicODModel - Data Driven)
# =============================================================================

class CyclicODModel(nn.Module):
    def __init__(
            self,
            num_links: int,
            num_od_pairs: int,
            architecture: DictConfig,
            **kwargs
    ):
        super().__init__()
        logger.info(f"INIT: CyclicODModel Data-Driven (Deep Encoders + Improved Matcher).")

        arch = architecture
        feature_dim = arch.feature_dim
        num_structures = arch.num_structures
        
        # Dimensiones Ocultas
        h_enc = arch.hidden_dim_from_link
        h_dec = arch.hidden_dim_to_od
        h_bwd_enc = kwargs.get('hidden_dim_from_od', h_dec)
        h_bwd_dec = kwargs.get('hidden_dim_to_link', h_enc)

        dropout = arch.get('dropout', 0.1)

        # Escaladores
        self.register_buffer("link_scale", torch.tensor(kwargs.get('link_scale', 1.0), dtype=torch.float32))
        self.register_buffer("od_scale", torch.tensor(kwargs.get('od_scale', 1.0), dtype=torch.float32))

        # --- A. FORWARD NETWORK (Estimación) ---
        self.f_encoder = ODEncoder(num_links, h_enc, feature_dim, dropout)
        self.f_decoder = ODDecoder(num_od_pairs, h_dec, feature_dim, dropout)

        # --- B. BACKWARD NETWORK (Asignación) ---
        self.b_encoder = ODEncoder(num_od_pairs, h_bwd_enc, feature_dim, dropout)
        self.b_decoder = ODDecoder(num_links, h_bwd_dec, feature_dim, dropout)

        # --- C. MATCHER ---
        self.matcher = GraphMatcher(feature_dim, num_structures)

    def forward(
            self,
            observed_flows: torch.Tensor,
            flow_mask: Optional[torch.Tensor] = None,
            true_od_demand: Optional[torch.Tensor] = None,
            od_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        # 1. Preparación
        if flow_mask is None: flow_mask = torch.ones_like(observed_flows)
        x_in = (observed_flows / self.link_scale) * flow_mask

        # 2. Forward Pass
        hx = self.f_encoder(x_in)
        gx = self.matcher(hx)      # Llamada a forward()
        raw_od_hat = self.f_decoder(gx)
        od_hat = raw_od_hat * self.od_scale

        # 3. Backward Pass (Teacher Forcing Mix)
        use_teacher_forcing = self.training and (true_od_demand is not None)
        if use_teacher_forcing:
            true_od_norm = true_od_demand / self.od_scale
            if od_mask is not None:
                od_input_mix = (true_od_norm * od_mask) + (raw_od_hat.detach() * (1.0 - od_mask))
            else:
                od_input_mix = true_od_norm
        else:
            od_input_mix = raw_od_hat

        hy = self.b_encoder(od_input_mix)
        gy = self.matcher(hy)      # Simetría cíclica
        raw_flow_hat = self.b_decoder(gy)
        flow_hat = raw_flow_hat * self.link_scale

        # 4. Update (Solo training)
        if self.training:
            self.matcher.update(hx.detach(), hy.detach())

        return {
            "estimated_demand": od_hat,
            "reconstructed_flows": flow_hat,
            "hx": hx, "gx": gx,
            "hy": hy, "gy": gy
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
        self.mse = nn.MSELoss()

    def forward(self, outputs=None, targets=None, **kwargs):
        """Adaptador Universal: Acepta diccionarios o kwargs desempacados."""
        
        # Extracción segura de kwargs (compatibilidad con trainer antiguo)
        pred_flow = kwargs.get('predicted_flows')
        if pred_flow is None and outputs: pred_flow = outputs.get('reconstructed_flows')
            
        true_flow = kwargs.get('true_flows')
        if true_flow is None and targets: true_flow = targets.get('flows')

        flow_mask = kwargs.get('flow_mask')
        if flow_mask is None and targets: flow_mask = targets.get('flow_mask')
        
        if pred_flow is None or true_flow is None:
            # Retorno seguro si faltan datos
            return {
                "total_loss": torch.tensor(0.0, requires_grad=True, device=self.link_scale.device),
                "l_flow": torch.tensor(0.0, device=self.link_scale.device),
                "l_od": torch.tensor(0.0, device=self.link_scale.device)
            }

        if flow_mask is None: flow_mask = torch.ones_like(true_flow)

        # Cálculo Flow Loss
        scaled_pred_f = pred_flow / self.link_scale
        scaled_true_f = true_flow / self.link_scale
        loss_flow = (self.mse(scaled_pred_f, scaled_true_f) * flow_mask).sum() / (flow_mask.sum() + 1e-6)

        # Cálculo OD Loss
        loss_od = torch.tensor(0.0, device=pred_flow.device)
        
        pred_od = kwargs.get('predicted_od')
        if pred_od is None:
            pred_od = kwargs.get('estimated_demand')
        if pred_od is None and outputs:
            pred_od = outputs.get('estimated_demand')
        
        true_od = kwargs.get('true_od')
        if true_od is None:
            true_od = kwargs.get('od')
        if true_od is None:
            true_od = kwargs.get('true_od_demand')
        if true_od is None and targets:
            true_od = targets.get('od')
        
        od_mask = kwargs.get('od_mask')
        if od_mask is None and targets: od_mask = targets.get('od_mask')

        if pred_od is not None and true_od is not None and od_mask is not None:
            if od_mask.sum() > 0:
                scaled_pred_od = pred_od / self.od_scale
                scaled_true_od = true_od / self.od_scale
                loss_od = (self.mse(scaled_pred_od, scaled_true_od) * od_mask).sum() / (od_mask.sum() + 1e-6)

        total_loss = (self.w_flow * loss_flow) + (self.w_od * loss_od)

        return {"total_loss": total_loss, "l_flow": loss_flow, "l_od": loss_od}

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