import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from typing import Optional, Dict, Any


# =============================================================================
# MEJORAS IMPLEMENTADAS:
# 1. Graph Matcher con regularización y flujo de gradientes consistente
# 2. Generación más robusta de h_y usando un encoder separado
# 3. Función de pérdida adaptativa y balanceada
# 4. Mecanismo de atención mejorado
# 5. Validación de convergencia en SUE
# =============================================================================

class ODEncoder(nn.Module):
    """Codifica los aforos en un vector de características latentes."""

    def __init__(self, num_links: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_links, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)  # Normalización final para estabilidad
        )

    def forward(self, counts_vector: torch.Tensor) -> torch.Tensor:
        return self.network(counts_vector)


class ODDecoder(nn.Module):
    """Decodifica el vector latente regularizado en la demanda OD final."""

    def __init__(self, num_od_pairs: int, hidden_dim: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_od_pairs),
            nn.Softplus()
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        return self.network(g_x)


class ImprovedGraphMatcher(nn.Module):
    """
    Graph Matcher mejorado con:
    - Regularización de matrices M y V
    - Mecanismo de atención más robusto
    - Flujo de gradientes consistente
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

        # Matrices M y V con inicialización mejorada
        self.register_buffer('M', torch.randn(feature_dim, num_structures) * 0.1)
        self.register_buffer('V', torch.ones(1, num_structures))

        # Red de atención aprendible para combinar estructuras
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, num_structures),
            nn.Softmax(dim=-1)
        )

        # Contador para estabilidad de actualizaciones
        self.register_buffer('update_count', torch.tensor(0.0))

    def _update_matrices_regularized(self, h_x: torch.Tensor, h_y: torch.Tensor):
        """Actualización de matrices M y V con regularización."""
        batch_size = h_x.size(0)

        # Normalización mejorada
        h_x_norm = F.normalize(h_x, p=2, dim=1)
        h_y_norm = F.normalize(h_y, p=2, dim=1)

        # Actualización adaptativa de M
        similarity_matrix = torch.bmm(h_x_norm.unsqueeze(2), h_y_norm.unsqueeze(1))
        similarity_vector = torch.mean(similarity_matrix.squeeze(), dim=0)

        # Regularización: mantener M cerca de la inicialización
        target_M = similarity_vector.unsqueeze(1).expand(-1, self.num_structures)
        regularized_M = target_M + self.reg_strength * torch.randn_like(target_M) * 0.01

        # Actualización con momento adaptativo
        # Asegurar que update_count se convierte a float para comparaciones
        uc = self.update_count.item() if isinstance(self.update_count, torch.Tensor) else float(self.update_count)
        momentum = min(self.lambda_m * (1 + uc * 0.001), 0.1)
        self.M.data = (1 - momentum) * self.M.data + momentum * regularized_M

        # Actualización de V con atención
        h_x_att_M = h_x.unsqueeze(2) * self.M
        h_x_att_M_norm = F.normalize(h_x_att_M, p=2, dim=1)
        h_y_expanded_norm = h_y_norm.unsqueeze(2)

        cosine_sim_v = torch.sum(h_x_att_M_norm * h_y_expanded_norm, dim=1)
        quality_per_structure = torch.mean(cosine_sim_v, dim=0, keepdim=True)

        # Regularización de V (mantener valores positivos y estables)
        quality_per_structure = torch.clamp(quality_per_structure, min=0.1, max=2.0)

        momentum_v = min(self.lambda_v * (1 + uc * 0.001), 0.1)
        self.V.data = (1 - momentum_v) * self.V.data + momentum_v * quality_per_structure

        # Incrementar contador (mutar el buffer)
        if isinstance(self.update_count, torch.Tensor):
            self.update_count.data = self.update_count.data + 1.0
        else:
            self.update_count = uc + 1.0

    def forward(self, h_x: torch.Tensor, h_y: torch.Tensor = None) -> torch.Tensor:
        # Actualizar matrices solo en entrenamiento y con referencia
        if self.training and h_y is not None:
            # Usar detach para evitar gradientes en la actualización de matrices
            with torch.no_grad():
                self._update_matrices_regularized(h_x.detach(), h_y.detach())

        # Aplicar transformación con atención aprendible
        attention_weights = self.attention_net(h_x)  # [batch, num_structures]

        # Aplicar matrices M y V con atención
        h_x_transformed = h_x.unsqueeze(2) * self.M  # [batch, feature_dim, num_structures]
        h_x_weighted = h_x_transformed * self.V  # Aplicar V

        # Combinar estructuras con atención
        g_x = torch.sum(h_x_weighted * attention_weights.unsqueeze(1), dim=2)

        return g_x


class ImprovedAssignmentValidator(nn.Module):
    """Validador mejorado con convergencia verificada y estabilización."""

    def __init__(self, num_links: int, t0: torch.Tensor, capacity: torch.Tensor,
                 route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: DictConfig,  # MODIFICACIÓN: Recibe config
                 max_iters: int = 10, convergence_threshold: float = 1e-4):
        super().__init__()
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.register_buffer('t0', t0)

        # Función de costos con inicialización mejorada
        self.cost_function = hydra.utils.instantiate(
            vdf_config,
            t0=t0,
            capacity=capacity,
            num_link_groups=num_link_groups,
            link_group=link_group,
            _recursive_=False
        )

        self.assignment_layer = StaticAssignmentLayer(route_masks, od_pair_indices, num_od_pairs)

        # Registro de convergencia
        self.register_buffer('last_convergence_iter', torch.tensor(0.0))

    def forward(self, estimated_demands: torch.Tensor, warmup: bool = False) -> tuple:
        batch_size = estimated_demands.shape[0]
        device = estimated_demands.device

        # Costos de flujo libre
        freeflow_costs_base = self.cost_function(torch.zeros_like(self.t0))
        freeflow_costs = freeflow_costs_base.unsqueeze(0).expand(batch_size, -1)

        if warmup:
            reconstructed_flows, route_probs = self.assignment_layer(freeflow_costs, estimated_demands)
            convergence_info = {"converged": True, "iterations": 1}
        else:
            flows, route_probs = self.assignment_layer(freeflow_costs, estimated_demands)
            prev_flows = flows.clone()

            converged = False
            actual_iters = 1

            for it in range(1, self.max_iters + 1):
                costs = self.cost_function(flows)
                new_flows, current_probs = self.assignment_layer(costs, estimated_demands)

                route_probs = current_probs

                # MSA con step size adaptativo
                alpha_msa = 1.0 / (it + 1)
                flows = flows + alpha_msa * (new_flows - flows)

                # Verificar convergencia
                if it > 2:  # No verificar en las primeras iteraciones
                    flow_change = torch.norm(flows - prev_flows, dim=1) / (torch.norm(flows, dim=1) + 1e-9)
                    max_change = torch.max(flow_change)

                    if max_change < self.convergence_threshold:
                        converged = True
                        actual_iters = it
                        break

                prev_flows = flows.clone()

            reconstructed_flows = flows
            convergence_info = {"converged": converged, "iterations": actual_iters}

            # Actualizar estadísticas de convergencia
            self.last_convergence_iter.data = torch.tensor(float(actual_iters))

        # MODIFICACIÓN: Uso de getattr seguro por si la VDF instanciada no tiene alpha/beta (ej. modelos no-BPR)
        learned_alpha = getattr(self.cost_function, 'get_alpha', lambda: None)()
        learned_beta = getattr(self.cost_function, 'get_beta', lambda: None)()

        # Fallback para compatibilidad visual si devuelve None
        if learned_alpha is None: learned_alpha = torch.tensor(0.15, device=self.t0.device)
        if learned_beta is None: learned_beta = torch.tensor(4.0, device=self.t0.device)

        return reconstructed_flows, learned_alpha, learned_beta, convergence_info, route_probs

"""
class LinkCostFunction(nn.Module):
    Función de coste BPR con restricciones mejoradas.

    def __init__(self, t0: torch.Tensor, capacity: torch.Tensor, num_link_groups: int, link_group: torch.Tensor):
        super().__init__()
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))

        # Inicialización más realista de parámetros BPR
        self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), 0.15))
        self.beta_raw = nn.Parameter(torch.full((num_link_groups,), 4.0))

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        # Restricciones más estrictas en los parámetros
        alpha = torch.clamp(F.softplus(self.alpha_raw), min=0.01, max=2.0)
        beta = torch.clamp(1.0 + F.softplus(self.beta_raw), min=1.1, max=10.0)

        alpha_links = alpha[self.link_group]
        beta_links = beta[self.link_group]

        # Evitar divisiones por cero y valores extremos
        flow_ratio = torch.clamp(link_flows / (self.capacity + 1e-9), max=5.0)
        bpr_cost = self.t0 * (1 + alpha_links * flow_ratio ** beta_links)

        return bpr_cost
"""


class StaticAssignmentLayer(nn.Module):
    """
    Capa de asignación optimizada para Tensores Esparsos (sin einsum).
    Reemplaza la lógica densa para evitar errores de memoria/runtime.
    """

    def __init__(self, route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, mu: float = 1.0):
        super().__init__()

        # Guardamos dimensiones originales
        self.num_od, self.k_paths, self.num_links = route_masks.shape
        self.register_buffer('od_pair_indices', od_pair_indices)
        self.num_od_pairs = num_od_pairs

        # --- LÓGICA COPIADA DE CYCLIC_MODEL (Flattening 3D -> 2D) ---
        # Convertimos la máscara [OD, K, Links] a [OD*K, Links] para usar sparse.mm
        if route_masks.is_sparse:
            route_masks = route_masks.coalesce()
            indices = route_masks.indices()
            values = route_masks.values()

            # Calcular nuevos índices de fila: row = od_idx * K + k_idx
            new_rows = indices[0] * self.k_paths + indices[1]
            new_cols = indices[2]

            new_indices = torch.stack([new_rows, new_cols])

            # Matriz esparsa 2D: [Rows=RutasTotales, Cols=Links]
            self.register_buffer(
                'sparse_mask_2d',
                torch.sparse_coo_tensor(
                    new_indices,
                    values,
                    size=(self.num_od * self.k_paths, self.num_links)
                )
            )
        else:
            # Fallback denso convertido a sparse
            self.register_buffer('sparse_mask_2d',
                                 route_masks.reshape(-1, self.num_links).to_sparse())

        # Parámetro mu aprendible
        self.mu_raw = nn.Parameter(torch.tensor(mu))

    @property
    def mu(self):
        return torch.clamp(F.softplus(self.mu_raw), min=0.1, max=10.0)

    def forward(self, link_costs: torch.Tensor, demands: torch.Tensor) -> tuple:
        """
        Calcula flujos usando multiplicación matricial esparsa (sparse.mm).
        """
        batch_size = link_costs.shape[0]

        # =====================================================================
        # PASO 1: Calcular Costo de Ruta (Link -> Ruta)
        # Reemplazo de: torch.einsum('bl,okl->bok', link_costs, self.route_masks)
        # Lógica: RouteCosts = (Mask @ LinkCosts.T).T
        # =====================================================================

        # LinkCosts.T -> [Links, Batch]
        costs_t = torch.transpose(link_costs, 0, 1)

        # Sparse MM: [OD*K, Links] @ [Links, Batch] -> [OD*K, Batch]
        route_costs_flat_t = torch.sparse.mm(self.sparse_mask_2d, costs_t)

        # Reshape: [Batch, OD, K]
        route_costs = route_costs_flat_t.transpose(0, 1).view(batch_size, self.num_od, self.k_paths)

        # =====================================================================
        # PASO 2: Logit Probabilities (Estabilización)
        # =====================================================================
        min_costs, _ = torch.min(route_costs, dim=2, keepdim=True)
        stable_costs = route_costs - min_costs.detach()
        stable_costs = torch.clamp(stable_costs, max=50.0)

        exp_utility = torch.exp(-self.mu * stable_costs)
        sum_utility = torch.sum(exp_utility, dim=2, keepdim=True)
        route_probs = exp_utility / (sum_utility + 1e-9)

        # Asignar Demanda: [Batch, OD, K]
        route_flows = route_probs * demands.unsqueeze(2)

        # =====================================================================
        # PASO 3: Proyectar a Links (Ruta -> Link)
        # Reemplazo de: torch.einsum('bok,okl->bl', route_flows, self.route_masks)
        # Lógica: LinkFlows = (Mask.T @ RouteFlowsFlat.T).T
        # =====================================================================

        # Aplanar flujos de ruta: [Batch, OD*K]
        route_flows_flat = route_flows.view(batch_size, -1)

        # Transponer para multiplicar: [OD*K, Batch]
        rf_t = torch.transpose(route_flows_flat, 0, 1)

        # Truco PyTorch: mask.t() en sparse es rápido (solo invierte índices)
        mask_t = self.sparse_mask_2d.t()  # [Links, OD*K]

        # Sparse MM: [Links, OD*K] @ [OD*K, Batch] -> [Links, Batch]
        link_flows_t = torch.sparse.mm(mask_t, rf_t)

        # Volver a formato batch: [Batch, Links]
        link_flows = link_flows_t.transpose(0, 1)

        # Retornamos TUPLA (Flujos, Probabilidades) para compatibilidad con pipeline
        return link_flows, route_probs


class AdaptiveCombinedLoss(nn.Module):
    """Función de pérdida adaptativa y balanceada."""

    def __init__(self, w_counts: float = 1.0, w_od: float = 1.0, w_reg: float = 0.01,
                 adaptive_weights: bool = True):
        super().__init__()
        self.register_buffer('w_counts', torch.tensor(w_counts))
        self.register_buffer('w_od', torch.tensor(w_od))
        self.w_reg = w_reg
        self.adaptive_weights = adaptive_weights
        self.mse_loss = nn.MSELoss()

        # Para tracking de pérdidas históricas
        self.register_buffer('loss_history_counts', torch.tensor(0.0))
        self.register_buffer('loss_history_od', torch.tensor(0.0))
        self.register_buffer('update_count', torch.tensor(0.0))

    def forward(self,
                predicted_flows: torch.Tensor,  # Pipeline envía esto
                true_flows: torch.Tensor,  # Pipeline envía esto
                flow_mask: torch.Tensor,  # Pipeline envía esto
                predicted_od: torch.Tensor,  # Pipeline envía esto
                true_od: torch.Tensor,  # Pipeline envía esto
                od_mask: torch.Tensor,  # Pipeline envía esto
                learned_alpha: Optional[torch.Tensor] = None,
                learned_beta: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        # Lógica Interna Adaptada

        # 1. Loss de Flujos
        if flow_mask.any():
            masked_pred = predicted_flows * flow_mask
            masked_true = true_flows * flow_mask
            l_counts = self.mse_loss(masked_pred, masked_true)
        else:
            l_counts = torch.tensor(0.0, device=predicted_flows.device)

        # 2. Loss de OD
        l_od = torch.tensor(0.0, device=predicted_flows.device)
        od_ratio = 0.0
        if od_mask is not None and od_mask.any():
            masked_pred_od = predicted_od * od_mask
            masked_true_od = true_od * od_mask
            l_od = self.mse_loss(masked_pred_od, masked_true_od)
            od_ratio = od_mask.float().mean().item()

        # 3. Regularización
        l_reg = torch.tensor(0.0, device=predicted_flows.device)
        if learned_alpha is not None and learned_beta is not None:
            l_reg = (torch.norm(learned_alpha - 0.15, p=2) +
                     torch.norm(learned_beta - 4.0, p=2))

        # 4. Adaptación de Pesos (Tu lógica original)
        if self.adaptive_weights and self.training:
            self._adapt_weights(l_counts.detach(), l_od.detach(), od_ratio)

        total_loss = (self.w_counts * l_counts +
                      self.w_od * l_od +
                      self.w_reg * l_reg)

        return {
            "total_loss": total_loss,
            "l_flow": l_counts,  # Renombrado para compatibilidad con logs del pipeline (antes l_counts)
            "l_od": l_od,
            "l_reg": l_reg,
            "w_flow": self.w_counts.item(),  # Renombrado para logs
            "w_od": self.w_od.item()
        }

    def _adapt_weights(self, l_counts: torch.Tensor, l_od: torch.Tensor, od_ratio: float):
        """Adaptación automática de pesos basada en el histórico de pérdidas."""
        alpha = 0.9  # Factor de suavizado

        # Actualizar promedios móviles
        #        if self.update_count == 0:
        #            self.loss_history_counts.data = l_counts
        #            self.loss_history_od.data = l_od if l_od > 0 else torch.tensor(1.0)
        #        else:
        #            self.loss_history_counts.data = alpha * self.loss_history_counts + (1 - alpha) * l_counts
        #            if l_od > 0:
        #                self.loss_history_od.data = alpha * self.loss_history_od + (1 - alpha) * l_od
        upd = self.update_count.item() if isinstance(self.update_count, torch.Tensor) else float(self.update_count)
        if upd == 0.0:
            self.loss_history_counts.data = l_counts
            self.loss_history_od.data = l_od if l_od > 0 else torch.tensor(1.0)
        else:
            self.loss_history_counts.data = alpha * self.loss_history_counts + (1 - alpha) * l_counts
            if l_od > 0:
                self.loss_history_od.data = alpha * self.loss_history_od + (1 - alpha) * l_od

        # Balancear pesos basado en magnitudes relativas
        if self.loss_history_od > 1e-6:
            ratio = self.loss_history_counts / self.loss_history_od
            # Ajustar w_od inversamente proporcional al ratio y a la disponibilidad de datos OD
            self.w_od.data = torch.clamp(ratio * (0.1 + od_ratio), min=0.1, max=5.0)

        # Incrementar contador de actualizaciones
        if isinstance(self.update_count, torch.Tensor):
            self.update_count.data = self.update_count.data + 1.0
        else:
            self.update_count = upd + 1.0


class UltraCyclicODModel(nn.Module):
    """Modelo mejorado con arquitectura más robusta."""

    def __init__(self, num_links, num_od_pairs, hidden_dim, feature_dim, num_structures,
                 t0, capacity, route_masks, od_pair_indices, num_link_groups, link_group,
                 vdf_config: DictConfig,  # MODIFICACIÓN: Argumento obligatorio nuevo
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        self.encoder = ODEncoder(num_links, hidden_dim, feature_dim, dropout)
        self.decoder = ODDecoder(num_od_pairs, hidden_dim, feature_dim, dropout)
        self.graph_matcher = ImprovedGraphMatcher(feature_dim, num_structures)
        self.validator = ImprovedAssignmentValidator(
            num_links, t0, capacity, route_masks, od_pair_indices,
            num_od_pairs, num_link_groups, link_group,
            vdf_config=vdf_config
        )

        # NO necesitamos encoder separado - usamos el mismo encoder para mantener 
        # h_x y h_y en el mismo espacio latente


    def forward(self, observed_counts: torch.Tensor, true_od_demand: torch.Tensor = None,
                warmup: bool = False) -> dict:
        is_batched = observed_counts.dim() == 2
        if not is_batched:
            observed_counts = observed_counts.unsqueeze(0)
            if true_od_demand is not None:
                true_od_demand = true_od_demand.unsqueeze(0)

        # 1. Codificar aforos observados
        h_x = self.encoder(observed_counts)

        # 2. Generar referencia h_y usando el MISMO encoder (espacio latente compartido)
        h_y = None
        if self.training and true_od_demand is not None:
            with torch.no_grad():
                # Simular flujos "ideales" a partir de la OD verdadera
                true_flows, _, _, _ = self.validator(true_od_demand, warmup=True)

                # CRÍTICO: Usar el MISMO encoder para mantener h_x y h_y 
                # en el mismo espacio latente - esto es clave para que el 
                # GraphMatcher pueda hacer comparaciones directas
                h_y = self.encoder(true_flows)

        # 3. Aplicar graph matcher
        g_x = self.graph_matcher(h_x, h_y)

        # 4. Decodificar demanda
        estimated_demand = self.decoder(g_x)

        # 5. Validar con SUE
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
        """
        Función de diagnóstico para verificar que h_x y h_y están en el mismo espacio.
        Debe llamarse durante la validación para monitorear la calidad del entrenamiento.
        """
        with torch.no_grad():
            # Generar h_x y h_y
            h_x = self.encoder(observed_counts)

            if true_od_demand is not None:
                true_flows, _, _, _ = self.validator(true_od_demand, warmup=True)
                h_y = self.encoder(true_flows)

                # Métricas de consistencia
                cosine_sim = F.cosine_similarity(h_x, h_y, dim=1).mean()
                l2_distance = F.mse_loss(h_x, h_y)

                return {
                    "cosine_similarity": cosine_sim.item(),
                    "l2_distance": l2_distance.item(),
                    "h_x_norm": h_x.norm(dim=1).mean().item(),
                    "h_y_norm": h_y.norm(dim=1).mean().item()
                }
        return None


# -----------------------------------------------------------------------------
# Compatibilidad con `train_cyclic_model_deprecated.py`
# -----------------------------------------------------------------------------
# `train_cyclic_model_deprecated.py` espera poder hacer:
# from src.models.Cyclic_Model.cyclic_model_ultra import CyclicODModelUltra, PartialDataLoss
# y luego instanciar `CyclicODModelUltra(..., cost_function_type=..., dropout=...)`
# y usar `PartialDataLoss(w_flow=..., w_od=..., w_reg=...)`.
# Para mantener compatibilidad, añadimos pequeñas "shims" (envoltorios) que
# exponen las mismas clases/firmas que el training script espera.
# -----------------------------------------------------------------------------


class PartialDataLoss(nn.Module):
    """
    Wrapper de compatibilidad que replica la API de `PartialDataLoss` usada
    por el training original. Implementación basada en la versión del
    `cyclic_model.py` para asegurar comportamiento consistente.
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

        # Pérdida de flujos (solo enlaces observados)
        if flow_mask.any():
            masked_pred_flows = predicted_flows * flow_mask
            masked_true_flows = true_flows * flow_mask
            l_flow = self.mse_loss(masked_pred_flows, masked_true_flows)
            flow_coverage = float(flow_mask.float().mean().item())
        else:
            l_flow = torch.tensor(0.0, device=predicted_flows.device)
            flow_coverage = 0.0

        # Pérdida de OD (solo demandas conocidas)
        if od_mask is not None and od_mask.any():
            masked_pred_od = predicted_od * od_mask
            masked_true_od = true_od * od_mask
            l_od = self.mse_loss(masked_pred_od, masked_true_od)
            od_coverage = float(od_mask.float().mean().item())
        else:
            l_od = torch.tensor(0.0, device=predicted_od.device)
            od_coverage = 0.0

        # Regularización de parámetros BPR
        l_reg = torch.tensor(0.0, device=predicted_flows.device)
        if learned_alpha is not None and learned_beta is not None:
            l_reg = (torch.norm(learned_alpha - 0.15, p=2) +
                     torch.norm(learned_beta - 4.0, p=2))

        # Pérdida total
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


class CyclicODModelUltra(UltraCyclicODModel):
    """
    Envoltorio de compatibilidad que acepta el argumento `cost_function_type`
    (que es pasado desde `train_cyclic_model_deprecated.py`) pero lo ignora o lo almacena
    para posibles usos futuros. Mantiene la misma firma que `CyclicODModel`.
    """

    def __init__(self, *args, cost_function_type: str = 'bpr', **kwargs):
        # Consumir cost_function_type para compatibilidad; UltraCyclicODModel no lo requiere
        super().__init__(*args, **kwargs)
        self.cost_function_type = cost_function_type

    def forward(self, observed_flows: torch.Tensor, flow_mask: torch.Tensor,
                true_od_demand: torch.Tensor = None, warmup: bool = False) -> dict:
        """
        Wrapper de compatibilidad con la firma de `CyclicODModel.forward`.

        - Aplica la máscara a los flujos observados (como hace el modelo original).
        - Llama al forward del UltraCyclicODModel que espera `observed_counts`.
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

        # Aplicar máscara (poner 0 donde no hay observación)
        observed_counts = observed_flows_proc * flow_mask_proc

        # Llamar al forward del Ultra (espera observed_counts)
        outputs = super().forward(observed_counts=observed_counts,
                                  true_od_demand=true_od_proc,
                                  warmup=warmup)

        # Si no era batched, deshacer la dimensión
        if not is_batched:
            outputs['estimated_demand'] = outputs['estimated_demand'].squeeze(0)
            outputs['reconstructed_flows'] = outputs['reconstructed_flows'].squeeze(0)
            # learned_alpha/beta pueden ser None o tensores; si son tensores con batch dim, manejarlo
            if isinstance(outputs.get('learned_alpha', None), torch.Tensor) and outputs['learned_alpha'].dim() == 2:
                outputs['learned_alpha'] = outputs['learned_alpha'].squeeze(0)
            if isinstance(outputs.get('learned_beta', None), torch.Tensor) and outputs['learned_beta'].dim() == 2:
                outputs['learned_beta'] = outputs['learned_beta'].squeeze(0)

        return outputs

# Facilitar import directo sin cambios en el training script
__all__ = [
    'CyclicODModelUltra',
    'PartialDataLoss',
    'UltraCyclicODModel',
    'AdaptiveCombinedLoss'
]
