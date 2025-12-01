"""
Modelo Deep Learning para Traffic Assignment con Datos Parciales.
(Versión Modularizada con Hydra)

Este modelo extiende la arquitectura cyclic para trabajar con:
- Demandas OD parcialmente conocidas
- Flujos de enlaces parcialmente observados

Arquitectura:
- ODEncoder: Codifica flujos observados a espacio latente
- GraphMatcher: Alinea espacio latente con estructura de red
- ODDecoder: Decodifica a demandas OD completas
- AssignmentValidator: Valida equilibrio mediante SUE con función de costo inyectada dinámicamente

Funciones de costo:
- Se definen en src/components/vdf/ y se inyectan vía configuración.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from typing import Dict, Optional, Any


# =============================================================================
# COMPONENTES DE RED NEURONAL (Encoder/Decoder/Matcher)
# =============================================================================

class ODEncoder(nn.Module):
    """Codifica flujos observados en un espacio latente."""

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
            nn.LayerNorm(feature_dim)
        )

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        return self.network(flows)


class ODDecoder(nn.Module):
    """Decodifica espacio latente a demandas OD completas."""

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
            nn.Softplus()  # Asegurar demandas no negativas
        )

    def forward(self, g_x: torch.Tensor) -> torch.Tensor:
        return self.network(g_x)


class GraphMatcher(nn.Module):
    """
    Alinea espacio latente con estructura de red mediante matrices aprendibles.
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

        # Matrices M y V para transformación estructural
        self.register_buffer('M', torch.randn(feature_dim, num_structures) * 0.1)
        self.register_buffer('V', torch.ones(1, num_structures))

        # Red de atención para combinar estructuras
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, num_structures),
            nn.Softmax(dim=-1)
        )

        self.register_buffer('update_count', torch.tensor(0.0))

    def _update_matrices_regularized(self, h_x: torch.Tensor, h_y: torch.Tensor):
        """Actualiza matrices M y V con regularización."""
        # Normalización
        h_x_norm = F.normalize(h_x, p=2, dim=1)
        h_y_norm = F.normalize(h_y, p=2, dim=1)

        # Actualizar M basado en similitud
        similarity_matrix = torch.bmm(h_x_norm.unsqueeze(2), h_y_norm.unsqueeze(1))
        similarity_vector = torch.mean(similarity_matrix.squeeze(), dim=0)

        target_M = similarity_vector.unsqueeze(1).expand(-1, self.num_structures)
        regularized_M = target_M + self.reg_strength * torch.randn_like(target_M) * 0.01

        momentum = min(self.lambda_m * (1 + self.update_count * 0.001), 0.1)
        self.M.data = (1 - momentum) * self.M.data + momentum * regularized_M

        # Actualizar V
        h_x_att_M = h_x.unsqueeze(2) * self.M
        h_x_att_M_norm = F.normalize(h_x_att_M, p=2, dim=1)
        h_y_expanded_norm = h_y_norm.unsqueeze(2)

        cosine_sim_v = torch.sum(h_x_att_M_norm * h_y_expanded_norm, dim=1)
        quality_per_structure = torch.mean(cosine_sim_v, dim=0, keepdim=True)
        quality_per_structure = torch.clamp(quality_per_structure, min=0.1, max=2.0)

        momentum_v = min(self.lambda_v * (1 + self.update_count * 0.001), 0.1)
        self.V.data = (1 - momentum_v) * self.V.data + momentum_v * quality_per_structure

        self.update_count += 1

    def forward(self, h_x: torch.Tensor, h_y: torch.Tensor = None) -> torch.Tensor:
        # Actualizar matrices solo en entrenamiento con referencia
        if self.training and h_y is not None:
            with torch.no_grad():
                self._update_matrices_regularized(h_x.detach(), h_y.detach())

        # Aplicar transformación con atención
        attention_weights = self.attention_net(h_x)  # [batch, num_structures]
        h_x_transformed = h_x.unsqueeze(2) * self.M  # [batch, feature_dim, num_structures]
        h_x_weighted = h_x_transformed * self.V
        g_x = torch.sum(h_x_weighted * attention_weights.unsqueeze(1), dim=2)

        return g_x


# =============================================================================
# ASSIGNMENT VALIDATOR (Con Inyección de VDF)
# =============================================================================

class AssignmentValidator(nn.Module):
    """
    Valida asignación mediante SUE con optimizaciones de memoria:
    1. Warm Start: Reutiliza flujos previos.
    2. Truncated Backprop: Solo calcula gradientes al final.
    3. Dynamic Iterations: Permite variar iteraciones durante entrenamiento.
    """

    def __init__(self, num_links: int, t0: torch.Tensor, capacity: torch.Tensor,
                 route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: DictConfig,
                 max_iters: int = 10, convergence_threshold: float = 1e-4):

        super().__init__()
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.register_buffer('t0', t0)

        # Instanciación VDF (Igual que antes)
        self.cost_function = hydra.utils.instantiate(
            vdf_config,
            t0=t0,
            capacity=capacity,
            num_link_groups=num_link_groups,
            link_group=link_group,
            _recursive_=False
        )

        self.assignment_layer = StochasticAssignmentLayer(route_masks=route_masks, mu=1.0)
        self.register_buffer('last_convergence_iter', torch.tensor(0.0))

        # ESTRATEGIA 1: Buffer para Warm Start
        # Guardamos el estado del flujo para reutilizarlo en la siguiente época
        self.register_buffer('running_flows', None)

    def forward(self, estimated_demands: torch.Tensor,
                warmup: bool = False,
                override_max_iters: Optional[int] = None) -> tuple:
        """
        Args:
            override_max_iters: ESTRATEGIA 3 (Permite variar iters desde el training loop)
        """

        # Determinar iteraciones (Estrategia 3)
        current_max_iters = override_max_iters if override_max_iters is not None else self.max_iters

        batch_size = estimated_demands.shape[0] if estimated_demands.dim() == 2 else 1
        if estimated_demands.dim() == 1: estimated_demands = estimated_demands.unsqueeze(0)

        # Costos base
        freeflow_costs_base = self.cost_function(torch.zeros_like(self.t0))
        freeflow_costs = freeflow_costs_base.unsqueeze(0).expand(batch_size, -1)

        # --- ESTRATEGIA 1: LOGICA DE WARM START ---
        # Si estamos entrenando y tenemos un historial válido, lo usamos.
        # Si cambiamos batch_size (ej. último batch) o es warmup, reseteamos.

        can_warm_start = (
                self.training
                and not warmup
                and self.running_flows is not None
                and self.running_flows.shape[0] == batch_size
        )

        if can_warm_start:
            # Usamos .detach() para romper el grafo hacia el pasado lejano
            flows = self.running_flows.detach().clone()
        else:
            # Cold Start (Flujo Libre)
            flows, _ = self.assignment_layer(freeflow_costs, estimated_demands)

        # Variable para almacenar las probs de la última iteración
        final_route_probs = None

        # --- BUCLE MSA ---
        converged = False
        actual_iters = 1

        # ESTRATEGIA 2: Configuración de Truncated Backprop
        # Solo calculamos gradientes en las últimas 'grad_steps' iteraciones
        grad_steps = 3

        if warmup:
            # En warmup solo hacemos 1 pasada rápida
            reconstructed_flows, final_route_probs = self.assignment_layer(freeflow_costs, estimated_demands)
            convergence_info = {"converged": True, "iterations": 1}
        else:
            # Si iters es muy bajo, calculamos gradiente siempre, si no, truncamos.
            grad_start_iter = max(0, current_max_iters - grad_steps)

            for it in range(1, current_max_iters + 1):
                # ESTRATEGIA 2: Toggle de Gradientes
                # Activamos gradientes solo si estamos en las últimas iteraciones Y en modo training
                requires_grad = self.training and (it > grad_start_iter)

                with torch.set_grad_enabled(requires_grad):
                    costs = self.cost_function(flows)
                    new_flows, current_probs = self.assignment_layer(costs, estimated_demands)

                    final_route_probs = current_probs

                    # MSA Step
                    alpha_msa = 1.0 / (it + 1)
                    flows = flows + alpha_msa * (new_flows - flows)

            # Guardar estado para la siguiente época (Warm Start)
            if self.training:
                self.running_flows = flows.detach()

            reconstructed_flows = flows
            convergence_info = {"converged": False, "iterations": current_max_iters}

        # Extracción de parámetros aprendidos (Igual que antes)
        learned_alpha = getattr(self.cost_function, 'get_alpha', lambda: None)()
        learned_beta = getattr(self.cost_function, 'get_beta', lambda: None)()

        return reconstructed_flows, learned_alpha, learned_beta, convergence_info, final_route_probs


class StochasticAssignmentLayer(nn.Module):
    """
    Capa de Asignación Estocástica Vectorizada (Optimizada para Tensores Esparsos).
    """

    def __init__(self, route_masks: torch.Tensor, mu: float = 1.0):
        super().__init__()

        # 1. Procesar Máscara Esparsa
        # Esperamos route_masks de tamaño (Num_OD, K_Paths, Num_Links)
        self.num_od, self.k_paths, self.num_links = route_masks.shape

        # Aplanar las dos primeras dimensiones (OD y K) para hacerla 2D
        # Nueva forma lógica: (Num_OD * K_Paths, Num_Links)
        if route_masks.is_sparse:
            route_masks = route_masks.coalesce()
            indices = route_masks.indices()
            values = route_masks.values()

            # Calcular nuevos índices de fila: row = od_idx * K + k_idx
            new_rows = indices[0] * self.k_paths + indices[1]
            new_cols = indices[2]  # Link index se mantiene

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
            # Fallback por si acaso le pasas un denso
            self.register_buffer('sparse_mask_2d', route_masks.reshape(-1, self.num_links).to_sparse())

        self.mu_raw = nn.Parameter(torch.tensor(float(mu)))

    @property
    def mu(self):
        return torch.clamp(F.softplus(self.mu_raw), min=0.1, max=10.0)

    def forward(self, link_costs: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
        """
        Args:
            link_costs: [Batch, Num_Links]
            demands: [Batch, Num_OD]
        """
        batch_size = link_costs.shape[0]

        # =====================================================================
        # PASO 1: Calcular Costo de Ruta (Link -> Ruta)
        # Queremos: route_costs [Batch, OD, K]
        # Operación: Sumar costos de links para cada ruta.
        # Matemáticamente: RouteCosts = LinkCosts @ Mask.T
        # =====================================================================

        # Truco para multiplicar (Batch, Link) x (Link, RutasTotales_Esparsa)
        # PyTorch sparse.mm requiere (Sparse x Dense). Usamos transposición:
        # (A @ B).T = B.T @ A.T
        # Result.T = (LinkCosts @ Mask.T).T = Mask @ LinkCosts.T

        # Mask [OD*K, L] (Sparse)
        # LinkCosts.T [L, B] (Dense)
        # Result_T [OD*K, B]

        costs_t = torch.transpose(link_costs, 0, 1)  # [L, B]
        route_costs_flat_t = torch.sparse.mm(self.sparse_mask_2d, costs_t)  # [OD*K, B]

        # Volver a formato [Batch, OD, K]
        route_costs = route_costs_flat_t.transpose(0, 1).view(batch_size, self.num_od, self.k_paths)

        # =====================================================================
        # PASO 2: Logit Probabilities (Igual que antes)
        # =====================================================================

        # Estabilización numérica
        min_costs, _ = torch.min(route_costs, dim=2, keepdim=True)
        stable_costs = route_costs - min_costs.detach()
        # Clamp para evitar exp() infinito
        stable_costs = torch.clamp(stable_costs, max=50.0)

        exp_utility = torch.exp(-self.mu * stable_costs)
        sum_utility = torch.sum(exp_utility, dim=2, keepdim=True)
        route_probs = exp_utility / (sum_utility + 1e-9)

        # Asignar Demanda: [Batch, OD, K]
        route_flows = route_probs * demands.unsqueeze(2)

        # =====================================================================
        # PASO 3: Proyectar a Links (Ruta -> Link)
        # Queremos: link_flows [Batch, Num_Links]
        # Operación: Sumar flujos de rutas que pasan por cada link.
        # Matemáticamente: LinkFlows = RouteFlowsFlat @ Mask
        # =====================================================================

        # RouteFlowsFlat: [Batch, OD*K]
        route_flows_flat = route_flows.view(batch_size, -1)

        # De nuevo el truco de la transpuesta para usar Sparse.mm:
        # Result.T = (RF @ Mask).T = Mask.T @ RF.T
        # Pero Mask.T es (L, OD*K).
        # Para hacer esto eficiente sin transponer la matriz esparsa explícitamente (que es lento),
        # usamos torch.sparse.mm con la transpuesta lógica si es posible,
        # o transponemos la esparsa una sola vez en __init__ si tenemos memoria.
        # Pero PyTorch permite .t() en sparse tensors rápido (solo cambia índices).

        mask_t = self.sparse_mask_2d.t()  # [L, OD*K] (Virtualmente gratis en sparse)
        rf_t = torch.transpose(route_flows_flat, 0, 1)  # [OD*K, B]

        link_flows_t = torch.sparse.mm(mask_t, rf_t)  # [L, B]

        link_flows = link_flows_t.transpose(0, 1)  # [B, L]

        return link_flows, route_probs

# =============================================================================
# FUNCIÓN DE PÉRDIDA
# =============================================================================

class PartialDataLoss(nn.Module):
    """
    Función de pérdida para entrenamiento con datos parciales.
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
            flow_coverage = flow_mask.float().mean().item()
        else:
            l_flow = torch.tensor(0.0, device=predicted_flows.device)
            flow_coverage = 0.0

        # Pérdida de OD (solo demandas conocidas)
        if od_mask.any():
            masked_pred_od = predicted_od * od_mask
            masked_true_od = true_od * od_mask
            l_od = self.mse_loss(masked_pred_od, masked_true_od)
            od_coverage = od_mask.float().mean().item()
        else:
            l_od = torch.tensor(0.0, device=predicted_od.device)
            od_coverage = 0.0

        # Regularización de parámetros BPR (si existen)
        l_reg = torch.tensor(0.0, device=predicted_flows.device)
        if learned_alpha is not None and learned_beta is not None:
            # Regularizar hacia valores típicos
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
            "w_flow": self.w_flow.item(),
            "w_od": self.w_od.item()
        }


# =============================================================================
# MODELO PRINCIPAL
# =============================================================================

class CyclicODModel(nn.Module):
    """
    Modelo principal para Traffic Assignment con datos parciales.
    Modularizado para recibir configuración de VDF.
    """

    def __init__(self,
                 num_links: int,
                 num_od_pairs: int,
                 hidden_dim: int,
                 feature_dim: int,
                 num_structures: int,
                 t0: torch.Tensor,
                 capacity: torch.Tensor,
                 route_masks: torch.Tensor,
                 od_pair_indices: torch.Tensor,
                 num_link_groups: int,
                 link_group: torch.Tensor,
                 vdf_config: DictConfig,  # Configuración VDF
                 dropout: float = 0.1,
                 **kwargs): # <--- AQUÍ ESTÁ LA MAGIA: kwargs absorbe loss_weights y otros extras

        super().__init__()

        # Opcional: Si quieres guardar los kwargs por si acaso (debugging)
        self.config_extras = kwargs

        self.encoder = ODEncoder(num_links, hidden_dim, feature_dim, dropout)
        self.decoder = ODDecoder(num_od_pairs, hidden_dim, feature_dim, dropout)
        self.graph_matcher = GraphMatcher(feature_dim, num_structures)

        # Pasamos la config de VDF al validador
        self.validator = AssignmentValidator(
            num_links, t0, capacity, route_masks, od_pair_indices,
            num_od_pairs, num_link_groups, link_group,
            vdf_config=vdf_config
        )

    def forward(self,
                observed_flows: torch.Tensor,
                flow_mask: torch.Tensor,
                true_od_demand: Optional[torch.Tensor] = None,
                warmup: bool = False,
                current_iter_count: Optional[int] = None) -> Dict[str, torch.Tensor]:

        is_batched = observed_flows.dim() == 2
        if not is_batched:
            observed_flows = observed_flows.unsqueeze(0)
            flow_mask = flow_mask.unsqueeze(0)
            if true_od_demand is not None:
                true_od_demand = true_od_demand.unsqueeze(0)

        masked_flows = observed_flows * flow_mask

        # 1. Codificar
        h_x = self.encoder(masked_flows)

        # 2. Referencia (training)
        h_y = None
        if self.training and true_od_demand is not None:
            with torch.no_grad():
                true_flows, _, _, _, _ = self.validator(true_od_demand, warmup=True)
                h_y = self.encoder(true_flows)

        # 3. Matching
        g_x = self.graph_matcher(h_x, h_y)

        # 4. Decodificar
        estimated_demand = self.decoder(g_x)

        # 5. Validar
        reconstructed_flows, learned_alpha, learned_beta, convergence_info, route_probs = self.validator(
            estimated_demand,
            warmup=warmup,
            override_max_iters=current_iter_count  # <--- Pasamos el valor aquí
        )

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