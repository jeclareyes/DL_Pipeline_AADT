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
from typing import Dict, Optional
import logging
import numpy as np

# Module logger for diagnostics
logger = logging.getLogger(__name__)


# =============================================================================
# COMPONENTES DE RED NEURONAL (Encoder/Decoder/Matcher)
# =============================================================================

class PaperMLP(nn.Module):
    """
    Two-layer MLP architecture based on Equations (6) and (8) from the paper.
    Enhanced with optional LayerNorm for latent space stabilization.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0,
                 nonneg: bool = False, use_norm: bool = False):
        super().__init__()

        # Base architecture: Linear -> LeakyReLU -> Dropout -> Linear
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        ]

        # 1. LATENT STABILIZATION: LayerNorm is added before the final activation
        # to ensure the latent features (hx, hy) are centered with unit variance.
        if use_norm:
            layers.append(nn.LayerNorm(out_dim))

        # 2. FINAL ACTIVATION:
        # - Softplus: Used for OD Demand estimation to ensure physical non-negativity.
        # - LeakyReLU: Used for intermediate latent representations.
        layers.append(nn.Softplus() if nonneg else nn.LeakyReLU(0.2))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        x shape: [Batch, in_dim]
        """
        return self.net(x)

class GraphMatcher(nn.Module):
    """
    Implements the structural matching logic between flow features (hx)
    and OD features (hy) using the double-layer attention mechanism.

    Ref: Equations (9), (10), and (11) from the paper.
    """

    def __init__(self, feature_dim: int, num_structures: int, m: float = 0.05, v: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_structures = num_structures
        self.lambda_m = m  # Similarity update rate (m in paper)
        self.lambda_v = v  # Value decay/update rate (v in paper)
        self.eps = eps

        # M: Similarity Matrix [feature_dim, num_structures]
        # V: Importance Vector [1, num_structures]
        # Initialized to ones as per paper section 3.2.2
        self.register_buffer("M", torch.ones(feature_dim, num_structures))
        self.register_buffer("V", torch.ones(1, num_structures))

        # Buffers to maintain history for sub-step updates (p batches concatenated)
        self.history_hxy = []
        self.history_hxh = []
        self.history_hyh = []

    @torch.no_grad()
    def update(self, h_x: torch.Tensor, h_y: torch.Tensor):
        """
        Structural Calibration: Updates M and V based on cross-space correlations.
        Calculates sub-step similarities for each of the ns structures.
        """
        # 1. Compute current batch statistics [feature_dim]
        curr_hxy = (h_x * h_y).sum(dim=0)
        curr_hxh = (h_x * h_x).sum(dim=0)
        curr_hyh = (h_y * h_y).sum(dim=0)

        # 2. Update history buffer
        self.history_hxy.append(curr_hxy)
        self.history_hxh.append(curr_hxh)
        self.history_hyh.append(curr_hyh)

        if len(self.history_hxy) > self.num_structures:
            self.history_hxy.pop(0)
            self.history_hxh.pop(0)
            self.history_hyh.pop(0)

        # 3. Structural update for each column p=1...ns
        # Each structure p represents the average similarity over the last p batches.
        for p in range(1, len(self.history_hxy) + 1):
            sum_hxy = torch.stack(self.history_hxy[-p:]).sum(dim=0)
            sum_hxh = torch.stack(self.history_hxh[-p:]).sum(dim=0)
            sum_hyh = torch.stack(self.history_hyh[-p:]).sum(dim=0)

            # Eq (9): Update Structural Match Matrix M
            target_M_p = sum_hxy / (torch.sqrt(sum_hxh) * torch.sqrt(sum_hyh) + self.eps)
            self.M[:, p - 1] = (1.0 - self.lambda_m) * self.M[:, p - 1] + self.lambda_m * target_M_p

            # Eq (10) & (11): Update Structural Value V
            # First, decay the previous value (Eq 10 logic)
            self.V[0, p - 1] *= (1.0 - self.lambda_m)

        # --- FUERA DEL BUCLE p ---
        # Eq (11): Vectorized update for ALL V at once
        # h_x: [B, F] -> [B, F, S] | h_y: [B, F] -> [B, F, 1]
        hx_prime_all = h_x.unsqueeze(2) * self.M.unsqueeze(0)
        hy_expanded = h_y.unsqueeze(2)

        num = (hx_prime_all * hy_expanded).sum(dim=1)  # [B, S]
        den = torch.sqrt((hx_prime_all ** 2).sum(dim=1)) * \
              torch.sqrt((h_y ** 2).sum(dim=1)).unsqueeze(1)
        cos_sim_vec = (num / (den + self.eps)).mean(dim=0)  # [S]

        # Actualización global del vector V
        self.V = cos_sim_vec.unsqueeze(0) + self.lambda_v * self.V

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        """
        Filters latent features using a snapshot of M and V to avoid
        in-place modification errors during backward pass.
        """
        # h: [B, F]
        h_exp = h.unsqueeze(2)

        # WE USE .clone() TO TAKE A SNAPSHOT FOR THE AUTOGRAD GRAPH
        # This prevents the "variable modified by inplace operation" error
        # when self.M and self.V are updated later in the same forward pass.
        M_snapshot = self.M.clone().unsqueeze(0) # [1, F, S]
        V_snapshot = self.V.clone().unsqueeze(0) # [1, 1, S]

        # Hadamard product: h ⊙ M ⊙ V
        g_structured = h_exp * M_snapshot * V_snapshot

        return g_structured.mean(dim=2)

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
                 lanes: torch.Tensor, route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 vdf_config: DictConfig, trips_scaler: float = 1.0,
                 max_iters: int = 5, convergence_threshold: float = 1e-4,
                 msa_convergence: Optional[Dict] = None):

        super().__init__()
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.register_buffer('t0', t0)
        self.trips_scaler = trips_scaler

        # Instanciación VDF (Igual que antes)
        self.cost_function = hydra.utils.instantiate(
            vdf_config,
            t0=t0,
            capacity=capacity,
            lanes=lanes,
            num_link_groups=num_link_groups,
            link_group=link_group,
            _recursive_=False
        )

        self.assignment_layer = StochasticAssignmentLayer(route_masks=route_masks, mu=1.0)
        self.register_buffer('last_convergence_iter', torch.tensor(0.0))

        # ESTRATEGIA 1: Buffer para Warm Start
        # Guardamos el estado del flujo para reutilizarlo en la siguiente época
        self.register_buffer('running_flows', None)

        # --- Convergence configuration ---
        # TODO estas configuraciones podrían moverse a un objeto aparte
        cfg = msa_convergence or {}
        self.strategy = cfg.get('strategy', 'relative_flow')
        self.flow_tol = float(cfg.get('flow_tol', 1e-4))
        self.gap_tol = float(cfg.get('gap_tol', 1e-4))
        self.min_iters = int(cfg.get('min_iters', 1))
        # override max_iters if provided in convergence cfg (keeps backward compat)
        self.max_iters = int(cfg.get('max_iters', self.max_iters))
        self.grad_steps = int(cfg.get('grad_steps', 10))
        self.eps = float(cfg.get('eps', 1e-9))
        # Verbose flag to control local logging from this class
        self.verbose = bool(cfg.get('verbose', False))

    # TODO: Mover estas funciones de convergencia a un módulo aparte si se vuelven más complejas
    def _strategy_relative_flow_change(self, prev: torch.Tensor, cur: torch.Tensor) -> float:
        # Compute batch-wise L2 relative change, return max across batch for conservative stop
        prev_flat = prev.view(prev.shape[0], -1)
        cur_flat = cur.view(cur.shape[0], -1)
        num = torch.norm(cur_flat - prev_flat, p=2, dim=1)
        den = torch.norm(prev_flat, p=2, dim=1).clamp(min=self.eps)
        rel = (num / den)
        return float(rel.max().item())

    def _strategy_relative_gap(self, link_flows: torch.Tensor, link_costs: torch.Tensor, demands: torch.Tensor) -> float:
        # link_flows: [B, L], link_costs: [B, L], demands: [B, OD]
        # link_term = sum_l f_l * c_l
        link_term = (link_flows * link_costs).sum(dim=1)  # [B]

        # Compute per-route minimum cost per OD using sparse mask
        # costs_t: [L, B]
        costs_t = link_costs.transpose(0, 1)
        route_costs_flat_t = torch.sparse.mm(self.assignment_layer.sparse_mask_2d, costs_t)  # [OD*K, B]
        batch_size = link_flows.shape[0]
        route_costs = route_costs_flat_t.transpose(0, 1).view(batch_size, self.assignment_layer.num_od, self.assignment_layer.k_paths)
        c_min_od, _ = torch.min(route_costs, dim=2)  # [B, OD]

        od_term = (demands * c_min_od).sum(dim=1)  # [B]

        gap = (link_term - od_term) / (link_term.clamp(min=self.eps))
        return float(gap.max().item())

    def _msa_run(self, flows: torch.Tensor, real_estimated_demands: torch.Tensor, current_max_iters: int):
        """
        Encapsulated MSA driver implementing configurable stopping strategies.
        Returns: flows, final_route_probs, converged(bool), actual_iters(int), rel_flow(float), rel_gap(float)
        """
        converged = False
        actual_iters = 0
        final_route_probs = None

        # Initialize diagnostics
        rel_flow = float('inf')
        rel_gap = float('inf')

        # Determine when to enable gradients (truncated backprop)
        grad_start_iter = max(0, current_max_iters - self.grad_steps)

        # Keep a snapshot of the previous flows for relative flow computation
        last_prev_flows = None

        # Log initial settings if verbose
        if logger.isEnabledFor(logging.DEBUG) and self.verbose:
            logger.debug(f"MSA start strategy={self.strategy} max_iters={current_max_iters} flow_tol={self.flow_tol} gap_tol={self.gap_tol} min_iters={self.min_iters} grad_steps={self.grad_steps} training={self.training}")

        for it in range(1, current_max_iters + 1):
            requires_grad = self.training and (it > grad_start_iter)

            # Snapshot flows BEFORE the update for relative flow metric
            last_prev_flows = flows.detach().clone()

            # Forward MSA step (with or without grad)
            with torch.set_grad_enabled(requires_grad):
                costs = self.cost_function(flows)
                new_flows, current_probs = self.assignment_layer(costs, real_estimated_demands)

                final_route_probs = current_probs
                alpha_msa = 1.0 / (it + 1)
                updated_flows = flows + alpha_msa * (new_flows - flows)

            # Compute cheap relative flow change (no grad needed) using the last snapshot
            with torch.no_grad():
                try:
                    rel_flow = self._strategy_relative_flow_change(last_prev_flows, updated_flows)
                except Exception:
                    rel_flow = float('inf')

            # Log iteration diagnostics
            if logger.isEnabledFor(logging.DEBUG) and self.verbose:
                logger.debug(f"MSA iter={it} alpha={alpha_msa:.6g} requires_grad={requires_grad} rel_flow={rel_flow:.6g}")

            # lazy gap computation: only compute when needed by strategy
            need_gap = False
            stop = False
            if it >= self.min_iters:
                if self.strategy == 'relative_flow':
                    stop = rel_flow <= self.flow_tol
                elif self.strategy == 'relative_gap':
                    need_gap = True
                elif self.strategy == 'hybrid':
                    # first cheap check
                    if rel_flow <= self.flow_tol:
                        need_gap = True
                    else:
                        stop = False
                else:
                    # fallback legacy
                    stop = rel_flow <= self.convergence_threshold

            if need_gap:
                # compute costs at the updated flows for accurate gap
                with torch.no_grad():
                    try:
                        costs_for_gap = self.cost_function(updated_flows)
                        rel_gap = self._strategy_relative_gap(updated_flows.detach(), costs_for_gap.detach(), real_estimated_demands.detach())
                    except Exception:
                        rel_gap = float('inf')

                if logger.isEnabledFor(logging.DEBUG) and self.verbose:
                    logger.debug(f"MSA iter={it} computed_rel_gap={rel_gap:.6g}")

                if self.strategy == 'relative_gap':
                    stop = rel_gap <= self.gap_tol
                elif self.strategy == 'hybrid':
                    stop = rel_gap <= self.gap_tol

            # Accept update
            flows = updated_flows
            actual_iters = it

            if stop:
                converged = True
                # If training and the iteration we just ran had no grad, ensure one grad-enabled final iteration
                if self.training and not requires_grad:
                    # log that we force a final grad-enabled step
                    if logger.isEnabledFor(logging.INFO) and self.verbose:
                        logger.info(f"MSA converged at iter={it} without grad; forcing one final grad-enabled iteration")
                    with torch.set_grad_enabled(True):
                        try:
                            costs = self.cost_function(flows)
                            new_flows, current_probs = self.assignment_layer(costs, real_estimated_demands)
                            final_route_probs = current_probs
                            alpha_msa = 1.0 / (it + 2)
                            flows = flows + alpha_msa * (new_flows - flows)
                        except Exception:
                            # if the forced grad step fails, keep the current flows
                            pass

                # Compute final diagnostics (rel_flow between last_prev_flows and flows, rel_gap at flows)
                with torch.no_grad():
                    try:
                        if last_prev_flows is not None:
                            rel_flow = self._strategy_relative_flow_change(last_prev_flows, flows)
                    except Exception:
                        rel_flow = float('inf')

                    try:
                        costs_final = self.cost_function(flows)
                        rel_gap = self._strategy_relative_gap(flows.detach(), costs_final.detach(), real_estimated_demands.detach())
                    except Exception:
                        rel_gap = float('inf')

                # Log convergence summary
                if logger.isEnabledFor(logging.INFO) and self.verbose:
                    logger.info(
                        f"MSA stop strategy={self.strategy} converged=True iters={actual_iters} rel_flow={rel_flow:.6g} rel_gap={rel_gap:.6g} grad_was_enabled={requires_grad}"
                    )
                break


        else:
            # Executed if loop finished without break
            if logger.isEnabledFor(logging.WARNING) and self.verbose:
                logger.warning(
                    f"MSA reached max_iters={current_max_iters} without meeting tolerance. last_rel_flow={rel_flow:.6g} last_rel_gap={rel_gap:.6g}"
                )

        # Ensure we always have sensible final diagnostics:
        # If rel_flow or rel_gap were never computed, compute them now
        need_final_metrics = (rel_gap is None) or (not torch.isfinite(torch.tensor(rel_flow))) or (
            not torch.isfinite(torch.tensor(rel_gap if rel_gap is not None else float('nan'))))

        if need_final_metrics:
            with torch.no_grad():
                try:
                    costs_final = self.cost_function(flows)
                    # Compute relative flow between last known previous state and final flows
                    try:
                        if last_prev_flows is not None:
                            rel_flow = self._strategy_relative_flow_change(last_prev_flows, flows)
                        else:
                            rel_flow = float('inf')
                    except Exception:
                        rel_flow = float('inf')

                    # Compute relative gap
                    try:
                        rel_gap = self._strategy_relative_gap(flows.detach(), costs_final.detach(), real_estimated_demands.detach())
                    except Exception:
                        rel_gap = float('inf')
                except Exception:
                    # Fail-safe: if cost_function or computations error, set infinities
                    rel_flow = float('inf')
                    rel_gap = float('inf')

        # Final sanity defaults
        if final_route_probs is None:
            final_route_probs = None

        # finalize metrics as plain floats
        rel_flow = float(rel_flow) if rel_flow is not None else float('inf')
        rel_gap = float(rel_gap) if rel_gap is not None else float('inf')

        return flows, final_route_probs, converged, actual_iters, rel_flow, rel_gap

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

        # Aplicar escalador de viajes
        real_estimated_demands = estimated_demands * self.trips_scaler

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
            flows, _ = self.assignment_layer(freeflow_costs, real_estimated_demands)

        # Variable para almacenar las probs de la última iteración
        final_route_probs = None

        # default diagnostics metrics (safe if warmup used)
        rel_flow = float('inf')
        rel_gap = float('inf')

        # --- BUCLE MSA ---
        converged = False
        actual_iters = 0

        if warmup:
            # En warmup solo hacemos 1 pasada rápida
            reconstructed_flows, final_route_probs = self.assignment_layer(freeflow_costs, real_estimated_demands)
            # Make warmup explicit in diagnostics
            convergence_info = {"converged": False, "iterations": 0, "rel_flow": None, "rel_gap": None, "warmup": True}
            # use the reconstructed flows as the flows to return
            flows = reconstructed_flows
        else:
            # Use the encapsulated MSA driver
            flows, final_route_probs, converged, actual_iters, rel_flow, rel_gap = self._msa_run(
                flows, real_estimated_demands, current_max_iters
            )

        # Guardar estado para la siguiente época (Warm Start)
        if self.training:
            self.running_flows = flows.detach()

        reconstructed_flows = flows
        # attach final metrics for diagnostics
        convergence_info = {"converged": converged, "iterations": int(actual_iters), "rel_flow": (float(rel_flow) if rel_flow not in (None, float('inf')) else None), "rel_gap": (float(rel_gap) if rel_gap not in (None, float('inf')) else None), "warmup": bool(warmup)}

        # Log summary of convergence if verbose
        if logger.isEnabledFor(logging.INFO) and getattr(self, 'verbose', False):
            if warmup:
                logger.info(f"AssignmentValidator forward (warmup) finished: warmup=True iterations=0")
            else:
                logger.info(f"AssignmentValidator forward finished strategy={self.strategy} convergence={convergence_info}")

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

    def forward(self, link_costs: torch.Tensor, demands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        # PASO 2: Logit Probabilities
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
# MODELO PRINCIPAL
# =============================================================================

class CyclicODModel(nn.Module):
    def __init__(
            self,
            num_links: int,
            num_od_pairs: int,
            architecture: DictConfig,
            t0: torch.Tensor,
            capacity: torch.Tensor,
            lanes: torch.Tensor,
            route_masks: torch.Tensor,
            od_pair_indices: torch.Tensor,
            num_link_groups: int,
            link_group: torch.Tensor,
            vdf_config: DictConfig,
            **kwargs
    ):
        super().__init__()

        logging.info(f"INITIALIZING: {kwargs.get('model_name', 'Unknown Model')}.")


        arch = architecture
        feature_dim = arch.feature_dim
        num_structures = arch.num_structures
        hidden_dim_from_link = arch.hidden_dim_from_link
        hidden_dim_from_od = arch.hidden_dim_from_od
        hidden_dim_to_od = arch.hidden_dim_to_od
        dropout_from_od = arch.dropout_from_od
        dropout_to_od = arch.dropout_to_od
        dropout_from_link = arch.dropout_from_link

        # 1. REGISTRO DE ESCALADORES (Buffers)
        # link_scale: Basado en la capacidad media para normalizar flujos de entrada.
        # od_scale: Basado en trips_scaler para normalizar demandas en el ciclo backward.
        self.link_scale_val = kwargs.get('link_scale', 1.0)
        self.od_scale_val = kwargs.get('od_scale', 1.0)
        self.register_buffer("link_scale", torch.tensor(self.link_scale_val, dtype=torch.float32))
        self.register_buffer("od_scale", torch.tensor(self.od_scale_val, dtype=torch.float32))
        logging.info(f"Actual scalers: link_scale={self.link_scale}, od_scale={self.od_scale}")

        # 2. INSTANCIACIÓN DE MLPs CON NORMALIZACIÓN LATENTE
        # f_encoder: Recibe flujos escalados -> Salida hx normalizada (use_norm=True).
        self.f_encoder = PaperMLP(num_links, hidden_dim_from_link, feature_dim, dropout_from_link, use_norm=True)

        # f_decoder: Recibe gx normalizado -> Salida od_hat en ESCALA REAL (use_norm=False).
        # Esto permite proyectar de un espacio [-1, 1] al dominio de miles de vehículos.
        self.f_decoder = PaperMLP(feature_dim, hidden_dim_to_od, num_od_pairs, dropout_to_od, nonneg=True, use_norm=False)

        # =====================================================================
        # START: ESTRATEGIA COLD START INITIALIZATION
        # =====================================================================
        # Objetivo: Forzar que el modelo empiece con V/C muy bajo para evitar
        # gradientes explosivos en el SUE durante las primeras épocas.
        # ---------------------------------------------------------------------
        import math
        # 1. Recuperar la media real que viene del pipeline
        # Si no llega, usamos 1.0 por seguridad.
        real_mean_target = kwargs.get('initial_mean', 1.0)

        # Recuperamos el scale (asegurando que sea float para matemáticas)
        scale_val = kwargs.get('od_scale', 1.0)
        if hasattr(scale_val, 'item'): scale_val = scale_val.item()
        if scale_val <= 1e-5: scale_val = 1.0

        # 2. Factor de Seguridad (El "Freno" Inicial)
        # Empezamos con el 5% de la demanda promedio histórica.
        safety_factor = 0.05

        # Calculamos qué valor debe escupir la red ANTES de multiplicarse por od_scale
        # Target_Net_Output * od_scale = Real_Mean * Safety_Factor
        target_normalized = (real_mean_target * safety_factor) / scale_val

        # Clamp para estabilidad numérica del logaritmo
        target_normalized = max(target_normalized, 1e-5)

        # 3. Calcular el Bias Inverso (Inverse Softplus)
        # Softplus(x) = y  =>  x = ln(exp(y) - 1)
        if target_normalized > 20:
            bias_init = target_normalized  # Para valores grandes es lineal
        else:
            bias_init = math.log(math.exp(target_normalized) - 1)

        logging.info(
            f"Cold Start Init: Real Mean={real_mean_target:.2f} | "
            f"OD Scale={scale_val:.2f} | Safety Factor={safety_factor} -> "
            f"Init Bias={bias_init:.4f}"
        )

        # 4. Inyectar en la última capa lineal del Decoder
        # PaperMLP.net es un Sequential.
        # [-1] es Softplus
        # [-2] es la Capa Lineal Final que queremos tocar
        final_linear = self.f_decoder.net[-2]

        if isinstance(final_linear, nn.Linear):
            # A. Fijar el bias calculado
            nn.init.constant_(final_linear.bias, bias_init)

            # B. Silenciar los pesos (Weights)
            # Usamos una desviación estándar minúscula para que el input (gx)
            # apenas afecte la salida en la primera época. El bias manda.
            nn.init.normal_(final_linear.weight, mean=0.0, std=0.0001)
        else:
            logging.warning("No se pudo inicializar bias: La capa -2 no es Linear.")
        # END: ESTRATEGIA COLD START INITIALIZATION
        # =====================================================================

        # b_encoder: Recibe demanda escalada -> Salida hy normalizada (use_norm=True).
        self.b_encoder = PaperMLP(num_od_pairs, hidden_dim_from_od, feature_dim, dropout_from_od, use_norm=True)

        self.matcher = GraphMatcher(feature_dim, num_structures)

        # 3. VALIDADOR FÍSICO (AssignmentValidator)
        # Sigue trabajando con trips_scaler para sus cálculos internos de costo.
        convergence_cfg = kwargs.get('msa_convergence', None)

        self.validator = AssignmentValidator(
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
            msa_convergence=convergence_cfg
        )

    def forward(
            self,
            observed_flows: torch.Tensor,
            flow_mask: torch.Tensor,
            true_od_demand: Optional[torch.Tensor] = None,
            od_mask: Optional[torch.Tensor] = None,
            warmup: bool = False,
            current_iter_count: Optional[int] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Manejo de dimensiones de lote (batch)
        is_batched = observed_flows.dim() == 2
        if not is_batched:
            observed_flows = observed_flows.unsqueeze(0)
            flow_mask = flow_mask.unsqueeze(0)
            if true_od_demand is not None: true_od_demand = true_od_demand.unsqueeze(0)
            if od_mask is not None: od_mask = od_mask.unsqueeze(0)

        # --- PASO 1: ENTRADA ESCALADA PARA RED NEURONAL ---
        # Escalamos por link_scale para que el encoder no reciba valores masivos.
        hx = self.f_encoder((observed_flows / self.link_scale) * flow_mask)
        gx = self.matcher.apply(hx)
        raw_normalized_od = self.f_decoder(gx)
        od_hat = raw_normalized_od * self.od_scale  # Escalamos de vuelta a escala real

        # --- PASO 2: HIBRIDACIÓN Y CALIBRACIÓN DEL MATCHER ---
        if self.training and true_od_demand is not None and od_mask is not None:
            # Hibridación para el ciclo backward (Escala Real)
            true_od_normalized = true_od_demand / self.od_scale
            demand_hybrid_norm = (true_od_normalized * od_mask) + (raw_normalized_od.detach() * (1.0 - od_mask))

            # El b_encoder recibe la demanda ESCALADA por od_scale para producir hy normalizado
            hy_ref = self.b_encoder(demand_hybrid_norm)
            self.matcher.update(hx.detach(), hy_ref.detach())

            # Demanda con gradiente para el SUE (Escala Real)
            demand_for_sue = od_hat

        else:
            demand_for_sue = od_hat

        # --- PASO 3: VALIDACIÓN FÍSICA (SUE) ---
        # El SUE recibe demanda en ESCALA REAL (vehículos).
        reconstructed_flows, learned_alpha, learned_beta, conv_info, route_probs = self.validator(
            demand_for_sue,
            warmup=warmup,
            override_max_iters=current_iter_count
        )

        if not is_batched:
            od_hat = od_hat.squeeze(0)
            reconstructed_flows = reconstructed_flows.squeeze(0)
            if route_probs is not None: route_probs = route_probs.squeeze(0)

        return {
            "estimated_demand": od_hat,
            "reconstructed_flows": reconstructed_flows,  # Flujos en Escala Real
            "convergence_info": conv_info,
            "learned_alpha": learned_alpha,
            "learned_beta": learned_beta,
            "route_probs": route_probs,
            "hx": hx,  # hx está normalizado por f_encoder
            "gx": gx  # gx está normalizado por el matcher
        }

# =============================================================================
# 3. FUNCIÓN DE PÉRDIDA COMPATIBLE CON TRAINER.PY
# =============================================================================

class Loss(nn.Module):
    def __init__(self,
                 link_scale: float = 1.0,
                 od_scale: float = 1.0,
                 w_flow: float = 1.0,
                 w_od: float = 1.0,
                 w_demand_reg: float = 1e-6,
                 t0_costs: Optional[torch.Tensor] = None):
        super().__init__()
        # Registramos los escaladores para asegurar que los errores estén en el rango [0, 1]
        self.register_buffer("link_scale", torch.tensor(float(link_scale)))
        self.register_buffer("od_scale", torch.tensor(float(od_scale)))

        self.register_buffer("w_flow", torch.tensor(float(w_flow)))
        self.register_buffer("w_od", torch.tensor(float(w_od)))
        self.register_buffer("w_demand_reg", torch.tensor(float(w_demand_reg)))

        self.mse = nn.MSELoss(reduction="none")

        ### TODO
        # PRIOR GRAVITACIONAL (El Ancla)
        # Si no se provee, usamos un prior uniforme para evitar el cero absoluto
        if t0_costs is not None:
            # Opción más robusta numéricamente que 1/x^2
            # Normalizamos costos para evitar exponentes gigantes
            norm_costs = t0_costs / (t0_costs.mean() + 1e-6)
            gravity = torch.exp(-2.0 * norm_costs)  # Beta aprox 2.0
            self.register_buffer("prior_demand", gravity / gravity.mean())
        else:
            self.register_buffer("prior_demand", None)
        ### TODO

    def forward(
            self,
            predicted_flows: torch.Tensor,
            true_flows: torch.Tensor,
            flow_mask: torch.Tensor,
            predicted_od: torch.Tensor,
            true_od: torch.Tensor,
            od_mask: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        link_scale, od_scale = self.link_scale,  self.od_scale
        scale_loss = True
        if not scale_loss:
            link_scale = torch.tensor(1.0, device=predicted_flows.device)
            od_scale = torch.tensor(1.0, device=predicted_od.device)

        # --- 1. NORMALIZACIÓN DE ERROR DE FLUJO ---
        # Escalamos ambos tensores antes del MSE para que el error sea relativo a la capacidad.
        # Loss = MSE(f_pred / scale, f_true / scale)
        scaled_pred_flow = predicted_flows / link_scale
        scaled_true_flow = true_flows / link_scale

        l_flow_raw = self.mse(scaled_pred_flow, scaled_true_flow)
        l_flow = (l_flow_raw * flow_mask).sum() / flow_mask.sum().clamp(min=1.0)

        # --- 2. NORMALIZACIÓN DE ERROR DE OD ---
        # Hacemos lo mismo con la demanda para que el gradiente sea comparable al de flujo.
        if od_mask.any():
            scaled_pred_od = predicted_od / od_scale
            scaled_true_od = true_od / od_scale
            l_od_raw = self.mse(scaled_pred_od, scaled_true_od)
            l_od = (l_od_raw * od_mask).sum() / od_mask.sum().clamp(min=1.0)
        else:
            l_od = torch.tensor(0.0, device=predicted_od.device)

        # --- 3. REGULARIZACIÓN DE DEMANDA DESCONOCIDA ---
        # La regularización también debe ocurrir en el espacio escalado.
        unknown_mask = 1.0 - od_mask
        # l_demand_reg = ((predicted_od / self.od_scale) * unknown_mask).pow(2).mean() TODO

        # Normalizamos la predicción para compararla con el prior
        pred_norm = predicted_od / od_scale

        if self.prior_demand is not None:
            # Opción A: Gravity Regularization (Evita que mueran en 0 y evita explosiones locas)
            # Empujamos la demanda desconocida hacia el modelo de gravedad, no hacia cero.
            target = self.prior_demand

            # Ajustamos la magnitud del target para que coincida con la predicción actual
            # (queremos copiar la FORMA del gravity model, no necesariamente su magnitud exacta)
            with torch.no_grad():
                scale_factor = pred_norm.mean() / target.mean().clamp(min=1e-6)
                target_scaled = target * scale_factor

            l_demand_reg = (self.mse(pred_norm, target_scaled) * unknown_mask).sum() / unknown_mask.sum().clamp(min=1.0)
        else:
            # Opción B (Fallback): Regularización L2 Suave (hacia la media, no hacia cero)
            # Penalizamos desviarse de la media del batch actual
            batch_mean = pred_norm.detach().mean()
            l_demand_reg = ((pred_norm - batch_mean) * unknown_mask).pow(2).mean()

        # Pérdida Total Balanceada
        total = (self.w_flow * l_flow) + (self.w_od * l_od) + (self.w_demand_reg * l_demand_reg)

        return {
            "total_loss": total,
            "l_flow": l_flow,
            "l_od": l_od,
            "l_demand_reg": l_demand_reg,
            "flow_coverage": flow_mask.float().mean().item(),
            "od_coverage": od_mask.float().mean().item()
        }

##%

import torch
import matplotlib.pyplot as plt
from collections import deque, defaultdict


class CGAMEDiagnosticTool:
    def __init__(self, window_size=50):
        self.window_size = window_size

        # --- 1. Historial de Series Temporales (Moving Averages / Trends) ---
        self.history = {
            # Estructurales (Originales)
            'latent_sim': [],
            'v_entropy': [],
            'hx_norm_ma': deque(maxlen=window_size),
            's_iters': [],
            'route_entropy': [],

            # Diagnóstico 1: Loss Dynamics
            'loss_flow': [],
            'loss_od': [],
            'loss_reg': [],

            # Diagnóstico 4: Salud del Gradiente
            'grad_enc': [],
            'grad_match': [],
            'grad_dec': [],

            # Diagnóstico 5: GRADIENTE
            'grad_norm': defaultdict(list),  # Potencia bruta del gradiente
            'update_ratio': defaultdict(list),  # Relación (Gradiente / Peso)
            'sparsity': defaultdict(list),  # % de Gradientes Cero (Neuronas muertas)
            'flow_health': [],              # Ratio Decoder/Encoder (Global)

        }

        # --- 2. Datos Acumulados para Distribuciones (Originales) ---
        self.dist_data = {
            'gx_vals': [],
            'gy_vals': [],
            'v_weights': []
        }

        # --- 3. Datos Snapshot (Para gráficos pesados: Residuals y OD Check) ---
        # Solo guardamos el último estado para no explotar la memoria
        self.snap_data = {
            'true_flows': None,
            'pred_flows': None,
            'true_od_masked': None,
            'pred_od_all': None
        }

        # --- NUEVO: EVOLUTION TRACKING (Spectral + Sentinels) ---
        # Definimos bins fijos logarítmicos para asegurar consistencia temporal
        # Rango: 0.1 a 50,000 (Ajustable según tu escala)
        self.evolution_bins = np.geomspace(0.1, 10000, 50)

        # Estructura para guardar: 'counts' (Matriz de densidad) y 'traces' (Líneas individuales)
        self.evo_data = {
            'flow_known': {'counts': [], 'traces': [], 'indices': None, 'label': 'Flow (Known)'},
            'flow_unknown': {'counts': [], 'traces': [], 'indices': None, 'label': 'Flow (Unknown)'},
            'od_known': {'counts': [], 'traces': [], 'indices': None, 'label': 'OD (Known)'},
            'od_unknown': {'counts': [], 'traces': [], 'indices': None, 'label': 'OD (Unknown)'}
        }

        # =====================================================================
        # NUEVO: PHYSICS TRACKING (V/C & VDF)
        # =====================================================================

        # 1. Congestión (V/C)
        # Guardaremos percentiles [25, 50, 75, 90, 99] por época
        # Estructura: self.vc_stats['Motorway'] = [[p25, p50... epoch0], [p25... epoch1]]
        self.vc_stats = defaultdict(list)
        self.link_types_map = None  # Se llenará en la primera iteración

        # 2. VDF Parameters
        # Estructura: self.vdf_history[group_id] = {'alpha': [], 'beta': []}
        self.vdf_history = defaultdict(lambda: {'alpha': [], 'beta': []})

        # =====================
        # BEGIN: GRADIENT ANALYSIS SETUP
        # =====================
        self.grad_history = defaultdict(lambda: defaultdict(list))
        self.epoch_indices = []
        # END: GRADIENT ANALYSIS SETUP
        # =========================

    @torch.no_grad()
    def update(self, outputs, model, loss_dict=None, targets=None, static_info=None):
        """
        Captures metrics from a single forward/backward pass.

        Args:
            outputs: Diccionario retornado por model()
            model: La instancia del modelo (para pesos y gradientes)
            loss_dict: (NUEVO) Diccionario retornado por criterion()
            targets: (NUEVO) Diccionario con {flows, mask, od, od_mask}
        """
        # =====================================================================
        # A. MÉTRICAS ORIGINALES (Latent, Structure, SUE)
        # =====================================================================
        gx = outputs.get('gx')
        hx = outputs.get('hx')
        est_demand = outputs.get('estimated_demand')

        # Validación de rangos
        if gx is not None and gx.dim() == 1: gx = gx.unsqueeze(0)
        if hx is not None and hx.dim() == 1: hx = hx.unsqueeze(0)
        if est_demand is not None and est_demand.dim() == 1: est_demand = est_demand.unsqueeze(0)

        # 1. Latent Alignment
        if est_demand is not None:
            # Replicar lógica del matcher para obtener gy
            est_demand_norm = est_demand / model.od_scale
            hy_pred = model.b_encoder(est_demand_norm)
            if hy_pred.dim() == 1: hy_pred = hy_pred.unsqueeze(0)
            gy_pred = model.matcher.apply(hy_pred)

            if gx is not None:
                cos_sim = torch.nn.functional.cosine_similarity(gx, gy_pred, dim=-1).mean()
                self.history['latent_sim'].append(cos_sim.item())

                # Samplear para histogramas (1 de cada 10 valores para ahorrar memoria)
                self.dist_data['gx_vals'].extend(gx.flatten()[::10].cpu().numpy())
                self.dist_data['gy_vals'].extend(gy_pred.flatten()[::10].cpu().numpy())

        # 2. Estabilidad de V
        V = model.matcher.V
        v_prob = torch.softmax(V, dim=-1)
        v_ent = -(v_prob * torch.log(v_prob + 1e-9)).sum()
        self.history['v_entropy'].append(v_ent.item())
        self.dist_data['v_weights'].extend(V.flatten().cpu().numpy())

        # 3. Estabilidad hx
        if hx is not None:
            self.history['hx_norm_ma'].append(torch.norm(hx, p=2, dim=-1).mean().item())

        # 4. SUE Iters & Route Entropy
        conv_info = outputs.get('convergence_info', {})
        self.history['s_iters'].append(conv_info.get('iterations', 0))

        route_probs = outputs.get('route_probs')
        if route_probs is not None:
            if route_probs.dim() == 2: route_probs = route_probs.unsqueeze(0)
            r_ent = -(route_probs * torch.log(route_probs + 1e-9)).sum(dim=-1).mean()
            self.history['route_entropy'].append(r_ent.item())

        # =====================================================================
        # B. DIAGNÓSTICO 1: LOSS DYNAMICS
        # =====================================================================
        if loss_dict is not None:
            self.history['loss_flow'].append(
                loss_dict.get('l_flow', 0).item() if hasattr(loss_dict.get('l_flow'), 'item') else 0)
            self.history['loss_od'].append(
                loss_dict.get('l_od', 0).item() if hasattr(loss_dict.get('l_od'), 'item') else 0)
            self.history['loss_reg'].append(
                loss_dict.get('l_demand_reg', 0).item() if hasattr(loss_dict.get('l_demand_reg'), 'item') else 0)

        # =====================================================================
        # C. DIAGNÓSTICO 4: SALUD DEL GRADIENTE (Gradient Norms)
        # =====================================================================
        # Nota: Esto debe ejecutarse después de loss.backward()

        def compute_grad_norm(module):
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()
            return total_norm

        self.history['grad_enc'].append(compute_grad_norm(model.f_encoder))
        self.history['grad_match'].append(compute_grad_norm(model.matcher))
        self.history['grad_dec'].append(compute_grad_norm(model.f_decoder))

        # =====================================================================
        # D. SNAPSHOT DATA (Para Diag 2 y 3)
        # =====================================================================
        if targets is not None:
            # Snapshot para Residuals (Diag 3)
            # Guardamos flows solo donde hay máscara
            mask = targets['mask'].bool()
            y_true = targets['flows'][mask].detach().cpu().numpy()
            y_pred = outputs['reconstructed_flows'][mask].detach().cpu().numpy()

            self.snap_data['true_flows'] = y_true
            self.snap_data['pred_flows'] = y_pred

            # Snapshot para OD Distribution (Diag 2)
            # Guardamos OD real (solo máscara) y Predicho (todo)
            od_mask = targets['od_mask'].bool()
            if targets.get('od') is not None:
                self.snap_data['true_od_masked'] = targets['od'][od_mask].detach().cpu().numpy()

            if est_demand is not None:
                self.snap_data['pred_od_all'] = est_demand.detach().cpu().numpy().flatten()

        # =====================================================================
        # E. EVOLUTION TRACKING (Spectral Density + Sentinels)
        # =====================================================================
        if targets is not None and outputs.get('reconstructed_flows') is not None:
            # 1. Preparar Datos Crudos
            # Flujos
            pred_flows = outputs['reconstructed_flows'].detach().flatten()
            mask_flow = targets['mask'].detach().flatten().bool()

            # OD (Si existe estimación)
            pred_od = outputs['estimated_demand'].detach().flatten() if outputs.get(
                'estimated_demand') is not None else None
            mask_od = targets['od_mask'].detach().flatten().bool() if targets.get(
                'od_mask') is not None else None

            # 2. Definir Grupos
            groups = {}

            # Flow Known vs Unknown
            if len(pred_flows) == len(mask_flow):
                groups['flow_known'] = pred_flows[mask_flow].cpu().numpy()
                groups['flow_unknown'] = pred_flows[~mask_flow].cpu().numpy()

            # OD Known vs Unknown
            if pred_od is not None and mask_od is not None:
                groups['od_known'] = pred_od[mask_od].cpu().numpy()
                groups['od_unknown'] = pred_od[~mask_od].cpu().numpy()

            # 3. Procesar cada grupo (Binning + Sentinel Selection)
            for key, data_array in groups.items():
                if len(data_array) == 0: continue

                # A. Spectral Density (Histograma del momento)
                # Usamos los bins fijos definidos en __init__
                counts, _ = np.histogram(data_array, bins=self.evolution_bins)
                self.evo_data[key]['counts'].append(counts)

                # B. Sentinel Traces (Seleccionar índices solo en la primera vez)
                if self.evo_data[key]['indices'] is None:
                    # Estrategia de Selección: 1 Max, 1 Median, 1 Random
                    n_samples = len(data_array)
                    if n_samples > 0:
                        idx_max = np.argmax(data_array)
                        idx_med = np.argsort(data_array)[n_samples // 2]
                        idx_rnd = np.random.randint(0, n_samples)
                        # Guardamos los índices relativos a este subgrupo
                        self.evo_data[key]['indices'] = [idx_max, idx_med, idx_rnd]
                    else:
                        self.evo_data[key]['indices'] = []

                # Guardar los valores de los centinelas en esta época
                current_indices = self.evo_data[key]['indices']
                if current_indices:
                    sentinel_values = data_array[current_indices]
                    self.evo_data[key]['traces'].append(sentinel_values)

        # =====================================================================
        # F. PHYSICS TRACKING (V/C & VDF)
        # =====================================================================

        # 1. Configuración Estática (Solo primera vez)
        if self.link_types_map is None and static_info is not None:
            self.link_types_map = static_info.get('link_types')  # Array [N_Links] con strings tipo "Motorway"
            self.capacity_ref = static_info.get('capacity')  # Tensor/Array [N_Links]
            if isinstance(self.capacity_ref, torch.Tensor):
                self.capacity_ref = self.capacity_ref.detach().cpu().numpy()

        # 2. Calcular V/C Ratios del Batch Actual
        if self.link_types_map is not None and outputs.get('reconstructed_flows') is not None:
            flows = outputs['reconstructed_flows'].detach().cpu().numpy().flatten()

            # Evitar división por cero
            caps = self.capacity_ref
            vc_ratios = np.divide(flows, caps, out=np.zeros_like(flows), where=caps > 0.1)

            # A. Global V/C Stats
            pcts = np.percentile(vc_ratios, [25, 50, 75, 90, 99])
            self.vc_stats['Global'].append(pcts)

            # B. Disaggregated V/C Stats (Por Link Type)
            unique_types = np.unique(self.link_types_map)
            for l_type in unique_types:
                mask = (self.link_types_map == l_type)
                if np.any(mask):
                    vals = vc_ratios[mask]
                    type_pcts = np.percentile(vals, [25, 50, 75, 90, 99])
                    self.vc_stats[str(l_type)].append(type_pcts)

        # 3. VDF Parameters Tracking
        # Intentamos recuperar alphas/betas del output (si el modelo los retorna)
        # o del modelo directamente si son accesibles
        alphas = outputs.get('learned_alpha')
        betas = outputs.get('learned_beta')

        if alphas is not None and betas is not None:
            # Asumimos que alphas es [Num_Link_Groups] o escalar
            if isinstance(alphas, torch.Tensor): alphas = alphas.detach().cpu().numpy().flatten()
            if isinstance(betas, torch.Tensor): betas = betas.detach().cpu().numpy().flatten()

            # Si es escalar, convertir a lista
            if np.ndim(alphas) == 0: alphas = [alphas]
            if np.ndim(betas) == 0: betas = [betas]

            for i, (a, b) in enumerate(zip(alphas, betas)):
                self.vdf_history[i]['alpha'].append(a)
                self.vdf_history[i]['beta'].append(b)

    def finalize_and_plot(self, save_path="cgame_diagnostics_advanced.png"):
        """Generates a comprehensive diagnostic dashboard (3x4 Grid)."""

        # Aumentamos tamaño para acomodar 12 subplots
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        axes = axes.flatten()  # Facilita indexar 0-11

        # --- ROW 1: DINÁMICA DE APRENDIZAJE ---

        # 1. Loss Dynamics (Log Scale) [Diag 1]
        ax = axes[0]
        ax.plot(self.history['loss_flow'], label='L_Flow', alpha=0.7)
        ax.plot(self.history['loss_od'], label='L_OD', alpha=0.7)
        ax.plot(self.history['loss_reg'], label='L_Reg', alpha=0.7, linestyle='--')
        ax.set_yscale('log')
        ax.set_title("Loss Components (Log Scale)")
        ax.legend()

        # 2. Gradient Norms [Diag 4]
        ax = axes[1]
        ax.plot(self.history['grad_enc'], label='Enc', color='blue', linewidth=1)
        ax.plot(self.history['grad_match'], label='Match', color='purple', linewidth=1)
        ax.plot(self.history['grad_dec'], label='Dec', color='red', linewidth=1)
        ax.set_title("Gradient Norms (L2)")
        ax.set_yscale('log')
        ax.legend()

        # 3. SUE Iterations
        ax = axes[2]
        ax.plot(self.history['s_iters'], color='green', alpha=0.6)
        ax.set_title("SUE MSA Iterations")

        # 4. Latent Alignment Trend
        ax = axes[3]
        ax.plot(self.history['latent_sim'], color='blue')
        ax.set_title("Latent Cosine Sim ($g_x, g_y$)")
        ax.set_ylim(0, 1)

        # --- ROW 2: ESTRUCTURA Y ENTROPÍA ---

        # 5. V Entropy
        ax = axes[4]
        ax.plot(self.history['v_entropy'], color='purple')
        ax.set_title("Matcher V Entropy")

        # 6. hx Norm
        ax = axes[5]
        ax.plot(list(self.history['hx_norm_ma']), color='red')
        ax.set_title(f"Mean $h_x$ Norm (MA)")

        # 7. Route Entropy
        ax = axes[6]
        ax.plot(self.history['route_entropy'], color='orange')
        ax.set_title("Route Prob Entropy")

        # 8. V Weights Dist
        ax = axes[7]
        ax.hist(self.dist_data['v_weights'], bins=30, color='orange')
        ax.set_title("V Weights Dist")

        # --- ROW 3: FÍSICA Y REALISMO (NUEVOS) ---

        # 9. Latent Values Dist
        ax = axes[8]
        ax.hist(self.dist_data['gx_vals'], bins=50, alpha=0.5, label='$g_x$', density=True)
        ax.hist(self.dist_data['gy_vals'], bins=50, alpha=0.5, label='$g_y$', density=True)
        ax.set_title("Latent Space Dist ($g_x$ vs $g_y$)")
        ax.legend()

        # 10. OD Realism Check (Log Scale) [Diag 2]
        ax = axes[9]
        if self.snap_data['pred_od_all'] is not None:
            # Plot Predicted (All)
            ax.hist(self.snap_data['pred_od_all'], bins=50, alpha=0.5, label='Pred (All)', color='red', log=True)
            # Plot True (Known)
            if self.snap_data['true_od_masked'] is not None:
                ax.hist(self.snap_data['true_od_masked'], bins=50, alpha=0.5, label='True (Known)', color='blue',
                        log=True)
            ax.set_title("OD Demand Distribution (Log Y)")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

        # 11. Residuals vs Magnitude [Diag 3]
        ax = axes[10]
        if self.snap_data['true_flows'] is not None:
            y_true = self.snap_data['true_flows']
            y_pred = self.snap_data['pred_flows']
            residuals = y_pred - y_true

            # Scatter Plot
            ax.scatter(y_true, residuals, alpha=0.3, s=5)
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlabel("True Flow")
            ax.set_ylabel("Residual (Pred - True)")
            ax.set_title("Residuals vs Magnitude")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

        # 12. Empty Slot (Future Use)
        axes[11].axis('off')
        axes[11].text(0.5, 0.5, "CGAME Diagnostic Tool\nv2.0", ha='center', fontsize=12)

        plt.savefig(save_path)
        print(f"Diagnostics saved to {save_path}")

    def plot_evolution(self, save_path="cgame_evolution.png"):
        """
        Generates the Spectral Density Heatmaps with Sentinel Traces.
        Focuses on the temporal evolution of distributions.
        """
        import matplotlib.colors as mcolors
        from matplotlib.ticker import ScalarFormatter

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Definir orden de ploteo
        plot_keys = ['flow_known', 'flow_unknown', 'od_known', 'od_unknown']

        for i, key in enumerate(plot_keys):
            ax = axes[i]
            data_struct = self.evo_data[key]

            # Verificar si hay datos
            if not data_struct['counts']:
                ax.text(0.5, 0.5, "No Data Available", ha='center')
                continue

            # --- 1. PLOT SPECTRAL DENSITY (HEATMAP) ---
            # Convertir lista de conteos a matriz (Epochs x Bins)
            density_matrix = np.array(data_struct['counts']).T  # Transponer para (Y=Bins, X=Epochs)

            # 2. DATA CHECK: Calculate max before converting to NaN
            v_max = np.max(density_matrix)

            # If the matrix is all zeros (common when od_mask is 100% known), 
            # v_max will be 0, which breaks LogNorm.
            if v_max <= 0:
                ax.text(0.5, 0.5, f"No active data for {key}\n(All values are 0 or mask is empty)", 
                        ha='center', fontsize=12, color='gray')
                ax.set_title(data_struct['label'])
                continue

            # Reemplazar ceros con NaN para que el fondo sea transparente/blanco
            density_matrix = density_matrix.astype(float)
            density_matrix[density_matrix == 0] = np.nan

            epochs = np.arange(len(data_struct['counts']))

            # Pcolormesh con escala Logarítmica para el COLOR (Density)
            # Usamos los bins definidos en init para el eje Y
            # 'cmap' sugerido: 'viridis' o 'plasma' (oscuro es poco, brillante es mucho)
            mesh = ax.pcolormesh(
                epochs,
                self.evolution_bins[:-1],  # Eje Y (Bin Edges inferiores)
                density_matrix,
                norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(density_matrix)),
                cmap='magma_r',  # Invertido: Claro=Fondo, Oscuro=Alta Densidad
                shading='auto',
                alpha=0.9
            )

            # --- 2. PLOT SENTINEL TRACES (LINES) ---
            if data_struct['traces']:
                traces_matrix = np.array(data_struct['traces'])  # (Epochs x 3)

                # Plot Max (Red), Median (Blue), Random (Green)
                labels = ['Max (Start)', 'Median (Start)', 'Random']
                colors = ['red', 'blue', 'green']

                for t_idx in range(traces_matrix.shape[1]):
                    ax.plot(epochs, traces_matrix[:, t_idx],
                            color=colors[t_idx % 3],
                            linewidth=1.5,
                            linestyle='--',
                            label=labels[t_idx] if i == 0 else "")  # Leyenda solo en el primero

            # --- FORMATTING ---
            # Aseguramos que los números se vean como enteros naturales
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            formatter.set_useOffset(False)  # Evita el "+1eX" en el eje
            ax.yaxis.set_major_formatter(formatter)

            ax.set_title(data_struct['label'])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Magnitude (Linear Scale)")
            ax.grid(True, ls="-", alpha=0.2)

            # Añadir colorbar pequeño
            cbar = plt.colorbar(mesh, ax=ax)
            cbar.set_label('Density (Count)')

        # Leyenda global para trazas
        fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))

        plt.suptitle("Temporal Evolution: Spectral Density & Sentinel Traces", fontsize=16, y=0.98)
        plt.savefig(save_path)
        print(f"Evolution diagnostics saved to {save_path}")


    def plot_physics(self, save_path="cgame_physics.png"):
        """
        Genera el dashboard físico: V/C Analysis y VDF Evolution.
        """
        import matplotlib.pyplot as plt

        # Determinar layout dinámico
        # Fila 1: V/C General + VDF Params
        # Filas siguientes: Small Multiples para V/C Types

        link_types = [k for k in self.vc_stats.keys() if k != 'Global']
        num_types = len(link_types)

        # Configurar figura:
        # Arriba: 3 columnas (V/C Global, Alpha Evol, Beta Evol)
        # Abajo: Grid para Link Types (ej. 3 columnas x N filas)

        rows_needed_types = (num_types + 2) // 3
        total_rows = 2 + rows_needed_types  # 1 fila resumen, 1 fila pain curves, N filas tipos

        fig = plt.figure(figsize=(20, 5 * total_rows))

        # --- ROW 1: RESUMEN GENERAL ---
        # 1. V/C Global
        ax1 = plt.subplot2grid((total_rows, 3), (0, 0))
        self._plot_vc_bands(ax1, self.vc_stats['Global'], "Global Network Congestion")

        # 2. VDF Parameters Evolution
        ax2 = plt.subplot2grid((total_rows, 3), (0, 1))
        for grp_id, hist in self.vdf_history.items():
            ax2.plot(hist['alpha'], label=f'Grp {grp_id}')
        ax2.set_title("VDF Alpha Evolution")
        ax2.set_xlabel("Epochs")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot2grid((total_rows, 3), (0, 2))
        for grp_id, hist in self.vdf_history.items():
            ax3.plot(hist['beta'], label=f'Grp {grp_id}')
        ax3.set_title("VDF Beta Evolution")
        ax3.set_xlabel("Epochs")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- ROW 2: PAIN CURVES (SNAPSHOTS) ---
        # Graficamos la curva de costo en Epoch 0, 50% y 100%
        ax_pain = plt.subplot2grid((total_rows, 3), (1, 0), colspan=3)
        self._plot_pain_curves(ax_pain)

        # --- ROWS 3+: V/C SMALL MULTIPLES ---
        for i, l_type in enumerate(link_types):
            row_idx = 2 + (i // 3)
            col_idx = i % 3
            ax = plt.subplot2grid((total_rows, 3), (row_idx, col_idx))
            self._plot_vc_bands(ax, self.vc_stats[l_type], f"Congestion: {l_type}")

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Physics diagnostics saved to {save_path}")

    def _plot_vc_bands(self, ax, stats_list, title):
        """Helper para pintar bandas de percentiles."""
        data = np.array(stats_list)  # [Epochs, 5] -> [25, 50, 75, 90, 99]
        epochs = np.arange(len(data))

        # P50 (Mediana)
        ax.plot(epochs, data[:, 1], color='black', linewidth=1.5, label='Median')

        # Banda IQR (25-75) - Zona Típica
        ax.fill_between(epochs, data[:, 0], data[:, 2], color='blue', alpha=0.2, label='IQR (25-75%)')

        # Banda Extrema (90-99) - Cuellos de Botella
        ax.fill_between(epochs, data[:, 3], data[:, 4], color='red', alpha=0.15, label='Stress (90-99%)')

        # Referencia Capacidad
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_title(title)
        ax.set_ylim(0, max(2.0, np.max(data) * 1.1))  # Limitar Y para ver detalle, pero permitir picos
        ax.set_ylabel("V/C Ratio")
        ax.grid(True, alpha=0.3)
        if "Global" in title: ax.legend(loc='upper left')

    def _plot_pain_curves(self, ax):
        """Dibuja la curva BPR en diferentes momentos del entrenamiento."""
        x = np.linspace(0, 1.5, 100)  # V/C de 0 a 1.5

        # Seleccionar snapshots: Inicio, Medio, Final
        epochs_recorded = len(self.vdf_history[0]['alpha'])
        if epochs_recorded < 2: return

        snapshots = [0, epochs_recorded // 2, epochs_recorded - 1]
        styles = [':', '--', '-']
        labels = ['Start', 'Mid', 'End']

        for grp_id, hist in self.vdf_history.items():
            # Solo graficar el primer grupo para no saturar, o iterar colores
            color = f"C{grp_id}"

            for shot, style, lbl in zip(snapshots, styles, labels):
                if shot < len(hist['alpha']):
                    a = hist['alpha'][shot]
                    b = hist['beta'][shot]

                    # BPR Formula: 1 + alpha * (x)^beta
                    y = 1.0 + a * np.power(x, b)

                    ax.plot(x, y, linestyle=style, color=color, linewidth=2,
                            label=f"Grp{grp_id} ({lbl}): a={a:.2f}, b={b:.2f}")

        ax.set_title("Evolution of Cost Function (Pain Curve)")
        ax.set_xlabel("V/C Ratio")
        ax.set_ylabel("Travel Time Multiplier (T/t0)")
        ax.axvline(1.0, color='k', linestyle='-', alpha=0.1)
        ax.grid(True)
        ax.legend()


    ### -----------------------------------
    # BEGIN: GRADIENT ANALYSIS

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    def _identify_block(self, param_name):
        """Clasifica los parámetros en órganos vitales."""
        if "f_encoder" in param_name: return "Forward Encoder"
        if "b_encoder" in param_name: return "Backward Encoder"
        if "matcher" in param_name: return "Graph Matcher"
        if "decoder" in param_name: return "Decoder"
        if "validator" in param_name: return "SUE Validator"  # Si tiene params aprendibles
        return "Other"

    def capture_gradient_history(self, model, epoch):
        """
        Toma una biopsia completa de los gradientes.
        Llamar justo después de loss.backward() y antes de optimizer.step().
        """
        self.epoch_indices.append(epoch)

        # Contenedores temporales para esta epoch
        block_norms = defaultdict(list)
        block_ratios = defaultdict(list)
        block_sparsity = defaultdict(list)

        for name, param in model.named_parameters():
            if param.grad is not None:
                block = self._identify_block(name)

                # 1. Norma del Gradiente (Potencia)
                g_norm = param.grad.data.norm(2).item()
                block_norms[block].append(g_norm)

                # 2. Ratio Peso/Actualización (Estabilidad)
                # ¿Qué tan grande es el empuje comparado con el tamaño del peso?
                p_norm = param.data.norm(2).item()
                if p_norm > 1e-9:
                    ratio = g_norm / p_norm
                else:
                    ratio = 0.0
                block_ratios[block].append(ratio)

                # 3. Sparsity (Necrosis)
                # Porcentaje de elementos en el tensor de gradiente que son exactamente cero
                n_zeros = torch.sum(param.grad.data == 0).item()
                n_elements = param.grad.data.numel()
                sparsity = (n_zeros / n_elements) * 100.0
                block_sparsity[block].append(sparsity)

        # Agregación por bloque (Promedio)
        target_blocks = ["Forward Encoder", "Graph Matcher", "Decoder", "Backward Encoder"]

        # Variables para flujo global
        norm_encoder = 0.0
        norm_decoder = 0.0

        for block in target_blocks:
            if block in block_norms and block_norms[block]:
                # Promedios
                avg_norm = np.mean(block_norms[block])
                avg_ratio = np.mean(block_ratios[block])
                avg_sparsity = np.mean(block_sparsity[block])

                # Guardar para el bloque específico
                self.history['grad_norm'][block].append(avg_norm)
                self.history['update_ratio'][block].append(avg_ratio)
                self.history['sparsity'][block].append(avg_sparsity)

                # Capturar valores clave para el diagnóstico de flujo
                if block == "Forward Encoder": norm_encoder = avg_norm
                if block == "Decoder": norm_decoder = avg_norm

            else:
                # Si el bloque no tiene params con gradiente (ej. Matcher sin pesos)
                # No guardamos ceros para no ensuciar las escalas logarítmicas
                pass

        # Calcular Salud del Flujo (Decoder -> Encoder)
        # Si ratio ~ 1.0, el gradiente pasa perfecto. Si ratio ~ 0.0, muere en el camino.
        if norm_decoder > 1e-9:
            flow_health = norm_encoder / norm_decoder
        else:
            flow_health = 0.0
        self.history['flow_health'].append(flow_health)

    def plot_gradient_health(self, save_path="gradient_health_report.png"):
        """
        Genera un panel de control de 3 niveles con diagnóstico textual.
        """
        if not self.epoch_indices:
            print("No hay datos de gradiente para graficar.")
            return

        epochs = self.epoch_indices
        blocks = ["Forward Encoder", "Graph Matcher", "Decoder"]
        colors = {"Forward Encoder": "#1f77b4", "Graph Matcher": "#ff7f0e", "Decoder": "#2ca02c",
                  "Backward Encoder": "#9467bd"}

        # Configurar figura: 3 filas
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        plt.subplots_adjust(bottom=0.15, hspace=0.3)

        # --- 1. Potencia (Norma L2) ---
        for block in blocks:
            if block in self.history['grad_norm']:
                ax1.plot(epochs, self.history['grad_norm'][block], label=block, color=colors.get(block, 'k'))
        ax1.set_yscale('log')
        ax1.set_title('A. Potencia del Gradiente (Norma L2) [Log Scale]')
        ax1.set_ylabel('||Grad||')
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc='upper right')

        # --- 2. Estabilidad (Update Ratio) ---
        for block in blocks:
            if block in self.history['update_ratio']:
                data = self.history['update_ratio'][block]
                ax2.plot(epochs, data, label=block, color=colors.get(block, 'k'))

        # Zonas de referencia
        ax2.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5, label='Healthy Limit')
        ax2.axhline(y=1e-5, color='red', linestyle=':', alpha=0.5, label='Frozen Limit')

        ax2.set_yscale('log')
        ax2.set_title('B. Relación Peso/Actualización (||Grad|| / ||Weight||)')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, which="both", alpha=0.3)

        # --- 3. Necrosis (Sparsity) ---
        for block in blocks:
            if block in self.history['sparsity']:
                ax3.plot(epochs, self.history['sparsity'][block], label=block, color=colors.get(block, 'k'))

        ax3.set_title('C. Necrosis Neuronal (% Gradientes Cero)')
        ax3.set_ylabel('% Muerto')
        ax3.set_xlabel('Epochs')
        ax3.set_ylim(-5, 105)
        ax3.grid(True, alpha=0.3)

        # --- DIAGNÓSTICO TEXTUAL ---
        report = self._generate_detailed_diagnosis()
        fig.text(0.5, 0.02, report, ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#333"),
                 family='monospace')

        # Guardar
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reporte forense guardado en: {save_path}")

    def _generate_diagnosis_text(self):
        """Analiza los últimos valores y redacta un diagnóstico."""
        diagnosis = ["--- REPORTE FORENSE DE GRADIENTES (ÚLTIMA EPOCH) ---"]

        # Obtener últimos valores
        last_norms = {b: self.grad_history['norm'][b][-1] if self.grad_history['norm'][b] else 0
                      for b in ["Forward Encoder", "Graph Matcher", "Decoder"]}

        # 1. Chequeo de Vanishing Gradient Global
        decoder_grad = last_norms["Decoder"]
        encoder_grad = last_norms["Forward Encoder"]

        if decoder_grad < 1e-7:
            diagnosis.append("CRÍTICO: Muerte Cerebral. El gradiente no llega ni al Decoder.")
            diagnosis.append("   -> Posible Causa: Loss Function desconectada o parámetros congelados.")

        # 2. Chequeo de Bloqueo en el Matcher (Cuello de Botella)
        elif decoder_grad > 1e-4 and encoder_grad < 1e-7:
            diagnosis.append("BLOQUEO: El Matcher está actuando como un tapón.")
            diagnosis.append(
                f"   -> Decoder recibe señal ({decoder_grad:.2e}), pero Encoder no ({encoder_grad:.2e}).")
            diagnosis.append("   -> El gradiente muere al intentar cruzar la atención o el SUE.")

        # 3. Chequeo de Gradiente Sano
        elif encoder_grad > 1e-5:
            diagnosis.append("SALUDABLE: El flujo de gradiente recorre toda la red.")

        # 4. Chequeo de Explosión
        if any(v > 100 for v in last_norms.values()):
            diagnosis.append("PELIGRO: Explosión de Gradiente detectada (>100).")
            diagnosis.append("   -> Recomendación: Aumentar Clipping o reducir Learning Rate.")

        return "\n".join(diagnosis)

    def _generate_detailed_diagnosis(self):
        """Analiza los datos acumulados para dar un veredicto."""
        try:
            last_flow = self.history['flow_health'][-1]
            last_decoder = self.history['grad_norm']['Decoder'][-1] if self.history['grad_norm']['Decoder'] else 0
            last_encoder = self.history['grad_norm']['Forward Encoder'][-1] if self.history['grad_norm'][
                'Forward Encoder'] else 0

            lines = ["--- DIAGNÓSTICO AUTOMÁTICO ---"]

            # 1. Chequeo de Salud Global (Decoder)
            if last_decoder < 1e-6:
                lines.append("PARO CARDÍACO: Gradiente en Decoder casi nulo.")
                lines.append("   -> Causa: Loss Function mal escalada o meseta plana.")
            elif last_decoder > 1000:
                lines.append("EXPLOSIÓN: Gradientes gigantescos en Decoder.")
                lines.append("   -> Acción: Activar 'scale_loss=True' y revisar 'link_scale'.")
            else:
                lines.append(f"Entrada Gradiente: Saludable ({last_decoder:.2e})")

            # 2. Chequeo de Flujo (Cuello de Botella)
            if last_decoder > 1e-5:
                if last_flow < 0.01:
                    lines.append("OBSTRUCCIÓN SEVERA: El gradiente no cruza al Encoder.")
                    lines.append(f"   -> Solo llega el {last_flow * 100:.4f}% de la señal.")
                    lines.append("   -> Culpable probable: Graph Matcher o SUE (Jacobiana cero).")
                elif last_flow < 0.5:
                    lines.append("PÉRDIDA PARCIAL: El gradiente se debilita en el camino.")
                else:
                    lines.append("FLUJO LIBRE: El gradiente viaja correctamente.")

            # 3. Chequeo de Estancamiento (Ratio)
            enc_ratio = self.history['update_ratio']['Forward Encoder'][-1] if self.history['update_ratio'][
                'Forward Encoder'] else 0
            if enc_ratio < 1e-6 and last_encoder > 0:
                lines.append("CONGELAMIENTO: Gradiente existe pero es muy débil para mover pesos.")
                lines.append("   -> Acción: Incrementar Learning Rate.")

            return "\n".join(lines)
        except Exception:
            return "Datos insuficientes para diagnóstico."

    # END: GRADIENTE ANALYSIS
    #----------------------------------------------
