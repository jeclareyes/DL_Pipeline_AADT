"""
Modelo Deep Learning para Traffic Assignment con Datos Parciales.

Este modelo extiende la arquitectura cyclic para trabajar con:
- Demandas OD parcialmente conocidas
- Flujos de enlaces parcialmente observados

El modelo aprende a:
1. Completar demandas OD faltantes
2. Estimar flujos en enlaces no observados
3. Respetar restricciones de equilibrio de tráfico

Arquitectura:
- ODEncoder: Codifica flujos observados a espacio latente
- GraphMatcher: Alinea espacio latente con estructura de red
- ODDecoder: Decodifica a demandas OD completas
- AssignmentValidator: Valida equilibrio mediante SUE con función de costo

Funciones de costo soportadas:
- BPR (Bureau of Public Roads)
- Cónica (futuro)
- Akçelik (futuro)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# =============================================================================
# COMPONENTES DEL MODELO
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
        """
        Args:
            flows: [batch_size, num_links] o [num_links]
        Returns:
            Embedding latente [batch_size, feature_dim] o [feature_dim]
        """
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
        """
        Args:
            g_x: [batch_size, feature_dim] o [feature_dim]
        Returns:
            Demandas OD [batch_size, num_od_pairs] o [num_od_pairs]
        """
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
        """
        Args:
            h_x: Embedding de entrada [batch_size, feature_dim]
            h_y: Embedding de referencia (opcional, solo training) [batch_size, feature_dim]
        Returns:
            Embedding transformado [batch_size, feature_dim]
        """
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
# FUNCIONES DE COSTO (Modular)
# =============================================================================

class CostFunction(nn.Module):
    """Clase base para funciones de costo."""

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        """
        Args:
            link_flows: [batch_size, num_links]
        Returns:
            Costos: [batch_size, num_links]
        """
        raise NotImplementedError


class BPRCostFunction(nn.Module):
    """
    Función de costo BPR (Bureau of Public Roads).

    BPR: t(x) = t0 * [1 + alpha * (x/c)^beta]
    """

    def __init__(self, t0: torch.Tensor, capacity: torch.Tensor,
                 num_link_groups: int, link_group: torch.Tensor,
                 learnable_params: bool = True):
        """
        Args:
            t0: Tiempos de flujo libre [num_links]
            capacity: Capacidades de enlaces [num_links]
            num_link_groups: Número de grupos de enlaces
            link_group: Asignación de enlaces a grupos [num_links]
            learnable_params: Si True, alpha y beta son aprendibles
        """
        super().__init__()
        self.register_buffer('t0', t0)
        self.register_buffer('capacity', capacity)
        self.register_buffer('link_group', link_group.to(torch.long))
        self.learnable_params = learnable_params

        if learnable_params:
            # Parámetros aprendibles por grupo
            self.alpha_raw = nn.Parameter(torch.full((num_link_groups,), 0.15))
            self.beta_raw = nn.Parameter(torch.full((num_link_groups,), 4.0))
        else:
            # Parámetros fijos
            self.register_buffer('alpha_raw', torch.full((num_link_groups,), 0.15))
            self.register_buffer('beta_raw', torch.full((num_link_groups,), 4.0))

    def get_alpha(self) -> torch.Tensor:
        """Obtiene valores de alpha con restricciones."""
        if self.learnable_params:
            return torch.clamp(F.softplus(self.alpha_raw), min=0.01, max=2.0)
        else:
            return self.alpha_raw

    def get_beta(self) -> torch.Tensor:
        """Obtiene valores de beta con restricciones."""
        if self.learnable_params:
            return torch.clamp(1.0 + F.softplus(self.beta_raw), min=1.1, max=10.0)
        else:
            return self.beta_raw

    def forward(self, link_flows: torch.Tensor) -> torch.Tensor:
        """
        Calcula costos BPR.

        Args:
            link_flows: [batch_size, num_links] o [num_links]
        Returns:
            Costos: [batch_size, num_links] o [num_links]
        """
        alpha = self.get_alpha()
        beta = self.get_beta()

        alpha_links = alpha[self.link_group]
        beta_links = beta[self.link_group]

        # Evitar divisiones por cero y valores extremos
        flow_ratio = torch.clamp(link_flows / (self.capacity + 1e-9), max=5.0)
        bpr_cost = self.t0 * (1 + alpha_links * flow_ratio ** beta_links)

        return bpr_cost


class ConicCostFunction(CostFunction):
    """
    Función de costo Cónica (para implementación futura).

    Placeholder para función de costo más realista que BPR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Conic cost function pendiente de implementación")


class AkcelikCostFunction(CostFunction):
    """
    Función de costo Akçelik (para implementación futura).

    Placeholder para función de costo con capacidad limitada.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("Akçelik cost function pendiente de implementación")


def get_cost_function(cost_type: str, **kwargs) -> CostFunction:
    """
    Factory para crear funciones de costo.

    Args:
        cost_type: 'bpr', 'conic', 'akcelik'
        **kwargs: Argumentos para la función de costo

    Returns:
        Instancia de CostFunction
    """
    cost_functions = {
        'bpr': BPRCostFunction,
        'conic': ConicCostFunction,
        'akcelik': AkcelikCostFunction
    }

    if cost_type.lower() not in cost_functions:
        raise ValueError(f"Unknown cost function: {cost_type}. "
                        f"Available: {list(cost_functions.keys())}")

    return cost_functions[cost_type.lower()](**kwargs)


# =============================================================================
# ASSIGNMENT VALIDATOR
# =============================================================================

class AssignmentValidator(nn.Module):
    """
    Valida asignación mediante Stochastic User Equilibrium (SUE).
    """

    def __init__(self, num_links: int, t0: torch.Tensor, capacity: torch.Tensor,
                 route_masks: torch.Tensor, od_pair_indices: torch.Tensor,
                 num_od_pairs: int, num_link_groups: int, link_group: torch.Tensor,
                 cost_function_type: str = 'bpr',
                 max_iters: int = 10, convergence_threshold: float = 1e-4):
        """
        Args:
            num_links: Número de enlaces
            t0: Tiempos de flujo libre [num_links]
            capacity: Capacidades [num_links]
            route_masks: Máscaras de rutas [num_od, num_routes, num_links]
            od_pair_indices: Índices de pares OD por ruta [num_routes]
            num_od_pairs: Número de pares OD
            num_link_groups: Número de grupos de enlaces
            link_group: Asignación de enlaces a grupos [num_links]
            cost_function_type: Tipo de función de costo ('bpr', 'conic', 'akcelik')
            max_iters: Iteraciones máximas de SUE
            convergence_threshold: Umbral de convergencia
        """
        super().__init__()
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.register_buffer('t0', t0)

        # Crear función de costo (modular)
        self.cost_function = get_cost_function(
            cost_function_type,
            t0=t0,
            capacity=capacity,
            num_link_groups=num_link_groups,
            link_group=link_group,
            learnable_params=True
        )

        # Capa de asignación estocástica
        self.assignment_layer = StochasticAssignmentLayer(
            route_masks=route_masks,  # Asegúrate que este sea [OD, K, L]
            mu=1.0
        )

        self.register_buffer('last_convergence_iter', torch.tensor(0.0))

    def forward(self, estimated_demands: torch.Tensor, warmup: bool = False) -> tuple:
        """
        Ejecuta SUE iterativo.

        Args:
            estimated_demands: [batch_size, num_od_pairs] o [num_od_pairs]
            warmup: Si True, solo una iteración (flujo libre)

        Returns:
            (flows, alpha, beta, convergence_info)
        """
        batch_size = estimated_demands.shape[0] if estimated_demands.dim() == 2 else 1
        device = estimated_demands.device

        if estimated_demands.dim() == 1:
            estimated_demands = estimated_demands.unsqueeze(0)

        # Costos de flujo libre
        freeflow_costs_base = self.cost_function(torch.zeros_like(self.t0))
        freeflow_costs = freeflow_costs_base.unsqueeze(0).expand(batch_size, -1)

        if warmup:
            # Solo una iteración con flujo libre
            reconstructed_flows = self.assignment_layer(freeflow_costs, estimated_demands)
            convergence_info = {"converged": True, "iterations": 1}
        else:
            # SUE iterativo (MSA - Method of Successive Averages)
            flows = self.assignment_layer(freeflow_costs, estimated_demands)
            prev_flows = flows.clone()

            converged = False
            actual_iters = 1

            for it in range(1, self.max_iters + 1):
                costs = self.cost_function(flows)
                new_flows = self.assignment_layer(costs, estimated_demands)

                # MSA step
                alpha_msa = 1.0 / (it + 1)
                flows = flows + alpha_msa * (new_flows - flows)

                # Verificar convergencia
                if it > 2:
                    flow_change = torch.norm(flows - prev_flows, dim=1) / (torch.norm(flows, dim=1) + 1e-9)
                    max_change = torch.max(flow_change)

                    if max_change < self.convergence_threshold:
                        converged = True
                        actual_iters = it
                        break

                prev_flows = flows.clone()

            reconstructed_flows = flows
            convergence_info = {"converged": converged, "iterations": actual_iters}
            self.last_convergence_iter.data = torch.tensor(float(actual_iters))

        # Obtener parámetros aprendidos de la función de costo
        if isinstance(self.cost_function, BPRCostFunction):
            learned_alpha = self.cost_function.get_alpha()
            learned_beta = self.cost_function.get_beta()
        else:
            learned_alpha = None
            learned_beta = None

        return reconstructed_flows, learned_alpha, learned_beta, convergence_info


class StochasticAssignmentLayer(nn.Module):
    """
    Capa de Asignación Estocástica Vectorizada (3D).

    Maneja las dimensiones: [Batch, OD_Pairs, Rutas_K, Links]
    sin necesidad de aplanar índices.
    """

    def __init__(self, route_masks: torch.Tensor, mu: float = 1.0):
        """
        Args:
            route_masks: Tensor de forma [num_od, num_routes (K), num_links].
                         (1 si la ruta pasa por el link, 0 si no).
            mu: Parámetro inicial de dispersión (logit).
        """
        super().__init__()
        # Registramos como buffer para que sea parte del estado pero no se entrene (fijo)
        self.register_buffer('route_masks', route_masks.float())

        # Parámetro mu aprendible
        self.mu_raw = nn.Parameter(torch.tensor(float(mu)))

    @property
    def mu(self):
        """Garantiza que mu sea positivo y esté en un rango razonable."""
        return torch.clamp(F.softplus(self.mu_raw), min=0.1, max=10.0)

    def forward(self, link_costs: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
        """
        Args:
            link_costs: [batch_size, num_links]
            demands:    [batch_size, num_od_pairs]

        Returns:
            link_flows: [batch_size, num_links]
        """
        # Dimensiones para referencia en comentarios:
        # b: batch_size
        # o: num_od_pairs
        # k: num_routes (K rutas por par OD)
        # l: num_links

        # 1. Calcular Costo de cada Ruta (Vectorizado)
        # Multiplicamos los costos de los links por la máscara de rutas.
        # Operación: Sumar costo de links 'l' para cada ruta 'k' del par 'o'.
        # Entrada: link_costs [b, l], route_masks [o, k, l]
        # Salida:  route_costs [b, o, k]
        route_costs = torch.einsum('bl,okl->bok', link_costs, self.route_masks)

        # 2. Estabilización Numérica
        # Restamos el mínimo costo dentro de las K opciones de cada par OD
        # para evitar explosión exponencial en el siguiente paso.
        # keepdim=True mantiene la dimensión k como 1 para broadcasting
        min_costs, _ = torch.min(route_costs, dim=2, keepdim=True)
        stable_costs = route_costs - min_costs.detach()

        # Clamp opcional para seguridad extrema
        stable_costs = torch.clamp(stable_costs, max=50.0)

        # 3. Modelo Logit (Softmax sobre la dimensión K)
        # Calculamos la utilidad (exponencial negativa del costo)
        exp_utility = torch.exp(-self.mu * stable_costs)  # [b, o, k]

        # Suma de utilidades por par OD (denominador)
        sum_utility = torch.sum(exp_utility, dim=2, keepdim=True)  # [b, o, 1]

        # Probabilidad de elegir la ruta k
        route_probs = exp_utility / (sum_utility + 1e-9)  # [b, o, k]

        # 4. Asignar Demanda a Rutas
        # Multiplicamos la probabilidad de la ruta por la demanda total del par OD
        # demands se expande de [b, o] a [b, o, 1] para multiplicar
        route_flows = route_probs * demands.unsqueeze(2)  # [b, o, k]

        # 5. Proyectar Flujos de Ruta a Flujos de Link
        # Sumamos todo el tráfico que pasa por cada link 'l'.
        # Operación: route_flows [b, o, k] * route_masks [o, k, l] -> Sumar sobre o, k
        # Salida: link_flows [b, l]
        link_flows = torch.einsum('bok,okl->bl', route_flows, self.route_masks)

        return link_flows


# =============================================================================
# FUNCIÓN DE PÉRDIDA
# =============================================================================

class PartialDataLoss(nn.Module):
    """
    Función de pérdida para entrenamiento con datos parciales.

    Componentes:
    1. Pérdida de flujos observados (MSE)
    2. Pérdida de demandas OD conocidas (MSE)
    3. Regularización de parámetros BPR
    """

    def __init__(self, w_flow: float = 1.0, w_od: float = 1.0, w_reg: float = 0.01):
        """
        Args:
            w_flow: Peso de pérdida de flujos
            w_od: Peso de pérdida de OD
            w_reg: Peso de regularización
        """
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
        """
        Calcula pérdida total.

        Args:
            predicted_flows: Flujos predichos [batch_size, num_links]
            true_flows: Flujos verdaderos [batch_size, num_links]
            flow_mask: Máscara de flujos conocidos [batch_size, num_links] (1=conocido, 0=desconocido)
            predicted_od: Demandas OD predichas [batch_size, num_od_pairs]
            true_od: Demandas OD verdaderas [batch_size, num_od_pairs]
            od_mask: Máscara de OD conocidas [batch_size, num_od_pairs] (1=conocido, 0=desconocido)
            learned_alpha: Parámetros alpha aprendidos (opcional)
            learned_beta: Parámetros beta aprendidos (opcional)

        Returns:
            Dict con pérdidas individuales y total
        """
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

        # Regularización de parámetros BPR
        l_reg = torch.tensor(0.0, device=predicted_flows.device)
        if learned_alpha is not None and learned_beta is not None:
            # Regularizar hacia valores típicos de BPR
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

    Pipeline:
    1. Encoder: flujos observados -> espacio latente
    2. GraphMatcher: alineación estructural
    3. Decoder: espacio latente -> demandas OD completas
    4. Validator: SUE con función de costo -> flujos validados
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
                 cost_function_type: str = 'bpr',
                 dropout: float = 0.1):
        """
        Args:
            num_links: Número de enlaces
            num_od_pairs: Número de pares OD
            hidden_dim: Dimensión oculta
            feature_dim: Dimensión del espacio latente
            num_structures: Número de estructuras en GraphMatcher
            t0: Tiempos de flujo libre
            capacity: Capacidades de enlaces
            route_masks: Máscaras de rutas
            od_pair_indices: Índices de pares OD
            num_link_groups: Número de grupos de enlaces
            link_group: Asignación de enlaces a grupos
            cost_function_type: Tipo de función de costo ('bpr', 'conic', 'akcelik')
            dropout: Tasa de dropout
        """
        super().__init__()

        self.encoder = ODEncoder(num_links, hidden_dim, feature_dim, dropout)
        self.decoder = ODDecoder(num_od_pairs, hidden_dim, feature_dim, dropout)
        self.graph_matcher = GraphMatcher(feature_dim, num_structures)
        self.validator = AssignmentValidator(
            num_links, t0, capacity, route_masks, od_pair_indices,
            num_od_pairs, num_link_groups, link_group,
            cost_function_type=cost_function_type
        )

    def forward(self,
                observed_flows: torch.Tensor,
                flow_mask: torch.Tensor,
                true_od_demand: Optional[torch.Tensor] = None,
                warmup: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modelo.

        Args:
            observed_flows: Flujos observados [batch_size, num_links] o [num_links]
            flow_mask: Máscara de flujos conocidos [batch_size, num_links] o [num_links]
            true_od_demand: Demandas OD verdaderas (opcional, para training)
            warmup: Si True, solo una iteración de SUE

        Returns:
            Dict con predicciones y metadata
        """
        is_batched = observed_flows.dim() == 2
        if not is_batched:
            observed_flows = observed_flows.unsqueeze(0)
            flow_mask = flow_mask.unsqueeze(0)
            if true_od_demand is not None:
                true_od_demand = true_od_demand.unsqueeze(0)

        # Aplicar máscara a flujos de entrada (poner 0 en desconocidos)
        masked_flows = observed_flows * flow_mask

        # 1. Codificar flujos observados
        h_x = self.encoder(masked_flows)

        # 2. Generar referencia h_y (solo en training con ground truth)
        h_y = None
        if self.training and true_od_demand is not None:
            with torch.no_grad():
                # Simular flujos "ideales" a partir de OD verdadera
                true_flows, _, _, _ = self.validator(true_od_demand, warmup=True)
                # Usar mismo encoder para mantener espacio latente compartido
                h_y = self.encoder(true_flows)

        # 3. Graph matching
        g_x = self.graph_matcher(h_x, h_y)

        # 4. Decodificar a demandas OD
        estimated_demand = self.decoder(g_x)

        # 5. Validar con SUE
        reconstructed_flows, learned_alpha, learned_beta, convergence_info = self.validator(
            estimated_demand, warmup=warmup)

        # Desempaquetar si no era batched
        if not is_batched:
            estimated_demand = estimated_demand.squeeze(0)
            reconstructed_flows = reconstructed_flows.squeeze(0)

        return {
            "estimated_demand": estimated_demand,
            "reconstructed_flows": reconstructed_flows,
            "learned_alpha": learned_alpha,
            "learned_beta": learned_beta,
            "convergence_info": convergence_info
        }

