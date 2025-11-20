"""
Implementación del algoritmo Frank-Wolfe para Traffic Assignment CON CONGESTIÓN.

Este módulo implementa el método Frank-Wolfe con función de costo BPR (Bureau of Public Roads)
que varía según el flujo asignado, modelando así la congestión en la red.

Función BPR:
    t(x) = t0 * [1 + b * (x/c)^power]

Donde:
    - t(x): tiempo de viaje con flujo x
    - t0: tiempo de flujo libre (free_flow_time)
    - b: parámetro de congestión (típicamente 0.15)
    - power: exponente (típicamente 4)
    - x: flujo actual en el enlace
    - c: capacidad del enlace

Referencias:
- Sheffi, Y. (1985). Urban Transportation Networks
- Patriksson, M. (1994). The Traffic Assignment Problem
"""
import networkx as nx
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .base_assignment import BaseTrafficAssignment
from utils.routing_cache import RoutingCache


class FrankWolfeAssignmentCongestion(BaseTrafficAssignment):
    """
    Asignación de tráfico usando Frank-Wolfe con función de costo BPR (congestión).

    A diferencia de la versión sin congestión, aquí los costos de los enlaces
    se actualizan en cada iteración según el flujo asignado usando la función BPR.

    Attributes:
        cost_function: Tipo de función de costo ('bpr')
        routing_cache: Cache para rutas (menos útil con costos dinámicos)
        use_cache: Si True, usa cache de rutas
        k_paths: Número de rutas alternativas a considerar
        bpr_params: Parámetros por defecto para BPR si no están en el grafo
        cost_scale_factor: Factor de escala para normalizar costos en Dijkstra
    """

    def __init__(self,
                 graph: nx.DiGraph,
                 od_matrix: sparse.spmatrix,
                 cost_function: str = 'bpr',
                 solution_attr: str = 'solution_congestion',
                 use_cache: bool = False,  # Cache menos útil con costos dinámicos
                 k_paths: int = 10):
        """
        Inicializa Frank-Wolfe con congestión.

        Args:
            graph: Grafo de red
            od_matrix: Matriz OD
            cost_function: Función de costo ('bpr')
            solution_attr: Donde guardar la solución
            use_cache: Activar cache de rutas (menos efectivo con costos dinámicos)
            k_paths: Número de rutas alternativas por par OD
        """
        # No usamos cost_attr fijo, sino que calculamos dinámicamente
        super().__init__(graph, od_matrix, cost_attr='dynamic_cost', solution_attr=solution_attr)

        self.cost_function = cost_function
        self.k_paths = k_paths
        self.use_cache = use_cache

        # Factor de escala para normalización de costos (solo para Dijkstra)
        # Divide costos grandes para evitar overflow numérico
        self.cost_scale_factor = 1e-3

        # Parámetros por defecto para BPR (si no están en el grafo)
        self.bpr_params = {
            'default_b': 0.15,
            'default_power': 4.0,
            'min_capacity': 100.0  # Capacidad mínima para evitar explosión numérica
        }

        if use_cache:
            self.routing_cache = RoutingCache()
            print("   ⚠ Advertencia: Cache activado con costos dinámicos (puede ser menos efectivo)")
        else:
            self.routing_cache = None

        # Inicializar atributo de costo dinámico en el grafo
        self._initialize_dynamic_costs()

    def _initialize_dynamic_costs(self):
        """
        Inicializa los costos dinámicos en el grafo.

        Al inicio (sin flujo), el costo es igual al free_flow_time.
        """
        for u, v, data in self.graph.edges(data=True):
            data['dynamic_cost'] = data.get('free_flow_time', data.get('length', 1.0))

    def compute_bpr_cost(self, u: int, v: int, flow: float) -> float:
        """
        Calcula el costo de un enlace usando la función BPR.

        BPR: t(x) = t0 * [1 + b * (x/c)^power]

        Args:
            u: Nodo origen del enlace
            v: Nodo destino del enlace
            flow: Flujo actual en el enlace

        Returns:
            Costo (tiempo) del enlace con el flujo dado
        """
        edge_data = self.graph[u][v]

        # Obtener parámetros BPR DIRECTAMENTE del dataframe (sin reemplazar)
        t0 = edge_data.get('free_flow_time', edge_data.get('length', 1.0))
        capacity = edge_data.get('capacity', self.bpr_params['min_capacity'])
        b = edge_data.get('b', self.bpr_params['default_b'])
        power = edge_data.get('power', self.bpr_params['default_power'])

        # Validación: evitar divisiones por cero
        if t0 <= 0:
            t0 = edge_data.get('length', 1.0)
            if t0 <= 0:
                t0 = 1.0

        if capacity <= 0:
            capacity = self.bpr_params['min_capacity']

        # Evitar flujos negativos
        flow = max(0.0, flow)

        # Calcular ratio volumen/capacidad
        # NOTA: Con capacity=1 y flujos altos, vc_ratio será muy alto
        # pero esto es correcto según los datos originales
        vc_ratio = flow / capacity

        # Función BPR estándar
        # Si b es muy pequeño (ej: 10^-18) y power es grande, el término de congestión
        # puede ser casi insignificante, lo cual está bien si así son los datos
        cost = t0 * (1.0 + b * (vc_ratio ** power))

        # Asegurar que el costo es al menos t0 (nunca menor que flujo libre)
        cost = max(cost, t0)

        return cost

    def update_dynamic_costs(self):
        """
        Actualiza los costos dinámicos de todos los enlaces según el flujo actual.

        Este método se llama en cada iteración de Frank-Wolfe para actualizar
        los costos antes de calcular las rutas más cortas.
        """
        min_cost = float('inf')
        max_cost = 0.0
        negative_count = 0
        zero_count = 0

        for u, v in self.graph.edges():
            current_flow = self.get_edge_flow(u, v)
            new_cost = self.compute_bpr_cost(u, v, current_flow)

            # Validación adicional: asegurar costo estrictamente positivo
            if new_cost <= 0:
                # Usar free_flow_time como fallback
                new_cost = self.graph[u][v].get('free_flow_time',
                           self.graph[u][v].get('length', 1.0))
                if new_cost <= 0:
                    new_cost = 1.0
                    zero_count += 1

            if new_cost < 0:
                negative_count += 1
                new_cost = 1.0  # Forzar a positivo

            self.graph[u][v]['dynamic_cost'] = new_cost

            # Tracking para debug
            min_cost = min(min_cost, new_cost)
            max_cost = max(max_cost, new_cost)

        # Debug info
        if zero_count > 0 or negative_count > 0:
            print(f"      ⚠ Costos corregidos: {zero_count} ceros, {negative_count} negativos")

        # Verificación final: ningún costo debe ser <= 0
        invalid_costs = [(u, v, data['dynamic_cost'])
                        for u, v, data in self.graph.edges(data=True)
                        if data.get('dynamic_cost', 1.0) <= 0]

        if invalid_costs:
            print(f"      ❌ ERROR: {len(invalid_costs)} enlaces con costo inválido!")
            for u, v, cost in invalid_costs[:5]:  # Mostrar primeros 5
                print(f"         Enlace {u}→{v}: cost={cost}")
                # Forzar corrección
                self.graph[u][v]['dynamic_cost'] = 1.0

    def all_or_nothing_assignment(self) -> Dict[Tuple[int, int], float]:
        """
        Realiza asignación All-or-Nothing (AON) con costos dinámicos actualizados.

        A diferencia de la versión sin congestión, aquí los costos se recalculan
        en cada iteración según el flujo actual.

        Returns:
            Dict con flujos AON: {(from_node, to_node): flow}
        """
        aon_flows = {}

        # Inicializar flujos AON en cero
        for u, v in self.graph.edges():
            aon_flows[(u, v)] = 0.0

        # Obtener pares OD con demanda
        od_pairs = self._get_od_pairs_with_demand()

        print(f"   🔍 Calculando All-or-Nothing para {len(od_pairs)} pares OD...")

        # Contadores de diagnóstico
        no_path_count = 0
        node_not_found_count = 0
        value_error_count = 0
        no_path_pairs = []  # Para guardar detalles de pares sin ruta

        # Función de peso que se calcula dinámicamente
        def weight_function(u, v, edge_attrs):
            """Función de peso para Dijkstra - NORMALIZADA para estabilidad numérica."""
            current_flow = edge_attrs.get(self.solution_attr, 0.0)

            # Calcular costo BPR REAL
            real_cost = self.compute_bpr_cost(u, v, current_flow)

            # 🔑 NORMALIZAR solo para Dijkstra (escalar a rango razonable)
            # Esto evita overflow pero NO afecta el costo total final
            normalized_cost = real_cost * self.cost_scale_factor

            return normalized_cost

        # Barra de progreso con tqdm
        for origin, destination, demand in tqdm(od_pairs,
                                                desc="   Procesando pares OD",
                                                unit="par",
                                                ncols=100,
                                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            # Calcular ruta más corta con función de peso dinámica
            try:
                shortest_path = nx.shortest_path(
                    self.graph,
                    origin,
                    destination,
                    weight=weight_function  # Usar función con costos normalizados
                )

            except nx.NetworkXNoPath:
                no_path_count += 1
                if len(no_path_pairs) < 10:  # Guardar solo los primeros 10
                    # Verificar si los nodos existen
                    origin_exists = origin in self.graph
                    dest_exists = destination in self.graph
                    no_path_pairs.append({
                        'origin': origin,
                        'destination': destination,
                        'demand': demand,
                        'origin_exists': origin_exists,
                        'dest_exists': dest_exists,
                        'reason': 'NetworkXNoPath'
                    })
                continue
            except nx.NodeNotFound as e:
                node_not_found_count += 1
                if len(no_path_pairs) < 10:
                    no_path_pairs.append({
                        'origin': origin,
                        'destination': destination,
                        'demand': demand,
                        'origin_exists': origin in self.graph,
                        'dest_exists': destination in self.graph,
                        'reason': f'NodeNotFound: {e}'
                    })
                continue
            except ValueError as e:
                # Si sigue habiendo error de pesos negativos, reportar y continuar
                value_error_count += 1
                if len(no_path_pairs) < 10:
                    no_path_pairs.append({
                        'origin': origin,
                        'destination': destination,
                        'demand': demand,
                        'origin_exists': origin in self.graph,
                        'dest_exists': destination in self.graph,
                        'reason': f'ValueError: {e}'
                    })
                continue

            # Actualizar flujos en todos los enlaces de la ruta
            for i in range(len(shortest_path) - 1):
                u = shortest_path[i]
                v = shortest_path[i + 1]

                if (u, v) in aon_flows:
                    aon_flows[(u, v)] += demand

        # Reportar problemas encontrados
        total_failed = no_path_count + node_not_found_count + value_error_count
        if total_failed > 0:
            print(f"\n      ⚠️ DIAGNÓSTICO DE PARES SIN RUTA:")
            print(f"         Total pares fallidos: {total_failed} de {len(od_pairs)} ({total_failed/len(od_pairs)*100:.2f}%)")
            print(f"         - Sin ruta (NetworkXNoPath): {no_path_count}")
            print(f"         - Nodo no encontrado: {node_not_found_count}")
            print(f"         - Error de valor: {value_error_count}")

            if no_path_pairs:
                print(f"\n      📋 Ejemplos de pares problemáticos:")
                for pair in no_path_pairs[:5]:
                    print(f"         {pair['origin']} → {pair['destination']} (demanda: {pair['demand']:.2f})")
                    print(f"            Origen existe: {pair['origin_exists']}, Destino existe: {pair['dest_exists']}")
                    print(f"            Razón: {pair['reason']}")

            # Sugerencias de solución
            print(f"\n      💡 POSIBLES CAUSAS Y SOLUCIONES:")
            if node_not_found_count > 0:
                print(f"         1. Nodos OD no están en el grafo → Verificar que los nodos 1-110 existan")
            if no_path_count > 0:
                print(f"         2. Red tiene componentes desconectadas → Verificar conectividad del grafo")
                print(f"            Ejecutar: python src/diagnostics_bpr.py para análisis completo")
            if value_error_count > 0:
                print(f"         3. Costos negativos o inválidos → Ya corregido en compute_bpr_cost()")

        # 🔑 DESPUÉS de asignar flujos, guardar costos REALES para análisis
        for (u, v), flow in aon_flows.items():
            real_cost = self.compute_bpr_cost(u, v, flow)  # Costo SIN normalizar
            self.graph[u][v]['current_cost'] = real_cost  # Guardar costo real

        return aon_flows

    def _get_od_pairs_with_demand(self) -> List[Tuple[int, int, float]]:
        """
        Extrae pares OD con demanda positiva de la matriz OD.

        Returns:
            Lista de tuplas (origin, destination, demand)
        """
        od_pairs = []

        # Convertir matriz sparse a formato COO para iterar
        od_coo = self.od_matrix.tocoo()

        for i, j, demand in zip(od_coo.row, od_coo.col, od_coo.data):
            if demand > 0:
                # Convertir índices base-0 a nodos base-1
                origin = i + 1
                destination = j + 1
                od_pairs.append((origin, destination, demand))

        return od_pairs

    def line_search(self,
                    current_flows: Dict[Tuple[int, int], float],
                    aon_flows: Dict[Tuple[int, int], float]) -> float:
        """
        Encuentra el step size óptimo mediante line search con función BPR.

        Minimiza: f(α) = Σ ∫[0 to x(α)] cost(w) dw

        Para BPR, la integral tiene solución analítica:
        ∫ t0 * [1 + b*(w/c)^power] dw = t0*w + t0*b/(power+1) * (w/c)^power * w

        Args:
            current_flows: Flujos actuales x
            aon_flows: Flujos All-or-Nothing y

        Returns:
            Step size óptimo α ∈ [0, 1]
        """
        # Grid search sobre [0, 1]
        alphas = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0
        best_alpha = 0.0
        best_objective = float('inf')

        for alpha in alphas:
            # Calcular valor objetivo para este alpha
            objective = 0.0

            for (u, v), current_flow in current_flows.items():
                aon_flow = aon_flows.get((u, v), 0.0)
                new_flow = (1 - alpha) * current_flow + alpha * aon_flow

                # Integral de la función BPR de 0 a new_flow
                edge_data = self.graph[u][v]
                t0 = edge_data.get('free_flow_time', edge_data.get('length', 1.0))
                capacity = max(edge_data.get('capacity', 1.0), 0.1)
                b = edge_data.get('b', self.bpr_params['default_b'])
                power = edge_data.get('power', self.bpr_params['default_power'])

                # Integral BPR: ∫[0 to x] t0*[1 + b*(w/c)^p] dw
                # = t0*x + t0*b/(p+1) * x^(p+1) / c^p
                if new_flow > 0:
                    term1 = t0 * new_flow
                    term2 = t0 * b / (power + 1) * (new_flow ** (power + 1)) / (capacity ** power)
                    objective += term1 + term2

            if objective < best_objective:
                best_objective = objective
                best_alpha = alpha

        return best_alpha

    def compute_gap(self,
                    current_flows: Dict[Tuple[int, int], float],
                    aon_flows: Dict[Tuple[int, int], float]) -> float:
        """
        Calcula el gap de convergencia (relative gap) con costos BPR.

        Gap = (TSTT_current - TSTT_aon) / TSTT_current

        donde TSTT = Total System Travel Time calculado con BPR

        Args:
            current_flows: Flujos actuales
            aon_flows: Flujos All-or-Nothing

        Returns:
            Gap relativo
        """
        tstt_current = 0.0
        tstt_aon = 0.0

        for (u, v), current_flow in current_flows.items():
            # Costo con flujo actual (BPR)
            cost_current = self.compute_bpr_cost(u, v, current_flow)
            tstt_current += cost_current * current_flow

            # Costo con flujo AON (BPR)
            aon_flow = aon_flows.get((u, v), 0.0)
            cost_aon = self.compute_bpr_cost(u, v, aon_flow)
            tstt_aon += cost_aon * aon_flow

        if tstt_current > 0:
            gap = abs(tstt_current - tstt_aon) / tstt_current
        else:
            gap = 0.0

        return gap

    def compute_total_cost(self) -> float:
        """
        Calcula el costo total del sistema usando la función BPR REAL (sin normalizar).

        Returns:
            TSTT (Total System Travel Time) con BPR
        """
        total_cost = 0.0
        for u, v in self.graph.edges():
            flow = self.get_edge_flow(u, v)

            # ⚠️ NO usar weight_function aquí (está normalizado)
            # Usar costo real guardado O recalcular
            if 'current_cost' in self.graph[u][v]:
                cost = self.graph[u][v]['current_cost']  # Costo real guardado
            else:
                cost = self.compute_bpr_cost(u, v, flow)  # Recalcular si no existe

            total_cost += cost * flow

        return total_cost

    def solve(self,
              max_iterations: int = 100,
              convergence_threshold: float = 0.01,
              verbose: bool = True) -> Dict:
        """
        Resuelve UE usando Frank-Wolfe con función BPR (congestión).

        Args:
            max_iterations: Máximo número de iteraciones
            convergence_threshold: Gap objetivo para convergencia
            verbose: Imprimir progreso

        Returns:
            Dict con estadísticas de convergencia
        """
        if verbose:
            print(f"\n🔧 Iniciando Frank-Wolfe con Congestión (BPR)")
            print(f"   - Función de costo: {self.cost_function.upper()}")
            print(f"   - Solución: {self.solution_attr}")
            print(f"   - Max iteraciones: {max_iterations}")
            print(f"   - Umbral convergencia: {convergence_threshold}")
            print(f"   - Parámetros BPR por defecto: b={self.bpr_params['default_b']}, power={self.bpr_params['default_power']}")

        # Inicialización: AON con costos de flujo libre
        print(f"\n   Iteración 0: Inicialización (AON con flujo libre)...")
        aon_flows = self.all_or_nothing_assignment()

        # Actualizar flujos en el grafo
        for (u, v), flow in aon_flows.items():
            self.graph[u][v][self.solution_attr] = flow

        # Actualizar costos dinámicos con el flujo inicial
        self.update_dynamic_costs()

        # Variables para tracking
        gap = float('inf')
        iteration = 0

        # Iteraciones Frank-Wolfe
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n   Iteración {iteration}:")

            # 1. Actualizar costos dinámicos según flujo actual
            self.update_dynamic_costs()

            # 2. All-or-Nothing assignment con costos actualizados
            aon_flows = self.all_or_nothing_assignment()

            # 3. Obtener flujos actuales
            current_flows = {}
            for u, v in self.graph.edges():
                current_flows[(u, v)] = self.get_edge_flow(u, v)

            # 4. Line search para encontrar step size
            alpha = self.line_search(current_flows, aon_flows)

            # 5. Actualizar flujos
            for (u, v), current_flow in current_flows.items():
                aon_flow = aon_flows.get((u, v), 0.0)
                new_flow = (1 - alpha) * current_flow + alpha * aon_flow
                self.graph[u][v][self.solution_attr] = new_flow

            # 6. Calcular gap
            gap = self.compute_gap(current_flows, aon_flows)

            # 7. Guardar historial
            self.iteration_history.append({
                'iteration': iteration,
                'gap': gap,
                'alpha': alpha,
                'total_cost': self.compute_total_cost()
            })

            if verbose:
                print(f"      → gap={gap:.6f}, alpha={alpha:.4f}, TSTT={self.compute_total_cost():.2f}")

            # 8. Verificar convergencia
            if gap < convergence_threshold:
                if verbose:
                    print(f"\n   ✓ Convergencia alcanzada en iteración {iteration}")
                    print(f"     Gap final: {gap:.6f}")
                break
        else:
            if verbose:
                print(f"\n   ⚠ Máximo de iteraciones alcanzado")
                print(f"     Gap final: {gap:.6f}")

        self.convergence_gap = gap
        self.iterations = iteration

        # Actualizar costos finales
        self.update_dynamic_costs()

        # Guardar cache si está activo (aunque es menos útil con costos dinámicos)
        cache_stats = None
        if self.use_cache and self.routing_cache is not None:
            self.routing_cache.save()
            cache_stats = self.routing_cache.get_statistics()

        return {
            'converged': gap < convergence_threshold,
            'iterations': iteration,
            'final_gap': gap,
            'total_cost': self.compute_total_cost(),
            'cache_stats': cache_stats,
            'cost_function': self.cost_function
        }


if __name__ == '__main__':
    """
    Test standalone del algoritmo Frank-Wolfe con congestión.
    
    Para test completo, usar traffic_assignment_congestion.py
    """
    print("Frank-Wolfe Traffic Assignment con Congestión (BPR)")
    print("Use traffic_assignment_congestion.py para ejecución completa")
