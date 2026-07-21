"""
Implementación del algoritmo Frank-Wolfe para Traffic Assignment.

El método Frank-Wolfe (también conocido como Convex Combinations) resuelve
el problema de equilibrio al usuario (User Equilibrium) mediante:

1. All-or-Nothing assignment (find shortest paths)
2. Line search para encontrar step size óptimo
3. Actualización de flujos: x_new = (1-α)*x_old + α*y_aon

Ventajas:
- Convergencia garantizada para problemas convexos
- Más rápido que MSA en la práctica
- Usado en software comercial (VISUM, TransCAD)

Referencias:
- Sheffi, Y. (1985). Urban Transportation Networks
- Patriksson, M. (1994). The Traffic Assignment Problem
"""
import networkx as nx
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .base_assignment import BaseTrafficAssignment
from utils.routing_cache import RoutingCache, find_k_shortest_with_cache


class FrankWolfeAssignment(BaseTrafficAssignment):
    """
    Asignación de tráfico usando el algoritmo Frank-Wolfe.

    Resuelve el problema de equilibrio al usuario (UE) con costos fijos
    (sin congestión) o con función de costo generalizada.

    Attributes:
        routing_cache: Cache para k-shortest paths
        use_cache: Si True, usa cache de rutas
        k_paths: Número de rutas alternativas a considerar
    """

    def __init__(self,
                 graph: nx.DiGraph,
                 od_matrix: sparse.spmatrix,
                 cost_attr: str = 'length',
                 solution_attr: str = 'solution_nocongestion',
                 use_cache: bool = True,
                 k_paths: int = 50):
        """
        Inicializa Frank-Wolfe.

        Args:
            graph: Grafo de red
            od_matrix: Matriz OD
            cost_attr: Atributo de costo (ej: 'length', 'free_flow_time')
            solution_attr: Donde guardar la solución
            use_cache: Activar cache de rutas
            k_paths: Número de rutas alternativas por par OD
        """
        super().__init__(graph, od_matrix, cost_attr, solution_attr)

        self.k_paths = k_paths
        self.use_cache = use_cache

        if use_cache:
            self.routing_cache = RoutingCache()
        else:
            self.routing_cache = None

    def all_or_nothing_assignment(self) -> Dict[Tuple[int, int], float]:
        """
        Realiza asignación All-or-Nothing (AON).

        Asigna toda la demanda de cada par OD a la ruta más corta
        según los costos actuales.

        Returns:
            Dict con flujos AON: {(from_node, to_node): flow}

        Notas:
            - Solo usa la ruta más corta (Dijkstra) para eficiencia
            - Cache almacena solo la primera ruta más corta
        """
        aon_flows = {}

        # Inicializar flujos AON en cero
        for u, v in self.graph.edges():
            aon_flows[(u, v)] = 0.0

        # Obtener pares OD con demanda
        od_pairs = self._get_od_pairs_with_demand()

        print(f"   🔍 Calculando All-or-Nothing para {len(od_pairs)} pares OD...")

        # Contador de progreso
        cache_hits = 0
        cache_misses = 0

        # Barra de progreso con tqdm mostrando cada par OD
        for origin, destination, demand in tqdm(od_pairs,
                                                desc="   Procesando pares OD",
                                                unit="par",
                                                ncols=100,
                                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            # Intentar obtener del cache
            cached_path = None
            if self.use_cache and self.routing_cache is not None:
                cached_paths = self.routing_cache.get(origin, destination)
                if cached_paths and len(cached_paths) > 0:
                    cached_path = cached_paths[0]  # Primera ruta
                    cache_hits += 1

            if cached_path is not None:
                # Usar ruta del cache
                shortest_path = cached_path
            else:
                # Calcular ruta más corta con Dijkstra simple
                try:
                    shortest_path = nx.shortest_path(
                        self.graph,
                        origin,
                        destination,
                        weight=self.cost_attr
                    )

                    # Guardar en cache (como lista con un solo elemento)
                    if self.use_cache and self.routing_cache is not None:
                        self.routing_cache.put(origin, destination, [shortest_path])

                    cache_misses += 1

                except nx.NetworkXNoPath:
                    # No hay ruta disponible
                    continue
                except nx.NodeNotFound:
                    # Nodo no existe
                    continue

            # Actualizar flujos en todos los enlaces de la ruta
            for i in range(len(shortest_path) - 1):
                u = shortest_path[i]
                v = shortest_path[i + 1]

                if (u, v) in aon_flows:
                    aon_flows[(u, v)] += demand

        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            hit_rate = cache_hits / total_requests * 100
            print(f"      💾 Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")

        return aon_flows

    def _get_od_pairs_with_demand(self) -> List[Tuple[int, int, float]]:
        """
        Extrae pares OD con demanda positiva de la matriz OD.

        Returns:
            Lista de tuplas (origin, destination, demand)

        Notas:
            - La matriz OD usa indexación base-0
            - Las zonas en el grafo son nodos 1-110
            - TODO: Verificar si el mapeo zona→nodo es 1:1 o necesita conversión
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
        Encuentra el step size óptimo mediante line search.

        Minimiza: f(α) = Σ cost(x + α(y - x)) * (x + α(y - x))

        Args:
            current_flows: Flujos actuales x
            aon_flows: Flujos All-or-Nothing y

        Returns:
            Step size óptimo α ∈ [0, 1]

        Notas:
            - Para costos fijos (sin BPR), α óptimo puede ser analítico
            - TODO: Implementar line search exacto vs. aproximado
            - Por ahora usamos búsqueda simple en grid
        """
        # Para costos fijos (sin congestión), podemos usar bisection search
        # o simplemente retornar un valor conservador

        # Grid search simple sobre [0, 1]
        alphas = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.1, ..., 1.0
        best_alpha = 0.0
        best_cost = float('inf')

        for alpha in alphas:
            # Calcular costo para este alpha
            total_cost = 0.0

            for (u, v), current_flow in current_flows.items():
                aon_flow = aon_flows.get((u, v), 0.0)
                new_flow = (1 - alpha) * current_flow + alpha * aon_flow

                # Costo del enlace (fijo, sin congestión)
                cost = self.get_edge_cost(u, v)
                total_cost += cost * new_flow

            if total_cost < best_cost:
                best_cost = total_cost
                best_alpha = alpha

        return best_alpha

    def compute_gap(self,
                    current_flows: Dict[Tuple[int, int], float],
                    aon_flows: Dict[Tuple[int, int], float]) -> float:
        """
        Calcula el gap de convergencia (relative gap).

        Gap = (TSTT_current - TSTT_aon) / TSTT_current

        donde TSTT = Total System Travel Time

        Args:
            current_flows: Flujos actuales
            aon_flows: Flujos All-or-Nothing

        Returns:
            Gap relativo
        """
        tstt_current = 0.0
        tstt_aon = 0.0

        for (u, v), current_flow in current_flows.items():
            cost = self.get_edge_cost(u, v)
            tstt_current += cost * current_flow

            aon_flow = aon_flows.get((u, v), 0.0)
            tstt_aon += cost * aon_flow

        if tstt_current > 0:
            gap = abs(tstt_current - tstt_aon) / tstt_current
        else:
            gap = 0.0

        return gap

    def solve(self,
              max_iterations: int = 100,
              convergence_threshold: float = 0.01,
              verbose: bool = True) -> Dict:
        """
        Resuelve UE usando Frank-Wolfe.

        Args:
            max_iterations: Máximo número de iteraciones
            convergence_threshold: Gap objetivo para convergencia
            verbose: Imprimir progreso

        Returns:
            Dict con estadísticas de convergencia
        """
        if verbose:
            print(f"\n🔧 Iniciando Frank-Wolfe Assignment")
            print(f"   - Costo: {self.cost_attr}")
            print(f"   - Solución: {self.solution_attr}")
            print(f"   - Max iteraciones: {max_iterations}")
            print(f"   - Umbral convergencia: {convergence_threshold}")
            print(f"   - K-paths: {self.k_paths}")
            print(f"   - Cache activo: {self.use_cache}")

        # Inicialización: AON con demanda total
        print(f"\n   Iteración 0: Inicialización (AON)...")
        aon_flows = self.all_or_nothing_assignment()

        # Actualizar flujos en el grafo
        for (u, v), flow in aon_flows.items():
            self.graph[u][v][self.solution_attr] = flow

        # Iteraciones Frank-Wolfe
        for iteration in range(1, max_iterations + 1):
            # 1. All-or-Nothing assignment
            aon_flows = self.all_or_nothing_assignment()

            # 2. Obtener flujos actuales
            current_flows = {}
            for u, v in self.graph.edges():
                current_flows[(u, v)] = self.get_edge_flow(u, v)

            # 3. Line search para encontrar step size
            alpha = self.line_search(current_flows, aon_flows)

            # 4. Actualizar flujos
            for (u, v), current_flow in current_flows.items():
                aon_flow = aon_flows.get((u, v), 0.0)
                new_flow = (1 - alpha) * current_flow + alpha * aon_flow
                self.graph[u][v][self.solution_attr] = new_flow

            # 5. Calcular gap
            gap = self.compute_gap(current_flows, aon_flows)

            # 6. Guardar historial
            self.iteration_history.append({
                'iteration': iteration,
                'gap': gap,
                'alpha': alpha,
                'total_cost': self.compute_total_cost()
            })

            if verbose and (iteration % 10 == 0 or iteration == 1):
                print(f"   Iteración {iteration}: gap={gap:.6f}, alpha={alpha:.4f}")

            # 7. Verificar convergencia
            if gap < convergence_threshold:
                if verbose:
                    print(f"\n   ✓ Convergencia alcanzada en iteración {iteration}")
                    print(f"     Gap final: {gap:.6f}")
                break
        else:
            if verbose:
                print(f"\n   ⚠ Máximo de iteraciones alcanzado sin convergencia")
                print(f"     Gap final: {gap:.6f}")

        self.convergence_gap = gap
        self.iterations = iteration

        # Guardar cache si está activo
        if self.use_cache and self.routing_cache is not None:
            self.routing_cache.save()
            cache_stats = self.routing_cache.get_statistics()
            if verbose:
                print(f"\n   📊 Estadísticas de cache:")
                print(f"      - Hits: {cache_stats['hits']}")
                print(f"      - Misses: {cache_stats['misses']}")
                print(f"      - Hit rate: {cache_stats['hit_rate']*100:.1f}%")
                print(f"      - Pares cacheados: {cache_stats['total_pairs_cached']}")

        return {
            'converged': gap < convergence_threshold,
            'iterations': iteration,
            'final_gap': gap,
            'total_cost': self.compute_total_cost(),
            'cache_stats': cache_stats if self.use_cache else None
        }


if __name__ == '__main__':
    """
    Test standalone del algoritmo Frank-Wolfe.
    
    Para test completo, usar desde data_manager.py
    """
    print("Frank-Wolfe Traffic Assignment Module")
    print("Use desde data_manager.py para ejecución completa")
