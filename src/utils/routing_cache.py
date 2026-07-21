"""
Sistema de cache para rutas calculadas (k-shortest paths).

Guarda las rutas calculadas en disco para evitar recalcular en cada ejecución.
Usa pickle para almacenar diccionarios de rutas por par OD.

Estructura del cache:
{
    (origin, destination): [
        [node1, node2, ..., nodeN],  # Ruta 1
        [node1, node3, ..., nodeN],  # Ruta 2
        ...
    ]
}
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict


class RoutingCache:
    """
    Gestor de cache para rutas precalculadas.

    Attributes:
        cache_path: Ruta al archivo pickle de cache
        cache: Diccionario con rutas {(o,d): [[ruta1], [ruta2], ...]}
        hits: Contador de cache hits
        misses: Contador de cache misses
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Inicializa el sistema de cache.

        Args:
            cache_path: Ruta al archivo de cache. Si None, usa ruta por defecto.
        """
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[2]
            cache_path = project_root / 'data' / 'processed' / 'routing_cache' / 'kshortest_paths.pkl'

        self.cache_path = Path(cache_path)
        self.cache: Dict[Tuple[int, int], List[List[int]]] = {}
        self.hits = 0
        self.misses = 0

        # Cargar cache existente si existe
        self.load()

    def load(self):
        """Carga el cache desde disco si existe."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"   ✓ Cache cargado: {len(self.cache)} pares OD con rutas precalculadas")
            except Exception as e:
                print(f"   ⚠ Error al cargar cache: {e}. Iniciando cache vacío.")
                self.cache = {}
        else:
            print(f"   ℹ Cache no encontrado. Se creará uno nuevo.")
            self.cache = {}

    def save(self):
        """Guarda el cache en disco."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✓ Cache guardado: {len(self.cache)} pares OD")

    def get(self, origin: int, destination: int) -> Optional[List[List[int]]]:
        """
        Obtiene rutas del cache.

        Args:
            origin: Nodo origen
            destination: Nodo destino

        Returns:
            Lista de rutas si existe en cache, None si no
        """
        key = (origin, destination)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, origin: int, destination: int, paths: List[List[int]]):
        """
        Guarda rutas en el cache.

        Args:
            origin: Nodo origen
            destination: Nodo destino
            paths: Lista de rutas (cada ruta es una lista de nodos)
        """
        key = (origin, destination)
        self.cache[key] = paths

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas del cache.

        Returns:
            Dict con hits, misses, hit_rate, total_pairs
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_pairs_cached': len(self.cache),
            'cache_file': str(self.cache_path)
        }

    def clear(self):
        """Limpia completamente el cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        if self.cache_path.exists():
            self.cache_path.unlink()
        print("   ✓ Cache limpiado")


def k_shortest_paths(graph: nx.DiGraph,
                     source: int,
                     target: int,
                     k: int = 50,
                     weight: str = 'length') -> List[List[int]]:
    """
    Encuentra las k rutas más cortas usando el algoritmo de Yen.

    Args:
        graph: Grafo NetworkX
        source: Nodo origen
        target: Nodo destino
        k: Número de rutas a encontrar
        weight: Atributo a usar como peso

    Returns:
        Lista de hasta k rutas (cada ruta es lista de nodos)

    Notas:
        - Si no existen k rutas, retorna las que encuentre
        - Usa el algoritmo de Yen (k-shortest paths sin loops)
    """
    try:
        # NetworkX tiene implementación directa de k-shortest paths
        # Retorna generador, convertimos a lista limitada a k
        paths_gen = nx.shortest_simple_paths(graph, source, target, weight=weight)
        paths = []

        for i, path in enumerate(paths_gen):
            if i >= k:
                break
            paths.append(path)

        return paths

    except nx.NetworkXNoPath:
        # No hay ruta entre origen y destino
        return []
    except nx.NodeNotFound:
        # Nodo no existe en el grafo
        return []
    except Exception as e:
        print(f"   ⚠ Error calculando k-shortest paths ({source}→{target}): {e}")
        return []


def find_k_shortest_with_cache(graph: nx.DiGraph,
                                origin: int,
                                destination: int,
                                k: int = 50,
                                weight: str = 'length',
                                cache: Optional[RoutingCache] = None) -> List[List[int]]:
    """
    Encuentra k-shortest paths con soporte de cache.

    Args:
        graph: Grafo NetworkX
        origin: Nodo origen
        destination: Nodo destino
        k: Número de rutas
        weight: Atributo de peso
        cache: Instancia de RoutingCache (opcional)

    Returns:
        Lista de rutas
    """
    # Intentar obtener del cache
    if cache is not None:
        cached_paths = cache.get(origin, destination)
        if cached_paths is not None:
            return cached_paths

    # Calcular rutas
    paths = k_shortest_paths(graph, origin, destination, k, weight)

    # Guardar en cache si existe
    if cache is not None and len(paths) > 0:
        cache.put(origin, destination, paths)

    return paths

