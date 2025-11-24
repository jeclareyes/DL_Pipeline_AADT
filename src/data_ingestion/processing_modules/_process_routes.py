import networkx as nx
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from tqdm import tqdm  # Para barra de progreso

# Import condicional de Rustworkx para evitar errores si no está instalado en modo 'networkx'
try:
    import rustworkx as rx
    HAS_RUSTWORKX = True
except ImportError:
    HAS_RUSTWORKX = False

# Configurar logger
log = logging.getLogger(__name__)


# --- FUNCIONES AUXILIARES (Fuera de clases para facilitar Multiprocessing) ---
#%%

# ==========================================
# BLOQUE 1: WORKERS DE NETWORKX (Procesos)
# ==========================================
# Deben estar en el scope global para que pickle/multiprocessing funcionen bien.

global_graph_ref = None

def init_worker_nx(graph: nx.Graph):
    """Inicializador para procesos de NetworkX: Carga el grafo en memoria global del hijo."""
    global global_graph_ref
    global_graph_ref = graph

def process_chunk_nx(args: Tuple) -> List[Tuple[Tuple[int, int], List[List[int]]]]:
    """Worker para NetworkX (CPU Bound - Procesos)."""
    od_pairs_chunk, k, weight, algorithm = args
    results = []
    graph = global_graph_ref  # Acceso a variable global

    for source, target in od_pairs_chunk:
        try:
            paths = []
            if algorithm in ['yen_astar', 'yen']:
                path_generator = nx.shortest_simple_paths(graph, source, target, weight=weight)
                for i, path in enumerate(path_generator):
                    if i >= k: break
                    paths.append(path)

            if paths:
                results.append(((source, target), paths))
        except Exception:
            continue
    return results

# ==========================================
# BLOQUE 2: WORKERS DE RUSTWORKX (Hilos)
# ==========================================
# Rustworkx libera el GIL, permitiendo usar Threads (memoria compartida real).

def process_chunk_rx(
        rx_graph: Any,
        od_pairs: List[Tuple[int, int]],
        nx_to_rx_map: Dict[Any, int],
        rx_to_nx_map: Dict[int, Any],
        k: int
) -> List[Tuple[Tuple[int, int], List[List[Any]]]]:
    """Worker para Rustworkx (IO/Rust Bound - Hilos)."""
    results = []
    # Función lambda identidad para peso (ya que el peso es el dato de la arista)
    weight_fn = lambda x: x

    for source_nx, target_nx in od_pairs:
        try:
            # 1. Traducir IDs NX -> RX
            source_rx = nx_to_rx_map[source_nx]
            target_rx = nx_to_rx_map[target_nx]

            # 2. Calcular en Rust
            paths_rx_indices = rx.graph_yen_k_shortest_paths(
                rx_graph, source_rx, target_rx, k=k, weight_fn=weight_fn
            )

            # 3. Traducir Rutas RX -> NX
            paths_nx = []
            for path_indices in paths_rx_indices:
                paths_nx.append([rx_to_nx_map[idx] for idx in path_indices])

            if paths_nx:
                results.append(((source_nx, target_nx), paths_nx))

        except (KeyError, Exception):
            continue

    return results

#%%
# ==========================================
# CLASE PRINCIPAL MODULAR
# ==========================================

class RouteComputer:
    """
    Clase híbrida que selecciona la estrategia de ejecución según la configuración.
    """

    def __init__(self, config: Dict):
        self.config = config

        # Parámetros comunes
        self.k = config.get('k_paths', 10)
        self.weight = config.get('weight', 'free_flow_time')
        self.n_workers = config.get('n_workers', 4)
        self.chunk_size = config.get('chunk_size', 50)

        # Selección de Motor
        self.engine = config.get('engine', 'networkx').lower()

        # Validación
        if self.engine == 'rustworkx' and not HAS_RUSTWORKX:
            log.warning("Se solicitó 'rustworkx' pero no está instalado. Cambiando a 'networkx'.")
            self.engine = 'networkx'

    def compute_all(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        """Punto de entrada único que despacha a la estrategia correcta."""

        log.info(f"Iniciando cálculo de rutas. Motor: {self.engine.upper()}")

        # Optimización común: Mezclar pares para balancear carga
        od_pairs_shuffled = list(od_pairs)
        random.shuffle(od_pairs_shuffled)

        if self.engine == 'rustworkx':
            return self._execute_rustworkx(graph, od_pairs_shuffled)
        else:
            return self._execute_networkx(graph, od_pairs_shuffled)

    # --- ESTRATEGIA A: RUSTWORKX (Threads + C++) ---
    def _execute_rustworkx(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        results = {}

        # 1. Conversión de Grafo (Solo ocurre si se elige este motor)
        rx_graph, nx_to_rx, rx_to_nx = self._convert_nx_to_rx(graph)

        # 2. Chunking
        chunks = [od_pairs[i:i + self.chunk_size] for i in range(0, len(od_pairs), self.chunk_size)]

        log.info(f"RX: Procesando {len(chunks)} lotes con {self.n_workers} hilos.")

        # 3. ThreadPool (Memoria compartida eficiente)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(process_chunk_rx, rx_graph, chunk, nx_to_rx, rx_to_nx, self.k)
                for chunk in chunks
            ]

            for future in tqdm(as_completed(futures), total=len(chunks), desc="Rutas (Rustworkx)"):
                chunk_res = future.result()
                for k_od, v_paths in chunk_res:
                    results[k_od] = v_paths

        return results

    def _convert_nx_to_rx(self, nx_graph: nx.Graph):
        """Convierte NetworkX a Rustworkx optimizando pesos."""
        log.info("Convirtiendo grafo a formato Rustworkx...")
        is_directed = nx_graph.is_directed()
        rx_graph = rx.PyDiGraph() if is_directed else rx.PyGraph()

        nx_nodes = list(nx_graph.nodes())
        rx_indices = rx_graph.add_nodes_from(nx_nodes)

        nx_to_rx = {n: i for n, i in zip(nx_nodes, rx_indices)}
        rx_to_nx = {i: n for n, i in zip(nx_nodes, rx_indices)}

        edges_to_add = []
        for u, v, data in nx_graph.edges(data=True):
            # Extraemos el peso float directamente
            w = float(data.get(self.weight, 1.0))
            edges_to_add.append((nx_to_rx[u], nx_to_rx[v], w))

        rx_graph.add_edges_from(edges_to_add)
        return rx_graph, nx_to_rx, rx_to_nx

    # --- ESTRATEGIA B: NETWORKX (Procesos + Python) ---
    def _execute_networkx(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        results = {}

        # 1. Chunking
        chunks = [od_pairs[i:i + self.chunk_size] for i in range(0, len(od_pairs), self.chunk_size)]

        # Argumentos para workers
        tasks = [(chunk, self.k, self.weight, 'yen_astar') for chunk in chunks]

        log.info(f"NX: Procesando {len(chunks)} lotes con {self.n_workers} procesos.")

        # 2. ProcessPool (Aislamiento de memoria)
        # Nota: Usamos init_worker_nx para pasar el grafo
        with ProcessPoolExecutor(max_workers=self.n_workers,
                                 initializer=init_worker_nx,
                                 initargs=(graph,)) as executor:

            futures = [executor.submit(process_chunk_nx, t) for t in tasks]

            for future in tqdm(as_completed(futures), total=len(tasks), desc="Rutas (NetworkX)"):
                chunk_res = future.result()
                for k_od, v_paths in chunk_res:
                    results[k_od] = v_paths

        return results

# CLASES DE HANDLER Y CACHE

class _RouteCache:
    """Maneja exclusivamente la Lectura/Escritura del cache en disco."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def exists(self) -> bool:
        return self.file_path.exists() and self.file_path.stat().st_size > 0

    def load(self) -> Dict:
        log.info(f"Cargando cache de rutas desde: {self.file_path}")
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            log.info(f"Cache cargado exitosamente. {len(data)} registros.")
            return data
        except Exception as e:
            log.error(f"Error cargando cache: {e}")
            return {}

    def save(self, data: Dict):
        log.info(f"Guardando cache de rutas en: {self.file_path}")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

class RouteHandler:
    """
    Clase Orquestadora (Facade).
    Conecta Configuración + Grafo + Cache + Computación.
    """

    def __init__(self,
                 graph: nx.Graph,
                 node_df: Any,  # DataFrame de nodos si es necesario para filtrar ODs
                 output_route: Union[str, Path],
                 config: Union[Dict, DictConfig]):

        self.graph = graph
        self.node_df = node_df

        # Manejo robusto de configuración (Hydra DictConfig -> Dict estándar)
        if isinstance(config, DictConfig):
            self.config = OmegaConf.to_container(config, resolve=True)
        else:
            self.config = config

        self.output_path = Path(output_route)
        self.force_recompute = self.config.get('force_recompute', False)

        # Instanciar sub-módulos
        self.cache = _RouteCache(self.output_path)
        self.computer = RouteComputer(self.config)

    def _get_od_pairs(self) -> List[Tuple[int, int]]:
        """
        Extrae pares OD del grafo.
        Incluye nodos cuyo atributo 'type' sea 'taz' o 'aux' (case-insensitive).
        Se mantiene el fallback a una muestra de nodos si no se encuentran nodos TAZ/AUX.
        """
        # Ejemplo simple: Todos los nodos contra todos (cuidado con n^2)
        # Idealmente filtrarías solo nodos que son centroides (TAZs)
        nodes = list(self.graph.nodes())

        # Incluir nodos con type == 'taz' o 'aux' (insensible a mayúsculas)
        taz_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if str(d.get('type', '')).lower() in ('taz', 'aux')
        ]

        if not taz_nodes:
            log.warning(
                "No se detectaron nodos TAZ o AUX. Usando muestra de nodos o todos (peligroso para grafos grandes).")
            taz_nodes = nodes[:50]  # Fallback por seguridad para pruebas

        # Generar pares (incluye self-pairs i,i usando product en lugar de permutations)
        import itertools
        return list(itertools.product(taz_nodes, taz_nodes))

    def run(self) -> Dict:
        """Pipeline principal."""

        # 1. Verificar si existe cache y si no forzamos recalculo
        if self.cache.exists() and not self.force_recompute:
            routes = self.cache.load()
            if routes:
                return routes

        # 2. Si llegamos aquí, hay que calcular
        log.info("Iniciando proceso de cálculo de nuevas rutas...")

        # Obtener pares origen-destino
        od_pairs = self._get_od_pairs()
        log.info(f"Total pares OD a procesar: {len(od_pairs)}")

        # Calcular
        routes = self.computer.compute_all(self.graph, od_pairs)

        # 3. Guardar
        if routes:
            self.cache.save(routes)

        return routes


# --- ENTRY POINT PARA EJECUCIÓN STANDALONE (Hydra) ---
@hydra.main(version_base=None, config_path="../../../../configs",
            config_name="config")  # Ajusta path según tu estructura
def main(cfg: DictConfig):
    """
    Permite correr este archivo independientemente para pruebas.
    Asume que 'cfg' contiene todo el yaml global.
    """
    # Simular carga de grafo para prueba
    log.info("Modo Standalone: Creando grafo de prueba...")
    G = nx.grid_2d_graph(5, 5)
    # Convertir nodos tupla a int para simular tus datos
    G = nx.convert_node_labels_to_integers(G)
    # Añadir pesos
    for u, v in G.edges():
        G[u][v]['free_flow_time'] = 1.0
        G.nodes[u]['type'] = 'taz'  # Marcar como zonas para probar

    # Extraer configuración relevante del YAML global
    # Nota: Aquí asumo que tu YAML tiene la estructura que mostraste
    route_conf = cfg.data_ingestion.data_processing.route_calculation
    output_path = cfg.data_ingestion.data_processing.output_routes.routing_cache_route

    handler = RouteHandler(
        graph=G,
        node_df=None,
        output_route=output_path,
        config=route_conf
    )

    handler.run()


if __name__ == "__main__":
    main()