import networkx as nx
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from omegaconf import DictConfig, OmegaConf
import hydra
import random
from tqdm import tqdm  # Para barra de progreso

# Configurar logger
log = logging.getLogger(__name__)


# --- FUNCIONES AUXILIARES (Fuera de clases para facilitar Multiprocessing) ---

# Configurar logger
log = logging.getLogger(__name__)

# --- GLOBAL VAR & WORKER ---
# Variable global para que cada proceso hijo tenga acceso al grafo
# sin necesidad de serializarlo/deserializarlo constantemente en cada tarea.
global_graph_ref = None

def init_worker(graph: nx.Graph):
    """
    Se ejecuta una vez al iniciar cada proceso hijo.
    Carga el grafo en la memoria local del proceso.
    """
    global global_graph_ref
    global_graph_ref = graph

def process_chunk(args: Tuple) -> List[Tuple[Tuple[int, int], List[List[int]]]]:
    """
    Procesa un LOTE de pares OD usando el grafo global.
    """
    od_pairs_chunk, k, weight, algorithm = args
    results = []

    # Usamos la referencia global (rápido, sin overhead de pickle)
    graph = global_graph_ref

    for source, target in od_pairs_chunk:
        try:
            paths = []
            if algorithm in ['yen_astar', 'yen']:
                # nx.shortest_simple_paths implementa Yen
                path_generator = nx.shortest_simple_paths(graph, source, target, weight=weight)
                for i, path in enumerate(path_generator):
                    if i >= k: break
                    paths.append(path)

            # (Opcional) Si agregas otros algoritmos en el futuro
            elif algorithm == 'dijkstra':
                paths = [nx.dijkstra_path(graph, source, target, weight=weight)]

            if paths:
                results.append(((source, target), paths))

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Es normal que algunos pares no tengan conexión
            continue
        except Exception as e:
            # Log de error silencioso para no romper el pool
            # print(f"Error en {source}->{target}: {e}")
            continue

    return results

# --- CLASES ---


class RouteComputer:
    """Aloja los algoritmos y la estrategia de ejecución (Secuencial/Paralela)."""

    def __init__(self, config: Dict):
        self.k = config.get('k_paths', 10)
        self.algorithm = config.get('algorithm', 'yen_astar')
        self.weight = config.get('weight', 'free_flow_time')
        self.mode = config.get('execution_mode', 'sequential')
        self.n_workers = config.get('n_workers', 4)  # Opcional, para paralelo
        self.chunk_size = config.get('chunk_size', 50)  # Clave para reducir overhead


    def compute_all(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        """Calcula rutas para una lista de pares OD."""

        log.info(f"🚀 Iniciando cálculo de rutas. Modo: {self.mode}. Algoritmo: {self.algorithm}")
        results = {}

        if self.mode == 'parallel':
            results = self._execute_parallel(graph, od_pairs)
        else:
            results = self._execute_sequential(graph, od_pairs)

        return results

    def _execute_sequential(self, graph, od_pairs):
        # Simulamos la variable global para reutilizar la función process_chunk
        global global_graph_ref
        global_graph_ref = graph

        # Procesamos todo como un solo gran chunk
        results_list = process_chunk((od_pairs, self.k, self.weight, self.algorithm))
        return dict(results_list)

    def _execute_parallel(self, graph, od_pairs):
        results = {}

        # 1. OPTIMIZACIÓN: Mezclar pares para balancear carga entre CPU cores
        # Esto evita que un core se lleve todos los caminos largos y otros los cortos
        od_pairs_shuffled = list(od_pairs)
        random.shuffle(od_pairs_shuffled)

        # 2. Chunking (Lotes)
        chunks = [
            od_pairs_shuffled[i:i + self.chunk_size]
            for i in range(0, len(od_pairs_shuffled), self.chunk_size)
        ]

        tasks = [(chunk, self.k, self.weight, self.algorithm) for chunk in chunks]

        log.info(f"🔥 Distribuyendo {len(tasks)} tareas en {self.n_workers} núcleos...")

        # 3. ProcessPool con Initializer
        # initializer=init_worker envía el grafo UNA VEZ por proceso, no por tarea.
        with ProcessPoolExecutor(max_workers=self.n_workers,
                                 initializer=init_worker,
                                 initargs=(graph,)) as executor:

            futures = [executor.submit(process_chunk, t) for t in tasks]

            # Barra de progreso
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Calculando Rutas"):
                chunk_res = future.result()
                for k_od, v_paths in chunk_res:
                    results[k_od] = v_paths

        return results

class _RouteCache:
    """Maneja exclusivamente la Lectura/Escritura del cache en disco."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def exists(self) -> bool:
        return self.file_path.exists() and self.file_path.stat().st_size > 0

    def load(self) -> Dict:
        log.info(f"🔄 Cargando cache de rutas desde: {self.file_path}")
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            log.info(f"✅ Cache cargado exitosamente. {len(data)} registros.")
            return data
        except Exception as e:
            log.error(f"❌ Error cargando cache: {e}")
            return {}

    def save(self, data: Dict):
        log.info(f"💾 Guardando cache de rutas en: {self.file_path}")
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
                "⚠️ No se detectaron nodos TAZ o AUX. Usando muestra de nodos o todos (peligroso para grafos grandes).")
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
        log.info("⚙️ Iniciando proceso de cálculo de nuevas rutas...")

        # Obtener pares origen-destino
        od_pairs = self._get_od_pairs()
        log.info(f"📊 Total pares OD a procesar: {len(od_pairs)}")

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
    log.info("🧪 Modo Standalone: Creando grafo de prueba...")
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