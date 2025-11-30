import networkx as nx
import pickle
import logging
import traceback
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
    """
    Worker NetworkX (CPU Bound).
    Maneja explícitamente (A, A) y errores.
    """
    od_pairs_chunk, k, weight = args
    results = []
    graph = global_graph_ref

    for source, target in od_pairs_chunk:
        try:
            paths = []

            # Algoritmo: Shortest Simple Paths (Variante de Yen en NX)
            # Para (A, A), la ruta 0 es [A]. Las siguientes k-1 son ciclos que salen y vuelven a A.
            path_generator = nx.shortest_simple_paths(graph, source, target, weight=weight)

            for i, path in enumerate(path_generator):
                if i >= k: break
                paths.append(path)

            # Si se pidieron K y se encontraron menos, se devuelve lo que hay.
            # (El post-procesamiento deberá encargarse del padding si se requiere vectorización estricta)
            if paths:
                results.append(((source, target), paths))

        except nx.NetworkXNoPath:
            print("!!! WARNING: No hay ruta entre nodos NX ")
            # Esto es un comportamiento esperado si no hay conexión, no es un error de código.
            continue
        except Exception as e:
            # REQ 3: No permitir excepciones silenciosas. Imprimir traceback.
            print(f"!!! CRITICAL ERROR en par NX {source}->{target}: {e}")
            traceback.print_exc()
            # No hacemos 'continue' ciego, dejamos constancia del fallo.
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
    """
    Worker Rustworkx (Threads).
    """
    results = []
    # Lambda identidad porque el peso ya está en la arista como float
    weight_fn = lambda x: x

    for source_nx, target_nx in od_pairs:
        try:
            source_rx = nx_to_rx_map[source_nx]
            target_rx = nx_to_rx_map[target_nx]

            # Rustworkx Yen
            # Nota: Yen en Rustworkx maneja ciclos para source==target correctamente
            paths_rx_indices = rx.graph_k_shortest_path_lengths(
                graph=rx_graph,
                start=source_rx,
                k=k,
                edge_cost=weight_fn,
                goal=target_rx
            )

            paths_nx = []
            for path_indices in paths_rx_indices:
                paths_nx.append([rx_to_nx_map[idx] for idx in path_indices])

            if paths_nx:
                results.append(((source_nx, target_nx), paths_nx))

        except KeyError as e:
            print(f"!!! ERROR: Nodo no encontrado en mapeo RX {source_nx}->{target_nx}: {e}")
        except Exception as e:
            # REQ 3: Errores ruidosos
            print(f"!!! CRITICAL ERROR en par RX {source_nx}->{target_nx}: {e}")
            traceback.print_exc()

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
        self.parallel_mode: bool = config.get('parallel_mode', False)
        self.sequential = (self.n_workers <= 1 or self.parallel_mode is False)

        # Selección de Motor
        self.engine = config.get('engine', 'networkx').lower()

        # Validación
        if self.engine == 'rustworkx' and not HAS_RUSTWORKX:
            log.warning("Se solicitó 'rustworkx' pero no está instalado. Cambiando a 'networkx'.")
            self.engine = 'networkx'

    def compute_all(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        """
        Punto de entrada. Prepara el grafo y despacha a la estrategia (Secuencial o Paralela).
        """
        log.info(f"Iniciando cálculo. Motor: {self.engine.upper()} | Workers: {self.n_workers} | Secuencial: {self.sequential}")

        # REQ 1: Asegurar DiGraph simple con peso mínimo
        processing_graph = self._ensure_simple_digraph(graph)

        # Optimización: Mezclar pares
        od_pairs_shuffled = list(od_pairs)
        random.shuffle(od_pairs_shuffled)

        if self.engine == 'rustworkx':
            return self._execute_rustworkx(processing_graph, od_pairs_shuffled)
        else:
            return self._execute_networkx(processing_graph, od_pairs_shuffled)

    def _ensure_simple_digraph(self, graph: nx.Graph) -> nx.DiGraph:
        """
        Convierte MultiDiGraph a DiGraph colapsando aristas por el peso mínimo.
        Si ya es DiGraph, verifica integridad.
        """
        if not graph.is_directed():
            # Si es no dirigido, lo convertimos a dirigido (cada arista va en ambos sentidos)
            graph = graph.to_directed()

        if isinstance(graph, nx.MultiDiGraph):
            log.info("Grafo detectado como MultiDiGraph. Colapsando a DiGraph (Min Weight)...")
            G_simple = nx.DiGraph()

            # Copiar nodos y atributos
            G_simple.add_nodes_from(graph.nodes(data=True))

            # Iterar aristas y guardar la de menor peso
            # Esto puede ser costoso en grafos gigantes, pero es necesario por el REQ 1.
            for u, v, data in graph.edges(data=True):
                w = data.get(self.weight, float('inf'))

                if G_simple.has_edge(u, v):
                    current_w = G_simple[u][v].get(self.weight, float('inf'))
                    if w < current_w:
                        # Actualizamos con la arista más rápida/corta
                        G_simple.add_edge(u, v, **data)
                else:
                    G_simple.add_edge(u, v, **data)

            log.info(f"Grafo colapsado. Nodos: {len(G_simple.nodes())}, Aristas: {len(G_simple.edges())}")
            return G_simple

        return graph


    # --- ESTRATEGIA A: RUSTWORKX (Threads + C++) ---
    def _execute_rustworkx(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        results = {}
        rx_graph, nx_to_rx, rx_to_nx = self._convert_nx_to_rx(graph)

        # REQ 5: Bypass Secuencial (Main Thread)
        if self.sequential:
            log.info("Ejecutando Rustworkx en modo SECUENCIAL (Main Thread)...")
            od_pairs_pbar = tqdm(od_pairs, desc="RX Rutas (Secuencial)", unit="pairs")

            chunk_res = process_chunk_rx(rx_graph, od_pairs, nx_to_rx, rx_to_nx, self.k)
            for k_od, v_paths in chunk_res:
                results[k_od] = v_paths
            return results

        # Modo Paralelo (Threads)
        chunks = [od_pairs[i:i + self.chunk_size] for i in range(0, len(od_pairs), self.chunk_size)]

        log.info(f"RX: {len(chunks)} lotes -> ThreadPoolExecutor({self.n_workers})")
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Mapeamos cada future a la longitud de su chunk para actualizar la barra correctamente
            future_to_len = {
                executor.submit(process_chunk_rx, rx_graph, chunk, nx_to_rx, rx_to_nx, self.k): len(chunk)
                for chunk in chunks
            }

            # Barra de progreso total basada en el número total de pares OD
            with tqdm(total=len(od_pairs), desc="RX Rutas (Paralelo)", unit="pairs") as pbar:
                for future in as_completed(future_to_len):
                    chunk_len = future_to_len[future]
                    try:
                        chunk_res = future.result()
                        for k_od, v_paths in chunk_res:
                            results[k_od] = v_paths
                    except Exception as e:
                        log.error(f"Error crítico en chunk RX: {e}")
                    finally:
                        # Actualizamos la barra sumando la cantidad de pares procesados en este chunk
                        pbar.update(chunk_len)
        return results

    def _convert_nx_to_rx(self, nx_graph: nx.Graph):
        # Conversión optimizada
        rx_graph = rx.PyDiGraph()
        nx_nodes = list(nx_graph.nodes())
        rx_indices = rx_graph.add_nodes_from(nx_nodes)

        nx_to_rx = {n: i for n, i in zip(nx_nodes, rx_indices)}
        rx_to_nx = {i: n for n, i in zip(nx_nodes, rx_indices)}

        edges_to_add = []
        for u, v, data in nx_graph.edges(data=True):
            try:
                # Asegurar float para RX
                w = float(data.get(self.weight, 1.0))
            except (ValueError, TypeError):
                log.error(f"Peso inválido en arista {u}->{v}. Usando 1.0 por defecto.")
                w = 1.0
            edges_to_add.append((nx_to_rx[u], nx_to_rx[v], w))

        rx_graph.add_edges_from(edges_to_add)
        return rx_graph, nx_to_rx, rx_to_nx


    # --- ESTRATEGIA B: NETWORKX (Procesos + Python) ---
    def _execute_networkx(self, graph: nx.Graph, od_pairs: List[Tuple[int, int]]) -> Dict:
        results = {}

        # REQ 5: Bypass Secuencial (Main Thread)
        if self.sequential:
            log.info("Ejecutando NetworkX en modo SECUENCIAL (Main Thread)...")
            # Seteamos la variable global manualmente sin multiprocessing
            global global_graph_ref
            global_graph_ref = graph

            # Envolvemos od_pairs en tqdm para ver el progreso par a par
            od_pairs_pbar = tqdm(od_pairs, desc="NX Rutas (Secuencial)", unit="pairs")

            # Procesamos todo en un solo chunk
            chunk_res = process_chunk_nx((od_pairs, self.k, self.weight))
            for k_od, v_paths in chunk_res:
                results[k_od] = v_paths
            return results

        # Modo Paralelo (Procesos)
        # REQ 4: Yen corre dentro de los workers, los workers corren en paralelo.
        chunks = [od_pairs[i:i + self.chunk_size] for i in range(0, len(od_pairs), self.chunk_size)]
        tasks = [(chunk, self.k, self.weight) for chunk in chunks]

        log.info(f"NX: {len(chunks)} lotes -> ProcessPoolExecutor({self.n_workers})")
        with ProcessPoolExecutor(max_workers=self.n_workers, initializer=init_worker_nx, initargs=(graph,)) as executor:
            # Mapeamos cada future a la longitud de su chunk
            # task[0] es el chunk de pares OD
            future_to_len = {executor.submit(process_chunk_nx, t): len(t[0]) for t in tasks}

            # Barra de progreso total basada en el número total de pares OD
            with tqdm(total=len(od_pairs), desc="NX Rutas (Paralelo)", unit="pairs") as pbar:
                for future in as_completed(future_to_len):
                    chunk_len = future_to_len[future]
                    try:
                        chunk_res = future.result()
                        for k_od, v_paths in chunk_res:
                            results[k_od] = v_paths
                    except Exception as e:
                        log.error(f"Error crítico en chunk NX: {e}")
                    finally:
                        # Avanzamos la barra según el tamaño del chunk completado
                        pbar.update(chunk_len)
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

from collections import Counter
import numpy as np
import torch

class RoutePadder:
    def __init__(self, padding_token="<PAD>", pad_idx=0, unk_token="<UNK>", unk_idx=1):
        """
        Inicializa el procesador.

        Args:
            padding_token: String para representar el relleno.
            pad_idx: Entero reservado para el padding (usualmente 0).
            unk_token: String para nodos desconocidos (por seguridad).
            unk_idx: Entero reservado para desconocidos.
        """

        self.pad_token = padding_token
        self.pad_idx = pad_idx
        self.unk_token = unk_token
        self.unk_idx = unk_idx

        # Mapeos (se llenarán en el fit)
        self.node2idx = {padding_token: pad_idx, unk_token: unk_idx}
        self.idx2node = {pad_idx: padding_token, unk_idx: unk_token}

        # Estadísticas
        self.max_len_detected = 0
        self.vocab_size = 2  # Empieza en 2 porque ya tenemos PAD y UNK

    def fit(self, raw_data, fixed_max_len=None):
        """
        1. Escanea todos los nodos para crear el diccionario (vocabulario).
        2. Determina la longitud máxima de las rutas.

        Args:
            raw_data: Tu diccionario {(O, D): [[n1, n2...], ...]}
            fixed_max_len: (Opcional) Si quieres forzar un recorte (ej. 50).
                           Si es None, usa la ruta más larga encontrada.
        """
        all_lengths = []
        unique_nodes = set()

        print("--- Analizando datos y construyendo vocabulario ---")

        for key, k_paths in raw_data.items():
            for path in k_paths:
                # Guardar longitud para estadísticas
                all_lengths.append(len(path))
                # Guardar nodos únicos
                unique_nodes.update(path)

        # Convertir set a lista ordenada para determinismo
        sorted_nodes = sorted(list(unique_nodes))

        # Llenar diccionarios
        current_idx = self.vocab_size  # Empezar después de PAD y UNK
        for node in sorted_nodes:
            self.node2idx[node] = current_idx
            self.idx2node[current_idx] = node
            current_idx += 1

        self.vocab_size = len(self.node2idx)
        self.max_len_detected = max(all_lengths)

        # Definir la longitud de corte final
        self.seq_len = fixed_max_len if fixed_max_len is not None else self.max_len_detected

        print(f"Vocabulario creado: {self.vocab_size} nodos únicos.")
        print(f"Ruta más larga encontrada: {self.max_len_detected}")
        print(f"Longitud de tensor establecida en: {self.seq_len}")
        print(f"Longitud promedio: {np.mean(all_lengths):.2f}")

    def transform(self, raw_data):
        """
        Convierte el diccionario crudo en Tensores con Padding.
        Devuelve una estructura [Num_Pares_OD, K_Rutas, Longitud_Fija]
        """
        od_pairs = list(raw_data.keys())
        num_od = len(od_pairs)
        # Asumimos que todos tienen el mismo K (basado en tu descripción K=10)
        # Tomamos el K del primer elemento para configurar dimensiones
        k_routes = len(next(iter(raw_data.values())))

        # 1. Tensor de Datos (Inicializado con PAD_IDX = 0)
        # Shape: [11664, 10, MAX_LEN]
        tensor_out = torch.full((num_od, k_routes, self.seq_len), self.pad_idx, dtype=torch.int32)

        # 2. Tensor de Máscara (1 es dato real, 0 es padding)
        # Útil para Transformers (Attention Mask)
        mask_out = torch.zeros((num_od, k_routes, self.seq_len), dtype=torch.bool)  # o Bool

        # 3. Tensor de Identificadores OD (Para saber quién es quién)
        # Guardaremos los índices de los nodos O y D
        od_indices = torch.zeros((num_od, 2), dtype=torch.int32)

        print("\n--- Transformando a Tensores ---")

        for i, (od_key, paths) in enumerate(raw_data.items()):
            # Guardar quién es el Origen y Destino
            o_id = self.node2idx.get(od_key[0], self.unk_idx)
            d_id = self.node2idx.get(od_key[1], self.unk_idx)
            od_indices[i] = torch.tensor([o_id, d_id])

            for k, path in enumerate(paths):
                # Recortar si excede seq_len
                path_truncated = path[:self.seq_len]

                # Convertir strings a ints
                idx_seq = [self.node2idx.get(n, self.unk_idx) for n in path_truncated]

                # Longitud real de esta ruta
                curr_len = len(idx_seq)

                # Asignar al tensor (el resto ya es 0/PAD por la inicialización)
                tensor_out[i, k, :curr_len] = torch.tensor(idx_seq)

                # Actualizar la máscara (1 donde hay datos)
                mask_out[i, k, :curr_len] = 1.0

        return {
            "routes": tensor_out,  # [N_OD, K, Len] - Los datos
            "mask": mask_out,  # [N_OD, K, Len] - Dónde mirar
            "od_ids": od_indices  # [N_OD, 2]      - Metadatos
        }

    def decode(self, token_ids):
        """Ayuda visual: Convierte una secuencia de ints de vuelta a strings"""
        # Eliminar padding (0) para visualización
        tokens = [self.idx2node.get(idx.item(), self.unk_token) for idx in token_ids if idx != self.pad_idx]
        return tokens


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
        self.export_as_dict = self.config.get('export_as_dict', True)

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

        # 3. Vectorizar y guardar
        if routes and self.export_as_dict:
            log.info("Guardando rutas calculadas en cache como diccionatio...")
            self.cache.save(routes)
            final_routes = routes

        elif routes and not self.export_as_dict:
            log.info("Guardando rutas calculadas en cache como tensores...")
            # Vectorización
            # 1. Instanciar la clase
            processor = RoutePadder()

            # 2. Fit (Aprender vocabulario y longitudes)
            # Aquí detectará que la ruta más larga es 6 y configurará todo acorde.
            processor.fit(routes)

            # 3. Transform (Crear los tensores)
            tensors = processor.transform(routes)
            final_routes = tensors

        else:
            log.warning("No se calcularon rutas. El conjunto de datos resultante estará vacío.")
            final_routes = {}

        return final_routes


# --- ENTRY POINT PARA EJECUCIÓN STANDALONE (Hydra) ---
@hydra.main(version_base=None, config_path="../../../configs",
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