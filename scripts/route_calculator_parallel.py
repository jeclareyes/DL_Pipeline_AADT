"""
Lightweight parallel K-shortest routes runner for Windows.
This script avoids sending the whole NetworkX graph object in Pool.initargs
(which forces pickling large objects). Instead, each worker loads the graph
from a pickle path inside the initializer.

Usage (from repo root):
    python scripts\route_calculator_parallel.py --graph-path data/processed/Linköping/Linköping_graph.pkl --n-od 50 --k 5 --n-workers 4

The script prints timings for sequential and parallel execution so you can
compare and verify that the parallel mode actually runs on Windows.
"""
import argparse
import pickle
import time
import numpy as np
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm

# ============= VARIABLES GLOBALES PARA WORKERS =============
# Worker defaults to keep static type-checkers happy; real values set in initializer
_worker_graph = None
_worker_k = 10
_worker_weight = 'free_flow_time'


def _init_worker(graph_path, k, weight):
    """Initializer for each process: load the graph from disk once per worker."""
    global _worker_graph, _worker_k, _worker_weight
    # Load graph locally in the worker process
    with open(graph_path, 'rb') as f:
        _worker_graph = pickle.load(f)
    _worker_k = k
    _worker_weight = weight


def k_shortest_paths(graph, origin, destination, k=10, weight='free_flow_time'):
    try:
        paths = []
        gen = nx.shortest_simple_paths(graph, origin, destination, weight=weight)
        for i, p in enumerate(gen):
            if i >= k:
                break
            paths.append(p)
        return paths
    except Exception:
        return []


def _calculate_od_pair(od_pair):
    origin, destination = od_pair
    try:
        return k_shortest_paths(_worker_graph, origin, destination, k=_worker_k, weight=_worker_weight)
    except Exception as e:
        print(f"Worker error for {origin}->{destination}: {e}")
        return []


def execute_k_routes_sequential(od_pairs, graph, k=10, weight='free_flow_time'):
    results = []
    for origin, destination in tqdm(od_pairs, desc="Calculando rutas (seq)"):
        try:
            results.append(k_shortest_paths(graph, origin, destination, k=k, weight=weight))
        except Exception as e:
            print(f"Error {origin}->{destination}: {e}")
            results.append([])
    return results


def execute_k_routes_parallel(od_pairs, graph_path, k=10, weight='free_flow_time', n_workers=None, chunksize=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    if chunksize is None:
        chunksize = max(1, len(od_pairs) // (n_workers * 4))

    print("\n" + "="*60)
    print("Configuración de paralelización:")
    print(f"  - CPUs disponibles: {mp.cpu_count()}")
    print(f"  - Procesos a usar: {n_workers}")
    print(f"  - Pares OD: {len(od_pairs)}")
    print(f"  - Chunksize: {chunksize}")
    print(f"  - Número de chunks: ~{len(od_pairs) // chunksize}")
    print("="*60 + "\n")

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(graph_path, k, weight)) as pool:
        results = list(tqdm(pool.imap(_calculate_od_pair, od_pairs, chunksize=chunksize), total=len(od_pairs), desc="Calculando rutas (par)"))

    return results


def create_routes_tensor(all_k_routes, k_value, padding_value=-1):
    max_length = max((len(route) for od_routes in all_k_routes for route in od_routes), default=0)
    num_od = len(all_k_routes)
    routes_tensor = np.full((num_od, k_value, max_length), padding_value, dtype=int)
    for i, od_routes in enumerate(all_k_routes):
        for j, route in enumerate(od_routes[:k_value]):
            routes_tensor[i, j, :len(route)] = route
    return routes_tensor


def load_graph(graph_path):
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def build_test_od_pairs(graph, n_pairs=100):
    # Use 'taz' and 'aux' types if present, fallback to arbitrary nodes
    taz_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'taz']
    aux_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'aux']
    combined = taz_nodes + aux_nodes
    if not combined:
        combined = list(graph.nodes())
    od_pairs = []
    for i in range(n_pairs):
        o = combined[i % len(combined)]
        d = combined[(i*7) % len(combined)]
        if o == d:
            d = combined[(i*7 + 1) % len(combined)]
        od_pairs.append((o, d))
    return od_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-path', type=str, default='C:/Users/jecla/Documents/Barcelona_GNN/data/processed/Linköping/Linköping_graph.pkl')
    # parser.add_argument('--graph-path', type=str, required=True)
    parser.add_argument('--n-od', type=int, default=11664)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--chunksize', type=int, default=None)
    args = parser.parse_args()

    mp.freeze_support()

    print("Loading graph (main process)...")
    graph = load_graph(args.graph_path)
    print(f"Graph loaded: {graph}")

    od_pairs = build_test_od_pairs(graph, n_pairs=args.n_od)
    print(f"Built {len(od_pairs)} OD pairs for testing")

    # Sequential run (small sample)
    t0 = time.time()
    seq = execute_k_routes_sequential(od_pairs, graph, k=args.k)
    t_seq = time.time() - t0
    print(f"Sequential time: {t_seq:.2f}s ({t_seq/len(od_pairs)*1000:.2f}ms per OD)")

    # Parallel run
    t0 = time.time()
    par = execute_k_routes_parallel(od_pairs, args.graph_path, k=args.k, n_workers=args.n_workers, chunksize=args.chunksize)
    t_par = time.time() - t0
    print(f"Parallel time: {t_par:.2f}s ({t_par/len(od_pairs)*1000:.2f}ms per OD)")

    print(f"Speedup: {t_seq / t_par:.2f}x (efficiency: {t_seq / t_par / mp.cpu_count() * 100:.1f}%)")

    routes_tensor = create_routes_tensor(par, args.k)
    print(f"Routes tensor shape: {routes_tensor.shape}")


if __name__ == '__main__':
    main()

"""
C:/Users/jecla/Documents/Barcelona_GNN/scripts/route_calculator_parallel.py --graph-path C:/Users/jecla/Documents/Barcelona_GNN/data/processed/Linköping/Linköping_graph.pkl --n-od 1000 --k 10 --n-workers 4
"""
