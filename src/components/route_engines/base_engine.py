from abc import ABC, abstractmethod
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from tqdm import tqdm

Route = List[int]
ODPair = tuple[int, int]
RouteBatchResult = list[tuple[ODPair, list[Route]]]

class RouteEngine(ABC):
    """
    Abstract base class for route calculation engines.
    """
    
    @abstractmethod
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the engine with a NetworkX graph.
        Engines that use a different backend (like RustworkX) should perform
        the conversion from the NetworkX graph during initialization.
        """
        pass

    @abstractmethod
    def get_k_routes(
        self,
        origin_id: int,
        destination_id: int,
        k: int,
        weight: str,
        constraints: Dict[str, bool],
        connector_link_types: Set[int],
    ) -> List[Route]:
        """Return up to ``k`` routes for one OD batch.

        The asset builder invokes this method separately for interzonal
        simple paths and intrazonal cycles. The mapping therefore describes a
        single OD class: ``allow_loops`` is false for interzonal routing and
        true only for an explicitly selected intrazonal cycle policy.
        """
        pass


# Centralized OD orchestration shared by every route engine.  The engine is
# created in each worker so backend-specific state never leaks across
# processes and the asset layer remains independent of routing details.
def _make_engine(engine_name: str, graph: nx.DiGraph) -> "RouteEngine":
    from . import get_route_engine

    return get_route_engine(engine_name, graph)


_WORKER_ENGINE: RouteEngine | None = None
_WORKER_K = 1
_WORKER_WEIGHT = "free_flow_time"
_WORKER_CONSTRAINTS: dict[str, bool] = {}
_WORKER_CONNECTORS: set[int] = set()


def _initialize_worker(
    engine_name: str,
    graph: nx.DiGraph,
    k: int,
    weight: str,
    constraints: dict[str, bool],
    connector_link_types: set[int],
) -> None:
    global _WORKER_ENGINE, _WORKER_K, _WORKER_WEIGHT
    global _WORKER_CONSTRAINTS, _WORKER_CONNECTORS
    _WORKER_ENGINE = _make_engine(engine_name, graph)
    _WORKER_K = int(k)
    _WORKER_WEIGHT = str(weight)
    _WORKER_CONSTRAINTS = dict(constraints)
    _WORKER_CONNECTORS = set(connector_link_types)


def _solve_batch(batch: list[ODPair]) -> RouteBatchResult:
    if _WORKER_ENGINE is None:
        raise RuntimeError("Route worker was not initialized.")
    return [
        (
            (int(origin), int(destination)),
            _WORKER_ENGINE.get_k_routes(
                origin_id=int(origin),
                destination_id=int(destination),
                k=_WORKER_K,
                weight=_WORKER_WEIGHT,
                constraints=_WORKER_CONSTRAINTS,
                connector_link_types=_WORKER_CONNECTORS,
            ),
        )
        for origin, destination in batch
    ]


def _chunks(items: list[ODPair], batch_size: int) -> list[list[ODPair]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def generate_routes_by_od(
    *,
    graph: nx.DiGraph,
    od_pairs: Iterable[ODPair],
    engine_name: str,
    k: int,
    weight: str,
    constraints: dict[str, bool],
    connector_link_types: set[int],
    parallel: bool = True,
    workers: int | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> dict[ODPair, list[Route]]:
    """Generate OD routes through a centralized engine, optionally in processes."""
    tasks = [(int(origin), int(destination)) for origin, destination in od_pairs]
    if not tasks:
        return {}

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = int(workers)
    use_parallel = bool(parallel) and workers > 1 and len(tasks) > batch_size

    results: dict[ODPair, list[Route]] = {}
    progress = tqdm(
        total=len(tasks),
        desc=f"Routing {engine_name}",
        unit="od",
        disable=not show_progress,
    )
    try:
        if not use_parallel:
            engine = _make_engine(engine_name, graph)
            for origin, destination in tasks:
                results[(origin, destination)] = engine.get_k_routes(
                    origin_id=origin,
                    destination_id=destination,
                    k=int(k),
                    weight=weight,
                    constraints=constraints,
                    connector_link_types=connector_link_types,
                )
                progress.update(1)
        else:
            batches = _chunks(tasks, int(batch_size))
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(
                    engine_name,
                    graph,
                    int(k),
                    weight,
                    constraints,
                    connector_link_types,
                ),
            ) as executor:
                futures = [executor.submit(_solve_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    for od_pair, routes in future.result():
                        results[od_pair] = routes
                        progress.update(1)
    finally:
        progress.close()

    return {od_pair: results[od_pair] for od_pair in tasks}
