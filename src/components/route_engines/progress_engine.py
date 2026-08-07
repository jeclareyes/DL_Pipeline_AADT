#src/components/route_engines/progress_engine.py

"""Progress-aware wrapper for route-generation engines.

This module provides :class:`ProgressRouteEngine`, a small decorator-style
wrapper around any object that implements the project's :class:`RouteEngine`
interface.

Role within the project
-----------------------
Route generation can require solving thousands of origin-destination (OD)
pairs. The concrete route engines—such as ``NetworkXEngine``,
``RustworkXEngine``, and ``RustworkXOptimizedEngine``—are responsible only for
computing routes. They should not need to know how progress is displayed in the
terminal.

``ProgressRouteEngine`` keeps those responsibilities separate:

* the wrapped engine performs the routing algorithm;
* this wrapper owns and updates the ``tqdm`` progress bar;
* logging emitted while the bar is active is redirected through ``tqdm`` so
  warnings and informational messages do not leave duplicated or incomplete
  progress-bar lines in the terminal.

The wrapper follows the same ``get_k_routes`` call signature as ``RouteEngine``.
As a result, calling code can use it almost exactly as it would use a concrete
route engine, while gaining terminal progress reporting without modifying the
routing algorithm itself.

Typical usage
-------------

.. code-block:: python

    engine = get_route_engine("rustworkx_optimized", graph)

    with ProgressRouteEngine(
        engine=engine,
        total=len(od_pairs),
        desc="Routing rustworkx_optimized",
    ) as progress_engine:
        for origin_id, destination_id in od_pairs:
            routes = progress_engine.get_k_routes(
                origin_id=origin_id,
                destination_id=destination_id,
                k=10,
                weight="free_flow_time",
                constraints=constraints,
                connector_link_types=connector_link_types,
            )

Using the wrapper as a context manager is strongly recommended. It guarantees
that both the progress bar and the temporary logging redirection are restored
when processing finishes, including when an exception interrupts the loop.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .base_engine import Route, RouteEngine


class ProgressRouteEngine:
    """Add terminal progress reporting to an existing route engine.

    The class uses composition rather than inheritance: it stores a concrete
    ``RouteEngine`` instance and delegates every route-calculation request to
    that instance. After each completed OD request, it advances a ``tqdm``
    progress bar by one unit.

    While the wrapper is active, Python logging output is redirected through
    ``tqdm``. This matters because a normal ``StreamHandler`` writes directly to
    the terminal and can interrupt the carriage-return mechanism used by a
    progress bar. ``logging_redirect_tqdm`` temporarily replaces compatible
    console handlers so each log message is printed cleanly above the bar, after
    which the current bar is redrawn as the terminal's last line.

    Parameters
    ----------
    engine:
        Concrete route engine that performs the actual K-shortest-path
        computation. It must provide the ``RouteEngine.get_k_routes`` interface.
    total:
        Total number of OD-pair calculations expected during this run. This is
        used by ``tqdm`` to calculate percentages, rates, and remaining time.
    desc:
        Human-readable label displayed to the left of the progress bar.
    **tqdm_kwargs:
        Additional keyword arguments forwarded directly to ``tqdm``. Examples
        include ``dynamic_ncols=True``, ``leave=False``, or ``mininterval=0.5``.

    Notes
    -----
    This wrapper deliberately does not inspect, validate, or alter routes. Its
    only responsibilities are delegation, progress accounting, and terminal
    output coordination.
    """

    def __init__(
        self,
        engine: RouteEngine,
        total: int,
        desc: str = "Routing",
        **tqdm_kwargs: Any,
    ) -> None:
        """Initialize the wrapper, logging redirection, and progress bar.

        Logging redirection is started before the progress bar is created so
        that any console log produced during bar initialization is handled
        consistently. If ``tqdm`` itself fails to initialize, the logging
        context is immediately restored before the exception is re-raised.
        """
        self._engine = engine
        self._closed = False

        # ``logging_redirect_tqdm`` is normally used in a ``with`` statement.
        # Here its lifetime must match the lifetime of this wrapper, so we enter
        # it manually in ``__init__`` and leave it in ``close``.
        self._logging_context = logging_redirect_tqdm()
        self._logging_context.__enter__()

        try:
            # One progress unit represents one completed OD-pair request.
            # ``disable=None`` lets tqdm decide automatically whether progress
            # rendering is appropriate for the current output stream.
            self._tqdm = tqdm(
                total=total,
                desc=desc,
                unit="od",
                disable=None,
                **tqdm_kwargs,
            )
        except Exception:
            # Avoid leaving Python's logging handlers redirected if progress-bar
            # construction fails before the object becomes usable.
            self._logging_context.__exit__(None, None, None)
            raise

    def get_k_routes(
        self,
        origin_id: int,
        destination_id: int,
        k: int,
        weight: str,
        constraints: dict[str, bool],
        connector_link_types: set[int],
    ) -> list[Route]:
        """Calculate routes for one OD pair and advance the progress bar.

        All routing arguments are passed unchanged to the wrapped engine. The
        bar is updated only after the delegated call returns successfully. This
        means the displayed count represents completed OD calculations rather
        than merely started calculations.

        Parameters
        ----------
        origin_id:
            Identifier of the origin node.
        destination_id:
            Identifier of the destination node.
        k:
            Maximum number of routes requested from the wrapped engine.
        weight:
            Edge attribute used as the route-cost weight.
        constraints:
            Route-generation options such as duplicate, loop, and intrazonal
            route permissions.
        connector_link_types:
            Link-type identifiers treated as zone connectors by the engine.

        Returns
        -------
        list[Route]
            Routes returned by the wrapped engine. Each route is represented as
            an ordered list of node identifiers.

        Raises
        ------
        Exception
            Any exception raised by the wrapped engine is propagated unchanged.
            In that case, the progress count is not incremented.
        """
        result = self._engine.get_k_routes(
            origin_id=origin_id,
            destination_id=destination_id,
            k=k,
            weight=weight,
            constraints=constraints,
            connector_link_types=connector_link_types,
        )

        self._tqdm.update(1)
        return result

    def set_description(self, desc: str) -> None:
        """Replace the text label displayed by the active progress bar.

        This is useful when the same wrapper remains active while the caller
        changes phases, engines, batches, or other contextual information.
        """
        self._tqdm.set_description(desc)

    def close(self) -> None:
        """Close the progress bar and restore the original logging handlers.

        The method is idempotent: calling it more than once has no effect after
        the first successful call. Idempotency is useful because cleanup may be
        triggered explicitly and again through context-manager exit.

        The logging context is restored in a ``finally`` block so handler
        restoration still occurs if ``tqdm.close`` unexpectedly raises.
        """
        if self._closed:
            return

        self._closed = True

        try:
            self._tqdm.close()
        finally:
            self._logging_context.__exit__(None, None, None)

    def __enter__(self) -> ProgressRouteEngine:
        """Return this wrapper when entering a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release terminal resources when leaving a ``with`` block.

        Exception information is accepted to satisfy the context-manager
        protocol, but exceptions are not suppressed. Returning ``None`` allows
        any active exception to continue propagating after cleanup.
        """
        self.close()