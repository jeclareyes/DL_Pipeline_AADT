# src/data_handling/readers/tntp_routes_reader.py

"""
TNTP Routes Reader
==================

This module reads and normalizes route files in a TNTP-like format.

Project context
---------------
In the AADT / traffic assignment pipeline, the routes file defines the feasible
route set for each origin-destination pair. These routes later become:

- the route-choice space used by route-based traffic assignment models;
- the route-link incidence structure used by neural network models;
- the topological bridge between OD demand and link-level traffic flows.

This reader is intentionally limited to route data ingestion and normalization.
It does not build graphs, compute routes, validate link existence, create sparse
route-link tensors, or prepare PyTorch objects. Those responsibilities belong to
downstream builders, validators, adapters and artifact builders.

Supported input format
----------------------
The preferred project format is one line per OD pair, where each line contains
a Python-like list of node-based routes:

    [[1, 73, 49, 66, 4], [1, 68, 69, 49, 66, 4]]
    [[1, 68, 6], [1, 73, 70, 6]]
    []

In this compact format, OD identifiers are not stored in the route file itself.
Therefore, the caller should provide either:
- od_pairs: explicit ordered OD pairs; or
- zone_ids: zone IDs used to reconstruct ordered OD pairs.

The reader also supports explicit OD lines, for example:

    1 4 : [[1, 73, 49, 66, 4], [1, 68, 69, 49, 66, 4]]
    OD 1 4 : [[1, 73, 49, 66, 4]]

Output
------
The main output is a RoutesReadResult containing:
- routes_by_od: dictionary {(origin, destination): [[route_1], [route_2], ...]};
- routes_df: long-format DataFrame with one row per route;
- metadata: dictionary with file path, number of OD pairs, number of routes,
  route count summaries and parsing information.

Design principles
-----------------
- Do only one thing: read and normalize route data.
- Receive only a TNTP route file as the file input. No pickle support.
- Preserve the route representation expected by RouteModelAdapter:
  {(origin_node, destination_node): [[node_1, node_2, ...], ...]}.
- Fail early when the compact format is used without OD reconstruction metadata.
- Avoid executing arbitrary code. Route lists are parsed with ast.literal_eval.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


logger = logging.getLogger(__name__)

ODPair = Tuple[int, int]
Route = List[int]
RoutesByOD = Dict[ODPair, List[Route]]


@dataclass(frozen=True)
class RoutesReadResult:
    """
    Container returned by TNTPRoutesReader.

    Attributes
    ----------
    routes_by_od : RoutesByOD
        Dictionary mapping each OD pair to a list of node-based routes.

    routes_df : pd.DataFrame
        Long-format route table with one row per route.

    metadata : Dict[str, Any]
        Metadata describing the parsing result, including source file,
        number of OD pairs, number of routes and route-count summaries.
    """

    routes_by_od: RoutesByOD
    routes_df: pd.DataFrame
    metadata: Dict[str, Any]


class TNTPRoutesReader:
    """
    Read and normalize a TNTP-like route file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the route file. Only text-based TNTP-like files are supported.

    od_pairs : Optional[Sequence[Tuple[int, int]]], default=None
        Ordered OD pairs used to map compact route lines to OD identities.
        Required when the route file does not explicitly include OD IDs.

    zone_ids : Optional[Sequence[int]], default=None
        Zone IDs used to reconstruct ordered OD pairs when od_pairs is not
        provided. OD pairs are generated using the same convention used by
        scenario_creation_pipeline.handle_routes:
        for origin in zone_ids:
            for destination in zone_ids:
                if origin != destination:
                    include pair.

    strict : bool, default=True
        If True, malformed lines, invalid routes or OD-count mismatches raise
        errors. If False, invalid entries are skipped when possible.

    allow_empty_routes : bool, default=True
        If True, an OD pair may have an empty route list, represented as [].
        This is useful for diagnostics because the route-generation stage may
        explicitly record OD pairs without feasible paths.

    max_routes_per_od : Optional[int], default=None
        Optional limit on the number of routes retained per OD pair.
    """

    COMMENT_PREFIXES = ("#", "//", "~")

    def __init__(
        self,
        path: Union[str, Path],
        od_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        zone_ids: Optional[Sequence[int]] = None,
        strict: bool = True,
        allow_empty_routes: bool = True,
        max_routes_per_od: Optional[int] = None,
    ) -> None:
        self.path = Path(path).resolve(strict=False)
        self.od_pairs = self._normalize_od_pairs(od_pairs) if od_pairs is not None else None
        self.zone_ids = [int(zone_id) for zone_id in zone_ids] if zone_ids is not None else None
        self.strict = bool(strict)
        self.allow_empty_routes = bool(allow_empty_routes)
        self.max_routes_per_od = max_routes_per_od

        if not self.path.exists():
            raise FileNotFoundError(f"Routes file not found: {self.path}")

        if self.max_routes_per_od is not None and self.max_routes_per_od <= 0:
            raise ValueError("max_routes_per_od must be positive when provided.")

    def read(self) -> RoutesReadResult:
        """
        Read, parse, normalize and validate the route file.

        Returns
        -------
        RoutesReadResult
            Object containing routes_by_od, routes_df and metadata.

        Raises
        ------
        FileNotFoundError
            If the route file does not exist.

        ValueError
            If route lines cannot be mapped to OD pairs or routes are invalid.
        """

        logger.info("Reading TNTP routes file: %s", self.path)

        lines = self._read_lines()
        route_lines = self._extract_route_lines(lines)

        if not route_lines:
            raise ValueError(f"No route records found in file: {self.path}")

        parsed_records = self._parse_route_lines(route_lines)
        routes_by_od = self._build_routes_by_od(parsed_records)
        routes_df = self._build_routes_dataframe(routes_by_od)
        self._validate(routes_by_od)

        metadata = self._build_metadata(
            routes_by_od=routes_by_od,
            routes_df=routes_df,
            num_route_lines=len(route_lines),
        )

        logger.info(
            "Routes file loaded successfully | od_pairs=%d | routes=%d",
            metadata["num_od_pairs"],
            metadata["num_routes"],
        )

        return RoutesReadResult(
            routes_by_od=routes_by_od,
            routes_df=routes_df,
            metadata=metadata,
        )

    def _read_lines(self) -> List[str]:
        """
        Read all lines from the routes file.

        Returns
        -------
        List[str]
            Raw file lines.
        """

        with self.path.open("r", encoding="utf-8", errors="replace") as file:
            return file.readlines()

    def _extract_route_lines(self, lines: List[str]) -> List[str]:
        """
        Remove empty lines, metadata lines and comments.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        Returns
        -------
        List[str]
            Cleaned route lines.
        """

        route_lines = []
        inside_metadata = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            upper = stripped.upper()

            if upper.startswith("<") and upper != "<END OF METADATA>":
                inside_metadata = True
                continue

            if upper == "<END OF METADATA>":
                inside_metadata = False
                continue

            if inside_metadata:
                continue

            if stripped.startswith(self.COMMENT_PREFIXES):
                continue

            if stripped.endswith(";"):
                stripped = stripped[:-1].strip()

            route_lines.append(stripped)

        return route_lines

    def _parse_route_lines(self, route_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse route lines into intermediate records.

        Each record contains:
        - optional origin/destination if explicitly available;
        - route list parsed from the line;
        - source line number.

        Parameters
        ----------
        route_lines : List[str]
            Cleaned route lines.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed route-line records.
        """

        records = []

        for line_number, line in enumerate(route_lines, start=1):
            try:
                origin, destination, route_text = self._split_optional_od_prefix(line)
                routes = self._parse_routes_literal(route_text)
                routes = self._normalize_routes(routes)

                if self.max_routes_per_od is not None:
                    routes = routes[: self.max_routes_per_od]

                records.append(
                    {
                        "line_number": line_number,
                        "origin": origin,
                        "destination": destination,
                        "routes": routes,
                    }
                )

            except Exception as exc:
                message = f"Could not parse route line {line_number}: {line}. Reason: {exc}"

                if self.strict:
                    raise ValueError(message) from exc

                logger.warning(message)

        return records

    def _split_optional_od_prefix(
        self,
        line: str,
    ) -> Tuple[Optional[int], Optional[int], str]:
        """
        Split a route line into optional OD prefix and route-list text.

        Supported explicit formats include:
        - "1 4 : [[1, 2, 4]]"
        - "OD 1 4 : [[1, 2, 4]]"
        - "origin=1 destination=4 : [[1, 2, 4]]"

        If no explicit OD prefix is found, the full line is treated as route text.

        Parameters
        ----------
        line : str
            Route line.

        Returns
        -------
        Tuple[Optional[int], Optional[int], str]
            Origin, destination and route-list text.
        """

        # Compact format: the line starts directly with the route list.
        if line.lstrip().startswith("["):
            return None, None, line

        if ":" not in line:
            raise ValueError("Route line does not contain a route list or OD separator ':'.")

        prefix, route_text = line.split(":", 1)
        prefix = prefix.strip()
        route_text = route_text.strip()

        patterns = [
            r"^OD\s+(?P<origin>\d+)\s+(?P<destination>\d+)$",
            r"^(?P<origin>\d+)\s+(?P<destination>\d+)$",
            r"^origin\s*=\s*(?P<origin>\d+)\s+destination\s*=\s*(?P<destination>\d+)$",
            r"^origin\s+(?P<origin>\d+)\s+destination\s+(?P<destination>\d+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, prefix, flags=re.IGNORECASE)

            if match:
                return (
                    int(match.group("origin")),
                    int(match.group("destination")),
                    route_text,
                )

        raise ValueError(f"Unsupported OD prefix format: '{prefix}'.")

    def _parse_routes_literal(self, route_text: str) -> Any:
        """
        Parse a Python-like route-list literal safely.

        Parameters
        ----------
        route_text : str
            Text containing a list of routes.

        Returns
        -------
        Any
            Parsed Python object.

        Notes
        -----
        ast.literal_eval is used instead of eval to avoid executing arbitrary code.
        """

        try:
            return ast.literal_eval(route_text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid route-list literal: {route_text}") from exc

    def _normalize_routes(self, raw_routes: Any) -> List[Route]:
        """
        Normalize parsed route objects to List[List[int]].

        Parameters
        ----------
        raw_routes : Any
            Parsed object from ast.literal_eval.

        Returns
        -------
        List[Route]
            Normalized list of node-based routes.

        Raises
        ------
        ValueError
            If the parsed object is not a valid route list.
        """

        if raw_routes == []:
            return []

        if not isinstance(raw_routes, list):
            raise ValueError("Routes must be represented as a list.")

        normalized_routes: List[Route] = []

        for route_idx, route in enumerate(raw_routes):
            if not isinstance(route, list):
                raise ValueError(
                    f"Route at index {route_idx} is not a list: {route}"
                )

            if len(route) < 2:
                raise ValueError(
                    f"Route at index {route_idx} must contain at least two nodes: {route}"
                )

            try:
                normalized_route = [int(node) for node in route]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Route at index {route_idx} contains non-integer node IDs: {route}"
                ) from exc

            normalized_routes.append(normalized_route)

        return normalized_routes

    def _build_routes_by_od(self, parsed_records: List[Dict[str, Any]]) -> RoutesByOD:
        """
        Build the canonical routes_by_od dictionary.

        If the file contains explicit OD prefixes, those are used. Otherwise,
        OD pairs are reconstructed from od_pairs or zone_ids.

        Parameters
        ----------
        parsed_records : List[Dict[str, Any]]
            Intermediate parsed records.

        Returns
        -------
        RoutesByOD
            Dictionary mapping OD pairs to route lists.
        """

        if not parsed_records:
            return {}

        has_explicit_od = all(
            record["origin"] is not None and record["destination"] is not None
            for record in parsed_records
        )

        has_compact_od = all(
            record["origin"] is None and record["destination"] is None
            for record in parsed_records
        )

        if not (has_explicit_od or has_compact_od):
            raise ValueError(
                "Route file mixes explicit-OD lines and compact lines. "
                "Use one route format consistently."
            )

        routes_by_od: RoutesByOD = {}

        if has_explicit_od:
            for record in parsed_records:
                od_pair = (int(record["origin"]), int(record["destination"]))

                if od_pair in routes_by_od:
                    raise ValueError(f"Duplicated OD pair found in route file: {od_pair}")

                routes_by_od[od_pair] = record["routes"]

            return routes_by_od

        ordered_od_pairs = self._get_ordered_od_pairs_for_compact_format()

        if len(ordered_od_pairs) != len(parsed_records):
            message = (
                "Compact route file line count does not match the number of "
                "provided/reconstructed OD pairs. "
                f"route_lines={len(parsed_records)} | od_pairs={len(ordered_od_pairs)}"
            )

            if self.strict:
                raise ValueError(message)

            logger.warning("%s. Truncating to the shortest length.", message)

        n = min(len(ordered_od_pairs), len(parsed_records))

        for idx in range(n):
            routes_by_od[ordered_od_pairs[idx]] = parsed_records[idx]["routes"]

        return routes_by_od

    def _get_ordered_od_pairs_for_compact_format(self) -> List[ODPair]:
        """
        Return ordered OD pairs for compact route files.

        Compact route files do not store OD IDs. Therefore, the reader needs
        either explicit od_pairs or zone_ids to reconstruct OD identity.

        Returns
        -------
        List[ODPair]
            Ordered OD pairs.
        """

        if self.od_pairs is not None:
            return list(self.od_pairs)

        if self.zone_ids is not None:
            return self._build_od_pairs_from_zone_ids(self.zone_ids)

        raise ValueError(
            "The route file uses compact format without OD identifiers. "
            "Provide either od_pairs or zone_ids to reconstruct OD identity."
        )

    @staticmethod
    def _build_od_pairs_from_zone_ids(zone_ids: Sequence[int]) -> List[ODPair]:
        """
        Build ordered non-intrazonal OD pairs from zone IDs.

        The ordering intentionally mirrors scenario_creation_pipeline.handle_routes:
        origin loop outside, destination loop inside, skipping origin == destination.

        Parameters
        ----------
        zone_ids : Sequence[int]
            Ordered zone IDs.

        Returns
        -------
        List[ODPair]
            Ordered non-intrazonal OD pairs.
        """

        normalized_zone_ids = [int(zone_id) for zone_id in zone_ids]

        include_intrazonal = True
        # TODO: implementar adecuadamente lo intrazonal

        return [
            (origin_id, destination_id)
            for origin_id in normalized_zone_ids
            for destination_id in normalized_zone_ids
            if include_intrazonal or origin_id != destination_id
        ]
    @staticmethod
    def _normalize_od_pairs(
        od_pairs: Sequence[Tuple[int, int]],
    ) -> List[ODPair]:
        """
        Normalize user-provided OD pairs.

        Parameters
        ----------
        od_pairs : Sequence[Tuple[int, int]]
            Ordered OD pairs.

        Returns
        -------
        List[ODPair]
            Normalized OD pairs as integer tuples.
        """

        return [
            (int(origin), int(destination))
            for origin, destination in od_pairs
        ]

    def _build_routes_dataframe(self, routes_by_od: RoutesByOD) -> pd.DataFrame:
        """
        Build a long-format route table.

        Parameters
        ----------
        routes_by_od : RoutesByOD
            Dictionary mapping OD pairs to route lists.

        Returns
        -------
        pd.DataFrame
            Route table with one row per route.
        """

        rows = []

        for od_index, ((origin, destination), routes) in enumerate(routes_by_od.items()):
            for route_index, route in enumerate(routes):
                rows.append(
                    {
                        "od_index": int(od_index),
                        "origin_id": int(origin),
                        "destination_id": int(destination),
                        "route_index": int(route_index),
                        "route": route,
                        "num_nodes": int(len(route)),
                        "num_links": int(len(route) - 1),
                    }
                )

        return pd.DataFrame(
            rows,
            columns=[
                "od_index",
                "origin_id",
                "destination_id",
                "route_index",
                "route",
                "num_nodes",
                "num_links",
            ],
        )

    def _validate(self, routes_by_od: RoutesByOD) -> None:
        """
        Validate the normalized route dictionary.

        This validation checks only route structure. It does not verify that
        route links exist in a graph; that should be done by the artifact
        validator after the graph has been built.

        Parameters
        ----------
        routes_by_od : RoutesByOD
            Dictionary mapping OD pairs to route lists.
        """

        if not routes_by_od:
            raise ValueError("No OD pairs were parsed from the routes file.")

        for od_pair, routes in routes_by_od.items():
            origin, destination = od_pair

            # if origin == destination:
            #     raise ValueError(f"Intrazonal OD pair found in routes: {od_pair}")

            if not routes and not self.allow_empty_routes:
                raise ValueError(f"OD pair has no routes and empty routes are not allowed: {od_pair}")

            for route_idx, route in enumerate(routes):
                if len(route) < 2:
                    raise ValueError(
                        f"Route {route_idx} for OD {od_pair} has fewer than two nodes."
                    )

                if route[0] != origin:
                    raise ValueError(
                        f"Route {route_idx} for OD {od_pair} starts at {route[0]}, "
                        f"but expected origin {origin}."
                    )

                if route[-1] != destination:
                    raise ValueError(
                        f"Route {route_idx} for OD {od_pair} ends at {route[-1]}, "
                        f"but expected destination {destination}."
                    )

    def _build_metadata(
        self,
        routes_by_od: RoutesByOD,
        routes_df: pd.DataFrame,
        num_route_lines: int,
    ) -> Dict[str, Any]:
        """
        Build metadata describing the loaded routes file.

        Parameters
        ----------
        routes_by_od : RoutesByOD
            Dictionary mapping OD pairs to route lists.

        routes_df : pd.DataFrame
            Long-format route table.

        num_route_lines : int
            Number of route lines parsed from the file.

        Returns
        -------
        Dict[str, Any]
            Reader metadata.
        """

        route_counts = {
            od_pair: len(routes)
            for od_pair, routes in routes_by_od.items()
        }

        counts = list(route_counts.values())
        num_routes = int(sum(counts))

        metadata: Dict[str, Any] = {
            "source_file": str(self.path),
            "num_route_lines": int(num_route_lines),
            "num_od_pairs": int(len(routes_by_od)),
            "num_routes": num_routes,
            "route_counts": route_counts,
            "od_pairs_without_routes": [
                od_pair
                for od_pair, count in route_counts.items()
                if count == 0
            ],
            "od_pairs_with_routes": [
                od_pair
                for od_pair, count in route_counts.items()
                if count > 0
            ],
            "max_routes_per_od_found": int(max(counts)) if counts else 0,
            "min_routes_per_od_found": int(min(counts)) if counts else 0,
            "mean_routes_per_od_found": float(sum(counts) / len(counts)) if counts else 0.0,
            "max_routes_per_od_retained": self.max_routes_per_od,
            "used_explicit_od_pairs": self._file_has_explicit_od(routes_by_od),
        }

        if not routes_df.empty:
            metadata.update(
                {
                    "max_route_num_nodes": int(routes_df["num_nodes"].max()),
                    "min_route_num_nodes": int(routes_df["num_nodes"].min()),
                    "mean_route_num_nodes": float(routes_df["num_nodes"].mean()),
                    "max_route_num_links": int(routes_df["num_links"].max()),
                    "min_route_num_links": int(routes_df["num_links"].min()),
                    "mean_route_num_links": float(routes_df["num_links"].mean()),
                }
            )
        else:
            metadata.update(
                {
                    "max_route_num_nodes": 0,
                    "min_route_num_nodes": 0,
                    "mean_route_num_nodes": 0.0,
                    "max_route_num_links": 0,
                    "min_route_num_links": 0,
                    "mean_route_num_links": 0.0,
                }
            )

        return metadata

    def _file_has_explicit_od(self, routes_by_od: RoutesByOD) -> bool:
        """
        Infer whether explicit OD pairs were likely used.

        This is based on whether the reader was able to parse the file without
        external od_pairs or zone_ids.

        Parameters
        ----------
        routes_by_od : RoutesByOD
            Dictionary mapping OD pairs to route lists.

        Returns
        -------
        bool
            True if external OD reconstruction metadata was not needed.
        """

        # If compact reconstruction metadata was supplied, the file may still
        # have explicit OD pairs, but this flag is mainly diagnostic.
        return self.od_pairs is None and self.zone_ids is None and bool(routes_by_od)


def read_tntp_routes(
    path: Union[str, Path],
    od_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    zone_ids: Optional[Sequence[int]] = None,
    strict: bool = True,
    allow_empty_routes: bool = True,
    max_routes_per_od: Optional[int] = None,
) -> RoutesReadResult:
    """
    Convenience function to read a TNTP-like routes file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the routes file.

    od_pairs : Optional[Sequence[Tuple[int, int]]], default=None
        Ordered OD pairs used for compact route files.

    zone_ids : Optional[Sequence[int]], default=None
        Zone IDs used to reconstruct OD pairs when od_pairs is not provided.

    strict : bool, default=True
        Whether to use strict validation behavior.

    allow_empty_routes : bool, default=True
        Whether OD pairs with empty route lists are allowed.

    max_routes_per_od : Optional[int], default=None
        Optional limit on retained routes per OD pair.

    Returns
    -------
    RoutesReadResult
        Normalized routes dictionary, route table and metadata.
    """

    reader = TNTPRoutesReader(
        path=path,
        od_pairs=od_pairs,
        zone_ids=zone_ids,
        strict=strict,
        allow_empty_routes=allow_empty_routes,
        max_routes_per_od=max_routes_per_od,
    )

    return reader.read()