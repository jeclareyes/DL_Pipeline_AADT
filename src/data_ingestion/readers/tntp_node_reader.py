# src/data_ingestion/readers/tntp_node_reader.py
"""
TNTP Node Reader
================

This module reads and normalizes node files in a TNTP-like format.

Project context
---------------
In the AADT / traffic assignment pipeline, the node file is one of the
fundamental raw inputs used to build a consistent training artifact.

The node table provides:
- the unique node identifier used across the network, routes and OD demand files;
- spatial coordinates used to build graph geometry and visualization objects;
- node type and node class information, such as zones and non-zones.

This reader is intentionally limited to data ingestion. It does not build
NetworkX graphs, compute routes, create OD matrices, or prepare tensors for
neural network training. Those responsibilities belong to downstream builders,
adapters, and artifact builders.

Expected input
--------------
A TNTP-like node file with a header and rows such as:

    node_id    x      y      type        class
    1          10.5   30.2   TAZ         Zones
    2          15.0   28.7   INTERSECTION Non-Zones

The reader is tolerant to some column aliases, for example:
- "Node", "node", "id" -> "node_id"
- "coord_x", "X" -> "x"
- "coord_y", "Y" -> "y"
- "node_type" -> "type"
- "node_class" -> "class"

Output
------
The main output is a NodeReadResult containing:
- nodes_df: normalized pandas DataFrame;
- metadata: dictionary with file path, number of nodes, coordinate ranges,
  node type counts and node class counts.

Design principles
-----------------
- Do only one thing: read and normalize node data.
- Avoid hard-coded project paths.
- Preserve useful extra columns.
- Fail early when required columns are missing.
- Return metadata useful for validation and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeReadResult:
    """
    Container returned by TNTPNodeReader.

    Attributes
    ----------
    nodes_df : pd.DataFrame
        Normalized node table.

    metadata : Dict[str, Any]
        Metadata describing the parsing result, including file path,
        number of nodes, coordinate ranges and category counts.
    """

    nodes_df: pd.DataFrame
    metadata: Dict[str, Any]


class TNTPNodeReader:
    """
    Read and normalize a TNTP-like node file.

    This class is responsible only for reading the raw node file and returning
    a clean DataFrame that downstream components can safely use.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the node file.

    strict : bool, default=True
        If True, invalid rows are not silently accepted and required column
        errors are raised immediately.

    preserve_extra_columns : bool, default=True
        If True, columns beyond the required schema are kept in the output.
        This is useful when node files contain additional attributes needed
        later for visualization, zoning, or scenario analysis.

    Required normalized columns
    ---------------------------
    - node_id
    - x
    - y
    - type

    Optional normalized columns
    ---------------------------
    - class
    """

    REQUIRED_COLUMNS = {"node_id", "x", "y", "type"}

    COLUMN_ALIASES = {
        "node": "node_id",
        "node_id": "node_id",
        "id": "node_id",

        "x": "x",
        "coord_x": "x",
        "x_coord": "x",

        "y": "y",
        "coord_y": "y",
        "y_coord": "y",

        "type": "type",
        "node_type": "type",

        "class": "class",
        "node_class": "class",
    }

    COMMENT_PREFIXES = ("#", "//")

    def __init__(
        self,
        path: Union[str, Path],
        strict: bool = True,
        preserve_extra_columns: bool = True,
    ) -> None:
        self.path = Path(path).resolve(strict=False)
        self.strict = bool(strict)
        self.preserve_extra_columns = bool(preserve_extra_columns)

        if not self.path.exists():
            raise FileNotFoundError(f"Node file not found: {self.path}")

    def read(self) -> NodeReadResult:
        """
        Read, parse, normalize and validate the node file.

        Returns
        -------
        NodeReadResult
            Object containing the normalized node DataFrame and metadata.

        Raises
        ------
        FileNotFoundError
            If the node file does not exist.

        ValueError
            If required columns are missing or the file contains no valid rows.
        """

        logger.info("Reading TNTP node file: %s", self.path)

        lines = self._read_lines()
        data_lines = self._extract_data_lines(lines)

        if not data_lines:
            raise ValueError(f"No node data found in file: {self.path}")

        nodes_df, skipped_rows = self._parse_data_lines(data_lines)
        nodes_df = self._normalize_dtypes(nodes_df)
        nodes_df = self._order_columns(nodes_df)
        self._validate(nodes_df)

        metadata = self._build_metadata(
            nodes_df=nodes_df,
            skipped_rows=skipped_rows,
        )

        logger.info(
            "Node file loaded successfully | nodes=%d | skipped_rows=%d",
            len(nodes_df),
            skipped_rows,
        )

        return NodeReadResult(
            nodes_df=nodes_df,
            metadata=metadata,
        )

    def _read_lines(self) -> List[str]:
        """
        Read all lines from the node file.

        Returns
        -------
        List[str]
            Raw file lines.
        """

        with self.path.open("r", encoding="utf-8", errors="replace") as file:
            return file.readlines()

    def _extract_data_lines(self, lines: List[str]) -> List[str]:
        """
        Remove empty lines, comments and trailing semicolons.

        This method keeps the header line because the parser needs it to infer
        the input schema.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        Returns
        -------
        List[str]
            Cleaned data lines, including the header.
        """

        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and explicit comment lines.
            if not stripped:
                continue

            if stripped.startswith(self.COMMENT_PREFIXES):
                continue

            # TNTP-like files commonly end rows with semicolons.
            if stripped.endswith(";"):
                stripped = stripped[:-1].strip()

            cleaned_lines.append(stripped)

        return cleaned_lines

    def _parse_data_lines(self, data_lines: List[str]) -> tuple[pd.DataFrame, int]:
        """
        Parse cleaned node lines into a DataFrame.

        The first valid line is treated as the header. Each following line is
        parsed according to that header.

        Parameters
        ----------
        data_lines : List[str]
            Cleaned lines including header and data rows.

        Returns
        -------
        tuple[pd.DataFrame, int]
            Parsed DataFrame and number of skipped malformed rows.
        """

        raw_header = data_lines[0].split()
        normalized_header = self._normalize_header(raw_header)

        missing = self.REQUIRED_COLUMNS - set(normalized_header)

        if missing:
            raise ValueError(
                "Node file is missing required columns after normalization: "
                f"{sorted(missing)}. Found columns: {normalized_header}"
            )

        rows = []
        skipped_rows = 0

        for line in data_lines[1:]:
            parts = line.split()

            # A malformed row is one whose number of values does not match
            # the number of columns in the header.
            if len(parts) != len(normalized_header):
                skipped_rows += 1

                if self.strict:
                    logger.warning(
                        "Skipping malformed node row with %d values, expected %d: %s",
                        len(parts),
                        len(normalized_header),
                        line,
                    )

                continue

            rows.append(dict(zip(normalized_header, parts)))

        if not rows:
            raise ValueError(f"No valid node rows found in file: {self.path}")

        return pd.DataFrame(rows), skipped_rows

    def _normalize_header(self, raw_header: List[str]) -> List[str]:
        """
        Normalize raw column names to the project canonical schema.

        Parameters
        ----------
        raw_header : List[str]
            Column names as found in the file.

        Returns
        -------
        List[str]
            Normalized column names.
        """

        normalized = []

        for column in raw_header:
            key = column.strip().lower()
            normalized.append(self.COLUMN_ALIASES.get(key, key))

        return normalized

    def _normalize_dtypes(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert core columns to stable data types.

        The expected output types are:
        - node_id: int
        - x: float
        - y: float
        - type: string
        - class: string, if available

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Parsed node table.

        Returns
        -------
        pd.DataFrame
            Node table with normalized data types.
        """

        df = nodes_df.copy()

        df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce")
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df["type"] = df["type"].astype(str).str.strip()

        if "class" in df.columns:
            df["class"] = df["class"].astype(str).str.strip()

        # Drop rows where mandatory numeric fields could not be parsed.
        before = len(df)
        df = df.dropna(subset=["node_id", "x", "y"])
        dropped = before - len(df)

        if dropped > 0:
            logger.warning(
                "Dropped %d node rows because node_id, x or y could not be parsed.",
                dropped,
            )

        df["node_id"] = df["node_id"].astype(int)
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)

        return df

    def _order_columns(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns so required fields appear first.

        Extra columns are preserved after the canonical columns when
        preserve_extra_columns=True.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Normalized node table.

        Returns
        -------
        pd.DataFrame
            Reordered node table.
        """

        canonical_columns = ["node_id", "x", "y", "type"]

        if "class" in nodes_df.columns:
            canonical_columns.append("class")

        if not self.preserve_extra_columns:
            return nodes_df[canonical_columns].copy()

        extra_columns = [
            column
            for column in nodes_df.columns
            if column not in canonical_columns
        ]

        return nodes_df[canonical_columns + extra_columns].copy()

    def _validate(self, nodes_df: pd.DataFrame) -> None:
        """
        Validate the normalized node table.

        This validation is intentionally limited to reader-level checks. More
        complex consistency checks, such as whether all route nodes exist in the
        network, should be performed by the training artifact validator.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Normalized node table.

        Raises
        ------
        ValueError
            If duplicated node IDs or missing required values are found.
        """

        missing_columns = self.REQUIRED_COLUMNS - set(nodes_df.columns)

        if missing_columns:
            raise ValueError(
                f"Normalized node table is missing columns: {sorted(missing_columns)}"
            )

        if nodes_df.empty:
            raise ValueError("Normalized node table is empty.")

        duplicated_mask = nodes_df["node_id"].duplicated(keep=False)

        if duplicated_mask.any():
            duplicated_ids = (
                nodes_df.loc[duplicated_mask, "node_id"]
                .astype(int)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Duplicated node IDs found in node file: "
                f"{duplicated_ids[:20]}"
            )

        if nodes_df[["node_id", "x", "y", "type"]].isna().any().any():
            raise ValueError("Missing values found in required node columns.")

    def _build_metadata(
        self,
        nodes_df: pd.DataFrame,
        skipped_rows: int,
    ) -> Dict[str, Any]:
        """
        Build metadata describing the loaded node file.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Normalized node table.

        skipped_rows : int
            Number of malformed rows skipped during parsing.

        Returns
        -------
        Dict[str, Any]
            Reader metadata.
        """

        metadata: Dict[str, Any] = {
            "source_file": str(self.path),
            "num_nodes": int(len(nodes_df)),
            "columns": nodes_df.columns.tolist(),
            "skipped_rows": int(skipped_rows),
            "node_id_min": int(nodes_df["node_id"].min()),
            "node_id_max": int(nodes_df["node_id"].max()),
            "x_range": (
                float(nodes_df["x"].min()),
                float(nodes_df["x"].max()),
            ),
            "y_range": (
                float(nodes_df["y"].min()),
                float(nodes_df["y"].max()),
            ),
            "node_type_counts": nodes_df["type"].value_counts().to_dict(),
        }

        if "class" in nodes_df.columns:
            metadata["node_class_counts"] = (
                nodes_df["class"]
                .value_counts()
                .to_dict()
            )

        return metadata


def read_tntp_nodes(
    path: Union[str, Path],
    strict: bool = True,
    preserve_extra_columns: bool = True,
) -> NodeReadResult:
    """
    Convenience function to read a TNTP-like node file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the node file.

    strict : bool, default=True
        Whether to use strict validation behavior.

    preserve_extra_columns : bool, default=True
        Whether to keep non-standard columns in the output DataFrame.

    Returns
    -------
    NodeReadResult
        Normalized node DataFrame and metadata.
    """

    reader = TNTPNodeReader(
        path=path,
        strict=strict,
        preserve_extra_columns=preserve_extra_columns,
    )

    return reader.read()