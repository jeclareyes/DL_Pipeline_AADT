# src/data_handling/readers/tntp_network_reader.py

"""
TNTP Network Reader
===================

This module reads and normalizes network files in a TNTP-like format.

Project context
---------------
In the AADT / traffic assignment pipeline, the network file is one of the core
raw inputs used to build the training artifact. It defines the directed road
links that later become:

- rows in the canonical link table;
- edges in the NetworkX graph;
- physical attributes used by traffic assignment and neural network models;
- the reference order for link-level targets, masks and model tensors.

This reader is intentionally limited to data ingestion. It does not build a
NetworkX graph, merge observed flows, compute routes, or prepare tensors.
Those responsibilities belong to downstream builders, adapters and artifact
builders.

Expected input
--------------
A TNTP-like network file with metadata and link rows, for example:

    <NUMBER OF ZONES> 4
    <NUMBER OF NODES> 20
    <FIRST THRU NODE> 5
    <NUMBER OF LINKS> 60
    <END OF METADATA>

    ~ init_node term_node capacity_per_lane lanes total_capacity effective_capacity length free_flow_time b power speed toll link_type
      1         5         1000     1     0.5    0.04           0.15 4     50    0    99;

The reader is tolerant to minor format variations:
- header may be provided after "~";
- rows may end with semicolons;
- columns may be separated by spaces or tabs;
- some optional columns may be missing.

Output
------
The main output is a NetworkReadResult containing:
- network_df: normalized pandas DataFrame;
- metadata: dictionary with file path, TNTP metadata, number of links,
  available columns and basic link statistics.

Design principles
-----------------
- Do only one thing: read and normalize network link data.
- Avoid hard-coded project paths.
- Preserve useful extra columns.
- Fail early when required columns are missing.
- Return metadata useful for validation and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkReadResult:
    """
    Container returned by TNTPNetworkReader.

    Attributes
    ----------
    network_df : pd.DataFrame
        Normalized link table read from the TNTP network file.

    metadata : Dict[str, Any]
        Metadata describing the parsing result, including source file,
        TNTP metadata tags, number of links and column information.
    """

    network_df: pd.DataFrame
    metadata: Dict[str, Any]


class TNTPNetworkReader:
    """
    Read and normalize a TNTP-like network file.

    This class is responsible only for reading the raw network file and
    returning a clean DataFrame that downstream components can safely use.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the network file.

    strict : bool, default=True
        If True, missing required columns or empty data raise errors.

    preserve_extra_columns : bool, default=True
        If True, non-standard columns are kept in the output DataFrame.

    Required normalized columns
    ---------------------------
    - init_node
    - term_node
    - effective_capacity
    - length
    - free_flow_time

    Common optional columns
    -----------------------
    - link_id
    - reverse_link_id
    - capacity_per_lane
    - lanes
    - total_capacity
    - effective_capacity
    - b
    - power
    - speed
    - vdf
    - toll
    - link_type
    """

    REQUIRED_COLUMNS = {
        "init_node",
        "term_node",
        "effective_capacity",
        "length",
        "free_flow_time",
    }

    CANONICAL_ORDER = [
        "link_id",
        "reverse_link_id",
        "init_node",
        "term_node",
        "capacity_per_lane",
        "lanes",
        "total_capacity",
        "effective_capacity",
        "length",
        "free_flow_time",
        "b",
        "power",
        "speed",
        "vdf",
        "toll",
        "link_type",
    ]

    COLUMN_ALIASES = {
        "init_node": "init_node",
        "from_node": "init_node",
        "from": "init_node",
        "u": "init_node",
        "tail": "init_node",

        "term_node": "term_node",
        "to_node": "term_node",
        "to": "term_node",
        "v": "term_node",
        "head": "term_node",

        "capacity_per_lane": "capacity_per_lane",
        "total_capacity": "total_capacity",
        "effective_capacity": "effective_capacity",
        "assignment_capacity": "effective_capacity",

        "lanes": "lanes",
        "lane": "lanes",

        "length": "length",
        "distance": "length",

        "free_flow_time": "free_flow_time",
        "fft": "free_flow_time",
        "fftime": "free_flow_time",
        "freeflowtime": "free_flow_time",
        "t0": "free_flow_time",

        "b": "b",
        "alpha": "b",

        "power": "power",
        "beta": "power",

        "speed": "speed",
        "speed_limit": "speed",

        "vdf": "vdf",
        "vdf_id": "vdf",

        "toll": "toll",

        "link_type": "link_type",
        "type": "link_type",

        "link_id": "link_id",
        "id": "link_id",

        "reverse_link_id": "reverse_link_id",
        "rev_link_id": "reverse_link_id",
    }

    METADATA_TAGS = {
        "<NUMBER OF ZONES>": "number_of_zones",
        "<NUMBER OF NODES>": "number_of_nodes",
        "<FIRST THRU NODE>": "first_thru_node",
        "<NUMBER OF LINKS>": "number_of_links",
    }

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
            raise FileNotFoundError(f"Network file not found: {self.path}")

    def read(self) -> NetworkReadResult:
        """
        Read, parse, normalize and validate the network file.

        Returns
        -------
        NetworkReadResult
            Object containing the normalized network DataFrame and metadata.

        Raises
        ------
        FileNotFoundError
            If the network file does not exist.

        ValueError
            If required columns are missing or no valid link rows are found.
        """

        logger.info("Reading TNTP network file: %s", self.path)

        lines = self._read_lines()
        file_metadata = self._parse_metadata(lines)
        header = self._find_header(lines, file_metadata)
        data_lines = self._extract_data_lines(lines)

        if not data_lines:
            raise ValueError(f"No network data rows found in file: {self.path}")

        network_df = self._parse_data_lines(
            data_lines=data_lines,
            header=header,
        )

        network_df = self._normalize_column_names(network_df)
        network_df = self._apply_default_columns(network_df)
        network_df = self._normalize_dtypes(network_df)
        network_df = self._order_columns(network_df)
        self._validate(network_df)

        metadata = self._build_metadata(
            network_df=network_df,
            file_metadata=file_metadata,
        )

        logger.info(
            "Network file loaded successfully | links=%d | columns=%d",
            len(network_df),
            len(network_df.columns),
        )

        return NetworkReadResult(
            network_df=network_df,
            metadata=metadata,
        )

    def _read_lines(self) -> List[str]:
        """
        Read all lines from the network file.

        Returns
        -------
        List[str]
            Raw file lines.
        """

        with self.path.open("r", encoding="utf-8", errors="replace") as file:
            return file.readlines()

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """
        Parse TNTP metadata tags before the data section.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        Returns
        -------
        Dict[str, Any]
            Parsed metadata dictionary.
        """

        metadata: Dict[str, Any] = {}

        for line in lines:
            stripped = line.strip()

            for tag, key in self.METADATA_TAGS.items():
                if stripped.upper().startswith(tag):
                    try:
                        metadata[key] = int(stripped.split()[-1])
                    except (ValueError, IndexError):
                        logger.warning("Could not parse metadata line: %s", stripped)

            # Some generated TNTP files may store the original header in metadata.
            if stripped.upper().startswith("<ORIGINAL HEADER>") and "~" in stripped:
                header_text = stripped.split("~", 1)[1].strip()
                metadata["original_header"] = [
                    col.strip()
                    for col in header_text.split()
                    if col.strip()
                ]

            if stripped.upper() == "<END OF METADATA>":
                break

        return metadata

    def _find_header(
        self,
        lines: List[str],
        file_metadata: Dict[str, Any],
    ) -> Optional[List[str]]:
        """
        Find the network table header.

        The method first checks metadata for an original header. If not found,
        it searches for a line containing init_node and term_node.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        file_metadata : Dict[str, Any]
            Parsed TNTP metadata.

        Returns
        -------
        Optional[List[str]]
            Header columns if found, otherwise None.
        """

        if "original_header" in file_metadata:
            return file_metadata["original_header"]

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            # Remove the leading TNTP comment/header marker if present.
            if stripped.startswith("~"):
                stripped = stripped[1:].strip()

            lower = stripped.lower()

            if "init_node" in lower and "term_node" in lower:
                return stripped.split()

        return None

    def _extract_data_lines(self, lines: List[str]) -> List[str]:
        """
        Extract only link data rows from a TNTP network file.

        The function supports both standard TNTP files with a metadata section and
        simplified TNTP-like files that start directly with the table header. Header,
        comment, metadata, and empty lines are ignored. Semicolons at the end of rows
        are removed.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        Returns
        -------
        List[str]
            Cleaned data rows.
        """

        data_lines = []
        after_metadata = False
        header_found = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            # Detect the end of a standard TNTP metadata section.
            if stripped.upper() == "<END OF METADATA>":
                after_metadata = True
                continue

            # Skip metadata lines when present.
            if stripped.startswith("<") and stripped.endswith(">"):
                continue

            # Skip comment lines.
            if stripped.startswith("~"):
                continue

            lower = stripped.lower()

            # Detect and skip the table header.
            is_header = (
                "init_node" in lower
                and "term_node" in lower
            )

            if is_header:
                header_found = True
                after_metadata = True
                continue

            # If the file has no metadata, data starts after the header.
            if not after_metadata and not header_found:
                continue

            if stripped.endswith(";"):
                stripped = stripped[:-1].strip()

            # Keep only lines that appear to contain numeric link records.
            if any(char.isdigit() for char in stripped):
                data_lines.append(stripped)

        return data_lines


    def _parse_data_lines(
        self,
        data_lines: List[str],
        header: Optional[List[str]],
    ) -> pd.DataFrame:
        """
        Parse cleaned link rows into a DataFrame.

        Parameters
        ----------
        data_lines : List[str]
            Cleaned link rows.

        header : Optional[List[str]]
            Header columns. If None, fallback column inference is used.

        Returns
        -------
        pd.DataFrame
            Raw parsed network DataFrame.
        """

        data_string = "\n".join(data_lines)

        raw = pd.read_csv(
            io.StringIO(data_string),
            sep=r"\s+",
            header=None,
            engine="python",
        )

        if header is not None:
            normalized_header = [col.strip() for col in header if col.strip()]

            if len(normalized_header) != raw.shape[1]:
                message = (
                    "Header column count does not match data column count. "
                    f"Header has {len(normalized_header)} columns, "
                    f"data has {raw.shape[1]} columns."
                )

                if self.strict:
                    raise ValueError(message)

                logger.warning("%s Falling back to inferred column names.", message)
                raw.columns = self._infer_columns(raw.shape[1])
            else:
                raw.columns = normalized_header
        else:
            raw.columns = self._infer_columns(raw.shape[1])

        return raw

    def _infer_columns(self, n_columns: int) -> List[str]:
        """
        Infer network columns when no header is available.

        The fallback follows common TNTP column order conventions.

        Parameters
        ----------
        n_columns : int
            Number of columns detected in the data rows.

        Returns
        -------
        List[str]
            Inferred column names.
        """

        base_columns = [
            "init_node",
            "term_node",
            "effective_capacity",
            "length",
            "free_flow_time",
            "b",
            "power",
            "speed",
            "toll",
            "link_type",
        ]

        if n_columns == len(base_columns):
            return base_columns

        if n_columns == len(base_columns) + 1:
            return [
                "init_node",
                "term_node",
                "effective_capacity",
                "lanes",
                "length",
                "free_flow_time",
                "b",
                "power",
                "speed",
                "toll",
                "link_type",
            ]

        if n_columns == len(base_columns) + 2:
            # Common generated format includes both lanes and vdf.
            return [
                "init_node",
                "term_node",
                "effective_capacity",
                "lanes",
                "length",
                "free_flow_time",
                "b",
                "power",
                "speed",
                "vdf",
                "toll",
                "link_type",
            ]

        # Generic fallback: use canonical columns first and preserve extras.
        columns = []

        for idx in range(n_columns):
            if idx < len(self.CANONICAL_ORDER):
                columns.append(self.CANONICAL_ORDER[idx])
            else:
                columns.append(f"extra_col_{idx}")

        return columns

    def _normalize_column_names(self, network_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to the project canonical schema.

        Parameters
        ----------
        network_df : pd.DataFrame
            Raw parsed network DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized column names.
        """

        rename_map = {}

        for column in network_df.columns:
            key = str(column).strip().lower()
            rename_map[column] = self.COLUMN_ALIASES.get(key, key)

        df = network_df.rename(columns=rename_map).copy()

        # If aliasing produced duplicated names, keep the first occurrence.
        df = df.loc[:, ~df.columns.duplicated()].copy()

        return df

    def _apply_default_columns(self, network_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add safe defaults for optional columns when they are missing.

        These defaults are intentionally conservative and only apply to optional
        attributes. Required columns are validated later.

        Parameters
        ----------
        network_df : pd.DataFrame
            Network DataFrame with normalized columns.

        Returns
        -------
        pd.DataFrame
            Network DataFrame with optional defaults.
        """

        df = network_df.copy()

        if "lanes" not in df.columns:
            df["lanes"] = 1

        if "b" not in df.columns:
            df["b"] = 0.15

        if "power" not in df.columns:
            df["power"] = 4.0

        if "speed" not in df.columns:
            df["speed"] = pd.NA

        if "vdf" not in df.columns:
            df["vdf"] = pd.NA

        if "toll" not in df.columns:
            df["toll"] = 0.0

        if "link_type" not in df.columns:
            df["link_type"] = 0

        if "link_id" not in df.columns:
            # Assign stable sequential link IDs if the file does not provide them.
            df["link_id"] = range(1, len(df) + 1)

        if "reverse_link_id" not in df.columns:
            df["reverse_link_id"] = pd.NA

        # Modern capacity contract:
        # - capacity_per_lane: physical lane-level capacity
        # - total_capacity: capacity_per_lane * lanes
        # - effective_capacity: total_capacity adjusted by scenario multiplier

        if "total_capacity" not in df.columns and "capacity_per_lane" in df.columns and "lanes" in df.columns:
            df["total_capacity"] = (
                pd.to_numeric(df["capacity_per_lane"], errors="coerce")
                * pd.to_numeric(df["lanes"], errors="coerce")
            )

        # Some TNTP exports use -1 for an unspecified scenario-adjusted
        # capacity.  It is a sentinel, not a physical zero/negative capacity;
        # use the already-normalized lane capacity in that case.  Keep other
        # non-positive values intact so the strict validation below still
        # rejects genuinely invalid network data.
        if "effective_capacity" in df.columns and "total_capacity" in df.columns:
            unspecified_capacity = df["effective_capacity"].eq(-1)
            df.loc[unspecified_capacity, "effective_capacity"] = df.loc[
                unspecified_capacity, "total_capacity"
            ]

        return df

    def _normalize_dtypes(self, network_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert core network columns to stable numeric data types.

        Parameters
        ----------
        network_df : pd.DataFrame
            Network DataFrame with normalized columns.

        Returns
        -------
        pd.DataFrame
            Network DataFrame with normalized data types.
        """

        df = network_df.copy()

        integer_columns = [
            "link_id",
            "reverse_link_id",
            "init_node",
            "term_node",
            "lanes",
            "vdf",
            "link_type",
        ]

        float_columns = [
            "capacity_per_lane",
            "total_capacity",
            "effective_capacity",
            "length",
            "free_flow_time",
            "b",
            "power",
            "speed",
            "toll",
        ]

        for column in integer_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in float_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        # Required integer columns cannot contain missing values.
        df = df.dropna(subset=["link_id", "init_node", "term_node"])

        df["link_id"] = df["link_id"].astype(int)
        df["init_node"] = df["init_node"].astype(int)
        df["term_node"] = df["term_node"].astype(int)

        if "reverse_link_id" in df.columns:
            df["reverse_link_id"] = df["reverse_link_id"].astype("Int64")

        if "lanes" in df.columns:
            df["lanes"] = df["lanes"].fillna(1).astype(int)

        if "vdf" in df.columns:
            df["vdf"] = df["vdf"].astype("Int64")

        if "link_type" in df.columns:
            df["link_type"] = df["link_type"].fillna(0).astype(int)

        return df

    def _order_columns(self, network_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns so canonical network fields appear first.

        Parameters
        ----------
        network_df : pd.DataFrame
            Normalized network DataFrame.

        Returns
        -------
        pd.DataFrame
            Reordered network DataFrame.
        """

        canonical_present = [
            column
            for column in self.CANONICAL_ORDER
            if column in network_df.columns
        ]

        if not self.preserve_extra_columns:
            return network_df[canonical_present].copy()

        extra_columns = [
            column
            for column in network_df.columns
            if column not in canonical_present
        ]

        return network_df[canonical_present + extra_columns].copy()

    def _validate(self, network_df: pd.DataFrame) -> None:
        """
        Validate the normalized network table.

        This validation is intentionally limited to reader-level checks.
        More complex consistency checks, such as graph connectivity or route
        validity, should be performed by downstream validators.

        Parameters
        ----------
        network_df : pd.DataFrame
            Normalized network DataFrame.

        Raises
        ------
        ValueError
            If duplicated link IDs, missing required columns or invalid values
            are found.
        """

        missing_columns = self.REQUIRED_COLUMNS - set(network_df.columns)

        if missing_columns:
            raise ValueError(
                "Normalized network table is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if network_df.empty:
            raise ValueError("Normalized network table is empty.")

        duplicated_links = network_df["link_id"].duplicated(keep=False)

        if duplicated_links.any():
            duplicated_ids = (
                network_df.loc[duplicated_links, "link_id"]
                .astype(int)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Duplicated link IDs found in network file: "
                f"{duplicated_ids[:20]}"
            )

        required_numeric = [
            "capacity_per_lane",
            "total_capacity",
            "effective_capacity",
            "length",
            "free_flow_time",
        ]

        if network_df[required_numeric].isna().any().any():
            raise ValueError(
                "Missing or non-numeric values found in required network columns: "
                f"{required_numeric}"
            )

        if (network_df["effective_capacity"] <= 0).any():
            raise ValueError("Network contains links with effective_capacity <= 0.")

        if (network_df["length"] < 0).any():
            raise ValueError("Network contains links with length < 0.")

        if (network_df["free_flow_time"] < 0).any():
            raise ValueError("Network contains links with free_flow_time < 0.")

    def _build_metadata(
        self,
        network_df: pd.DataFrame,
        file_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build metadata describing the loaded network file.

        Parameters
        ----------
        network_df : pd.DataFrame
            Normalized network table.

        file_metadata : Dict[str, Any]
            Metadata parsed from the TNTP header.

        Returns
        -------
        Dict[str, Any]
            Reader metadata.
        """

        metadata: Dict[str, Any] = {
            "source_file": str(self.path),
            "file_metadata": file_metadata,
            "num_links": int(len(network_df)),
            "columns": network_df.columns.tolist(),
            "init_node_min": int(network_df["init_node"].min()),
            "init_node_max": int(network_df["init_node"].max()),
            "term_node_min": int(network_df["term_node"].min()),
            "term_node_max": int(network_df["term_node"].max()),
            "capacity_per_lane_total": float(network_df["capacity_per_lane"].sum()) if "capacity_per_lane" in network_df.columns else None,
            "capacity_per_lane_mean": float(network_df["capacity_per_lane"].mean()) if "capacity_per_lane" in network_df.columns else None,
            "total_capacity_total": float(network_df["total_capacity"].sum()) if "total_capacity" in network_df.columns else None,
            "total_capacity_mean": float(network_df["total_capacity"].mean()) if "total_capacity" in network_df.columns else None,
            "effective_capacity_total": float(network_df["effective_capacity"].sum()) if "effective_capacity" in network_df.columns else None,
            "effective_capacity_mean": float(network_df["effective_capacity"].mean()) if "effective_capacity" in network_df.columns else None,
            "length_total": float(network_df["length"].sum()),
            "free_flow_time_mean": float(network_df["free_flow_time"].mean()),
            "capacity_convention": {
                "capacity_per_lane": "lane_level_capacity" if "capacity_per_lane" in network_df.columns else None,
                "total_capacity": "lane_level_capacity_times_lanes" if "total_capacity" in network_df.columns else None,
                "effective_capacity": "scenario_adjusted_capacity" if "effective_capacity" in network_df.columns else None,
            },
        }

        if "link_type" in network_df.columns:
            metadata["link_type_counts"] = (
                network_df["link_type"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

        if "vdf" in network_df.columns:
            metadata["vdf_counts"] = (
                network_df["vdf"]
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )

        return metadata


def read_tntp_network(
    path: Union[str, Path],
    strict: bool = True,
    preserve_extra_columns: bool = True,
) -> NetworkReadResult:
    """
    Convenience function to read a TNTP-like network file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the network file.

    strict : bool, default=True
        Whether to use strict validation behavior.

    preserve_extra_columns : bool, default=True
        Whether to keep non-standard columns in the output DataFrame.

    Returns
    -------
    NetworkReadResult
        Normalized network DataFrame and metadata.
    """

    reader = TNTPNetworkReader(
        path=path,
        strict=strict,
        preserve_extra_columns=preserve_extra_columns,
    )

    return reader.read()
