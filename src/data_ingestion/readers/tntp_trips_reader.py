# src/data_ingestion/readers/tntp_trips_reader.py

"""
TNTP Trips Reader
=================

This module reads and normalizes trip demand files in a TNTP-like format.

Project context
---------------
In the AADT / traffic assignment pipeline, the trips file defines the
origin-destination demand that later becomes:

- the OD demand matrix used by assignment and training;
- the demand target or prior for neural network models;
- the reference OD space used to align routes, OD pairs and model tensors.

This reader is intentionally limited to data ingestion and OD demand
normalization. It does not build graphs, compute routes, assign traffic flows,
or prepare PyTorch tensors. Those responsibilities belong to downstream
builders, adapters and artifact builders.

Supported input formats
-----------------------
The reader supports two common TNTP-like demand formats.

1. Standard single-matrix format:

    <NUMBER OF ZONES> 4
    <TOTAL OD FLOW> 1000
    <END OF METADATA>

    Origin 1
        2 : 100; 3 : 120; 4 : 80;
    Origin 2
        1 : 90; 3 : 110;

2. Multiday / time-dependent format:

    <NUMBER OF ZONES> 4
    <END OF METADATA>

    <MATRIX DATE> 2022-10-01 00:00
    Origin 1
        2 : 10; 3 : 15;

    <MATRIX DATE> 2022-10-01 01:00
    Origin 1
        2 : 20; 3 : 25;

Output
------
The main output is a TripsReadResult containing:
- trips_df: normalized long-format OD demand table;
- od_matrix: sparse OD matrix;
- metadata: dictionary with file path, number of zones, dates, total demand,
  aggregation method and matrix statistics.

Design principles
-----------------
- Do only one thing: read and normalize trip demand data.
- Support both static and multiday TNTP-like formats.
- Return both long-format demand and sparse matrix representation.
- Avoid hard-coded project paths.
- Preserve enough metadata for reproducibility and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripsReadResult:
    """
    Container returned by TNTPTripsReader.

    Attributes
    ----------
    trips_df : pd.DataFrame
        Normalized long-format OD demand table.

    od_matrix : sparse.csr_matrix
        Sparse OD matrix after the selected aggregation.

    metadata : Dict[str, Any]
        Metadata describing the parsing result, including number of zones,
        demand totals, dates and matrix statistics.
    """

    trips_df: pd.DataFrame
    od_matrix: sparse.csr_matrix
    metadata: Dict[str, Any]


class TNTPTripsReader:
    """
    Read and normalize a TNTP-like trips file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the trips file.

    aggregation : str, default="average_daily"
        Aggregation rule used when multiple matrix blocks or dates are found.

        Supported values:
        - "sum": sum all parsed demand records.
        - "average_daily": aggregate demand by date first and then average
          across available dates.
        - "average_matrix": average across matrix blocks directly.

    matrix_format : str, default="csr"
        Sparse matrix format returned by the reader. Currently the public
        result always returns CSR, but this parameter is kept to make the
        conversion explicit and future-proof.

    include_zero_flows : bool, default=False
        If True, OD records with zero flow are preserved in trips_df. The sparse
        matrix naturally stores only non-zero values.

    strict : bool, default=True
        If True, missing demand records or invalid metadata raise errors.
    """

    SUPPORTED_AGGREGATIONS = {"sum", "average_daily", "average_matrix"}

    def __init__(
        self,
        path: Union[str, Path],
        aggregation: str = "average_daily",
        matrix_format: str = "csr",
        include_zero_flows: bool = False,
        strict: bool = True,
    ) -> None:
        self.path = Path(path).resolve(strict=False)
        self.aggregation = str(aggregation)
        self.matrix_format = str(matrix_format)
        self.include_zero_flows = bool(include_zero_flows)
        self.strict = bool(strict)

        if self.aggregation not in self.SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported trips aggregation: {self.aggregation}. "
                f"Supported values: {sorted(self.SUPPORTED_AGGREGATIONS)}"
            )

        if not self.path.exists():
            raise FileNotFoundError(f"Trips file not found: {self.path}")

    def read(self) -> TripsReadResult:
        """
        Read, parse, aggregate and convert the trips file to sparse OD matrix.

        Returns
        -------
        TripsReadResult
            Object containing normalized OD records, sparse OD matrix and metadata.

        Raises
        ------
        FileNotFoundError
            If the trips file does not exist.

        ValueError
            If no valid OD records are found.
        """

        logger.info("Reading TNTP trips file: %s", self.path)

        lines = self._read_lines()
        file_metadata = self._parse_metadata(lines)
        raw_trips_df = self._parse_trip_records(lines)

        if raw_trips_df.empty:
            raise ValueError(f"No valid OD demand records found in file: {self.path}")

        aggregated_trips_df = self._aggregate_trips(
            raw_trips_df=raw_trips_df,
            file_metadata=file_metadata,
        )

        trips_df = self._normalize_trips_df(aggregated_trips_df)

        origin_zone_ids = self._extract_zone_ids_from_origin_lines(lines)

        zone_ids = self._infer_zone_ids(
            trips_df=trips_df,
            raw_trips_df=raw_trips_df,
            file_metadata=file_metadata,
            origin_zone_ids=origin_zone_ids,
        )

        zone_id_to_idx, idx_to_zone_id = self._build_zone_mappings(zone_ids)

        od_matrix = self._build_sparse_matrix(
            trips_df=trips_df,
            zone_id_to_idx=zone_id_to_idx,
        )

        metadata = self._build_metadata(
            raw_trips_df=raw_trips_df,
            trips_df=trips_df,
            od_matrix=od_matrix,
            file_metadata=file_metadata,
            zone_ids=zone_ids,
            zone_id_to_idx=zone_id_to_idx,
            idx_to_zone_id=idx_to_zone_id,
        )

        self._validate(
            trips_df=trips_df,
            od_matrix=od_matrix,
            metadata=metadata,
        )

        logger.info(
            "Trips file loaded successfully | od_pairs=%d | total_demand=%.4f | zones=%d",
            len(trips_df),
            float(trips_df["flow"].sum()),
            len(zone_ids),
        )

        return TripsReadResult(
            trips_df=trips_df,
            od_matrix=od_matrix,
            metadata=metadata,
        )

    def _read_lines(self) -> List[str]:
        """
        Read all lines from the trips file.

        Returns
        -------
        List[str]
            Raw file lines.
        """

        with self.path.open("r", encoding="utf-8", errors="replace") as file:
            return file.readlines()

    def _parse_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """
        Parse metadata tags from the trips file.

        Recognized metadata includes:
        - number of zones;
        - total OD flow;
        - dates embedded in metadata lines.

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
        dates_in_metadata = []

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("<NUMBER OF ZONES>"):
                try:
                    metadata["number_of_zones"] = int(stripped.split()[-1])
                except (ValueError, IndexError):
                    logger.warning("Could not parse number of zones from line: %s", stripped)

            elif upper.startswith("<TOTAL OD FLOW"):
                try:
                    metadata["total_od_flow"] = float(stripped.split()[-1])
                except (ValueError, IndexError):
                    logger.warning("Could not parse total OD flow from line: %s", stripped)

                # Some multiday files include dates inside total-flow metadata tags.
                dates_in_metadata.extend(self._extract_dates_from_text(stripped))

            else:
                dates_in_metadata.extend(self._extract_dates_from_text(stripped))

            if upper == "<END OF METADATA>":
                break

        if dates_in_metadata:
            metadata["dates_in_metadata"] = sorted(set(dates_in_metadata))

        return metadata

    def _parse_trip_records(self, lines: List[str]) -> pd.DataFrame:
        """
        Parse all OD records from the trips file.

        The parser scans the file sequentially and tracks:
        - the current matrix block;
        - the current date/time, if available;
        - the current origin node.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        Returns
        -------
        pd.DataFrame
            Raw long-format OD demand records.
        """

        records: List[Dict[str, Any]] = []

        data_started = False
        current_origin: Optional[int] = None
        current_date: Optional[str] = None
        current_time: Optional[str] = None
        current_datetime: Optional[str] = None
        matrix_index = -1

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            upper = stripped.upper()

            if upper == "<END OF METADATA>":
                data_started = True
                continue

            if not data_started:
                continue

            matrix_info = self._parse_matrix_header(stripped)

            if matrix_info is not None:
                matrix_index += 1
                current_date, current_time, current_datetime = matrix_info
                current_origin = None
                continue

            origin = self._parse_origin_line(stripped)

            if origin is not None:
                current_origin = origin
                continue

            if current_origin is not None and ":" in stripped:
                records.extend(
                    self._parse_od_pairs_line(
                        line=stripped,
                        origin=current_origin,
                        date=current_date,
                        time=current_time,
                        datetime_value=current_datetime,
                        matrix_index=max(matrix_index, 0),
                    )
                )

        df = pd.DataFrame(records)

        # Standard static TNTP files may have no explicit matrix header. In that
        # case matrix_index remains 0 for all records through the max(..., 0) call.
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "origin",
                    "destination",
                    "flow",
                    "date",
                    "time",
                    "datetime",
                    "matrix_index",
                ]
            )

        return df

    def _parse_matrix_header(
        self,
        line: str,
    ) -> Optional[Tuple[Optional[str], Optional[str], Optional[str]]]:
        """
        Parse a matrix/date header line.

        Supported examples:
        - <MATRIX DATE> 2022-10-01 00:00
        - <matrix> 2022-10-01 00:00
        - <matrix date> 2022-10-01 00:00

        Parameters
        ----------
        line : str
            Input line.

        Returns
        -------
        Optional[Tuple[Optional[str], Optional[str], Optional[str]]]
            Parsed date, time and datetime string. Returns None if the line is
            not a matrix header.
        """

        pattern = (
            r"<\s*matrix(?:\s+date)?\s*>?\s*"
            r"(?P<date>\d{4}-\d{2}-\d{2})?"
            r"\s*"
            r"(?P<time>\d{2}:\d{2})?"
        )

        match = re.match(pattern, line, flags=re.IGNORECASE)

        if not match:
            return None

        date_value = match.group("date")
        time_value = match.group("time")

        if date_value and time_value:
            datetime_value = f"{date_value} {time_value}"
        elif date_value:
            datetime_value = date_value
        else:
            datetime_value = None

        return date_value, time_value, datetime_value

    def _parse_origin_line(self, line: str) -> Optional[int]:
        """
        Parse an origin declaration line.

        Parameters
        ----------
        line : str
            Input line.

        Returns
        -------
        Optional[int]
            Origin node ID if the line declares an origin, otherwise None.
        """

        match = re.match(r"Origin\s+(?P<origin>\d+)", line, flags=re.IGNORECASE)

        if not match:
            return None

        return int(match.group("origin"))

    def _parse_od_pairs_line(
        self,
        line: str,
        origin: int,
        date: Optional[str],
        time: Optional[str],
        datetime_value: Optional[str],
        matrix_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Parse a line containing destination-flow pairs.

        Example
        -------
        "2 : 100; 3 : 120; 4 : 80;"

        Parameters
        ----------
        line : str
            Input line with OD entries.

        origin : int
            Current origin node ID.

        date : Optional[str]
            Current matrix date, if available.

        time : Optional[str]
            Current matrix time, if available.

        datetime_value : Optional[str]
            Current matrix datetime value, if available.

        matrix_index : int
            Sequential matrix block index.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed OD records.
        """

        records = []
        entries = line.split(";")

        for entry in entries:
            entry = entry.strip()

            if not entry or ":" not in entry:
                continue

            destination_text, flow_text = entry.split(":", 1)

            try:
                destination = int(destination_text.strip())
                flow = float(flow_text.strip())
            except ValueError:
                continue

            if flow == 0 and not self.include_zero_flows:
                continue

            records.append(
                {
                    "origin": int(origin),
                    "destination": int(destination),
                    "flow": float(flow),
                    "date": date,
                    "time": time,
                    "datetime": datetime_value,
                    "matrix_index": int(matrix_index),
                }
            )

        return records

    def _aggregate_trips(
        self,
        raw_trips_df: pd.DataFrame,
        file_metadata: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Aggregate raw OD records according to the configured rule.

        Parameters
        ----------
        raw_trips_df : pd.DataFrame
            Raw parsed OD demand records.

        file_metadata : Dict[str, Any]
            Parsed metadata from the file.

        Returns
        -------
        pd.DataFrame
            Aggregated OD table with columns origin, destination and flow.
        """

        df = raw_trips_df.copy()

        # If dates are missing in the body but available in metadata, use the
        # first metadata date as a fallback. This mirrors common TNTP variants
        # where the date is stored only in the metadata tags.
        if (
            "date" in df.columns
            and df["date"].isna().all()
            and file_metadata.get("dates_in_metadata")
        ):
            df["date"] = file_metadata["dates_in_metadata"][0]

        if self.aggregation == "sum":
            return (
                df.groupby(["origin", "destination"], as_index=False)["flow"]
                .sum()
            )

        if self.aggregation == "average_matrix":
            matrix_count = max(int(df["matrix_index"].nunique()), 1)

            aggregated = (
                df.groupby(["origin", "destination"], as_index=False)["flow"]
                .sum()
            )

            aggregated["flow"] = aggregated["flow"] / matrix_count
            return aggregated

        if self.aggregation == "average_daily":
            if "date" not in df.columns or df["date"].isna().all():
                # Without dates, average_daily degenerates to sum because there
                # is no reliable way to identify separate days.
                logger.warning(
                    "average_daily aggregation requested, but no dates were found. "
                    "Falling back to sum aggregation."
                )

                return (
                    df.groupby(["origin", "destination"], as_index=False)["flow"]
                    .sum()
                )

            daily = (
                df.groupby(["date", "origin", "destination"], as_index=False)["flow"]
                .sum()
            )

            num_days = max(int(daily["date"].nunique()), 1)

            averaged = (
                daily.groupby(["origin", "destination"], as_index=False)["flow"]
                .sum()
            )

            averaged["flow"] = averaged["flow"] / num_days
            return averaged

        raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def _normalize_trips_df(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OD table data types and ordering.

        Parameters
        ----------
        trips_df : pd.DataFrame
            Aggregated OD table.

        Returns
        -------
        pd.DataFrame
            Normalized OD table.
        """

        df = trips_df.copy()

        df["origin"] = pd.to_numeric(df["origin"], errors="coerce")
        df["destination"] = pd.to_numeric(df["destination"], errors="coerce")
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["origin", "destination", "flow"])
        dropped = before - len(df)

        if dropped > 0:
            logger.warning("Dropped %d OD rows with invalid origin, destination or flow.", dropped)

        df["origin"] = df["origin"].astype(int)
        df["destination"] = df["destination"].astype(int)
        df["flow"] = df["flow"].astype(float)

        # Sparse matrices do not store NaN values. If a file contains NaN-like
        # demand values, they are removed at reader level.
        df = df[np.isfinite(df["flow"])].copy()

        if not self.include_zero_flows:
            df = df[df["flow"] != 0].copy()

        df = df.sort_values(["origin", "destination"]).reset_index(drop=True)

        return df

    def _extract_zone_ids_from_origin_lines(
        self,
        lines: List[str],
    ) -> List[int]:
        """
        Extract real zone IDs from Origin lines in the TNTP trips file.

        Project-specific note
        ---------------------
        In the synthetic scenario generator, only zone nodes are written as Origin
        blocks. Therefore, Origin lines provide the most reliable source of the real
        zone IDs used by the demand matrix.

        Parameters
        ----------
        lines : List[str]
            Raw file lines.

        Returns
        -------
        List[int]
            Ordered unique zone IDs extracted from Origin declarations.
        """

        zone_ids = []
        seen = set()
        data_started = False

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            if upper == "<END OF METADATA>":
                data_started = True
                continue

            if not data_started:
                continue

            origin = self._parse_origin_line(stripped)

            if origin is not None and origin not in seen:
                zone_ids.append(int(origin))
                seen.add(int(origin))

        return zone_ids


    def _infer_zone_ids(
        self,
        trips_df: pd.DataFrame,
        raw_trips_df: pd.DataFrame,
        file_metadata: Dict[str, Any],
        origin_zone_ids: List[int],
    ) -> List[int]:
        """
        Infer real zone IDs used to build the compact OD matrix.

        Important
        ---------
        The returned IDs are real node IDs, not zero-based matrix indices.
        For example, if the zones are [1, 4, 7], the OD matrix will still have
        shape [3, 3], but the mapping will be:

            1 -> 0
            4 -> 1
            7 -> 2

        Parameters
        ----------
        trips_df : pd.DataFrame
            Normalized and aggregated OD table.

        raw_trips_df : pd.DataFrame
            Raw parsed OD records before aggregation.

        file_metadata : Dict[str, Any]
            Parsed file metadata.

        origin_zone_ids : List[int]
            Zone IDs extracted from Origin lines.

        Returns
        -------
        List[int]
            Ordered real zone IDs.
        """

        if origin_zone_ids:
            zone_ids = [int(zone_id) for zone_id in origin_zone_ids]
        else:
            # Fallback for non-standard files where Origin lines were not detected.
            # Use finite OD records only. This avoids non-zone destinations that were
            # exported with NaN demand.
            finite_raw = raw_trips_df.copy()

            if "flow" in finite_raw.columns:
                finite_raw["flow"] = pd.to_numeric(finite_raw["flow"], errors="coerce")
                finite_raw = finite_raw[np.isfinite(finite_raw["flow"])].copy()

            zone_ids = sorted(
                set(trips_df["origin"].astype(int)).union(
                    set(trips_df["destination"].astype(int))
                ).union(
                    set(finite_raw["origin"].dropna().astype(int))
                    if "origin" in finite_raw.columns else set()
                ).union(
                    set(finite_raw["destination"].dropna().astype(int))
                    if "destination" in finite_raw.columns else set()
                )
            )

        expected_num_zones = file_metadata.get("number_of_zones")

        if expected_num_zones is not None and int(expected_num_zones) != len(zone_ids):
            message = (
                "The number of inferred zone IDs does not match <NUMBER OF ZONES>. "
                f"inferred={len(zone_ids)} | metadata={int(expected_num_zones)} | "
                f"zone_ids={zone_ids}"
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

        if not zone_ids:
            raise ValueError("Could not infer any zone IDs from the trips file.")

        return zone_ids


    @staticmethod
    def _build_zone_mappings(
        zone_ids: List[int],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Build real-zone-ID to compact-index mappings.

        Parameters
        ----------
        zone_ids : List[int]
            Ordered real zone IDs.

        Returns
        -------
        Tuple[Dict[int, int], Dict[int, int]]
            zone_id_to_idx and idx_to_zone_id mappings.
        """

        zone_id_to_idx = {
            int(zone_id): int(idx)
            for idx, zone_id in enumerate(zone_ids)
        }

        idx_to_zone_id = {
            int(idx): int(zone_id)
            for zone_id, idx in zone_id_to_idx.items()
        }

        return zone_id_to_idx, idx_to_zone_id


    def _build_sparse_matrix(
        self,
        trips_df: pd.DataFrame,
        zone_id_to_idx: Dict[int, int],
    ) -> sparse.csr_matrix:
        """
        Build a compact sparse OD matrix using explicit zone ID mapping.

        Important
        ---------
        This method does not assume that zone IDs are consecutive or 1-based.

        Example
        -------
        If the real zone IDs are:

            [1, 4, 7]

        the compact OD matrix has shape:

            [3, 3]

        and the mapping is:

            1 -> 0
            4 -> 1
            7 -> 2

        Parameters
        ----------
        trips_df : pd.DataFrame
            Normalized OD table with real origin and destination node IDs.

        zone_id_to_idx : Dict[int, int]
            Mapping from real zone ID to compact matrix index.

        Returns
        -------
        sparse.csr_matrix
            Sparse OD matrix in compact zone-space.
        """

        origins_real = trips_df["origin"].to_numpy(dtype=int)
        destinations_real = trips_df["destination"].to_numpy(dtype=int)
        flows = trips_df["flow"].to_numpy(dtype=float)

        origin_indices = []
        destination_indices = []
        valid_flows = []

        invalid_records = []

        for origin_id, destination_id, flow in zip(origins_real, destinations_real, flows):
            origin_id = int(origin_id)
            destination_id = int(destination_id)

            origin_idx = zone_id_to_idx.get(origin_id)
            destination_idx = zone_id_to_idx.get(destination_id)

            if origin_idx is None or destination_idx is None:
                invalid_records.append(
                    {
                        "origin": origin_id,
                        "destination": destination_id,
                        "flow": float(flow),
                        "reason": "origin_or_destination_not_in_zone_mapping",
                    }
                )
                continue

            origin_indices.append(origin_idx)
            destination_indices.append(destination_idx)
            valid_flows.append(float(flow))

        if invalid_records:
            message = (
                f"{len(invalid_records)} OD records could not be mapped to compact "
                "zone-space because origin or destination is not in zone_id_to_idx."
            )

            if self.strict:
                raise ValueError(
                    f"{message} Sample: {invalid_records[:10]}"
                )

            logger.warning("%s They will be ignored. Sample: %s", message, invalid_records[:10])

        num_zones = len(zone_id_to_idx)

        matrix = sparse.coo_matrix(
            (
                np.asarray(valid_flows, dtype=np.float32),
                (
                    np.asarray(origin_indices, dtype=np.int64),
                    np.asarray(destination_indices, dtype=np.int64),
                ),
            ),
            shape=(num_zones, num_zones),
            dtype=np.float32,
        )

        return matrix.tocsr()


    def _validate(
        self,
        trips_df: pd.DataFrame,
        od_matrix: sparse.csr_matrix,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Validate the normalized trips output.

        Parameters
        ----------
        trips_df : pd.DataFrame
            Normalized OD table.

        od_matrix : sparse.csr_matrix
            Sparse OD matrix.

        metadata : Dict[str, Any]
            Reader metadata.

        Raises
        ------
        ValueError
            If the trips table or OD matrix is invalid.
        """

        if trips_df.empty:
            raise ValueError("Normalized trips table is empty.")

        if od_matrix.shape[0] != od_matrix.shape[1]:
            raise ValueError(f"OD matrix must be square. Found shape: {od_matrix.shape}")

        if od_matrix.nnz == 0:
            raise ValueError("OD matrix contains no non-zero demand entries.")

        if (trips_df["flow"] < 0).any():
            raise ValueError("Trips file contains negative OD flows.")

        expected_total = metadata.get("file_metadata", {}).get("total_od_flow")

        if expected_total is not None and self.aggregation == "sum":
            actual_total = float(trips_df["flow"].sum())
            absolute_error = abs(actual_total - float(expected_total))

            if absolute_error > 1e-2:
                logger.warning(
                    "Parsed total OD flow does not match metadata total. "
                    "parsed=%.6f | metadata=%.6f | abs_error=%.6f",
                    actual_total,
                    float(expected_total),
                    absolute_error,
                )


    def _build_metadata(
        self,
        raw_trips_df: pd.DataFrame,
        trips_df: pd.DataFrame,
        od_matrix: sparse.csr_matrix,
        file_metadata: Dict[str, Any],
        zone_ids: List[int],
        zone_id_to_idx: Dict[int, int],
        idx_to_zone_id: Dict[int, int],
    ) -> Dict[str, Any]:
        """
        Build metadata describing the loaded trips file.

        Parameters
        ----------
        raw_trips_df : pd.DataFrame
            Raw parsed OD records before aggregation.

        trips_df : pd.DataFrame
            Aggregated normalized OD table.

        od_matrix : sparse.csr_matrix
            Sparse OD matrix.

        file_metadata : Dict[str, Any]
            Metadata parsed from the file header.

        num_zones : int
            Inferred number of zones.

        Returns
        -------
        Dict[str, Any]
            Reader metadata.
        """

        num_zones = len(zone_ids)
        possible_pairs = num_zones * num_zones

        metadata: Dict[str, Any] = {
            "source_file": str(self.path),
            "file_metadata": file_metadata,
            "aggregation": self.aggregation,
            "num_zones": int(num_zones),
            "zone_ids": [int(zone_id) for zone_id in zone_ids],
            "zone_id_to_idx": {
                int(zone_id): int(idx)
                for zone_id, idx in zone_id_to_idx.items()
            },
            "idx_to_zone_id": {
                int(idx): int(zone_id)
                for idx, zone_id in idx_to_zone_id.items()
            },
            "matrix_indexing": "compact_zone_space",
            "matrix_indexing_note": (
                "Rows and columns of od_matrix use compact zone indices. "
                "Use zone_id_to_idx and idx_to_zone_id to map between real node IDs "
                "and matrix indices."
            ),
            "num_raw_records": int(len(raw_trips_df)),
            "num_od_pairs": int(len(trips_df)),
            "matrix_shape": tuple(int(x) for x in od_matrix.shape),
            "matrix_nnz": int(od_matrix.nnz),
            "density": float(od_matrix.nnz / possible_pairs) if possible_pairs else 0.0,
            "sparsity": float(1.0 - (od_matrix.nnz / possible_pairs)) if possible_pairs else 1.0,
            "total_demand": float(trips_df["flow"].sum()),
            "mean_od_flow": float(trips_df["flow"].mean()),
            "min_od_flow": float(trips_df["flow"].min()),
            "max_od_flow": float(trips_df["flow"].max()),
            "num_matrix_blocks": int(raw_trips_df["matrix_index"].nunique())
            if "matrix_index" in raw_trips_df.columns else 1,
        }

        if "date" in raw_trips_df.columns and raw_trips_df["date"].notna().any():
            dates = sorted(raw_trips_df["date"].dropna().unique().tolist())
            metadata["dates"] = dates
            metadata["num_dates"] = len(dates)

        if "datetime" in raw_trips_df.columns and raw_trips_df["datetime"].notna().any():
            metadata["num_datetimes"] = int(raw_trips_df["datetime"].dropna().nunique())

        return metadata


    @staticmethod
    def _extract_dates_from_text(text: str) -> List[str]:
        """
        Extract ISO date strings from a text line.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        List[str]
            Date strings matching YYYY-MM-DD.
        """

        return re.findall(r"\d{4}-\d{2}-\d{2}", text)


def read_tntp_trips(
    path: Union[str, Path],
    aggregation: str = "average_daily",
    matrix_format: str = "csr",
    include_zero_flows: bool = False,
    strict: bool = True,
) -> TripsReadResult:
    """
    Convenience function to read a TNTP-like trips file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the trips file.

    aggregation : str, default="average_daily"
        Aggregation rule for multiday or multi-matrix demand files.

    matrix_format : str, default="csr"
        Sparse matrix format requested.

    include_zero_flows : bool, default=False
        Whether zero-flow OD records should be preserved in trips_df.

    strict : bool, default=True
        Whether to use strict validation behavior.

    Returns
    -------
    TripsReadResult
        Normalized trips DataFrame, sparse OD matrix and metadata.
    """

    reader = TNTPTripsReader(
        path=path,
        aggregation=aggregation,
        matrix_format=matrix_format,
        include_zero_flows=include_zero_flows,
        strict=strict,
    )

    return reader.read()