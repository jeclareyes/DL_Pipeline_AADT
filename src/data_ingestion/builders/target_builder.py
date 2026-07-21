# src/data_ingestion/builders/target_builder.py
from __future__ import annotations

"""
Target Builder
==============

This module builds model targets and observation masks from processed traffic
tables.

Project context
---------------
In the AADT / traffic assignment pipeline, the training model needs aligned
target vectors for:

- link-level flows;
- origin-destination demand.

These targets must follow the same ordering used by the graph, the route set
and the model-ready tensors. The target builder centralizes this logic so the
training pipeline does not need to reconstruct masks, fill missing values, or
project OD demand into model space.

This builder receives already processed objects, such as:

- link_df from link_table_builder.py;
- trips_df from tntp_trips_reader.py;
- routes_by_od from tntp_routes_reader.py;
- optional indexing payloads from graph_builder.py.

This module does not read TNTP files, build graphs, compute routes, or save
artifacts.

Design principles
-----------------
- Preserve NaN information as observation masks.
- Fill missing target values with zero only after creating masks.
- Align flow targets with link_df row order.
- Align OD targets with routes_by_od OD-pair order.
- Return NumPy arrays and optional CPU tensors.
- Keep device transfer outside this builder.
"""

from networkx.generators import spectral_graph_forge

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger(__name__)

ODPair = Tuple[int, int]


@dataclass(frozen=True)
class TargetBuildResult:
    """
    Container returned by TargetBuilder.

    Attributes
    ----------
    targets : Dict[str, Any]
        Dictionary containing flow and OD targets, masks, NumPy arrays and
        optional PyTorch tensors.

    metadata : Dict[str, Any]
        Metadata describing target construction, observed shares and dimensions.
    """

    targets: Dict[str, Any]
    metadata: Dict[str, Any]


class TargetBuilder:
    """
    Build flow and OD targets for model training.

    Parameters
    ----------
    create_tensors : bool, default=True
        If True, PyTorch tensors are created in addition to NumPy arrays.

    tensor_device : str, default="cpu"
        Device used when creating tensors. For artifact generation, "cpu" is
        recommended.

    tensor_dtype : torch.dtype, default=torch.float32
        Tensor dtype for target and mask tensors.

    strict : bool, default=True
        If True, missing required columns or dimensional inconsistencies raise
        errors.

    missing_target_fill_value : float, default=0.0
        Value used to fill missing targets after masks have been created.
    """

    REQUIRED_LINK_COLUMNS = {
        "volume",
    }

    REQUIRED_TRIPS_COLUMNS = {
        "origin",
        "destination",
        "flow",
    }

    def __init__(
        self,
        create_tensors: bool = True,
        tensor_device: str = "cpu",
        tensor_dtype: torch.dtype = torch.float32,
        strict: bool = True,
        missing_target_fill_value: float = 0.0,
    ) -> None:
        self.create_tensors = bool(create_tensors)
        self.tensor_device = str(tensor_device)
        self.tensor_dtype = tensor_dtype
        self.strict = bool(strict)
        self.missing_target_fill_value = float(missing_target_fill_value)

    def build(
        self,
        link_df: pd.DataFrame,
        trips_df: pd.DataFrame,
        routes_by_od: Dict[ODPair, List[List[int]]],
        edge_indexing: Optional[Dict[str, Any]] = None,
        od_indexing: Optional[Dict[str, Any]] = None,
    ) -> TargetBuildResult:
        """
        Build all model targets and masks.

        Parameters
        ----------
        link_df : pd.DataFrame
            Canonical link table. The row order is assumed to be the model link
            order unless edge_indexing specifies otherwise.

        trips_df : pd.DataFrame
            Long-format OD demand table with columns origin, destination, flow.

        routes_by_od : Dict[ODPair, List[List[int]]]
            Route dictionary. Its key order defines the model OD order unless
            od_indexing specifies otherwise.

        edge_indexing : Optional[Dict[str, Any]], default=None
            Optional edge indexing payload from graph_builder.py.

        od_indexing : Optional[Dict[str, Any]], default=None
            Optional OD indexing payload.

        Returns
        -------
        TargetBuildResult
            Target dictionary and metadata.
        """

        logger.info("Building flow and OD targets.")

        self._require_columns(
            df=link_df,
            required_columns=self.REQUIRED_LINK_COLUMNS,
            df_name="link_df",
        )

        self._require_columns(
            df=trips_df,
            required_columns=self.REQUIRED_TRIPS_COLUMNS,
            df_name="trips_df",
        )

        flow_payload = self._build_flow_targets(
            link_df=link_df,
            edge_indexing=edge_indexing,
        )

        od_payload = self._build_od_targets(
            trips_df=trips_df,
            routes_by_od=routes_by_od,
            od_indexing=od_indexing,
        )

        targets = {
            **flow_payload,
            **od_payload,
        }

        if self.create_tensors:
            targets.update(self._build_tensor_payload(targets))

        metadata = self._build_metadata(
            targets=targets,
            link_df=link_df,
            trips_df=trips_df,
            routes_by_od=routes_by_od,
        )

        self._validate_targets(
            targets=targets,
            metadata=metadata,
        )

        logger.info(
            "Targets built successfully | links=%d | od_pairs=%d | observed_flows=%d | observed_od=%d",
            metadata["num_links"],
            metadata["num_od_pairs"],
            metadata["num_observed_flows"],
            metadata["num_observed_od_pairs"],
        )

        return TargetBuildResult(
            targets=targets,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Flow targets
    # ------------------------------------------------------------------

    def _build_flow_targets(
        self,
        link_df: pd.DataFrame,
        edge_indexing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build link-flow target vector and observation mask.

        The flow target must be aligned with the model link order. When
        edge_indexing is provided, the canonical model order is taken from
        edge_indexing["link_pair_indices"].

        This avoids comparing:

            predicted_flows[i] from model edge order

        against:

            link_df["volume"].iloc[i] from tabular/link_id order

        when both orders differ.
        """

        if edge_indexing is None or "link_pair_indices" not in edge_indexing:
            message = (
                "edge_indexing['link_pair_indices'] is required to build flow targets "
                "safely. Flow targets must be explicitly aligned with the model link "
                "order; relying on link_df row order as an implicit fallback is not "
                "allowed in strict mode."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(
                "%s Falling back to link_df row order because strict=False.",
                message,
            )

            link_rows = link_df.reset_index(drop=True).copy()

        else:
            link_rows = self._build_edge_aligned_link_rows(
                link_df=link_df,
                link_pair_indices=edge_indexing["link_pair_indices"],
            )

        raw_flow = pd.to_numeric(
            link_rows["volume"],
            errors="coerce",
        ).to_numpy(dtype=np.float32)

        observed_mask = (~np.isnan(raw_flow)).astype(np.float32)

        target = np.nan_to_num(
            raw_flow,
            nan=self.missing_target_fill_value,
        ).astype(np.float32)

        payload = {
            "flows_raw_np": raw_flow,
            "flows_target_np": target,
            "flows_observed_mask_np": observed_mask,
        }

        if "link_id" in link_rows.columns:
            payload["flow_target_link_ids"] = (
                pd.to_numeric(link_rows["link_id"], errors="coerce")
                .fillna(-1)
                .astype(np.int64)
                .to_numpy()
            )

        payload["flow_target_link_pair_indices"] = (
            link_rows[["init_node", "term_node"]]
            .astype(np.int64)
            .to_numpy()
        )

        return payload


    def _build_edge_aligned_link_rows(
        self,
        link_df: pd.DataFrame,
        link_pair_indices: Any,
    ) -> pd.DataFrame:
        """
        Reindex link_df to the model edge order.

        Parameters
        ----------
        link_df:
            Canonical link table.

        link_pair_indices:
            Directed edge order used by the model. Shape [num_links, 2].

        Returns
        -------
        pd.DataFrame
            link_df rows reordered to match link_pair_indices.
        """

        required_columns = {"init_node", "term_node", "volume"}
        missing_columns = required_columns - set(link_df.columns)

        if missing_columns:
            raise ValueError(
                "Cannot align flow targets because link_df is missing columns: "
                f"{sorted(missing_columns)}"
            )

        edge_order = np.asarray(link_pair_indices, dtype=np.int64)

        if edge_order.ndim != 2 or edge_order.shape[1] != 2:
            raise ValueError(
                "link_pair_indices must have shape [num_links, 2]. "
                f"Received shape {edge_order.shape}."
            )

        links = link_df.copy()

        links["init_node"] = pd.to_numeric(
            links["init_node"],
            errors="coerce",
        )
        links["term_node"] = pd.to_numeric(
            links["term_node"],
            errors="coerce",
        )

        links = links.dropna(
            subset=["init_node", "term_node"],
        ).copy()

        links["init_node"] = links["init_node"].astype(int)
        links["term_node"] = links["term_node"].astype(int)

        duplicated_edges = links.duplicated(
            subset=["init_node", "term_node"],
            keep=False,
        )

        if duplicated_edges.any():
            duplicated_sample = (
                links.loc[duplicated_edges, ["init_node", "term_node"]]
                .head(10)
                .to_dict("records")
            )

            raise ValueError(
                "Cannot build edge-aligned flow targets because link_df contains "
                "duplicated directed edges. Sample: "
                f"{duplicated_sample}"
            )

        edge_to_row = {
            (int(row.init_node), int(row.term_node)): row._asdict()
            for row in links.itertuples(index=False)
        }

        aligned_rows = []
        missing_edges = []

        for position, (u, v) in enumerate(edge_order):
            edge = (int(u), int(v))

            if edge not in edge_to_row:
                missing_edges.append(
                    {
                        "position": int(position),
                        "init_node": int(u),
                        "term_node": int(v),
                    }
                )
                continue

            aligned_rows.append(edge_to_row[edge])

        if missing_edges:
            raise ValueError(
                "Some model edges could not be found in link_df while building "
                "flow targets. First missing edges: "
                f"{missing_edges[:10]}"
            )

        aligned_link_df = pd.DataFrame(aligned_rows)

        if len(aligned_link_df) != edge_order.shape[0]:
            raise ValueError(
                "Aligned link table length mismatch. "
                f"Expected {edge_order.shape[0]}, got {len(aligned_link_df)}."
            )

        return aligned_link_df.reset_index(drop=True)


    # ------------------------------------------------------------------
    # OD targets
    # ------------------------------------------------------------------

    def _build_od_targets(
        self,
        trips_df: pd.DataFrame,
        routes_by_od: Dict[ODPair, List[List[int]]],
        od_indexing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build OD target vector aligned with model OD order.

        Parameters
        ----------
        trips_df : pd.DataFrame
            Long-format OD demand table.

        routes_by_od : Dict[ODPair, List[List[int]]]
            Route dictionary. Its key order defines OD order if od_indexing is
            not provided.

        od_indexing : Optional[Dict[str, Any]], default=None
            Optional OD indexing payload containing "od_pairs".

        Returns
        -------
        Dict[str, Any]
            OD target arrays, masks and OD pair ordering.
        """

        od_pairs = self._resolve_od_pairs(
            routes_by_od=routes_by_od,
            od_indexing=od_indexing,
        )

        trips = trips_df.copy()

        trips["origin"] = pd.to_numeric(
            trips["origin"],
            errors="coerce",
        )

        trips["destination"] = pd.to_numeric(
            trips["destination"],
            errors="coerce",
        )

        trips["flow"] = pd.to_numeric(
            trips["flow"],
            errors="coerce",
        )

        trips = trips.dropna(
            subset=["origin", "destination", "flow"],
        ).copy()

        trips["origin"] = trips["origin"].astype(int)
        trips["destination"] = trips["destination"].astype(int)
        trips["flow"] = trips["flow"].astype(float)

        od_flow_lookup = {
            (int(row.origin), int(row.destination)): float(row.flow)
            for row in trips.itertuples(index=False)
        }

        raw_od = np.array(
            [
                od_flow_lookup.get(od_pair, np.nan)
                for od_pair in od_pairs
            ],
            dtype=np.float32,
        )

        observed_mask = (~np.isnan(raw_od)).astype(np.float32)

        target = np.nan_to_num(
            raw_od,
            nan=self.missing_target_fill_value,
        ).astype(np.float32)

        od_pair_indices = np.array(
            od_pairs,
            dtype=np.int64,
        )

        return {
            "od_pairs": od_pairs,
            "od_pair_indices_np": od_pair_indices,
            "od_raw_np": raw_od,
            "od_target_np": target,
            "od_observed_mask_np": observed_mask,
        }

    def _resolve_od_pairs(
        self,
        routes_by_od: Dict[ODPair, List[List[int]]],
        od_indexing: Optional[Dict[str, Any]] = None,
    ) -> List[ODPair]:
        """
        Resolve the OD-pair order used for OD target construction.

        Parameters
        ----------
        routes_by_od : Dict[ODPair, List[List[int]]]
            Route dictionary.

        od_indexing : Optional[Dict[str, Any]], default=None
            Optional OD indexing payload.

        Returns
        -------
        List[ODPair]
            Ordered OD pairs.
        """

        if od_indexing is None or "od_pairs" not in od_indexing:
            message = (
                "od_indexing['od_pairs'] is required to build OD targets safely. "
                "OD targets must be explicitly aligned with the model OD order; "
                "relying on routes_by_od key order as an implicit fallback is not "
                "allowed in strict mode."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(
                "%s Falling back to routes_by_od key order because strict=False.",
                message,
            )

            return [
                (int(origin), int(destination))
                for origin, destination in routes_by_od.keys()
            ]   

        return [
            (int(origin), int(destination))
            for origin, destination in od_indexing["od_pairs"]
        ]

    # ------------------------------------------------------------------
    # Tensor payload
    # ------------------------------------------------------------------

    def _build_tensor_payload(
        self,
        targets: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Build PyTorch tensors from NumPy target arrays.

        Parameters
        ----------
        targets : Dict[str, Any]
            Target dictionary containing NumPy arrays.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tensor payload.
        """

        tensor_payload = {
            "flows_target_t": torch.tensor(
                targets["flows_target_np"],
                dtype=self.tensor_dtype,
                device=self.tensor_device,
            ),
            "flows_observed_mask_t": torch.tensor(
                targets["flows_observed_mask_np"],
                dtype=self.tensor_dtype,
                device=self.tensor_device,
            ),
            "od_target_t": torch.tensor(
                targets["od_target_np"],
                dtype=self.tensor_dtype,
                device=self.tensor_device,
            ),
            "od_observed_mask_t": torch.tensor(
                targets["od_observed_mask_np"],
                dtype=self.tensor_dtype,
                device=self.tensor_device,
            ),
        }

        if "od_pair_indices_np" in targets:
            tensor_payload["od_pair_indices_t"] = torch.tensor(
                targets["od_pair_indices_np"],
                dtype=torch.long,
                device=self.tensor_device,
            )

        return tensor_payload

    # ------------------------------------------------------------------
    # Validation and metadata
    # ------------------------------------------------------------------

    def _validate_targets(
        self,
        targets: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Validate target dimensions and values.

        Parameters
        ----------
        targets : Dict[str, Any]
            Target dictionary.

        metadata : Dict[str, Any]
            Target metadata.
        """

        num_links = metadata["num_links"]
        num_od_pairs = metadata["num_od_pairs"]

        if len(targets["flows_target_np"]) != num_links:
            raise ValueError("flows_target_np length does not match num_links.")

        if len(targets["flows_observed_mask_np"]) != num_links:
            raise ValueError("flows_observed_mask_np length does not match num_links.")

        if len(targets["od_target_np"]) != num_od_pairs:
            raise ValueError("od_target_np length does not match num_od_pairs.")

        if len(targets["od_observed_mask_np"]) != num_od_pairs:
            raise ValueError("od_observed_mask_np length does not match num_od_pairs.")

        if np.any(targets["flows_target_np"] < 0):
            raise ValueError("flows_target_np contains negative values.")

        if np.any(targets["od_target_np"] < 0):
            raise ValueError("od_target_np contains negative values.")

        self._validate_binary_mask(
            mask=targets["flows_observed_mask_np"],
            name="flows_observed_mask_np",
        )

        self._validate_binary_mask(
            mask=targets["od_observed_mask_np"],
            name="od_observed_mask_np",
        )

        if self.strict and metadata["num_observed_flows"] == 0:
            logger.warning(
                "No observed link flows were found. Training may still work if "
                "OD supervision or other losses are available."
            )

        if self.strict and metadata["num_observed_od_pairs"] == 0:
            logger.warning(
                "No observed OD targets were found. Training may still work if "
                "flow supervision or unsupervised losses are available."
            )

    @staticmethod
    def _validate_binary_mask(
        mask: np.ndarray,
        name: str,
    ) -> None:
        """
        Validate that a mask contains only 0 and 1.

        Parameters
        ----------
        mask : np.ndarray
            Mask array.

        name : str
            Mask name.
        """

        unique_values = set(np.unique(mask).tolist())

        if not unique_values.issubset({0.0, 1.0}):
            raise ValueError(
                f"{name} must contain only 0 and 1. Found values: {sorted(unique_values)}"
            )

    def _build_metadata(
        self,
        targets: Dict[str, Any],
        link_df: pd.DataFrame,
        trips_df: pd.DataFrame,
        routes_by_od: Dict[ODPair, List[List[int]]],
    ) -> Dict[str, Any]:
        """
        Build metadata describing target construction.

        Parameters
        ----------
        targets : Dict[str, Any]
            Target dictionary.

        link_df : pd.DataFrame
            Canonical link table.

        trips_df : pd.DataFrame
            Long-format OD demand table.

        routes_by_od : Dict[ODPair, List[List[int]]]
            Route dictionary.

        Returns
        -------
        Dict[str, Any]
            Target metadata.
        """

        flow_mask = targets["flows_observed_mask_np"]
        od_mask = targets["od_observed_mask_np"]

        num_links = int(len(targets["flows_target_np"]))
        num_od_pairs = int(len(targets["od_target_np"]))

        observed_flows = targets["flows_target_np"][flow_mask.astype(bool)]
        observed_od = targets["od_target_np"][od_mask.astype(bool)]

        metadata: Dict[str, Any] = {
            "num_links": num_links,
            "num_od_pairs": num_od_pairs,
            "num_routes_od_pairs": int(len(routes_by_od)),
            "num_rows_link_df": int(len(link_df)),
            "num_rows_trips_df": int(len(trips_df)),
            "num_observed_flows": int(flow_mask.sum()),
            "num_missing_flows": int(num_links - flow_mask.sum()),
            "observed_flow_share": float(flow_mask.mean()) if num_links > 0 else 0.0,
            "num_observed_od_pairs": int(od_mask.sum()),
            "num_missing_od_pairs": int(num_od_pairs - od_mask.sum()),
            "observed_od_share": float(od_mask.mean()) if num_od_pairs > 0 else 0.0,
            "missing_target_fill_value": self.missing_target_fill_value,
            "created_tensors": self.create_tensors,
            "tensor_device": self.tensor_device if self.create_tensors else None,
            "tensor_dtype": str(self.tensor_dtype) if self.create_tensors else None,
        }

        if len(observed_flows) > 0:
            metadata.update(
                {
                    "total_observed_flow": float(observed_flows.sum()),
                    "mean_observed_flow": float(observed_flows.mean()),
                    "min_observed_flow": float(observed_flows.min()),
                    "max_observed_flow": float(observed_flows.max()),
                }
            )
        else:
            metadata.update(
                {
                    "total_observed_flow": 0.0,
                    "mean_observed_flow": None,
                    "min_observed_flow": None,
                    "max_observed_flow": None,
                }
            )

        if len(observed_od) > 0:
            metadata.update(
                {
                    "total_observed_od": float(observed_od.sum()),
                    "mean_observed_od": float(observed_od.mean()),
                    "min_observed_od": float(observed_od.min()),
                    "max_observed_od": float(observed_od.max()),
                }
            )
        else:
            metadata.update(
                {
                    "total_observed_od": 0.0,
                    "mean_observed_od": None,
                    "min_observed_od": None,
                    "max_observed_od": None,
                }
            )

        return metadata

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        required_columns: Iterable[str],
        df_name: str,
    ) -> None:
        """
        Ensure a DataFrame contains required columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        required_columns : Iterable[str]
            Required column names.

        df_name : str
            Human-readable DataFrame name.

        Raises
        ------
        ValueError
            If required columns are missing.
        """

        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"{df_name} is missing required columns: {sorted(missing)}"
            )


def build_targets(
    link_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_by_od: Dict[ODPair, List[List[int]]],
    edge_indexing: Optional[Dict[str, Any]] = None,
    od_indexing: Optional[Dict[str, Any]] = None,
    create_tensors: bool = True,
    tensor_device: str = "cpu",
    tensor_dtype: torch.dtype = torch.float32,
    strict: bool = True,
    missing_target_fill_value: float = 0.0,
) -> TargetBuildResult:
    """
    Convenience function to build model targets and masks.

    Parameters
    ----------
    link_df : pd.DataFrame
        Canonical link table produced by link_table_builder.py.

    trips_df : pd.DataFrame
        Long-format OD demand table produced by tntp_trips_reader.py.

    routes_by_od : Dict[ODPair, List[List[int]]]
        Route dictionary produced by tntp_routes_reader.py.

    edge_indexing : Optional[Dict[str, Any]], default=None
        Edge indexing payload from graph_builder.py.

    od_indexing : Optional[Dict[str, Any]], default=None
        OD indexing payload.

    create_tensors : bool, default=True
        Whether to create PyTorch tensors.

    tensor_device : str, default="cpu"
        Device for tensor creation.

    tensor_dtype : torch.dtype, default=torch.float32
        Floating-point tensor dtype.

    strict : bool, default=True
        Whether to use strict validation behavior.

    missing_target_fill_value : float, default=0.0
        Fill value used after observation masks are created.

    Returns
    -------
    TargetBuildResult
        Target dictionary and metadata.
    """

    builder = TargetBuilder(
        create_tensors=create_tensors,
        tensor_device=tensor_device,
        tensor_dtype=tensor_dtype,
        strict=strict,
        missing_target_fill_value=missing_target_fill_value,
    )

    return builder.build(
        link_df=link_df,
        trips_df=trips_df,
        routes_by_od=routes_by_od,
        edge_indexing=edge_indexing,
        od_indexing=od_indexing,
    )