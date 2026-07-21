# src/data_ingestion/builders/link_table_builder.py

"""
Link Table Builder
==================

This module builds a canonical link table by merging the normalized TNTP
network table with the normalized TNTP flow table.

Project context
---------------
In the AADT / traffic assignment pipeline, the link table is the central
link-level representation used by downstream components.

It connects:

- physical network attributes from network.tntp;
- observed or assigned link volumes from flows.tntp;
- link identifiers used to align graph edges, flow targets and model tensors;
- optional cost/travel-time information used for diagnostics.

This builder is intentionally limited to link-table construction. It does not
read TNTP files, build NetworkX graphs, compute routes, create PyTorch tensors,
or save training artifacts.

Those responsibilities belong to:

- readers: read TNTP files;
- graph_builder.py: build the graph;
- target_builder.py: create targets and masks;
- training_artifact_builder.py: orchestrate and save the final artifact.

Design principles
-----------------
- Keep one row per directed link.
- Preserve all useful network attributes.
- Preserve missing volume values as NaN.
- Do not replace missing observations with zero.
- Fail early when required columns are missing.
- Make merge behavior explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkTableBuildResult:
    """
    Container returned by LinkTableBuilder.

    Attributes
    ----------
    link_df : pd.DataFrame
        Canonical link table.

    metadata : Dict[str, Any]
        Metadata describing the merge result, including number of links,
        matched flow records, missing volumes and duplicate-flow diagnostics.
    """

    link_df: pd.DataFrame
    metadata: Dict[str, Any]


class LinkTableBuilder:
    """
    Build a canonical link table from network and flow DataFrames.

    Parameters
    ----------
    strict : bool, default=True
        If True, duplicated network links or missing required columns raise
        errors. If False, some issues are reported in metadata and warnings.

    preserve_extra_flow_columns : bool, default=True
        If True, non-canonical columns from the flow table are retained after
        merging, using a "_flow" suffix when needed.

    aggregate_duplicate_flows : bool, default=True
        If True, duplicated flow records for the same directed link are
        aggregated before merging. Volume and cost are averaged by default.
        If False, duplicated flow records raise an error when strict=True.

    duplicate_flow_aggregation : str, default="mean"
        Aggregation rule for duplicated flow records.

        Supported values:
        - "mean"
        - "sum"
        - "first"
    """

    NETWORK_ENDPOINT_COLUMNS = ("init_node", "term_node")
    FLOW_ENDPOINT_COLUMNS = ("from_node", "to_node")

    REQUIRED_NETWORK_COLUMNS = {
        "init_node",
        "term_node",
        "effective_capacity",
        "length",
        "free_flow_time",
    }

    REQUIRED_FLOW_COLUMNS = {
        "from_node",
        "to_node",
        "volume",
    }

    CANONICAL_LINK_ORDER = [
        "link_id",
        "reverse_link_id",
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
        "volume",
        "cost",
    ]

    SUPPORTED_DUPLICATE_AGGREGATIONS = {"mean", "sum", "first"}

    def __init__(
        self,
        strict: bool = True,
        preserve_extra_flow_columns: bool = True,
        aggregate_duplicate_flows: bool = True,
        duplicate_flow_aggregation: str = "mean",
    ) -> None:
        self.strict = bool(strict)
        self.preserve_extra_flow_columns = bool(preserve_extra_flow_columns)
        self.aggregate_duplicate_flows = bool(aggregate_duplicate_flows)
        self.duplicate_flow_aggregation = str(duplicate_flow_aggregation)

        if self.duplicate_flow_aggregation not in self.SUPPORTED_DUPLICATE_AGGREGATIONS:
            raise ValueError(
                "Unsupported duplicate_flow_aggregation: "
                f"{self.duplicate_flow_aggregation}. Supported values: "
                f"{sorted(self.SUPPORTED_DUPLICATE_AGGREGATIONS)}"
            )

    def build(
        self,
        network_df: pd.DataFrame,
        flow_df: pd.DataFrame,
    ) -> LinkTableBuildResult:
        """
        Build the canonical link table.

        Parameters
        ----------
        network_df : pd.DataFrame
            Normalized network table produced by tntp_network_reader.py.

        flow_df : pd.DataFrame
            Normalized flow table produced by tntp_flow_reader.py.

        Returns
        -------
        LinkTableBuildResult
            Canonical link table and merge metadata.

        Raises
        ------
        ValueError
            If required columns are missing or severe consistency issues are
            found.
        """

        logger.info("Building canonical link table.")

        network = self._prepare_network_df(network_df)
        flows = self._prepare_flow_df(flow_df)

        duplicate_flow_info = self._diagnose_duplicate_flows(flows)

        if duplicate_flow_info["num_duplicate_flow_rows"] > 0:
            flows = self._handle_duplicate_flows(
                flows=flows,
                duplicate_flow_info=duplicate_flow_info,
            )

        link_df = self._merge_network_and_flows(
            network_df=network,
            flow_df=flows,
        )

        link_df = self._normalize_output_dtypes(link_df)
        link_df = self._order_columns(link_df)
        self._validate_link_df(link_df)

        metadata = self._build_metadata(
            network_df=network,
            flow_df=flows,
            link_df=link_df,
            duplicate_flow_info=duplicate_flow_info,
        )

        logger.info(
            "Link table built successfully | links=%d | observed_volumes=%d | missing_volumes=%d",
            metadata["num_links"],
            metadata["num_observed_volumes"],
            metadata["num_missing_volumes"],
        )

        return LinkTableBuildResult(
            link_df=link_df,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare_network_df(self, network_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize the network table before merging.

        Parameters
        ----------
        network_df : pd.DataFrame
            Network DataFrame.

        Returns
        -------
        pd.DataFrame
            Prepared network table.
        """

        self._require_columns(
            df=network_df,
            required_columns=self.REQUIRED_NETWORK_COLUMNS,
            df_name="network_df",
        )

        network = network_df.copy()

        network["init_node"] = pd.to_numeric(
            network["init_node"],
            errors="coerce",
        )

        network["term_node"] = pd.to_numeric(
            network["term_node"],
            errors="coerce",
        )

        before = len(network)

        network = network.dropna(
            subset=["init_node", "term_node"],
        ).copy()

        dropped = before - len(network)

        if dropped > 0:
            message = (
                f"Dropped {dropped} network rows because init_node or term_node "
                "could not be parsed."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

        network["init_node"] = network["init_node"].astype(int)
        network["term_node"] = network["term_node"].astype(int)

        self._validate_unique_directed_links(
            network,
            df_name="network_df",
            endpoint_columns=self.NETWORK_ENDPOINT_COLUMNS,
        )

        return network.reset_index(drop=True)

    def _prepare_flow_df(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize the flow table before merging.

        Parameters
        ----------
        flow_df : pd.DataFrame
            Flow DataFrame.

        Returns
        -------
        pd.DataFrame
            Prepared flow table with endpoint columns renamed to network schema.
        """

        self._require_columns(
            df=flow_df,
            required_columns=self.REQUIRED_FLOW_COLUMNS,
            df_name="flow_df",
        )

        flows = flow_df.copy()

        flows["from_node"] = pd.to_numeric(
            flows["from_node"],
            errors="coerce",
        )

        flows["to_node"] = pd.to_numeric(
            flows["to_node"],
            errors="coerce",
        )

        before = len(flows)

        flows = flows.dropna(
            subset=["from_node", "to_node"],
        ).copy()

        dropped = before - len(flows)

        if dropped > 0:
            message = (
                f"Dropped {dropped} flow rows because from_node or to_node "
                "could not be parsed."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

        flows["from_node"] = flows["from_node"].astype(int)
        flows["to_node"] = flows["to_node"].astype(int)

        flows["volume"] = pd.to_numeric(
            flows["volume"],
            errors="coerce",
        )

        if "cost" in flows.columns:
            flows["cost"] = pd.to_numeric(
                flows["cost"],
                errors="coerce",
            )

        flows = flows.rename(
            columns={
                "from_node": "init_node",
                "to_node": "term_node",
            }
        )

        return flows.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Duplicate flow handling
    # ------------------------------------------------------------------

    def _diagnose_duplicate_flows(self, flows: pd.DataFrame) -> Dict[str, Any]:
        """
        Diagnose duplicated flow records for the same directed link.

        Parameters
        ----------
        flows : pd.DataFrame
            Prepared flow table.

        Returns
        -------
        Dict[str, Any]
            Duplicate-flow diagnostics.
        """

        duplicated_mask = flows.duplicated(
            subset=["init_node", "term_node"],
            keep=False,
        )

        duplicated_rows = flows.loc[duplicated_mask].copy()

        duplicated_links = (
            duplicated_rows[["init_node", "term_node"]]
            .drop_duplicates()
            .apply(lambda row: (int(row["init_node"]), int(row["term_node"])), axis=1)
            .tolist()
            if not duplicated_rows.empty
            else []
        )

        return {
            "num_duplicate_flow_rows": int(duplicated_mask.sum()),
            "num_duplicate_flow_links": int(len(duplicated_links)),
            "duplicated_flow_links_sample": duplicated_links[:20],
        }

    def _handle_duplicate_flows(
        self,
        flows: pd.DataFrame,
        duplicate_flow_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Handle duplicated flow rows according to the configured policy.

        Parameters
        ----------
        flows : pd.DataFrame
            Prepared flow table.

        duplicate_flow_info : Dict[str, Any]
            Duplicate-flow diagnostics.

        Returns
        -------
        pd.DataFrame
            Flow table with unique directed links.
        """

        message = (
            "Duplicated flow records found for directed links. "
            f"duplicate_rows={duplicate_flow_info['num_duplicate_flow_rows']}, "
            f"duplicate_links={duplicate_flow_info['num_duplicate_flow_links']}"
        )

        if not self.aggregate_duplicate_flows:
            if self.strict:
                raise ValueError(message)

            logger.warning("%s. Keeping the first record per directed link.", message)

            return (
                flows
                .drop_duplicates(subset=["init_node", "term_node"], keep="first")
                .reset_index(drop=True)
            )

        logger.warning(
            "%s. Aggregating duplicated flow records using '%s'.",
            message,
            self.duplicate_flow_aggregation,
        )

        return self._aggregate_duplicate_flows(flows)

    def _aggregate_duplicate_flows(self, flows: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate duplicated flow records by directed link.

        Parameters
        ----------
        flows : pd.DataFrame
            Prepared flow table.

        Returns
        -------
        pd.DataFrame
            Aggregated flow table with one row per directed link.
        """

        endpoint_columns = ["init_node", "term_node"]

        numeric_columns = [
            column
            for column in flows.columns
            if column not in endpoint_columns
            and pd.api.types.is_numeric_dtype(flows[column])
        ]

        non_numeric_columns = [
            column
            for column in flows.columns
            if column not in endpoint_columns
            and column not in numeric_columns
        ]

        aggregation: Dict[str, Any] = {}

        for column in numeric_columns:
            if self.duplicate_flow_aggregation == "mean":
                aggregation[column] = "mean"
            elif self.duplicate_flow_aggregation == "sum":
                aggregation[column] = "sum"
            elif self.duplicate_flow_aggregation == "first":
                aggregation[column] = "first"

        for column in non_numeric_columns:
            aggregation[column] = "first"

        return (
            flows
            .groupby(endpoint_columns, as_index=False)
            .agg(aggregation)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Merge and output formatting
    # ------------------------------------------------------------------

    def _merge_network_and_flows(
        self,
        network_df: pd.DataFrame,
        flow_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge network attributes and flow observations.

        Parameters
        ----------
        network_df : pd.DataFrame
            Prepared network table.

        flow_df : pd.DataFrame
            Prepared flow table with init_node and term_node columns.

        Returns
        -------
        pd.DataFrame
            Canonical merged link table.
        """

        flow_columns = self._select_flow_columns_for_merge(flow_df)

        link_df = network_df.merge(
            flow_df[flow_columns],
            on=["init_node", "term_node"],
            how="left",
            suffixes=("", "_flow"),
            indicator=True,
        )

        link_df["has_flow_record"] = link_df["_merge"].eq("both")
        link_df = link_df.drop(columns=["_merge"])

        if "volume" not in link_df.columns:
            link_df["volume"] = np.nan

        return link_df

    def _select_flow_columns_for_merge(self, flow_df: pd.DataFrame) -> list[str]:
        """
        Select flow columns that should be merged into the link table.

        Parameters
        ----------
        flow_df : pd.DataFrame
            Prepared flow table.

        Returns
        -------
        list[str]
            Columns to merge.
        """

        required = ["init_node", "term_node", "volume"]

        optional = []

        if "cost" in flow_df.columns:
            optional.append("cost")

        if self.preserve_extra_flow_columns:
            extra = [
                column
                for column in flow_df.columns
                if column not in set(required + optional)
            ]

            return required + optional + extra

        return required + optional

    def _normalize_output_dtypes(self, link_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize output link-table dtypes.

        Parameters
        ----------
        link_df : pd.DataFrame
            Merged link table.

        Returns
        -------
        pd.DataFrame
            Link table with normalized dtypes.
        """

        df = link_df.copy()

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
            "capacity",
            "length",
            "free_flow_time",
            "b",
            "power",
            "speed",
            "toll",
            "volume",
            "cost",
        ]

        for column in integer_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in float_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in ["init_node", "term_node"]:
            df[column] = df[column].astype(int)

        if "link_id" in df.columns:
            df["link_id"] = df["link_id"].astype(int)

        if "reverse_link_id" in df.columns:
            df["reverse_link_id"] = df["reverse_link_id"].astype("Int64")

        if "lanes" in df.columns:
            df["lanes"] = df["lanes"].fillna(1).astype(int)

        if "vdf" in df.columns:
            df["vdf"] = df["vdf"].astype("Int64")

        if "link_type" in df.columns:
            df["link_type"] = df["link_type"].fillna(0).astype(int)

        if "has_flow_record" in df.columns:
            df["has_flow_record"] = df["has_flow_record"].astype(bool)

        return df

    def _order_columns(self, link_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder link table columns so canonical fields appear first.

        Parameters
        ----------
        link_df : pd.DataFrame
            Normalized link table.

        Returns
        -------
        pd.DataFrame
            Reordered link table.
        """

        canonical_present = [
            column
            for column in self.CANONICAL_LINK_ORDER
            if column in link_df.columns
        ]

        diagnostic_columns = [
            column
            for column in ["has_flow_record"]
            if column in link_df.columns
        ]

        extra_columns = [
            column
            for column in link_df.columns
            if column not in set(canonical_present + diagnostic_columns)
        ]

        return link_df[
            canonical_present + diagnostic_columns + extra_columns
        ].copy()

    # ------------------------------------------------------------------
    # Validation and metadata
    # ------------------------------------------------------------------

    def _validate_link_df(self, link_df: pd.DataFrame) -> None:
        """
        Validate the final canonical link table.

        Parameters
        ----------
        link_df : pd.DataFrame
            Canonical link table.

        Raises
        ------
        ValueError
            If required link-table conditions are not satisfied.
        """

        self._require_columns(
            df=link_df,
            required_columns=self.REQUIRED_NETWORK_COLUMNS | {"volume"},
            df_name="link_df",
        )

        if link_df.empty:
            raise ValueError("link_df is empty.")

        self._validate_unique_directed_links(
            link_df,
            df_name="link_df",
            endpoint_columns=self.NETWORK_ENDPOINT_COLUMNS,
        )

        if "link_id" in link_df.columns:
            duplicated_link_ids = link_df["link_id"].duplicated(keep=False)

            if duplicated_link_ids.any():
                duplicated_ids = (
                    link_df.loc[duplicated_link_ids, "link_id"]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    "Duplicated link_id values found in link_df: "
                    f"{duplicated_ids[:20]}"
                )

        if (link_df["effective_capacity"] <= 0).any():
            raise ValueError("link_df contains links with effective_capacity <= 0.")

        if (link_df["length"] < 0).any():
            raise ValueError("link_df contains links with length < 0.")

        if (link_df["free_flow_time"] < 0).any():
            raise ValueError("link_df contains links with free_flow_time < 0.")

        if (link_df["volume"].dropna() < 0).any():
            raise ValueError("link_df contains negative observed volumes.")

    def _build_metadata(
        self,
        network_df: pd.DataFrame,
        flow_df: pd.DataFrame,
        link_df: pd.DataFrame,
        duplicate_flow_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build metadata describing the link-table construction.

        Parameters
        ----------
        network_df : pd.DataFrame
            Prepared network table.

        flow_df : pd.DataFrame
            Prepared flow table.

        link_df : pd.DataFrame
            Final canonical link table.

        duplicate_flow_info : Dict[str, Any]
            Duplicate-flow diagnostics.

        Returns
        -------
        Dict[str, Any]
            Link-table metadata.
        """

        observed_mask = link_df["volume"].notna()

        flow_key_set = set(
            zip(
                flow_df["init_node"].astype(int),
                flow_df["term_node"].astype(int),
            )
        )

        network_key_set = set(
            zip(
                network_df["init_node"].astype(int),
                network_df["term_node"].astype(int),
            )
        )

        unmatched_flow_links = sorted(flow_key_set - network_key_set)
        network_links_without_flow_record = sorted(network_key_set - flow_key_set)

        metadata: Dict[str, Any] = {
            "num_network_links": int(len(network_df)),
            "num_flow_records_after_preprocessing": int(len(flow_df)),
            "num_links": int(len(link_df)),
            "num_links_with_flow_record": int(link_df["has_flow_record"].sum())
            if "has_flow_record" in link_df.columns else None,
            "num_links_without_flow_record": int((~link_df["has_flow_record"]).sum())
            if "has_flow_record" in link_df.columns else None,
            "num_observed_volumes": int(observed_mask.sum()),
            "num_missing_volumes": int((~observed_mask).sum()),
            "observed_volume_share": float(observed_mask.mean()) if len(link_df) > 0 else 0.0,
            "num_unmatched_flow_links": int(len(unmatched_flow_links)),
            "unmatched_flow_links_sample": unmatched_flow_links[:20],
            "network_links_without_flow_record_sample": network_links_without_flow_record[:20],
            "duplicate_flow_info": duplicate_flow_info,
            "columns": link_df.columns.tolist(),
        }

        observed_volume = link_df.loc[observed_mask, "volume"]

        if len(observed_volume) > 0:
            metadata.update(
                {
                    "total_observed_volume": float(observed_volume.sum()),
                    "mean_observed_volume": float(observed_volume.mean()),
                    "min_observed_volume": float(observed_volume.min()),
                    "max_observed_volume": float(observed_volume.max()),
                }
            )
        else:
            metadata.update(
                {
                    "total_observed_volume": 0.0,
                    "mean_observed_volume": None,
                    "min_observed_volume": None,
                    "max_observed_volume": None,
                }
            )

        return metadata

    @staticmethod
    def _validate_unique_directed_links(
        df: pd.DataFrame,
        df_name: str,
        endpoint_columns: Tuple[str, str],
    ) -> None:
        """
        Validate that a DataFrame has at most one row per directed link.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        df_name : str
            Human-readable DataFrame name.

        endpoint_columns : Tuple[str, str]
            Source and target endpoint columns.

        Raises
        ------
        ValueError
            If duplicated directed links are found.
        """

        duplicated_mask = df.duplicated(
            subset=list(endpoint_columns),
            keep=False,
        )

        if duplicated_mask.any():
            duplicated_links = (
                df.loc[duplicated_mask, list(endpoint_columns)]
                .drop_duplicates()
                .apply(
                    lambda row: (
                        int(row[endpoint_columns[0]]),
                        int(row[endpoint_columns[1]]),
                    ),
                    axis=1,
                )
                .tolist()
            )

            raise ValueError(
                f"{df_name} contains duplicated directed links: "
                f"{duplicated_links[:20]}"
            )

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


def build_link_table(
    network_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    strict: bool = True,
    preserve_extra_flow_columns: bool = True,
    aggregate_duplicate_flows: bool = True,
    duplicate_flow_aggregation: str = "mean",
) -> LinkTableBuildResult:
    """
    Convenience function to build the canonical link table.

    Parameters
    ----------
    network_df : pd.DataFrame
        Normalized network table produced by tntp_network_reader.py.

    flow_df : pd.DataFrame
        Normalized flow table produced by tntp_flow_reader.py.

    strict : bool, default=True
        Whether to use strict validation behavior.

    preserve_extra_flow_columns : bool, default=True
        Whether to keep non-canonical flow columns.

    aggregate_duplicate_flows : bool, default=True
        Whether duplicated flow records should be aggregated before merging.

    duplicate_flow_aggregation : str, default="mean"
        Aggregation rule for duplicated flow rows.

    Returns
    -------
    LinkTableBuildResult
        Canonical link table and metadata.
    """

    builder = LinkTableBuilder(
        strict=strict,
        preserve_extra_flow_columns=preserve_extra_flow_columns,
        aggregate_duplicate_flows=aggregate_duplicate_flows,
        duplicate_flow_aggregation=duplicate_flow_aggregation,
    )

    return builder.build(
        network_df=network_df,
        flow_df=flow_df,
    )
