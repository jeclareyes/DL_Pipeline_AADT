"""Dataset-driven node classification and OD-zone validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class NodeClassificationResult:
    """Validated semantic classification of the dataset's nodes."""

    zone_ids: list[int]
    non_zone_ids: list[int]
    normalized_class_by_node: dict[int, str]
    normalized_type_by_node: dict[int, str]


def validate_and_classify_nodes(
    node_df: pd.DataFrame,
    classification_cfg: Mapping[str, Any],
) -> NodeClassificationResult:
    """Validate class/type congruence and derive the OD zone space."""

    required_columns = {"node_id", "class", "type"}
    missing_columns = required_columns.difference(node_df.columns)
    if missing_columns:
        raise ValueError(
            "Node classification requires columns: "
            f"{sorted(required_columns)}; missing={sorted(missing_columns)}."
        )

    rules = classification_cfg.get("rules")
    normalization = classification_cfg.get("normalization", {})
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)) or not rules:
        raise ValueError("node_classification.rules must be a non-empty sequence.")
    if not isinstance(normalization, Mapping):
        raise TypeError("node_classification.normalization must be a mapping.")

    case_sensitive = bool(normalization.get("case_sensitive", False))
    strip_whitespace = bool(normalization.get("strip_whitespace", True))

    def normalize(value: Any) -> str:
        text = str(value)
        if strip_whitespace:
            text = "".join(text.split())
        return text if case_sensitive else text.casefold()

    normalized_rules: list[tuple[set[str], set[str], bool]] = []
    type_to_rule: dict[str, int] = {}
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise TypeError(f"node_classification.rules[{rule_index}] must be a mapping.")
        class_values = rule.get("class_values")
        type_values = rule.get("type_values")
        include_in_od = rule.get("include_in_od")
        if not isinstance(class_values, Sequence) or not class_values:
            raise ValueError(f"Rule {rule_index} must define class_values.")
        if not isinstance(type_values, Sequence) or not type_values:
            raise ValueError(f"Rule {rule_index} must define type_values.")
        if not isinstance(include_in_od, bool):
            raise TypeError(f"Rule {rule_index}.include_in_od must be boolean.")

        normalized_types = {normalize(value) for value in type_values}
        for node_type in normalized_types:
            previous_rule = type_to_rule.setdefault(node_type, rule_index)
            if previous_rule != rule_index:
                raise ValueError(
                    f"Node type {node_type!r} is assigned to multiple classes: "
                    f"rules {previous_rule} and {rule_index}."
                )

        normalized_rules.append(
            (
                {normalize(value) for value in class_values},
                normalized_types,
                include_in_od,
            )
        )

    zone_ids: list[int] = []
    non_zone_ids: list[int] = []
    normalized_class_by_node: dict[int, str] = {}
    normalized_type_by_node: dict[int, str] = {}

    for row in node_df[["node_id", "class", "type"]].itertuples(index=False):
        node_id = int(row.node_id)
        node_class = normalize(row[1])
        node_type = normalize(row[2])
        matching_rules = [
            rule
            for rule in normalized_rules
            if node_class in rule[0] and node_type in rule[1]
        ]
        if len(matching_rules) != 1:
            raise ValueError(
                "Node class/type combination is invalid or ambiguous: "
                f"node_id={node_id}, class={row[1]!r}, type={row[2]!r}, "
                f"matching_rules={len(matching_rules)}."
            )

        (zone_ids if matching_rules[0][2] else non_zone_ids).append(node_id)
        normalized_class_by_node[node_id] = node_class
        normalized_type_by_node[node_id] = node_type

    if not zone_ids:
        raise ValueError("Node classification produced an empty OD zone space.")

    return NodeClassificationResult(
        zone_ids=zone_ids,
        non_zone_ids=non_zone_ids,
        normalized_class_by_node=normalized_class_by_node,
        normalized_type_by_node=normalized_type_by_node,
    )
