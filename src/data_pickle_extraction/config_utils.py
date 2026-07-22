from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from .graph_validation_utils import (
        DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS,
        DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES,
    )
except ImportError:
    from graph_validation_utils import (  # type: ignore
        DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS,
        DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES,
    )


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return start


def _replace_legacy_scenario_name(value: Any, scenario_name: str) -> Any:
    if isinstance(value, str):
        return value.replace("${Scenario_Name}", scenario_name)
    if isinstance(value, dict):
        return {
            key: _replace_legacy_scenario_name(item, scenario_name)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_legacy_scenario_name(item, scenario_name) for item in value]
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. Expected a config.yaml file in the same directory as this script."
        )

    config = OmegaConf.load(str(config_path))
    raw_config = OmegaConf.to_container(config, resolve=False)

    if not isinstance(raw_config, dict):
        raise TypeError(
            "The YAML configuration must be loaded as a dictionary-like object. "
            f"Received: {type(raw_config).__name__}"
        )

    general_section = raw_config.get("General", {}) if isinstance(raw_config.get("General", {}), dict) else {}
    scenario_section = raw_config.get("scenario", {}) if isinstance(raw_config.get("scenario", {}), dict) else {}

    scenario_name = general_section.get("Scenario_Name") or scenario_section.get("name")
    if not scenario_name:
        raise KeyError("The configuration must define either General.Scenario_Name or scenario.name.")

    raw_config = _replace_legacy_scenario_name(raw_config, str(scenario_name))

    scenario_section = raw_config.get("scenario", {}) if isinstance(raw_config.get("scenario", {}), dict) else {}
    paths_section = raw_config.get("paths", {}) if isinstance(raw_config.get("paths", {}), dict) else {}
    legacy_paths_section = raw_config.get("Paths", {}) if isinstance(raw_config.get("Paths", {}), dict) else {}
    input_section = paths_section.get("input", {}) if isinstance(paths_section.get("input", {}), dict) else {}
    output_section = paths_section.get("output", {}) if isinstance(paths_section.get("output", {}), dict) else {}

    return {
        "scenario": {
            "name": scenario_name,
            "start_date": scenario_section.get("start_date", "2022-10-01"),
            "entries_per_line": scenario_section.get("entries_per_line", 6),
        },
        "paths": {
            "input": {
                "graph_path": input_section.get("graph_path", f"data/processed/{scenario_name}/{scenario_name}_graph.pkl"),
                "link_data_path": input_section.get("link_data_path", f"data/processed/{scenario_name}/{scenario_name}_link_data.parquet"),
                "od_matrix_path": input_section.get("od_matrix_path", f"data/processed/{scenario_name}/{scenario_name}_od_matrix.npz"),
            },
            "output": {
                "root": output_section.get("root", legacy_paths_section.get("general_saving_route", f"data/raw/{scenario_name}")),
                "info_dir": output_section.get("info_dir", legacy_paths_section.get("creation_information_filepath", f"data/raw/{scenario_name}/info")),
                "nodes_filepath": output_section.get("nodes_filepath", legacy_paths_section.get("nodes_filepath", f"data/raw/{scenario_name}/{scenario_name}.tntp")),
                "network_filepath": output_section.get("network_filepath", legacy_paths_section.get("network_filepath", f"data/raw/{scenario_name}/{scenario_name}.tntp")),
                "trips_filepath": output_section.get("trips_filepath", legacy_paths_section.get("trips_filepath", f"data/raw/{scenario_name}/{scenario_name}.tntp")),
                "routes_filepath": output_section.get("routes_filepath", legacy_paths_section.get("routes_filepath", f"data/raw/{scenario_name}/{scenario_name}.tntp")),
                "flows_filepath": output_section.get("flows_filepath", legacy_paths_section.get("flows_filepath", f"data/raw/{scenario_name}/{scenario_name}.tntp")),
                "manifest_filepath": output_section.get("manifest_filepath", legacy_paths_section.get("manifest_filepath", f"data/raw/{scenario_name}/info/reconstruction_manifest.json")),
            },
        },
        "trips": {
            "export_format": raw_config.get("trips", {}).get(
                "export_format",
                "average_day_hourly",
            ),
            "aggregation": raw_config.get("trips", {}).get(
                "aggregation",
                "mean_over_days",
            ),
            "start_date": raw_config.get("trips", {}).get(
                "start_date",
                scenario_section.get("start_date", "2022-10-01"),
            ),
            "hours_per_day": raw_config.get("trips", {}).get(
                "hours_per_day",
                24,
            ),
        },
        "routes": {
            "export": raw_config.get("routes", {}).get("export", True),
            "k_routes": raw_config.get("routes", {}).get("k_routes", 10),
            "weight": raw_config.get("routes", {}).get("weight", "free_flow_time"),
            "allow_intrazonal": raw_config.get("routes", {}).get("allow_intrazonal", False),
            "intrazonal_policy": raw_config.get("routes", {}).get("intrazonal_policy", "cycle"),
            "allow_loops": raw_config.get("routes", {}).get("allow_loops", False),
            "require_exact_k_routes": raw_config.get("routes", {}).get("require_exact_k_routes", True),
            "show_progress": raw_config.get("routes", {}).get("show_progress", True),
            "parallel": raw_config.get("routes", {}).get("parallel", False),
            "parallel_workers": raw_config.get("routes", {}).get("parallel_workers", None),
            "od_batch_size": raw_config.get("routes", {}).get("od_batch_size", 100),
        },
        "validation": {
            "reverse_attribute_symmetry": {
                "enabled": raw_config.get("validation", {}).get("reverse_attribute_symmetry", {}).get("enabled", True),
                "excluded_columns": raw_config.get("validation", {}).get("reverse_attribute_symmetry", {}).get("excluded_columns", list(DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS)),
                "excluded_prefixes": raw_config.get("validation", {}).get("reverse_attribute_symmetry", {}).get("excluded_prefixes", list(DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES)),
            },
        },
        "defaults": {
            "bpr_b": raw_config.get("defaults", {}).get("bpr_b", -1),
            "bpr_power": raw_config.get("defaults", {}).get("bpr_power", -1),
            "toll": raw_config.get("defaults", {}).get("toll", 0.0),
            "missing_reverse_link_id": raw_config.get("defaults", {}).get("missing_reverse_link_id", -1),
        },
    }


def resolve_project_path(path_value: str | Path, project_root: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if project_root is None:
        project_root = find_project_root(Path(__file__).resolve().parent)
    return (project_root / path).resolve()
