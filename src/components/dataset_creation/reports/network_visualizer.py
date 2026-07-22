from __future__ import annotations

from typing import Any

from .._scenario_creation_utils import _network_previsualization


def export_network_previsualization(data: dict[str, Any], config) -> dict[str, Any]:
    analyzer_stub = type(
        "AnalyzerStub",
        (),
        {"scenario": type("ScenarioStub", (), {"config": config, "nodes_df": data["nodes"]})()},
    )()
    return _network_previsualization(analyzer_stub, data["network"], {"graph": data["graph"]})
