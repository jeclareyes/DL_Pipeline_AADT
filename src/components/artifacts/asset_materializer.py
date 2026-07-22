"""Delegate asset construction to specialized builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .assignment_set_builder import AssignmentSetBuilder, AssignmentSetBuildResult
from .config_schemas import AssignmentSetSpecConfig, RouteSetSpecConfig, RouteSetRequirementConfig
from .route_set_builder import RouteSetBuilder, RouteSetBuildResult


class AssetMaterializer:
    """Materialize assets from a base artifact."""

    def __init__(self, base_artifact: Mapping[str, Any], output_root: str | Path):
        self.base_artifact = dict(base_artifact)
        self.output_root = Path(output_root)
        self.route_set_builder = RouteSetBuilder(self.base_artifact, self.output_root)
        self.assignment_set_builder = AssignmentSetBuilder(self.base_artifact, self.output_root)

    def build_route_set(
        self,
        spec: RouteSetSpecConfig,
        requirement: RouteSetRequirementConfig | None = None,
    ) -> RouteSetBuildResult:
        return self.route_set_builder.build(spec=spec, requirement=requirement)

    def build_assignment_set(
        self,
        spec: AssignmentSetSpecConfig,
        route_set_entry: Mapping[str, Any],
    ) -> AssignmentSetBuildResult:
        return self.assignment_set_builder.build(spec=spec, route_set_entry=route_set_entry)

