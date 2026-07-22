from __future__ import annotations

import logging
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.components.dataset_creation.config import DatasetConfig
from src.components.dataset_creation.orchestration import (
    dataset_creation_orchestration,
)
from src.utils.paths import get_project_root, resolve_path

"""
Synthetic dataset Creation Pipeline
====================================

This module defines the high-level pipeline for creating synthetic traffic
assignment datasets and exporting them in TNTP-compatible format.

The purpose of the dataset creation layer is to generate a complete, 
self-contained synthetic transportation datasets that can later be consumed 
by other modules or parts of the project for different purposes, such as training 
machine learning models, traffic assignment models, demand estimation models, 
flow estimation models, route analysis and optimization, and route generation.

After the dataset creation layer exports the generated dataset, the expected 
downstream workflow is:

 - Data processing: this stage builds a ready-to-use artifact for training 
 demand models, flow estimation models, or other downstream models. 

 - Model training: each model retrieves the required data from the artifact 
 according to its own objective, whether that objective is traffic assignment, 
 combined Origin-Destination Estimation and Flow Estimation, parameter 
 estimation, or another modelling task. 

 - Model testing: the testing pipeline evaluates each trained model. 
 Depending on the capabilities and features of each model, a dedicated 
 set of testing functions will be executed. 

 These functions may use outputs produced during training, such as link-wise 
 or route-wise flow estimates, demand estimates usually represented as OD pairs 
 or OD matrices, route proportion estimates, VDF or assignment parameter 
 estimates, and other model-specific outputs. 

 A major part of testing will also involve creating visual representations of 
 the network and its results. This functionality is still pending. # TODO

The pipeline is intentionally configuration-driven. Every dataset must be
reproducible from its YAML configuration file, random seed, and explicitly
declared generation parameters. The Python code should define the mechanics of
dataset creation, while the YAML file should define the dataset-specific
choices: spatial morphology, node types, link hierarchy, demand scale, demand
profile, VDF catalogue, assignment settings, export paths, and optional reports.

Core responsibilities
---------------------

This module is responsible for orchestrating the creation of the following
dataset components:

1. Nodes
   Generate synthetic network nodes with explicit identifiers, coordinates,
   classes, and types. Nodes may represent demand zones, auxiliary connectors,
   intersections, or future morphology-specific categories.

2. Network links
   Generate directed links between nodes, including physical and operational
   attributes such as capacity, lanes, length, free-flow time, speed, toll,
   link type, VDF identifier, and reverse-link relationships.

3. Demand matrices
   Generate zone-to-zone OD demand matrices either by using compact zone-space
   representations or full node-by-node matrices. Demand generation
   should support temporal profiles, stochastic noise, intrazonal policy, and
   future morphology-aware demand patterns.

4. Routes
   Produce route sets for OD pairs by delegating route computation to the
   project route generation infrastructure. The dataset creation pipeline
   should request routes and validate their compatibility with the generated
   network, but it should not become a specialized routing engine itself.

5. Traffic assignment and link flows
   Produce link flows by delegating assignment to the assignment_motors package.
   The dataset creation pipeline should prepare the OD matrix, links, routes,
   and assignment configuration, then store the resulting flows and link costs.

6. VDF-based impedance
   Link travel times and congestion effects must be computed through the VDF
   system. The dataset creation layer should attach VDF identifiers and
   parameters to links, but the implementation of volume-delay functions should
   remain delegated to the dedicated VDF module.

7. TNTP exports
   Export the canonical TNTP files required by downstream workflows:
   nodes, network, trips, routes, and flows.

8. dataset information artifacts
   Export an `info/` directory containing metadata, manifests, debug tables,
   visualizations, control reports, and reproducibility artifacts. These
   outputs should be modular and optional, so future datasets can enable,
   disable, replace, or extend individual reports without modifying the core
   generation logic.

Design philosophy
-----------------

The dataset creation pipeline should behave as an orchestrator, not as a
monolithic implementation of every subproblem.

Its role is to coordinate synthetic network generation, demand generation,
route computation, traffic assignment, VDF-based cost computation, validation,
and export. Specialized tasks should be delegated to specialized modules:

- Route generation should be delegated to route builders and route engines.
- Traffic assignment should be delegated to assignment_motors.
- VDF computation should be delegated to the VDF package.
- Reports, plots, tables, manifests, and artifacts should be delegated to
  dataset information/export modules.
- Synthetic morphology generation should gradually move into dedicated
  morphology generators.

This separation is important because dataset creation is expected to evolve
from simple random networks into more realistic synthetic transport systems.
Future generators should support hierarchical and morphology-aware networks,
for example:

- continuous highway or primary corridors crossing the study area;
- secondary arterials branching from high-capacity corridors;
- local street grids with shorter spacing and lower speeds;
- connectors between demand zones and the physical road network;
- spatially heterogeneous land-use or activity patterns;
- OD demand magnitudes that depend on zone importance, accessibility,
  network hierarchy, distance, and morphology.

In other words, the synthetic dataset should not only be a connected graph. It
should increasingly represent a plausible transportation system where supply
and demand are coherent with each other.

Expected pipeline structure
---------------------------

The intended conceptual flow is* :

    YAML dataset configuration
        -> synthetic node generation
        -> synthetic network morphology generation
        -> link attribute and VDF assignment
        -> OD demand generation
        -> route set generation
        -> traffic assignment
        -> TNTP export
        -> info artifact export
        -> master dataset artifact and manifest

*This flow is subject to change if future attempts to build more realistic, 
morphology-aware, and demand-aware datasets show that the sequence should be 
modified.

The core pipeline should remain stable even if individual generators,
assignment methods, route engines, or reports are replaced.

Configuration contract
----------------------

A dataset YAML file should be the single source of truth for dataset-specific
settings. Hard-coded paths, constants, default parameters, fallback parameters,
generation rules, assignment parameters, or report settings should be avoided 
unless they are universal defaults.

The dataset configuration should define, at minimum*:

- dataset identity and random seed;
- output paths;
- node classes, node types, and spatial bounds;
- network morphology parameters;
- link type catalogues;
- VDF catalogues or references to VDF configurations;
- demand generation parameters;
- route generation parameters;
- assignment parameters;
- reporting and artifact-export options.

Each TNTP file must satisfy a strict contract with the minimum columns 
expected for that file. If the contract is not met, the pipeline should raise 
an explicit error with an actionable message.

Fail-fast principle
-------------------

The pipeline should avoid silent failures. If required configuration keys,
columns, route sets, graph attributes, OD mappings, VDF parameters, or export
paths are missing or inconsistent, the pipeline should raise explicit errors
with actionable messages.

Silent fallbacks should only be allowed when they are explicitly configured and
reported in the dataset manifest.

Reproducibility principle
-------------------------

Every generated dataset should be reproducible. The exported manifest and
master artifact should record:

- dataset name;
- creation timestamp;
- full resolved configuration;
- random seed;
- package/runtime versions;
- output file paths;
- node, link, demand, route, and flow metadata;
- enabled reports and diagnostics;
- warnings, fallbacks, and validation results.

Relationship with downstream modules
------------------------------------

The TNTP files and dataset metadata produced here will be used by most parts 
of the project. However, the dataset creation pipeline should not be responsible 
for implementing or maintaining the contracts of downstream modules. Those 
contracts should be maintained by the downstream modules themselves, while the 
dataset creation pipeline should simply produce files that are compatible with 
its own TNTP contracts.

Nevertheless, this layer should document and explain the meaning of all 
metadata, as well as the contents and meaning of each TNTP file,
so that downstream modules can understand and use them correctly.

Refactoring direction
---------------------

This file should gradually become a thin orchestration layer. The following
responsibilities should be moved into separate modules as the dataset creation
system grows:

- configuration loading and validation;
- node generation;
- morphology-aware network generation;
- link attribute generation;
- demand generation;
- TNTP export;
- route generation adapters;
- assignment adapters;
- dataset artifact writing;
- manifest creation;
- plots and visual reports;
- assignment control reports;
- dataset validation.

The long-term goal is to make dataset creation extensible: adding a new
network morphology, demand model, report, route engine, VDF family, or
assignment method should not require rewriting the main pipeline.
"""


cs = ConfigStore.instance()
cs.store(name="base_config", node=DatasetConfig)


def run_pipeline(config: DictConfig):
    logging.info("Starting dataset creation pipeline...")

    schema = OmegaConf.structured(DatasetConfig)
    resolved_config = OmegaConf.merge(schema, config)

    typed_config = OmegaConf.to_object(resolved_config)
    
    if not isinstance(typed_config, DatasetConfig):
        raise TypeError(
            "Resolved dataset creation config is not a datasetCreationConfig."
        )

    dataset_name = typed_config.name

    output_dir = resolve_path(
        typed_config.paths.export_dirs.generated_dataset,
        relative_to=typed_config.paths.root_dir or get_project_root(),
    )

    if output_dir.exists() and not typed_config.overwrite_existing:
        raise FileExistsError(
            f"""Output directory '{output_dir}' already exists.
            Set 'overwrite_existing' to True in the configuration to overwrite it."""
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Loaded dataset configuration for '%s' successfully.",
        dataset_name,
    )

    dataset_creation_orchestration(typed_config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_pipeline(config)

if __name__ == "__main__":
    main()
