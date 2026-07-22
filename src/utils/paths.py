"""
# Central path resolution utilities.

This module is the single source of truth for resolving paths within the
project. All path-related logic must go through this module: reading files,
saving artifacts, building directories, resolving relative paths,
normalizing absolute paths, and expanding values such as `~` or environment
variables.

## Why it exists

The project can be executed from different contexts:

* from the repository root;
* from subdirectories such as `src/train/`;
* through Hydra;
* from CI/CD runners;
* from notebooks or auxiliary scripts.

Relying on the current working directory (`cwd`) or manually concatenating
paths makes them fragile and prone to breaking. This module avoids that
problem by centralizing path resolution and enforcing a consistent policy
throughout the codebase.

## How to use it

* Use `get_project_root()` whenever you need the stable root of the repository.
* Use `resolve_path()` to convert a relative or absolute path into a final,
  normalized `Path`.
* Pass `Path` objects to internal functions whenever possible.
* If a configuration value may be provided as a string, resolve it here before
  passing it to loaders, writers, or builders.

## Usage guidelines

* Do not duplicate the logic of `os.path.join`, `Path.cwd()`, or
  `Path(...).resolve()` in other modules if the problem is path resolution.
* Do not assume that Hydra or the calling script is being executed from the
  repository root.
* Do not mix strings and resolved paths longer than necessary: convert early
  and work with `Path` objects.

## Practical contract

* If a path is already absolute, it is preserved as is.
* If a path is relative, it is resolved against the project root or the
  explicit base specified by `relative_to`.
* If a path comes from configuration, this module must be the first point of
  normalization before opening, saving, or passing the path to another layer.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> Path:
    """
    Busca la raíz real del proyecto de forma dinámica ascendente.
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Busca un archivo ancla inequívoco del proyecto
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (
            parent / "requirements.txt"
        ).exists():
            return parent

    return Path.cwd()


ROOT_DIR = get_project_root()


def resolve_path(
    target_path: str | Path,
    relative_to: Path | None = None,
) -> Path:
    """
    Resuelve cualquier ruta. Si es absoluta la respeta, si es relativa
    la acopla a la base indicada (por defecto la raíz del proyecto).
    """
    path_obj = Path(target_path).expanduser()

    if relative_to is None:
        relative_to = get_project_root()

    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(relative_to).expanduser() / path_obj).resolve()
