# Repository Guidelines

## Project Structure & Module Organization
`src/aiida_vasp/` contains the plugin code in a standard `src` layout. Core areas are `calcs/` for calculation plugins, `parsers/` for VASP output parsing, `workchains/v2/` for workflow logic, `commands/` for CLI entry points, and `data/`, `protocols/`, `inputset/`, `utils/`, and `common/` for shared models and helpers. Tests live under `tests/`, with large regression fixtures in `tests/test_data/`. User-facing documentation is in `docs/source/`; `examples/` holds notebooks and sample exports.

## Build, Test, and Development Commands
Activate the workspace environment first with `source ../.venv/bin/activate`. Prefer `uv` for local setup and execution, for example `uv pip install -e .[tests,pre-commit,docs]`. Run the full suite with `uv run pytest` or `uv run pytest tests`. Target an area while iterating, for example `uv run pytest tests/parsers -q` or `uv run pytest tests/workchains/v2 -q`. Run local quality checks with `uv run pre-commit run --all-files`; this applies Ruff linting/formatting and basic file hygiene checks. Build docs with `uv run sphinx-build docs/source docs/_build`, or use `uv run sphinx-autobuild docs/source docs/_build` for live previews.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints where practical, and concise docstrings for public classes and functions. The codebase follows Ruff with a `120` character line length and single quotes in formatted output. Module names are lowercase; test files use `test_*.py`; AiiDA entry-point classes are typically `CamelCase` such as `VaspWorkChain`, while functions and helpers use `snake_case`.

## Testing Guidelines
This project uses `pytest`; coverage is tracked in CI with `pytest-cov`, but no explicit minimum is declared in the repository. Add or update tests with every behavior change, especially for parsers, workchains, and CLI commands. Reuse fixtures from `tests/conftest.py` and `tests/test_data/` instead of creating ad hoc sample files. For workchain development, prefer the bundled `mock-vasp` flow and only refresh registry data intentionally.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `Added option to bypass residual force check (#836)` and dependency updates like `Bump actions/upload-artifact from 5 to 6`. Keep the first line specific and scoped; add a body when context is needed. Open PRs against `develop`, link related issues in the template, mark whether the PR is ready or WIP, and describe the change clearly. Run `uv run pytest` and `uv run pre-commit run --all-files` before requesting review.
# Claude Instructions for aiida-vasp

## Testing Commands

All commands for testing should be prefixed with `source .venv/bin/activate`. For example:

```bash
source .venv/bin/activate && uv run pytest tests/workchains/v2 -q
```

Or use the uv run command:

```bash
uv run pytest tests/workchains/v2 -q
```

This ensures the correct virtual environment is activated before running commands.
