---
description: Manage uv-based Python environments in a workspace. Use when Python or packages are missing, when a uv venv is absent, or when you need to initialize or sync dependencies with uv and run Python via uv.
applyTo: '**/*.py'
---

# UV Python Environment

## Core workflow

- Work from the workspace root.
- If no `pyproject.toml` exists, initialize the project with `uv init .`.
- If `pyproject.toml` exists and dependencies are defined, create or repair the env with `uv sync --no-cache --upgrade`.
- Use `uv add <pkg>` for runtime dependencies and `uv add --dev <pkg>` for dev tools (e.g., `ipython`, `rich`, `see`).

## Activate and run

- Activate the venv with `source /path/to/workspace/.venv/bin/activate`.
- Prefer `uv run python <file.py>` to run scripts.
- If a Python snippet is needed, write it to a `.py` file first and then run it. Avoid `uv run python -c "<multiline>"`.
- When writing a helper script, avoid `argparse` CLI scaffolding. Prefer small, local, low-coupling functions that take arguments, and call them explicitly under `if __name__ == "__main__":` without a `main()` function when possible.

## Troubleshooting

- If Python is not found, activate the venv or run with `uv run`.
- If a package is missing, add it with `uv add` (or `uv add --dev` for dev-only tools), then retry.
