# Claude Instructions for aiida-vasp

## Testing Commands

All commands for testing should be prefixed with `source .venv/bin/activate`. For example:

```bash
source .venv/bin/activate && verdi run run_matpes_static_si.py
```

Or use the uv run command:

```bash
uv run verdi run run_matpes_static_si.py
```

This ensures the correct virtual environment is activated before running commands.
