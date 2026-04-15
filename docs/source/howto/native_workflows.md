# Native Workflows

The bundled native workflows in `aiida-vasp` include:

- `vasp.v2.vasp` -- single VASP calculation with error handling
- `vasp.v2.relax` -- geometry optimization with automatic restarts
- `vasp.v2.bands` -- semilocal band-structure workflow (SCF + NSCF)
- `vasp.v2.hybrid_bands` -- hybrid-functional band-structure workflow
- `vasp.v2.nscf` -- standalone NSCF workflow
- `vasp.v2.converge` -- convergence testing workflow
- `vasp.v2.neb` -- nudged elastic band workflow
- `vasp.v2.staged_relax` -- explicitly staged multi-step relaxation

See the [workflows concept page](../concepts/workflows.md) for design principles and usage patterns.
