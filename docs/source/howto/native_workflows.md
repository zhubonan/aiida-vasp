# Native Composite Workflows

This page documents the native composite workflows recently added to `aiida-vasp`.
These workflows are implemented directly in `aiida-vasp` using the existing `v2`
workchains as building blocks. They do not depend on `atomate2` at runtime.

## Double relaxation

The `VaspDoubleRelaxWorkChain` performs two back-to-back
relax-like `VaspWorkChain` calculations.
The shared `relax` branch still uses the `VaspRelaxWorkChain` protocol inputs to
define the relax settings and baseline INCAR/k-point defaults, but the runtime
execution is a plain two-stage `vasp.v2.vasp -> vasp.v2.vasp` chain with
`restart_folder` passed from stage 1 to stage 2.
This is useful when the first relaxation changes the lattice enough that a second
relaxation with refreshed basis and k-point settings is desirable.

Entry point:

```python
from aiida.plugins import WorkflowFactory

wc = WorkflowFactory('vasp.v2.double_relax')
```

Builder from protocol:

```python
builder = wc.get_builder_from_protocol(
    code=code,
    structure=structure,
)
```

Optional stage-local overrides can be supplied using the normalized stage namespaces:

- `stage_1.parameters`
- `stage_1.settings`
- `stage_1.options`
- `stage_1.relax_settings`
- `stage_2.parameters`
- `stage_2.settings`
- `stage_2.options`
- `stage_2.relax_settings`

The legacy flattened `stage_1_*` and `stage_2_*` ports are still accepted as
compatibility aliases.

For example, to tighten the second stage:

```python
from aiida import orm

builder.stage_2.relax_settings = orm.Dict(
    dict={
        'force_cutoff': 0.02,
        'convergence_volume': 0.005,
    }
)
```

## Relaxation plus band structure

The `VaspRelaxBandsWorkChain` makes the `relax -> scf -> bands/dos` sequence explicit.
It first runs a `VaspRelaxWorkChain` and then launches a `VaspBandsWorkChain`
from the relaxed structure.

Entry point:

```python
from aiida.plugins import WorkflowFactory

wc = WorkflowFactory('vasp.v2.relax_bands')
```

Builder from protocol:

```python
builder = wc.get_builder_from_protocol(
    code=code,
    structure=structure,
    relax_protocol='balanced',
    band_protocol='balanced',
)
```

The builder exposes:

- `relax`: inputs for `VaspRelaxWorkChain`
- `bands`: inputs for `VaspBandsWorkChain` without its nested `relax` namespace

Inside the nested `bands` namespace, the normalized interface is:

- `bands.nscf`: reusable SCF/NSCF execution inputs
- `bands.band_settings`: top-level path-generation settings for the nested bands workflow
- `bands.bs_kpoints`: optional explicit path kpoints

This is useful when the relaxation and the follow-up band-structure calculation
should be configured independently but submitted as one provenance-preserving workflow.

## Materials Project style workflows

The following entry points implement native `aiida-vasp` workflows that reproduce
the workflow logic extracted from the corresponding `atomate2` VASP flows:

| Entry point | Native workchain | Workflow logic |
| --- | --- | --- |
| `vasp.v2.mp_gga_double_relax` | `VaspMPGGADoubleRelaxWorkChain` | `MPRelaxSet` relax, then `MPRelaxSet` relax |
| `vasp.v2.mp_gga_relax_static` | `VaspMPGGARelaxStaticWorkChain` | MP GGA double relax, then `MPStaticSet` static |
| `vasp.v2.mp_meta_gga_double_relax` | `VaspMPMetaGGADoubleRelaxWorkChain` | pre-relax from `MPScanRelaxSet` with `GGA=PS` and `METAGGA=None`, then r2SCAN relax |
| `vasp.v2.mp_meta_gga_relax_static` | `VaspMPMetaGGARelaxStaticWorkChain` | MP meta-GGA double relax, then `MPScanStaticSet` static |
| `vasp.v2.mp24_double_relax` | `VaspMP24DoubleRelaxWorkChain` | `MP24RelaxSet(xc_functional='PBEsol')`, then `MP24RelaxSet(xc_functional='r2SCAN')` |
| `vasp.v2.mp24_relax_static` | `VaspMP24RelaxStaticWorkChain` | MP24 double relax, then `MP24StaticSet(xc_functional='r2SCAN')` |
| `vasp.v2.matpes_static` | `MatPesStaticWorkChain` | PBE static, then r2SCAN static with WAVECAR reuse |

All seven workflows are launched in the same way:

```python
from aiida.plugins import WorkflowFactory

wc = WorkflowFactory('vasp.v2.mp_gga_relax_static')
builder = wc.get_builder_from_protocol(code=code, structure=structure)
```

### Extracted input changes

The MP workflows use the existing pymatgen adaptor path inside `aiida-vasp`
to materialize stage-specific VASP inputs directly from the corresponding
pymatgen VASP sets.

The native workflows therefore capture both:

- the workflow topology from `atomate2`
- the important stage-specific input changes from the underlying input sets

The key stage-specific changes are:

- `mp_gga_*`:
  two MP GGA relaxations using `MPRelaxSet`, with the `relax_static` workflow
  finishing with `MPStaticSet`
- `mp_meta_gga_*`:
  a pre-relax using `MPScanRelaxSet` with `EDIFFG=-0.05`, `GGA=PS`,
  `LWAVE=True`, `LCHARG=True`, `LELF=False`, and `METAGGA=None`,
  followed by an r2SCAN relaxation and optional `MPScanStaticSet` static
- `mp24_*`:
  a first `MP24RelaxSet` stage with `xc_functional='PBEsol'`, then a second
  `MP24RelaxSet` stage with `xc_functional='r2SCAN'`, and optionally a final
  `MP24StaticSet` stage with `LELF=True` and `KPAR=1`

## MatPES static flow

The `MatPesStaticWorkChain` implements the MatPES static calculation flow,
which runs a PBE static calculation followed by an r2SCAN static calculation.
The WAVECAR from the PBE calculation is reused to accelerate convergence
in the r2SCAN calculation.

Entry point:

```python
from aiida.plugins import WorkflowFactory

wc = WorkflowFactory('vasp.v2.matpes_static')
```

Builder from protocol:

```python
builder = wc.get_builder_from_protocol(
    code=code,
    structure=structure,
)
```

The builder exposes:
- `static1`: inputs for the first PBE static calculation
- `static2`: inputs for the second r2SCAN static calculation

The workflow is designed for generating accurate potential energy surface
data where force and stress accuracy are paramount.

## Input generators

Matching `InputGenerator` classes are available for the new workflows:

- `VaspNscfInputGenerator`
- `VaspDoubleRelaxInputGenerator`
- `VaspRelaxBandsInputGenerator`
- `VaspMPGGADoubleRelaxInputGenerator`
- `VaspMPGGARelaxStaticInputGenerator`
- `VaspMPMetaGGADoubleRelaxInputGenerator`
- `VaspMPMetaGGARelaxStaticInputGenerator`
- `VaspMP24DoubleRelaxInputGenerator`
- `VaspMP24RelaxStaticInputGenerator`
- `VaspMatPesStaticInputGenerator`

These live in `aiida_vasp.protocols.generator`.

The canonical generator entry point is now `build(...)`. The older
`get_builder(...)` method is kept as a compatibility alias while the generator
API moves toward typed child accessors for composite workflows.

The generators are also self-documenting. After calling `build(...)`, inspect
the generator itself before applying updates:

```python
print(upd)
upd.describe()
```

This will show:

- the canonical top-level ports for the workflow
- which child accessors to use
- short notes about workflow-specific conventions
- small usage examples for the branch layout

For namespace-specific discovery, inspect the child view directly:

```python
print(upd.scf())
print(upd.reuse())
```

For example, the NSCF-focused generator can be configured through explicit
sub-workflow views:

```python
from aiida_vasp.protocols.generator import VaspNscfInputGenerator

upd = VaspNscfInputGenerator()
upd.build(structure=structure, code='my-vasp@cluster')
upd.scf().set_incar(ismear=0)
upd.bands().set_settings(parser_settings={'include_node': ['bands']})
upd.reuse().use_restart_folder(restart_folder)
upd.dos().enable(distance=0.04)
builder = upd.builder
```

Likewise, composite workflows such as `VaspRelaxBandsInputGenerator` expose
separate `relax()` and `bands()` views so that stage-local configuration does
not rely on recursive updates across unrelated namespaces.

This is especially useful for distinguishing semilocal and hybrid band
workflows:

- `VaspBandsInputGenerator` advertises `relax()` and `nscf()`
- `VaspHybridBandsInputGenerator` advertises `relax()` and `scf()`

The printed schema is the canonical guide to which accessors are supported by a
given generator.
