# ATOMATE2 Workflow Implementation Plan

## Goal

Implement the VASP workflow logic from `atomate2` natively in `aiida-vasp` using AiiDA `WorkChain`s.

This document tracks:

- which `atomate2` workflows have native `aiida-vasp` equivalents
- which existing `aiida-vasp` workflows already cover `atomate2` functionality
- which new `WorkChain`s still need to be implemented
- which helper utilities and protocol layers are still missing

## Design Principles

- Do not use `atomate2` as a runtime dependency.
- Treat `atomate2` as a specification source for:
  - workflow topology
  - stage-specific input changes
  - restart and file-reuse semantics
- Reuse existing `aiida-vasp` v2 workchains wherever possible:
  - `vasp.v2.vasp`
  - `vasp.v2.relax`
  - `vasp.v2.bands`
  - `vasp.v2.hybrid_bands`
  - `vasp.v2.neb`
  - `vasp.v2.converge`
- Implement new top-level orchestration as native `WorkChain`s using:
  - `define`
  - `outline`
  - `submit`
  - `ToContext`
  - namespaced child inputs and outputs
- Encode workflow presets and stage deltas through protocols and builder overrides, not through `atomate2` makers.

## Translation Model

The implementation mapping is:

- `atomate2` single-job makers
  -> existing `aiida-vasp` base workchains or direct protocolized uses of them
- `atomate2` flow makers
  -> new top-level orchestration `WorkChain`s
- `atomate2` helper jobs that generate structures, strains, displacements, or perform collation
  -> `calcfunction`s or small helper utilities in `aiida-vasp`

Common examples:

- `DoubleRelaxMaker`
  -> `VaspDoubleRelaxWorkChain`
  -> runs `vasp.v2.relax` twice
- `RelaxBandStructureMaker`
  -> `VaspRelaxBandsWorkChain`
  -> runs `vasp.v2.relax` then `vasp.v2.bands`
- MP staged workflows
  -> dedicated MP-native `WorkChain`s that inject stage-local parameter deltas into reused child workchains

## Implemented Native WorkChains

These workflows have already been added natively:

- `atomate2` `DoubleRelaxMaker`
  -> `vasp.v2.double_relax`
  -> implemented in `src/aiida_vasp/workchains/v2/core_flows.py`
- `atomate2` `RelaxBandStructureMaker`
  -> `vasp.v2.relax_bands`
  -> implemented in `src/aiida_vasp/workchains/v2/core_flows.py`
- `atomate2` `MPGGADoubleRelaxMaker`
  -> `vasp.v2.mp_gga_double_relax`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`
- `atomate2` `MPGGADoubleRelaxStaticMaker`
  -> `vasp.v2.mp_gga_relax_static`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`
- `atomate2` `MPMetaGGADoubleRelaxMaker`
  -> `vasp.v2.mp_meta_gga_double_relax`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`
- `atomate2` `MPMetaGGADoubleRelaxStaticMaker`
  -> `vasp.v2.mp_meta_gga_relax_static`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`
- `atomate2` `MP24DoubleRelaxMaker`
  -> `vasp.v2.mp24_double_relax`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`
- `atomate2` `MP24DoubleRelaxStaticMaker`
  -> `vasp.v2.mp24_relax_static`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`
- `atomate2` `MatPesStaticFlowMaker`
  -> `vasp.v2.matpes_static`
  -> implemented in `src/aiida_vasp/workchains/v2/mp.py`

## Existing aiida-vasp Coverage

These `atomate2` workflows are already substantially covered by existing `aiida-vasp` workchains and need protocol or interface work more than new topology:

- `StaticMaker`
  -> `vasp.v2.vasp`
- `RelaxMaker`
  -> `vasp.v2.relax`
- `TightRelaxMaker`
  -> `vasp.v2.relax` with tighter protocol
- `BandStructureMaker`
  -> `vasp.v2.bands`
- `UniformBandStructureMaker`
  -> `vasp.v2.bands` with uniform-only mode
- `LineModeBandStructureMaker`
  -> `vasp.v2.bands` with line-only mode
- `HSEBandStructureMaker`
  -> `vasp.v2.hybrid_bands`
- `HSEUniformBandStructureMaker`
  -> `vasp.v2.hybrid_bands` with uniform-only mode
- `HSELineModeBandStructureMaker`
  -> `vasp.v2.hybrid_bands` with line-only mode
- `NebFromImagesMaker` and `NebFromEndpointsMaker`
  -> `vasp.v2.neb`
- convergence workflows
  -> `vasp.v2.converge`

## Next Native WorkChains To Implement

### High Priority

- `atomate2` `OpticsMaker`
  -> target `vasp.v2.optics`
  -> reuse `vasp.v2.vasp`
  -> topology:
    `scf/static -> optics`
- `atomate2` `HSEOpticsMaker`
  -> target `vasp.v2.hse_optics` or a hybrid protocol mode of `vasp.v2.optics`
  -> reuse `vasp.v2.vasp`
- `atomate2` `EosMaker`
  -> target `vasp.v2.eos`
  -> reuse `vasp.v2.relax` + `vasp.v2.vasp`
  -> topology:
    optional equilibrium relax -> deformed relax/static branches -> EOS fit
- `atomate2` `ElasticMaker`
  -> target `vasp.v2.elastic`
  -> reuse `vasp.v2.relax` + `vasp.v2.vasp`
  -> topology:
    optional tight bulk relax -> deformation set -> relax/static per strain -> tensor fit
- `atomate2` `PhononMaker`
  -> target `vasp.v2.phonon`
  -> reuse `vasp.v2.relax` + `vasp.v2.vasp`
  -> topology:
    optional tight relax -> displacement supercells -> force calculations -> phonon collation

### Medium Priority

- `atomate2` `QhaMaker`
  -> target `vasp.v2.qha`
  -> reuse future `vasp.v2.eos` + `vasp.v2.phonon`
- `atomate2` `GruneisenMaker`
  -> target `vasp.v2.gruneisen`
  -> reuse future `vasp.v2.phonon`
- `atomate2` `MultiMDMaker`
  -> target `vasp.v2.multi_md`
  -> reuse `vasp.v2.vasp` or a dedicated MD child workflow
- `atomate2` `MPMorphVaspMDMaker`
  -> target `vasp.v2.mpmorph_md`
  -> reuse future MD workflows plus relax/static components

## Deferred Specialized Workflows

These should be implemented only after the core native stack is stable:

- `VaspLobsterMaker`
  -> future `vasp.v2.lobster`
- `MPVaspLobsterMaker`
  -> future MP protocol variant of `vasp.v2.lobster`
- `VaspAmsetMaker`
  -> future `vasp.v2.amset`
- `HSEVaspAmsetMaker`
  -> future HSE protocol variant
- `ElectronPhononMaker`
  -> future `vasp.v2.elph`
- `HSEElectronPhononMaker`
  -> future HSE variant
- `ApproxNebMaker`
  -> future `vasp.v2.approx_neb`
- `FerroelectricMaker`
  -> future `vasp.v2.ferroelectric`
- `AdsorptionMaker`
  -> future `vasp.v2.adsorption`
- defect workflows:
  - `FormationEnergyMaker`
  - `ConfigurationCoordinateMaker`
  - `NonRadiativeMaker`
- `ElectrodeInsertionMaker`
  -> future `vasp.v2.electrode_insertion`
- `MVLGWBandStructureMaker`
  -> future `vasp.v2.mvl_gw_bands`

## Missing Cross-Cutting Helper Pieces

The following infrastructure is still needed for the remaining workflow families:

- structure standardization helpers
  - primitive cell conversion
  - conventional cell conversion
- deformation generation helpers
  - EOS strain sets
  - elastic strain sets
- displacement generation helpers
  - phonon supercells
  - finite-displacement patterns
- aggregation helpers
  - EOS fitting
  - elastic tensor fitting
  - phonon and thermal-property collation
  - QHA aggregation
  - Gruneisen aggregation
- protocol families
  - `tight`
  - `hse`
  - `mp-gga`
  - `mp-meta-gga`
  - `mp24`
  - `eos`
  - `phonon`
  - `elastic`
  - `optics`

## Testing Status

Current tests are builder-heavy and entry-point-heavy.

Covered now:

- entry-point registration for the new native workflows
- `get_builder_from_protocol` for:
  - `double_relax`
  - `relax_bands`
  - MP double-relax flows
  - MP relax+static flows

Still missing:

- execution-level orchestration tests
- restart-folder propagation tests
- failure-path tests
- child-output exposure tests
- stage-local override execution tests

## Recommended Rollout Order

1. `vasp.v2.optics`
2. `vasp.v2.eos`
3. `vasp.v2.elastic`
4. `vasp.v2.phonon`
5. `vasp.v2.qha`
6. `vasp.v2.gruneisen`
7. specialized ecosystems after the above are stable

## Summary Matrix

| atomate2 workflow | aiida-vasp target | Reused base workchains | Status |
| --- | --- | --- | --- |
| `DoubleRelaxMaker` | `vasp.v2.double_relax` | `vasp.v2.relax` | Implemented |
| `RelaxBandStructureMaker` | `vasp.v2.relax_bands` | `vasp.v2.relax`, `vasp.v2.bands` | Implemented |
| `MPGGADoubleRelaxMaker` | `vasp.v2.mp_gga_double_relax` | `vasp.v2.relax` | Implemented |
| `MPGGADoubleRelaxStaticMaker` | `vasp.v2.mp_gga_relax_static` | `vasp.v2.double_relax`, `vasp.v2.vasp` | Implemented |
| `MPMetaGGADoubleRelaxMaker` | `vasp.v2.mp_meta_gga_double_relax` | `vasp.v2.relax` | Implemented |
| `MPMetaGGADoubleRelaxStaticMaker` | `vasp.v2.mp_meta_gga_relax_static` | `vasp.v2.double_relax`, `vasp.v2.vasp` | Implemented |
| `MP24DoubleRelaxMaker` | `vasp.v2.mp24_double_relax` | `vasp.v2.relax` | Implemented |
| `MP24DoubleRelaxStaticMaker` | `vasp.v2.mp24_relax_static` | `vasp.v2.double_relax`, `vasp.v2.vasp` | Implemented |
| `MatPesStaticFlowMaker` | `vasp.v2.matpes_static` | `vasp.v2.vasp` | Implemented |
| `OpticsMaker` | `vasp.v2.optics` | `vasp.v2.vasp` | Planned |
| `HSEOpticsMaker` | `vasp.v2.hse_optics` | `vasp.v2.vasp` | Planned |
| `EosMaker` | `vasp.v2.eos` | `vasp.v2.relax`, `vasp.v2.vasp` | Planned |
| `ElasticMaker` | `vasp.v2.elastic` | `vasp.v2.relax`, `vasp.v2.vasp` | Planned |
| `PhononMaker` | `vasp.v2.phonon` | `vasp.v2.relax`, `vasp.v2.vasp` | Planned |
| `QhaMaker` | `vasp.v2.qha` | `vasp.v2.eos`, `vasp.v2.phonon` | Planned |
| `GruneisenMaker` | `vasp.v2.gruneisen` | `vasp.v2.phonon` | Planned |
| `VaspLobsterMaker` | `vasp.v2.lobster` | custom + `vasp.v2.vasp` | Deferred |
| `VaspAmsetMaker` | `vasp.v2.amset` | multiple | Deferred |
| `ElectronPhononMaker` | `vasp.v2.elph` | multiple | Deferred |
| `FerroelectricMaker` | `vasp.v2.ferroelectric` | multiple | Deferred |
| `AdsorptionMaker` | `vasp.v2.adsorption` | multiple | Deferred |
| defect workflows | defect workflow family | multiple | Deferred |
| `ElectrodeInsertionMaker` | `vasp.v2.electrode_insertion` | multiple | Deferred |
| `MVLGWBandStructureMaker` | `vasp.v2.mvl_gw_bands` | custom | Deferred |
