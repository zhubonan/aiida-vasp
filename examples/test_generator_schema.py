#!/usr/bin/env python
"""Try the self-documenting input-generator API with a bulk silicon structure."""

from __future__ import annotations

import argparse
from pathlib import Path
from subprocess import check_output

from aiida import orm
from aiida.common.exceptions import NotExistent
from ase.build import bulk

from aiida_vasp.data.potcar import PotcarData
from aiida_vasp.protocols.generator import (
    VaspBandsInputGenerator,
    VaspHybridBandsInputGenerator,
    VaspNscfInputGenerator,
    VaspRelaxBandsInputGenerator,
)
from aiida_vasp.utils.temp_profile import load_temp_profile

ROOT = Path(__file__).resolve().parents[1]
POTCAR_FAMILY = 'PBE.EXAMPLE'


def print_header(title: str) -> None:
    """Print a simple section header."""
    print(f'\n{"=" * 20} {title} {"=" * 20}')


def ensure_localhost_computer() -> orm.Computer:
    """Get or create a localhost computer suitable for the example scripts."""
    try:
        computer = orm.load_computer('localhost')
    except NotExistent:
        computer = orm.Computer(
            label='localhost',
            hostname='localhost',
            transport_type='core.local',
            scheduler_type='core.direct',
        )
        computer.store()

    computer.set_workdir('/tmp/aiida_run/')
    try:
        computer.configure()
    except Exception:
        # Ignore if the local computer has already been configured.
        pass
    return computer


def ensure_mock_vasp_code(computer: orm.Computer) -> str:
    """Get or create a mock VASP code and return its fully qualified label."""
    code_label = 'mock-vasp'
    full_label = f'{code_label}@{computer.label}'

    try:
        orm.load_code(full_label)
        return full_label
    except NotExistent:
        pass

    executable = check_output(['which', 'mock-vasp'], universal_newlines=True).strip()
    code = orm.InstalledCode(computer, executable, default_calc_job_plugin='vasp.vasp')
    code.label = code_label
    code.store()
    return full_label


def ensure_example_potcars() -> None:
    """Upload the example POTCAR family used by the mock examples."""
    potcar_source = None
    for candidate in (
        ROOT / 'examples' / 'potcars',
        ROOT / 'docs' / 'source' / 'tutorials' / 'potcars',
        ROOT / 'tests' / 'test_data' / 'potcar',
    ):
        if any(candidate.rglob('POTCAR')):
            potcar_source = candidate
            break

    if potcar_source is None:
        raise RuntimeError('Could not find a POTCAR folder in the repository for the generator demo.')

    PotcarData.upload_potcar_family(
        potcar_source,
        POTCAR_FAMILY,
        'Example POTCAR family for generator demos',
        stop_if_existing=False,
    )


def make_silicon_structure() -> orm.StructureData:
    """Create a bulk silicon structure with ASE and wrap it for AiiDA."""
    silicon = bulk('Si', 'diamond', a=5.43)
    return orm.StructureData(ase=silicon)


def build_semilocal_bands(structure: orm.StructureData, code: str, protocol: str):
    """Build a semilocal bands generator."""
    gen = VaspBandsInputGenerator(protocol=protocol)
    gen.build(
        structure=structure,
        code=code,
        run_relax=True,
        overrides={
            'relax': {'vasp': {'potential_family': POTCAR_FAMILY}},
            'nscf': {'scf': {'potential_family': POTCAR_FAMILY}},
        },
    )
    gen.relax().relax().set_relax_settings(force_cutoff=0.03)
    gen.set_band_settings(run_dos=True, dos_kpoints_distance=0.03)
    gen.nscf().scf().set_incar(ismear=0)
    return gen


def build_hybrid_bands(structure: orm.StructureData, code: str, protocol: str):
    """Build a hybrid bands generator."""
    gen = VaspHybridBandsInputGenerator(protocol=protocol)
    gen.build(
        structure=structure,
        code=code,
        run_relax=True,
        overrides={
            'relax': {'vasp': {'potential_family': POTCAR_FAMILY}},
            'scf': {'potential_family': POTCAR_FAMILY},
        },
    )
    gen.relax().relax().set_relax_settings(force_cutoff=0.03)
    gen.scf().set_incar(ismear=0)
    gen.set_band_settings(kpoints_per_split=150)
    return gen


def build_standalone_nscf(structure: orm.StructureData, code: str, protocol: str):
    """Build a standalone NSCF generator."""
    gen = VaspNscfInputGenerator(protocol=protocol)
    gen.build(
        structure=structure,
        code=code,
        overrides={'scf': {'potential_family': POTCAR_FAMILY}},
    )
    gen.scf().set_incar(ismear=0)
    gen.enable_dos(distance=0.03)
    return gen


def build_relax_bands(structure: orm.StructureData, code: str, protocol: str):
    """Build a relax-plus-bands generator."""
    gen = VaspRelaxBandsInputGenerator(protocol=protocol)
    gen.build(
        structure=structure,
        code=code,
        overrides={
            'relax': {'vasp': {'potential_family': POTCAR_FAMILY}},
            'bands': {
                'relax': {'vasp': {'potential_family': POTCAR_FAMILY}},
                'nscf': {'scf': {'potential_family': POTCAR_FAMILY}},
            },
        },
    )
    gen.relax().relax().set_relax_settings(force_cutoff=0.03)
    gen.bands().set_band_settings(run_dos=True, dos_kpoints_distance=0.03)
    gen.bands().nscf().scf().set_incar(ismear=0)
    return gen


def show_generator(name: str, generator, show_builder: bool) -> None:
    """Print the generator schema and some important namespace views."""
    print_header(name)
    print(generator)

    for accessor in generator.accessors():
        method_name = accessor.removesuffix('()')
        try:
            view = getattr(generator, method_name)()
        except AttributeError as exc:
            print_header(f'{name} :: {accessor}')
            print(f'{accessor} unavailable: {exc}')
            continue

        print_header(f'{name} :: {accessor}')
        print(view)

        nested_accessors = getattr(view, 'ACCESSOR_DOCS', ())
        nested_names = [item.partition(': ')[0].removesuffix('()') for item in nested_accessors]
        if method_name == 'bands' and 'nscf' in nested_names:
            print_header(f'{name} :: {accessor}.nscf()')
            print(view.nscf())

    if show_builder:
        print_header(f'{name} :: builder')
        print(generator.builder)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workflow',
        choices=('all', 'bands', 'hybrid', 'nscf', 'relax-bands'),
        default='all',
        help='Which generator example to build.',
    )
    parser.add_argument(
        '--protocol',
        default='balanced',
        help='Protocol name passed to the input generators.',
    )
    parser.add_argument(
        '--show-builder',
        action='store_true',
        help='Also print the underlying builder contents.',
    )
    return parser.parse_args()


def main() -> None:
    """Build and display generator schemas for the selected workflows."""
    args = parse_args()

    load_temp_profile()
    computer = ensure_localhost_computer()
    code = ensure_mock_vasp_code(computer)
    ensure_example_potcars()
    structure = make_silicon_structure()

    print_header('Environment')
    print(f'Code: {code}')
    print(f'POTCAR family: {POTCAR_FAMILY}')
    print(f'Structure: {structure}')

    builders = {
        'bands': build_semilocal_bands,
        'hybrid': build_hybrid_bands,
        'nscf': build_standalone_nscf,
        'relax-bands': build_relax_bands,
    }

    selected = builders.keys() if args.workflow == 'all' else (args.workflow,)
    for name in selected:
        generator = builders[name](structure, code, args.protocol)
        show_generator(name, generator, show_builder=args.show_builder)


if __name__ == '__main__':
    main()
