#!/usr/bin/env python3
"""Run the silicon MatPES static tutorial via ``VaspMatPesStaticInputGenerator``.

This example assumes the current AiiDA profile is already configured using the
``aiida-vasp setup-local`` workflow, including a localhost VASP code and a
POTCAR family such as ``PBE.54``.
"""

from __future__ import annotations

from _example_common import (
    build_parser,
    create_silicon_structure,
    maybe_override_potential_family,
    print_called_summary,
    print_header,
    print_node_summary,
    resolve_code,
)

from aiida_vasp.protocols.generator import VaspMatPesStaticInputGenerator


def build_matpes_generator(mode: str, code_label: str | None, potential_family: str) -> VaspMatPesStaticInputGenerator:
    """Configure a MatPES static generator for the selected example mode."""
    structure = create_silicon_structure()
    code = resolve_code(code_label)

    generator = VaspMatPesStaticInputGenerator(protocol='balanced')

    overrides: dict = {}
    if mode == 'custom':
        overrides = {
            'static1': maybe_override_potential_family({}, potential_family),
            'static2': maybe_override_potential_family({}, potential_family),
        }

    generator.build(structure=structure, code=code, overrides=overrides)

    generator.static1().set_options(
        resources={
            'num_machines': 1,
            'num_mpiprocs_per_machine': 4,
        },
        max_wallclock_seconds=3600,
    )
    generator.static2().set_options(
        resources={
            'num_machines': 1,
            'num_mpiprocs_per_machine': 4,
        },
        max_wallclock_seconds=3600,
    )

    if mode == 'generator':
        generator.static1().set_incar(encut=600, ediff=1e-6)
        generator.static2().set_incar(encut=600, ediff=1e-6)
        generator.builder.metadata.label = 'Si MatPES Static (via generator)'
    elif mode == 'custom':
        generator.builder.metadata.label = 'Si MatPES Static (custom potentials)'
    else:
        generator.builder.metadata.label = 'Si MatPES Static PBE -> r2SCAN'

    return generator


def main() -> None:
    """Build and run a MatPES static workflow for silicon."""
    parser = build_parser('Run a silicon MatPES static workchain with VaspMatPesStaticInputGenerator.')
    parser.add_argument(
        'mode',
        nargs='?',
        default='basic',
        choices=('basic', 'generator', 'custom'),
        help='Example mode: basic uses the default protocol, generator adds explicit INCAR overrides, '
        'custom applies the selected POTCAR family to both static stages.',
    )
    args = parser.parse_args()

    generator = build_matpes_generator(args.mode, args.code, args.potential_family)

    print_header('Builder')
    print(generator)
    print(generator.builder)

    print_header('Running WorkChain')
    results = generator.run_get_node()

    print_header('WorkChain Summary')
    print_node_summary(results.node)

    print_header('Child Processes')
    print_called_summary(results.node.called)


if __name__ == '__main__':
    main()
