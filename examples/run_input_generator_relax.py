#!/usr/bin/env python3
"""Run the silicon relaxation tutorial via ``VaspRelaxInputGenerator``.

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

from aiida_vasp.protocols.generator import VaspRelaxInputGenerator


def main() -> None:
    """Build and run a relaxation workflow for silicon."""
    parser = build_parser('Run a silicon relaxation workchain with VaspRelaxInputGenerator.')
    args = parser.parse_args()

    structure = create_silicon_structure()
    code = resolve_code(args.code)

    generator = VaspRelaxInputGenerator(protocol='balanced')
    overrides = maybe_override_potential_family({}, args.potential_family, namespace='vasp')
    generator.build(structure=structure, code=code, overrides=overrides)

    print_header('Relax Settings Help')
    generator.get_input_help('relax_settings')

    print_header('Builder')
    print(generator)
    print(generator.builder)

    print_header('Running WorkChain')
    results = generator.run_get_node()

    print_header('WorkChain Summary')
    print_node_summary(results.node)

    relaxed_structure = results.node.outputs.relax.structure
    print_header('Relaxed Structure')
    print(f'Initial volume: {structure.get_cell_volume():.6f} A^3')
    print(f'Relaxed volume: {relaxed_structure.get_cell_volume():.6f} A^3')

    print_header('Child Processes')
    print_called_summary(results.node.called)


if __name__ == '__main__':
    main()
