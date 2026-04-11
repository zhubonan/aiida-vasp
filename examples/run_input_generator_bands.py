#!/usr/bin/env python3
"""Run the silicon band-structure tutorial via ``VaspBandsInputGenerator``.

This example assumes the current AiiDA profile is already configured using the
``aiida-vasp setup-local`` workflow, including a localhost VASP code and a
POTCAR family such as ``PBE.54``.
"""

from __future__ import annotations

from _example_common import (
    build_parser,
    create_silicon_structure,
    maybe_override_potential_family,
    print_header,
    print_node_summary,
    resolve_code,
)

from aiida_vasp.protocols.generator import VaspBandsInputGenerator


def main() -> None:
    """Build and run a semilocal bands plus DOS workflow for silicon."""
    parser = build_parser('Run a silicon band-structure workchain with VaspBandsInputGenerator.')
    args = parser.parse_args()

    structure = create_silicon_structure()
    code = resolve_code(args.code)

    generator = VaspBandsInputGenerator(protocol='balanced')
    overrides = maybe_override_potential_family({}, args.potential_family, namespace='scf')
    generator.build(structure=structure, code=code, overrides=overrides, run_relax=False)
    generator.nscf().enable_dos(distance=0.03)

    print_header('Builder')
    print(generator)
    print(generator.builder)

    print_header('Running WorkChain')
    results = generator.run_get_node()
    node = results.node

    print_header('WorkChain Summary')
    print_node_summary(node)

    print_header('Outputs')
    print(f'band_structure: {"band_structure" in node.outputs}')
    print(f'dos: {"dos" in node.outputs}')
    print(f'projectors: {"projectors" in node.outputs}')


if __name__ == '__main__':
    main()
