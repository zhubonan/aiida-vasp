#!/usr/bin/env python3
"""Run the silicon single-point tutorial via ``VaspInputGenerator``.

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
    print_pretty_dict,
    resolve_code,
)

from aiida_vasp.protocols.generator import VaspInputGenerator


def main() -> None:
    """Build and run a single-point workflow for silicon."""
    parser = build_parser('Run a silicon single-point workchain with VaspInputGenerator.')
    args = parser.parse_args()

    structure = create_silicon_structure()
    code = resolve_code(args.code)

    generator = VaspInputGenerator(protocol='balanced')
    overrides = maybe_override_potential_family({}, args.potential_family)
    generator.build(structure=structure, code=code, overrides=overrides)

    print_header('Builder')
    print(generator)
    print(generator.builder)

    print_header('Running WorkChain')
    results = generator.run_get_node()

    print_header('WorkChain Summary')
    print_node_summary(results.node)

    print_header('Misc Output')
    print_pretty_dict(results.node.outputs.misc.get_dict())


if __name__ == '__main__':
    main()
