#!/usr/bin/env python3
"""Shared helpers for plain Python example scripts."""

from __future__ import annotations

import argparse
import os
from pprint import pprint
from typing import Iterable

from aiida import load_profile, orm
from ase.build import bulk

load_profile()

DEFAULT_POTENTIAL_FAMILY = 'PBE.54'
DEFAULT_CODE_ENV = 'AIIDA_VASP_CODE'
DEFAULT_POTENTIAL_FAMILY_ENV = 'AIIDA_VASP_POTCAR_FAMILY'


def build_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with the shared example options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--code',
        default=os.environ.get(DEFAULT_CODE_ENV),
        help=f'AiiDA code label, e.g. vasp-std-642@localhost. Defaults to ${DEFAULT_CODE_ENV} or auto-detection.',
    )
    parser.add_argument(
        '--potential-family',
        default=os.environ.get(DEFAULT_POTENTIAL_FAMILY_ENV, DEFAULT_POTENTIAL_FAMILY),
        help=(f'POTCAR family to use. Defaults to ${DEFAULT_POTENTIAL_FAMILY_ENV} or {DEFAULT_POTENTIAL_FAMILY}.'),
    )
    return parser


def resolve_code(code_label: str | None) -> orm.AbstractCode:
    """Resolve the requested code or auto-detect a localhost VASP code."""
    if code_label:
        return orm.load_code(code_label)

    builder = orm.QueryBuilder()
    builder.append(orm.InstalledCode, project=['*'])
    candidates = [
        code
        for code in builder.all(flat=True)
        if code.computer.label == 'localhost' and code.label.startswith('vasp-std')
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise SystemExit(
            'No localhost InstalledCode with a label starting with `vasp-std` was found. '
            'Pass `--code <label>@localhost` or set AIIDA_VASP_CODE.'
        )

    available = ', '.join(sorted(code.full_label for code in candidates))
    raise SystemExit(
        'Multiple localhost codes matching `vasp-std*` were found. '
        f'Pass `--code` to disambiguate. Candidates: {available}'
    )


def create_silicon_structure() -> orm.StructureData:
    """Return the silicon primitive cell used in the tutorials."""
    silicon = bulk('Si', 'diamond', 5.4)
    structure = orm.StructureData(ase=silicon)
    structure.label = 'Silicon diamond primitive cell'
    return structure


def maybe_override_potential_family(
    overrides: dict | None,
    potential_family: str,
    *,
    default_family: str = DEFAULT_POTENTIAL_FAMILY,
    namespace: str | None = None,
) -> dict:
    """Inject a potential family override only when needed."""
    resolved = dict(overrides or {})
    if potential_family == default_family:
        return resolved

    if namespace is None:
        resolved['potential_family'] = potential_family
        return resolved

    namespace_overrides = dict(resolved.get(namespace, {}))
    namespace_overrides['potential_family'] = potential_family
    resolved[namespace] = namespace_overrides
    return resolved


def print_header(title: str) -> None:
    """Print a compact section header."""
    print(f'\n=== {title} ===')


def print_node_summary(node: orm.ProcessNode) -> None:
    """Print core node identifiers and final state."""
    print(f'PK: {node.pk}')
    print(f'UUID: {node.uuid}')
    print(f'Process label: {node.process_label}')
    print(f'Process state: {node.process_state}')
    print(f'Exit status: {node.exit_status}')


def print_called_summary(nodes: Iterable[orm.ProcessNode]) -> None:
    """Print a compact summary of called child processes."""
    called_nodes = list(nodes)
    if not called_nodes:
        print('No called child processes.')
        return

    for child in called_nodes:
        print(f'- PK {child.pk}: {child.process_label} [{child.process_state}] exit_status={child.exit_status}')


def print_pretty_dict(content: dict) -> None:
    """Pretty-print a dictionary."""
    pprint(content, sort_dicts=False)
