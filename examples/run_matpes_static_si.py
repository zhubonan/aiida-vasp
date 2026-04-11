#!/usr/bin/env python3
"""Example: Submit a MatPES static calculation for silicon.

This script demonstrates how to use the MatPesStaticWorkChain to run
a PBE static calculation followed by an r2SCAN static calculation with
WAVECAR reuse.
"""

from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

# Load the workchain
MatPesStaticWorkChain = WorkflowFactory('vasp.v2.matpes_static')


def create_diamond_si_structure():
    """Create a diamond silicon structure (conventional cell)."""
    from ase.build import bulk

    # Create diamond structure with 2 atoms (primitive cell)
    ase_si = bulk('Si', 'diamond', a=5.43)
    structure = orm.StructureData(ase=ase_si)
    return structure


def submit_matpes_static():
    """Submit a MatPES static calculation for silicon."""

    # Load your VASP code (adjust the label and computer as needed)
    code = orm.load_code('vasp-std@localhost')

    # Create the silicon structure
    structure = create_diamond_si_structure()

    # Create builder from protocol
    builder = MatPesStaticWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        options={
            'resources': {
                'num_machines': 1,
                'num_mpiprocs_per_machine': 4,
            },
            'max_wallclock_seconds': 3600,
        },
    )

    # Optional: Override parameters for specific stages
    # For example, increase ENCUT for both stages:
    # builder.static1.parameters = orm.Dict(dict={
    #     'incar': {'encut': 600}
    # })
    # builder.static2.parameters = orm.Dict(dict={
    #     'incar': {'encut': 600}
    # })

    # Optional: Set metadata
    builder.metadata.label = 'Si MatPES Static PBE -> r2SCAN'

    # Submit the workflow
    results, node = run_get_node(builder)
    print(f'Submitted MatPES static workflow: PK = {node.pk}')
    print(f'  Process type: {node.process_type}')
    print(f'  Label: {node.label}')

    return node


def run_with_input_generator():
    """Alternative: Use the InputGenerator for interactive configuration."""

    from aiida_vasp.protocols.generator import VaspMatPesStaticInputGenerator

    # Load code and create structure
    code = orm.load_code('vasp-std@localhost')
    structure = create_diamond_si_structure()

    # Build using the generator
    gen = VaspMatPesStaticInputGenerator()
    gen.build(
        structure=structure,
        code=code,
    )

    # Configure stage-specific parameters
    gen.static1().set_incar(encut=600, ediff=1e-6)
    gen.static2().set_incar(encut=600, ediff=1e-6)

    # Set computational resources
    gen.static1().set_resources(num_machines=1, num_mpiprocs_per_machine=4)
    gen.static2().set_resources(num_machines=1, num_mpiprocs_per_machine=4)

    # Submit
    builder = gen.builder
    builder.metadata.label = 'Si MatPES Static (via generator)'
    results, node = run_get_node(builder)
    print(f'Submitted workflow via generator: PK = {node.pk}')

    return node


def run_with_custom_potentials():
    """Run with explicit potential family and mapping."""

    code = orm.load_code('vasp-std@localhost')
    structure = create_diamond_si_structure()

    builder = MatPesStaticWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        overrides={
            # Override for static1 (PBE)
            'static1': {
                'potential_family': 'PBE.54',
            },
            # Override for static2 (r2SCAN)
            'static2': {
                'potential_family': 'PBE.54',
            },
        },
        options={
            'resources': {
                'num_machines': 1,
                'num_mpiprocs_per_machine': 4,
            },
            'max_wallclock_seconds': 3600,
        },
    )

    builder.metadata.label = 'Si MatPES Static (custom potentials)'
    results, node = run_get_node(builder)
    print(f'Submitted workflow: PK = {node.pk}')

    return node


if __name__ == '__main__':
    # Choose which example to run
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'basic'

    if mode == 'basic':
        submit_matpes_static()
    elif mode == 'generator':
        run_with_input_generator()
    elif mode == 'custom':
        run_with_custom_potentials()
    else:
        print(f'Unknown mode: {mode}')
        print('Available modes: basic, generator, custom')
