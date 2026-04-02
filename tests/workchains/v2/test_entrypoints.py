from aiida.plugins import WorkflowFactory


def test_entrypoints(aiida_profile):
    """Test that we can instantiate the workchains from the entry points."""
    _ = aiida_profile
    entrypoints = [
        'vasp.vasp',
        'vasp.relax',
        'vasp.converge',
        'vasp.bands',
        'vasp.v2.vasp',
        'vasp.v2.relax',
        'vasp.v2.converge',
        'vasp.v2.bands',
        'vasp.v2.nscf',
        'vasp.v2.hybrid_bands',
        'vasp.v2.staged_relax',
        'vasp.v2.double_relax',
        'vasp.v2.relax_bands',
        'vasp.v2.mp_gga_double_relax',
        'vasp.v2.mp_gga_relax_static',
        'vasp.v2.mp_meta_gga_double_relax',
        'vasp.v2.mp_meta_gga_relax_static',
        'vasp.v2.mp24_double_relax',
        'vasp.v2.mp24_relax_static',
    ]
    for point in entrypoints:
        WorkflowFactory(point)
