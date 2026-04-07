import sys
import types

import pytest
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager

from aiida_vasp.workchains.v2 import (
    VaspBandsWorkChain,
    VaspConvergenceWorkChain,
    VaspDoubleRelaxWorkChain,
    VaspHybridBandsWorkChain,
    VaspMP24DoubleRelaxWorkChain,
    VaspMP24RelaxStaticWorkChain,
    VaspMPGGADoubleRelaxWorkChain,
    VaspMPGGARelaxStaticWorkChain,
    VaspMPMetaGGADoubleRelaxWorkChain,
    VaspMPMetaGGARelaxStaticWorkChain,
    VaspNscfWorkChain,
    VaspRelaxBandsWorkChain,
    VaspRelaxWorkChain,
    VaspWorkChain,
)


@pytest.fixture
def basic_env(aiida_profile, mock_vasp, potcar_family_name, upload_potcar):
    """
    Test defining the inputs for a VASP workchain
    """
    pass


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_vasp_protocol(basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Test VASP workchain protocol"""

    builder = VaspWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
    )

    assert builder.structure == vasp_structure
    assert builder.code == mock_vasp
    assert builder.parameters is not None


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_vasp_protocol_pmg_overrides_none(monkeypatch, basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Known pymatgen-style protocols should not fail when ``overrides`` is omitted."""

    class FakeAdaptor:
        KNOWN_SETS = {'FakeSet'}

        def __init__(self, protocol, incar_overrides=None, pmg_kwargs=None):
            self.protocol = protocol
            self.incar_overrides = incar_overrides
            self.pmg_kwargs = pmg_kwargs

        def get_inputs(self, structure, is_workchain=True, overrides=None):
            potential_mapping = {kind: kind for kind in structure.get_kind_names()}
            if 'In' in potential_mapping:
                potential_mapping['In'] = 'In_d'
            return {
                'potential_family': potcar_family_name,
                'potential_mapping': potential_mapping,
                'parameters': {'incar': {'encut': 520}},
                'calc': {'metadata': {'options': {'resources': {'num_machines': 1}}}},
                'meta_parameters': {'ediff_per_atom': 1.0e-6},
                'kpoints_spacing': 0.05,
            }

    monkeypatch.setitem(
        sys.modules,
        'aiida_vasp.protocols.pmg',
        types.SimpleNamespace(PymatgenInputAdaptor=FakeAdaptor),
    )

    builder = VaspWorkChain.get_builder_from_protocol(code=mock_vasp, structure=vasp_structure, protocol='FakeSet')

    assert builder.structure == vasp_structure
    assert builder.parameters['incar']['encut'] == 520


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_vasp_protocol_pmg_kspacing_only(monkeypatch, basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Known pymatgen-style protocols should preserve INCAR KSPACING/KGAMMA when no explicit k-points are supplied."""

    class FakeAdaptor:
        KNOWN_SETS = {'FakeSet'}

        def __init__(self, protocol, incar_overrides=None, pmg_kwargs=None):
            self.protocol = protocol
            self.incar_overrides = incar_overrides
            self.pmg_kwargs = pmg_kwargs

        def get_inputs(self, structure, is_workchain=True, overrides=None):
            potential_mapping = {kind: kind for kind in structure.get_kind_names()}
            if 'In' in potential_mapping:
                potential_mapping['In'] = 'In_d'
            return {
                'potential_family': potcar_family_name,
                'potential_mapping': potential_mapping,
                'parameters': {'incar': {'encut': 520, 'kspacing': 0.22, 'kgamma': False}},
                'calc': {'metadata': {'options': {'resources': {'num_machines': 1}}}},
                'meta_parameters': {'ediff_per_atom': 1.0e-6},
            }

    monkeypatch.setitem(
        sys.modules,
        'aiida_vasp.protocols.pmg',
        types.SimpleNamespace(PymatgenInputAdaptor=FakeAdaptor),
    )

    builder = VaspWorkChain.get_builder_from_protocol(code=mock_vasp, structure=vasp_structure, protocol='FakeSet')

    assert builder.structure == vasp_structure
    assert builder.parameters['incar']['kspacing'] == 0.22
    assert builder.parameters['incar']['kgamma'] is False
    assert 'kpoints' not in builder
    assert 'kpoints_spacing' not in builder

    manager = get_manager()
    runner = manager.get_runner()
    process = instantiate_process(runner, VaspWorkChain, **builder)

    assert process.setup() is None
    assert process.init_inputs() is None
    assert 'kpoints' not in process.ctx.inputs


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_relax_protocol(basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Test VASP relax workchain protocol"""

    builder = VaspRelaxWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
    )

    assert builder.structure == vasp_structure
    assert builder.vasp.code == mock_vasp
    assert builder.vasp.parameters is not None
    assert builder.relax_settings['algo']


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_band_protocol(basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Test VASP band structure workchain protocol"""

    builder = VaspBandsWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={
            'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
            'relax': {'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        },
    )

    assert builder.structure == vasp_structure
    assert builder.nscf.scf.code == mock_vasp
    assert builder.nscf.scf.parameters is not None
    assert builder.band_settings is not None
    assert builder.relax.relax_settings is not None

    nscf_builder = VaspNscfWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={
            'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
        },
    )

    assert nscf_builder.structure == vasp_structure
    assert nscf_builder.scf.code == mock_vasp
    assert nscf_builder.scf.parameters is not None
    assert nscf_builder.band_settings is not None

    # No relax
    builder = VaspBandsWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        run_relax=False,
        overrides={
            'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
            'relax': {'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        },
    )

    assert builder.structure == vasp_structure
    assert builder.nscf.scf.code == mock_vasp
    assert builder.nscf.scf.parameters is not None
    assert builder.band_settings is not None
    assert not builder.relax.relax_settings

    builder = VaspHybridBandsWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={
            'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
            'relax': {'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        },
    )

    assert builder.structure == vasp_structure
    assert builder.scf.code.full_label == mock_vasp.full_label
    assert builder.scf.parameters is not None
    assert builder.band_settings is not None
    assert builder.relax.relax_settings is not None


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_conv_protocol(basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Test VASP convergence test workchain protocol"""

    builder = VaspConvergenceWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
    )

    assert builder.structure == vasp_structure
    assert builder.vasp.code == mock_vasp
    assert builder.vasp.parameters is not None
    assert builder.conv_settings is not None


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_double_relax_protocol(basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Test native double-relax builder generation."""

    builder = VaspDoubleRelaxWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        stage_2_overrides={'parameters': {'incar': {'encut': 600}}},
    )

    assert builder.structure == vasp_structure
    assert builder.relax.vasp.code == mock_vasp
    assert builder.stage_2.parameters['incar']['encut'] == 600


@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_relax_bands_protocol(basic_env, mock_vasp, vasp_structure, potcar_family_name):
    """Test native relax+bands builder generation."""

    builder = VaspRelaxBandsWorkChain.get_builder_from_protocol(
        code=mock_vasp,
        structure=vasp_structure,
        overrides={
            'relax': {'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
            'bands': {
                'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
            },
        },
    )

    assert builder.structure == vasp_structure
    assert builder.relax.vasp.code == mock_vasp
    assert builder.bands.nscf.scf.code == mock_vasp


@pytest.mark.parametrize(
    'workflow_class',
    [
        VaspMPGGADoubleRelaxWorkChain,
        VaspMPMetaGGADoubleRelaxWorkChain,
        VaspMP24DoubleRelaxWorkChain,
    ],
)
@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_mp_double_relax_builders(basic_env, mock_vasp, vasp_structure, workflow_class):
    """Test native MP double-relax builders."""

    builder = workflow_class.get_builder_from_protocol(code=mock_vasp, structure=vasp_structure)

    assert builder.structure == vasp_structure
    assert builder.relax.vasp.code == mock_vasp
    assert builder.relax.vasp.parameters is not None


@pytest.mark.parametrize(
    'workflow_class',
    [
        VaspMPGGARelaxStaticWorkChain,
        VaspMPMetaGGARelaxStaticWorkChain,
        VaspMP24RelaxStaticWorkChain,
    ],
)
@pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
def test_mp_relax_static_builders(basic_env, mock_vasp, vasp_structure, workflow_class):
    """Test native MP relax+static builders."""

    builder = workflow_class.get_builder_from_protocol(code=mock_vasp, structure=vasp_structure)

    assert builder.structure == vasp_structure
    assert builder.relax is not None
    assert builder.static.code == mock_vasp
    assert builder.static.parameters is not None
