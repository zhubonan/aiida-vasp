"""
Unit tests for aiida-vasp launch command family.
"""

import json
import sys
import types
from pathlib import Path

import pytest
from aiida.common import AttributeDict
from click.testing import CliRunner

from aiida_vasp.commands.launch import launch_workchain, list_presets, list_protocols, load_inputs_from_vasp_folder
from aiida_vasp.commands.utils import load_structure


@pytest.fixture
def cmd_params(tmp_path):
    """Common building blocks for launch command calls."""
    params = AttributeDict()
    params.TMP_PATH = str(tmp_path)
    params.STRUCTURE_FILE = str(tmp_path / 'POSCAR')
    params.VASP_FOLDER = str(tmp_path / 'vasp_calc')

    # Create a simple POSCAR file
    poscar_content = """Test structure
1.0
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
Si
1
Direct
0.0 0.0 0.0
"""
    with open(params.STRUCTURE_FILE, 'w') as f:
        f.write(poscar_content)

    return params


@pytest.fixture
def run_env(aiida_profile, mock_vasp_strict, mock_potcars):
    """Create a mock VASP code for testing."""
    return AttributeDict({'code': mock_vasp_strict})


def run_cmd(command=None, args=None, **kwargs):
    """Run aiida-vasp launch <command> [args]."""
    runner = CliRunner()
    params = args or []
    if command:
        if command == 'launch':
            return runner.invoke(launch_workchain, params, **kwargs)
        elif command == 'presets':
            return runner.invoke(list_presets, params, **kwargs)
        elif command == 'protocols':
            return runner.invoke(list_protocols, params, **kwargs)
    return runner.invoke(launch_workchain, params, **kwargs)


def test_launch_help():
    """Test that help command works."""
    result = run_cmd(command='launch', args=['--help'])
    assert result.exit_code == 0
    assert 'Launch a VASP workchain' in result.output


def test_launch_missing_structure_and_folder():
    """Test launch command fails without structure or vasp folder."""
    result = run_cmd(command='launch', args=['--code', 'test-code', '--label', 'test'])
    assert result.exit_code != 0
    assert 'Either --structure or --from-vasp-folder must be specified' in result.output


def test_launch_dryrun_with_structure(cmd_params, run_env):
    """Test dry run with structure file."""
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-calc',
            '--description',
            'Test calculation',
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output
    assert 'test-calc' in result.output
    assert 'Test calculation' in result.output


def test_launch_invalid_code(cmd_params):
    """Test launch with invalid code."""
    result = run_cmd(
        command='launch',
        args=['--structure', cmd_params.STRUCTURE_FILE, '--code', 'nonexistent-code', '--label', 'test'],
    )
    assert result.exit_code != 0
    assert 'nonexistent-code' in result.output


def test_launch_different_workchain_types(cmd_params, run_env):
    """Test launching different workchain types."""
    workchain_types = ['vasp', 'relax', 'band', 'converge']

    for wc_type in workchain_types:
        result = run_cmd(
            command='launch',
            args=[
                '--structure',
                cmd_params.STRUCTURE_FILE,
                '--code',
                f'{run_env.code.pk}',
                '--label',
                f'test-{wc_type}',
                '--workchain-type',
                wc_type,
                '--dryrun',
            ],
        )
        assert result.exit_code == 0
        assert 'DRY RUN' in result.output


def test_launch_with_options(cmd_params, run_env):
    """Test launch with computational options."""
    options_json = '{"max_wallclock_seconds": 3600, "max_memory_kb": 2000000}'

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-options',
            '--options',
            options_json,
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_launch_with_incar_overrides(cmd_params, run_env):
    """Test launch with INCAR overrides."""
    incar_overrides = '{"encut": 400, "ediff": 1e-6}'

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-incar',
            '--incar-overrides',
            incar_overrides,
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_launch_from_vasp_folder_missing(cmd_params, run_env):
    """Test launch from non-existent VASP folder."""
    result = run_cmd(
        command='launch',
        args=['--from-vasp-folder', cmd_params.VASP_FOLDER, '--code', f'{run_env.code.pk}', '--label', 'test'],
    )
    assert result.exit_code != 0
    assert 'VASP folder not found' in result.output


def test_launch_from_vasp_folder_missing_files(cmd_params, run_env):
    """Test launch from VASP folder missing required files."""
    # Create folder but without required files
    vasp_folder = Path(cmd_params.VASP_FOLDER)
    vasp_folder.mkdir()
    result = run_cmd(
        command='launch',
        args=['--from-vasp-folder', cmd_params.VASP_FOLDER, '--code', f'{run_env.code.pk}', '--label', 'test'],
    )
    assert result.exit_code != 0
    assert 'Missing required files' in result.output


def test_launch_from_vasp_folder_with_files(cmd_params, run_env):
    """Test launch from VASP folder with required files."""
    vasp_folder = Path(cmd_params.VASP_FOLDER)
    vasp_folder.mkdir()

    # Create required files
    incar_content = 'ENCUT = 400\nEDIFF = 1E-6\n'
    (vasp_folder / 'INCAR').write_text(incar_content)

    poscar_content = """Test structure
1.0
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
Si
1
Direct
0.0 0.0 0.0
"""
    (vasp_folder / 'POSCAR').write_text(poscar_content)

    kpoints_content = """Automatic mesh
0
Monkhorst-Pack
4 4 4
0 0 0
"""
    (vasp_folder / 'KPOINTS').write_text(kpoints_content)

    result = run_cmd(
        command='launch',
        args=[
            '--from-vasp-folder',
            cmd_params.VASP_FOLDER,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-from-folder',
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_load_inputs_from_vasp_folder_contains_converge_override(monkeypatch, tmp_path):
    """Convergence overrides should use the public workchain key and VASP-shaped inputs."""

    class DummyStructure:
        def get_formula(self):
            return 'Si'

    class DummyDict:
        def get_dict(self):
            return {'encut': 400}

    folder = tmp_path / 'vasp'
    folder.mkdir()
    for name in ('INCAR', 'POSCAR', 'KPOINTS', 'POTCAR'):
        (folder / name).write_text('')

    monkeypatch.setattr('aiida_vasp.calcs.immigrant.get_poscar_input', lambda _: DummyStructure())
    monkeypatch.setattr('aiida_vasp.calcs.immigrant.get_incar_input', lambda _: DummyDict())
    monkeypatch.setattr('aiida_vasp.calcs.immigrant.get_kpoints_input', lambda *_args, **_kwargs: 'KPOINTS_NODE')
    monkeypatch.setattr('aiida_vasp.calcs.immigrant.get_potcar_input', lambda *_args, **_kwargs: {'Si': 'POTCAR_NODE'})

    _, overrides_map = load_inputs_from_vasp_folder(folder)

    assert 'converge' in overrides_map
    assert overrides_map['converge'] == {
        'parameters': {'incar': {'encut': 400}},
        'potential': {'Si': 'POTCAR_NODE'},
        'kpoints': 'KPOINTS_NODE',
    }


def test_launch_wraps_unexpected_errors(cmd_params, monkeypatch):
    """Unexpected exceptions should surface as Click errors instead of raw tracebacks."""
    monkeypatch.setattr(
        'aiida_vasp.commands.utils.load_structure',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('boom')),
    )

    result = run_cmd(
        command='launch',
        args=['--structure', cmd_params.STRUCTURE_FILE, '--code', 'test-code', '--label', 'test'],
    )

    assert result.exit_code != 0
    assert 'Error: boom' in result.output


def test_launch_match_existing_with_folder(cmd_params, run_env):
    """Test matching an existing structure when combining direct and folder inputs."""
    vasp_folder = Path(cmd_params.VASP_FOLDER)
    vasp_folder.mkdir()
    (vasp_folder / 'INCAR').write_text('ENCUT = 400\n')
    (vasp_folder / 'POSCAR').write_text(Path(cmd_params.STRUCTURE_FILE).read_text())
    (vasp_folder / 'KPOINTS').write_text('Automatic mesh\n0\nMonkhorst-Pack\n4 4 4\n0 0 0\n')

    ext = load_structure(cmd_params.STRUCTURE_FILE).store()
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--from-vasp-folder',
            cmd_params.VASP_FOLDER,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-from-folder',
            '--dryrun',
            '--match-existing',
        ],
    )
    assert result.exit_code == 0
    assert 'Loaded structure from VASP folder' in result.output
    assert 'Loaded structure:' in result.output
    assert 'Using existing structure node' in result.output

    # Test match-existing
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            ext.uuid,
            '--from-vasp-folder',
            cmd_params.VASP_FOLDER,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-from-folder',
            '--dryrun',
            '--match-existing',
        ],
    )
    assert result.exit_code == 0
    assert 'Loaded structure:' in result.output
    assert 'Using existing structure node' in result.output

    # Test using UUID as structure input
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            ext.uuid,
            '--from-vasp-folder',
            cmd_params.VASP_FOLDER,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-from-folder',
            '--dryrun',
            '--match-existing',
        ],
    )
    assert result.exit_code == 0


def test_presets_list():
    """Test listing available presets."""
    result = run_cmd(command='presets')
    assert result.exit_code == 0
    assert 'Available presets:' in result.output


def test_presets_show_content():
    """Test showing preset content."""
    result = run_cmd(command='presets', args=['--show-content'])
    assert result.exit_code == 0


def test_presets_specific():
    """Test showing specific preset."""
    # First get available presets
    result = run_cmd(command='presets')
    if result.exit_code == 0 and 'default' in result.output:
        result = run_cmd(command='presets', args=['default'])
        assert result.exit_code == 0


def test_protocols_list():
    """Test listing available protocols."""
    result = run_cmd(command='protocols')
    assert result.exit_code == 0


def test_protocols_show_content():
    """Test showing protocol content."""
    result = run_cmd(command='protocols', args=['--show-content'])
    assert result.exit_code == 0


def test_protocols_specific_workflow():
    """Test showing protocols for specific workflow."""
    result = run_cmd(command='protocols', args=['vasp'])
    assert result.exit_code == 0


@pytest.mark.parametrize('preset', ['default'])
def test_launch_with_different_presets(cmd_params, run_env, preset):
    """Test launch with different presets."""
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            f'test-{preset}',
            '--preset',
            preset,
            '--dryrun',
        ],
    )
    # Should not fail even if preset doesn't exist (will use default)
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_launch_uses_preset_default_protocol(cmd_params, run_env, monkeypatch):
    """If ``--protocol`` is omitted, the preset's default protocol should be used."""
    preset_dir = Path(cmd_params.TMP_PATH) / 'presets'
    preset_dir.mkdir()
    (preset_dir / 'custom.yaml').write_text(
        '\n'.join(
            [
                'name: custom',
                'default_protocol: fast',
                'default_code: mock-vasp@localhost',
            ]
        )
    )
    monkeypatch.setattr('aiida_vasp.protocols.generator.get_preset_library_paths', lambda: (preset_dir,))

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-custom-protocol',
            '--workchain-type',
            'relax',
            '--preset',
            'custom',
            '--dryrun',
        ],
    )

    assert result.exit_code == 0
    assert 'Protocol: fast' in result.output


@pytest.mark.parametrize('protocol', ['balanced', 'MPRelaxSet'])
def test_launch_with_different_protocols(cmd_params, run_env, protocol):
    """Test launch with different protocols."""
    args = [
        '--structure',
        cmd_params.STRUCTURE_FILE,
        '--code',
        f'{run_env.code.pk}',
        '--label',
        f'test-{protocol}',
        '--protocol',
        protocol,
        '--dryrun',
    ]
    if 'MP' in protocol:
        args.extend(['--overrides', 'potential_family=PBE.54'])
        # Skip this test case if pymatgen is not installed
        try:
            import pymatgen  # noqa: PLC0415
        except ImportError:
            _ = pymatgen
            return
        from pymatgen.core import SETTINGS  # noqa: PLC0415

        if SETTINGS.get('PMG_VASP_PSP_DIR') is None:
            return

    result = run_cmd(command='launch', args=args)
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_launch_with_resources(cmd_params, run_env):
    """Test launch with resource specifications."""
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-resources',
            '--num-machines',
            '2',
            '--tot-num-mpiprocs',
            '16',
            '--max-wallclock-seconds',
            '3600',
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'num_machines: 2' in result.output
    assert 'tot_num_mpiprocs: 16' in result.output
    assert 'max_wallclock_seconds: 3600' in result.output
    assert 'DRY RUN' in result.output


def test_launch_with_pymatgen_pmg_set_vasp(cmd_params, run_env, monkeypatch):
    """Pymatgen sets should be usable through the launch CLI for VaspWorkChain."""

    class FakeAdaptor:
        KNOWN_SETS = {'FakeStaticSet'}
        last_pmg_kwargs = None

        def __init__(self, protocol, incar_overrides=None, pmg_kwargs=None):
            self.protocol = protocol
            self.incar_overrides = incar_overrides or {}
            self.pmg_kwargs = pmg_kwargs or {}
            type(self).last_pmg_kwargs = self.pmg_kwargs

        def get_inputs(self, structure, is_workchain=True, overrides=None):
            return {
                'potential_family': 'PBE.54',
                'potential_mapping': {'Si': 'Si'},
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

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-pmg-vasp',
            '--pmg-set',
            'FakeStaticSet',
            '--pmg-kwargs',
            '{"user_incar_settings": {"ISMEAR": 0}}',
            '--dryrun',
        ],
    )

    assert result.exit_code == 0
    assert 'Pymatgen set: FakeStaticSet' in result.output
    assert FakeAdaptor.last_pmg_kwargs == {'user_incar_settings': {'ISMEAR': 0}}


def test_launch_with_pymatgen_pmg_set_relax(cmd_params, run_env, monkeypatch):
    """Pymatgen sets should be usable through the launch CLI for VaspRelaxWorkChain."""

    class FakeAdaptor:
        KNOWN_SETS = {'FakeRelaxSet'}
        last_pmg_kwargs = None

        def __init__(self, protocol, incar_overrides=None, pmg_kwargs=None):
            self.protocol = protocol
            self.incar_overrides = incar_overrides or {}
            self.pmg_kwargs = pmg_kwargs or {}
            type(self).last_pmg_kwargs = self.pmg_kwargs

        def get_inputs(self, structure, is_workchain=True, overrides=None):
            return {
                'potential_family': 'PBE.54',
                'potential_mapping': {'Si': 'Si'},
                'parameters': {'incar': {'encut': 520, 'nsw': 99, 'ibrion': 2, 'isif': 3}},
                'calc': {'metadata': {'options': {'resources': {'num_machines': 1}}}},
                'meta_parameters': {'ediff_per_atom': 1.0e-6},
                'kpoints_spacing': 0.05,
            }

    monkeypatch.setitem(
        sys.modules,
        'aiida_vasp.protocols.pmg',
        types.SimpleNamespace(PymatgenInputAdaptor=FakeAdaptor),
    )

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-pmg-relax',
            '--workchain-type',
            'relax',
            '--pmg-set',
            'FakeRelaxSet',
            '--pmg-kwargs',
            '{"sort_structure": false}',
            '--dryrun',
        ],
    )

    assert result.exit_code == 0
    assert 'Pymatgen set: FakeRelaxSet' in result.output
    assert FakeAdaptor.last_pmg_kwargs == {'sort_structure': False}


def test_launch_band_workchain_settings(cmd_params, run_env):
    """Test launch band workchain with settings."""
    band_settings = '{"symprec": 0.001}'

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-band',
            '--workchain-type',
            'band',
            '--band-settings',
            band_settings,
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'symprec: 0.001' in result.output
    assert 'DRY RUN' in result.output


def test_launch_relax_workchain_settings(cmd_params, run_env):
    """Test launch relax workchain with settings."""
    relax_settings = '{"force_cutoff": 0.01}'

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-relax',
            '--workchain-type',
            'relax',
            '--relax-settings',
            relax_settings,
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'force_cutoff: 0.01' in result.output
    assert 'DRY RUN' in result.output


def test_launch_converge_workchain_settings(cmd_params, run_env):
    """Test launch converge workchain with settings."""
    converge_settings = '{"converge_cutoff": true}'

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-converge',
            '--workchain-type',
            'converge',
            '--converge-settings',
            converge_settings,
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_launch_with_invalid_json_options(cmd_params, run_env):
    """Test launch with invalid JSON in options."""
    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-invalid',
            '--options',
            'invalid-json',
            '--dryrun',
        ],
    )
    # Should handle invalid JSON gracefully
    assert result.exit_code != 0 or 'DRY RUN' in result.output


def test_launch_with_overrides_file(cmd_params, run_env):
    """Test launch with overrides from file."""
    overrides_file = cmd_params.TMP_PATH + '/overrides.json'
    overrides = {'parameters': {'incar': {'encut': 500}}}

    with open(overrides_file, 'w') as f:
        json.dump(overrides, f)

    result = run_cmd(
        command='launch',
        args=[
            '--structure',
            cmd_params.STRUCTURE_FILE,
            '--code',
            f'{run_env.code.pk}',
            '--label',
            'test-overrides-file',
            '--overrides',
            overrides_file,
            '--dryrun',
        ],
    )
    assert result.exit_code == 0
    assert 'DRY RUN' in result.output


def test_launch_missing_structure_file(cmd_params, run_env):
    """Test launch with non-existent structure file."""
    result = run_cmd(
        command='launch', args=['--structure', '/nonexistent/POSCAR', '--code', f'{run_env.code.pk}', '--label', 'test']
    )
    assert result.exit_code != 0
