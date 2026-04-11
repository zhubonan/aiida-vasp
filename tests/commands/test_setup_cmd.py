"""
Unit tests for aiida-vasp setup-local command.
"""

import shutil
from pathlib import Path

from aiida.orm import Computer, load_code
from click.testing import CliRunner

from aiida_vasp.commands.setup import setup_local
from aiida_vasp.data.potcar import PotcarData


def run_cmd(args=None, **kwargs):
    """Run aiida-vasp setup-local [args]."""
    runner = CliRunner()
    return runner.invoke(setup_local, args or [], **kwargs)


def _make_executable(path: Path) -> str:
    path.write_text('#!/bin/bash\nexit 0\n')
    path.chmod(0o755)
    return str(path)


def _create_localhost(tmp_path):
    computer = Computer(
        label='localhost',
        hostname='localhost',
        transport_type='core.local',
        scheduler_type='core.direct',
        workdir=str(tmp_path / 'workdir'),
    )
    computer.store()
    computer.configure()
    return computer


def test_setup_local_creates_three_codes(aiida_profile_clean, tmp_path):
    """The command should create std/gam/ncl InstalledCode entries on localhost."""
    _create_localhost(tmp_path)
    std = _make_executable(tmp_path / 'vasp_std')
    gam = _make_executable(tmp_path / 'vasp_gam')
    ncl = _make_executable(tmp_path / 'vasp_ncl')

    result = run_cmd(['--tag', '642', '--vasp-std', std, '--vasp-gam', gam, '--vasp-ncl', ncl, '--no-potcars'])

    assert result.exit_code == 0
    assert 'vasp-std-642@localhost' in result.output
    assert 'vasp-gam-642@localhost' in result.output
    assert 'vasp-ncl-642@localhost' in result.output
    assert load_code('vasp-std-642@localhost').default_calc_job_plugin == 'vasp.vasp'
    assert load_code('vasp-gam-642@localhost').default_calc_job_plugin == 'vasp.vasp'
    assert load_code('vasp-ncl-642@localhost').default_calc_job_plugin == 'vasp.vasp'


def test_setup_local_requires_localhost(aiida_profile_clean, tmp_path):
    """The command should direct users to verdi presto when localhost is missing."""
    std = _make_executable(tmp_path / 'vasp_std')
    gam = _make_executable(tmp_path / 'vasp_gam')
    ncl = _make_executable(tmp_path / 'vasp_ncl')

    try:
        Computer.collection.get(label='localhost')
        assert False, 'localhost should not exist before setup-local runs'
    except Exception:
        pass

    result = run_cmd(['--tag', '700', '--vasp-std', std, '--vasp-gam', gam, '--vasp-ncl', ncl, '--no-potcars'])

    assert result.exit_code != 0
    assert 'Run `verdi presto` first' in result.output


def test_setup_local_rejects_non_executable(aiida_profile_clean, tmp_path):
    """The command should reject non-executable paths."""
    _create_localhost(tmp_path)
    std = tmp_path / 'vasp_std'
    std.write_text('#!/bin/bash\nexit 0\n')
    gam = _make_executable(tmp_path / 'vasp_gam')
    ncl = _make_executable(tmp_path / 'vasp_ncl')

    result = run_cmd(['--tag', '800', '--vasp-std', str(std), '--vasp-gam', gam, '--vasp-ncl', ncl, '--no-potcars'])

    assert result.exit_code != 0
    assert '--vasp-std executable is not executable' in result.output


def test_setup_local_imports_potcars_from_local_path(aiida_profile_clean, tmp_path, temp_pot_folder):
    """The command should optionally import POTCARs from a local folder during setup."""
    _create_localhost(tmp_path)
    std = _make_executable(tmp_path / 'vasp_std')
    gam = _make_executable(tmp_path / 'vasp_gam')
    ncl = _make_executable(tmp_path / 'vasp_ncl')
    potcar_dir = tmp_path / 'potpaw_PBE.64'
    shutil.copytree(temp_pot_folder, potcar_dir)

    result = run_cmd(
        [
            '--tag',
            '642',
            '--vasp-std',
            std,
            '--vasp-gam',
            gam,
            '--vasp-ncl',
            ncl,
            '--potcar-path',
            str(potcar_dir),
        ]
    )

    assert result.exit_code == 0
    assert 'Imported POTCAR family: PBE.64' in result.output
    assert PotcarData.get_potcar_group('PBE.64') is not None


def test_setup_local_interactive_infers_executables(aiida_profile_clean, tmp_path, monkeypatch):
    """Interactive setup should ask for vasp_std and infer sibling gamma/ncl executables."""
    _create_localhost(tmp_path)
    std = _make_executable(tmp_path / 'vasp_std')
    _make_executable(tmp_path / 'vasp_gam')
    _make_executable(tmp_path / 'vasp_ncl')
    monkeypatch.setattr('aiida_vasp.commands.setup._which_executable', lambda _name: None)

    result = run_cmd(input=f'642\n{std}\nn\n')

    assert result.exit_code == 0
    assert 'Inferred vasp_gam' in result.output
    assert 'Inferred vasp_ncl' in result.output
    assert 'vasp-std-642@localhost' in result.output


def test_setup_local_reuses_existing_path_by_default(aiida_profile_clean, tmp_path):
    """Duplicate computer/path entries should be reused unless the user opts into duplication."""
    from aiida import orm

    computer = _create_localhost(tmp_path)
    std = _make_executable(tmp_path / 'vasp_std')
    gam = _make_executable(tmp_path / 'vasp_gam')
    ncl = _make_executable(tmp_path / 'vasp_ncl')
    existing = orm.InstalledCode(label='existing-std', computer=computer, filepath_executable=std)
    existing.default_calc_job_plugin = 'vasp.vasp'
    existing.store()

    result = run_cmd(['--tag', '642', '--vasp-std', std, '--vasp-gam', gam, '--vasp-ncl', ncl, '--no-potcars'])

    assert result.exit_code == 0
    assert 'existing-std@localhost' in result.output
