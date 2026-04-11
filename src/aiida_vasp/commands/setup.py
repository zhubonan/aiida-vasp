"""
Helpers for setting up local VASP execution resources.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import click

from . import cmd_aiida_vasp
from .potcar import FUNCTIONAL_CHOICES, _resolve_family_metadata


def _get_localhost():
    """Return the configured localhost computer."""
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    try:
        return orm.Computer.collection.get(label='localhost')
    except NotExistent:
        raise click.ClickException('No localhost computer found. Run `verdi presto` first, then rerun this command.')


def _validate_executable(path: str, option_name: str) -> Path:
    """Validate a provided executable path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise click.ClickException(f'{option_name} executable not found: {resolved}')
    if not resolved.is_file():
        raise click.ClickException(f'{option_name} executable is not a file: {resolved}')
    if not os.access(resolved, os.X_OK):
        raise click.ClickException(f'{option_name} executable is not executable: {resolved}')
    return resolved


def _find_code_by_computer_and_path(computer, executable: Path):
    """Return an existing InstalledCode for a computer/path pair, if present."""
    from aiida import orm

    executable = executable.resolve()
    query = orm.QueryBuilder().append(orm.InstalledCode)
    for code in query.all(flat=True):
        if code.computer != computer:
            continue
        try:
            if Path(code.filepath_executable).expanduser().resolve() == executable:
                return code
        except OSError:
            continue
    return None


def _load_code(full_label: str):
    """Load a code by full label, returning None if absent."""
    from aiida import orm
    from aiida.common.exceptions import NotExistent

    try:
        return orm.load_code(full_label)
    except NotExistent:
        return None


def _create_or_reuse_installed_code(computer, label: str, executable: Path, non_interactive: bool):
    """Create an InstalledCode unless a duplicate should be reused."""
    from aiida import orm

    full_label = f'{label}@{computer.label}'
    existing_label_code = _load_code(full_label)
    existing_path_code = _find_code_by_computer_and_path(computer, executable)

    if existing_label_code is not None:
        if non_interactive or click.confirm(
            f'Code {full_label} already exists. Reuse it instead of creating a duplicate?', default=True
        ):
            return existing_label_code, False
        raise click.ClickException(f'Cannot create duplicate code label: {full_label}')

    if existing_path_code is not None:
        if non_interactive:
            return existing_path_code, False
        if not click.confirm(
            f'Executable {executable} is already configured as {existing_path_code.full_label}. '
            'Create a duplicate code anyway?',
            default=False,
        ):
            return existing_path_code, False

    code = orm.InstalledCode(label=label, computer=computer, filepath_executable=str(executable))
    code.default_calc_job_plugin = 'vasp.vasp'
    code.description = f'Local VASP executable at {executable}'
    code.store()
    return code, True


def _which_executable(name: str) -> Path | None:
    """Return a resolved executable from PATH, if available."""
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    return None


def _infer_executable_from_std(vasp_std: Path, executable_name: str) -> Path | None:
    """Infer a sibling executable from the vasp_std path."""
    candidate = vasp_std.parent / executable_name
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate.resolve()
    return None


def _prompt_executable(option_name: str, prompt: str) -> Path:
    """Prompt until a valid executable is provided."""
    return _validate_executable(click.prompt(prompt, type=click.Path(exists=True)), option_name)


def _resolve_executable_paths(vasp_std: str | None, vasp_gam: str | None, vasp_ncl: str | None):
    """Resolve executable paths from options, PATH lookup, or interactive prompts."""
    resolved = {}
    supplied = {'vasp_std': vasp_std, 'vasp_gam': vasp_gam, 'vasp_ncl': vasp_ncl}
    option_names = {'vasp_std': '--vasp-std', 'vasp_gam': '--vasp-gam', 'vasp_ncl': '--vasp-ncl'}

    for name, value in supplied.items():
        if value:
            resolved[name] = _validate_executable(value, option_names[name])

    for name in ('vasp_std', 'vasp_gam', 'vasp_ncl'):
        if name not in resolved:
            found = _which_executable(name)
            if found:
                click.echo(f'Found {name}: {found}')
                resolved[name] = found

    if 'vasp_std' not in resolved:
        resolved['vasp_std'] = _prompt_executable('--vasp-std', 'Path to vasp_std')

    for name in ('vasp_gam', 'vasp_ncl'):
        if name not in resolved:
            inferred = _infer_executable_from_std(resolved['vasp_std'], name)
            if inferred:
                click.echo(f'Inferred {name}: {inferred}')
                resolved[name] = inferred
            else:
                resolved[name] = _prompt_executable(option_names[name], f'Path to {name}')

    return resolved


def _upload_potcar_family(path: str, name: str | None, description: str | None, stop_if_existing: bool = False) -> str:
    """Upload a POTCAR family from a local path."""
    from aiida_vasp.data.potcar import PotcarData

    resolved_name, resolved_description = _resolve_family_metadata(name, description, path=path)
    PotcarData.upload_potcar_family(
        path, resolved_name, resolved_description, stop_if_existing=stop_if_existing, dry_run=False
    )
    return resolved_name


def _upload_potcar_family_from_pymatgen(
    functional: str, name: str | None, description: str | None, stop_if_existing: bool = False
) -> str:
    """Upload a POTCAR family from the user's pymatgen configuration."""
    from pymatgen.io.vasp.inputs import SETTINGS, PotcarSingle

    from aiida_vasp.data.potcar import PotcarData
    from aiida_vasp.utils.pmg import convert_pymatgen_potcar_folder, temporary_folder

    resolved_name, resolved_description = _resolve_family_metadata(name, description, functional=functional)
    funcdir = PotcarSingle.functional_dir[functional]
    pmg_vasp_psp_dir = SETTINGS.get('PMG_VASP_PSP_DIR')
    if pmg_vasp_psp_dir is None:
        raise click.ClickException(
            'PMG_VASP_PSP_DIR is not set, please set it in your .pmgrc.yaml file or set the environment variable.'
        )
    source_folder = Path(pmg_vasp_psp_dir) / funcdir
    if not source_folder.exists():
        raise click.ClickException(f'The source folder {source_folder} does not exist.')

    with temporary_folder() as temp_folder:
        convert_pymatgen_potcar_folder(str(source_folder), temp_folder)
        PotcarData.upload_potcar_family(
            temp_folder, resolved_name, resolved_description, stop_if_existing=stop_if_existing, dry_run=False
        )
    return resolved_name


def _scan_pymatgen_potcars() -> dict[str, Path]:
    """Scan pymatgen's configured POTCAR directory for available functionals."""
    try:
        from pymatgen.io.vasp.inputs import SETTINGS, PotcarSingle
    except ImportError:
        return {}

    pmg_vasp_psp_dir = SETTINGS.get('PMG_VASP_PSP_DIR')
    if pmg_vasp_psp_dir is None:
        return {}

    root = Path(pmg_vasp_psp_dir)
    available = {}
    for functional in FUNCTIONAL_CHOICES:
        funcdir = PotcarSingle.functional_dir.get(functional)
        if not funcdir:
            continue
        folder = root / funcdir
        if folder.exists():
            available[functional] = folder
    return available


def _maybe_setup_potcars(
    potcar_path: str | None,
    potcar_from_pymatgen: bool,
    potcar_functional: str,
    potcar_name: str | None,
    potcar_description: str | None,
    no_potcars: bool,
) -> str | None:
    """Optionally import a POTCAR family during setup."""
    if potcar_path and potcar_from_pymatgen:
        raise click.ClickException('Use either --potcar-path or --potcar-from-pymatgen, not both.')

    if no_potcars:
        return None

    if potcar_path:
        return _upload_potcar_family(potcar_path, potcar_name, potcar_description)
    if potcar_from_pymatgen:
        available = _scan_pymatgen_potcars()
        if potcar_functional not in available:
            available_names = ', '.join(available) if available else 'none found'
            raise click.ClickException(
                f'Pymatgen POTCAR functional {potcar_functional} is not available ({available_names}).'
            )
        return _upload_potcar_family_from_pymatgen(potcar_functional, potcar_name, potcar_description)

    if not click.confirm('Import a POTCAR family now?', default=True):
        return None

    available = _scan_pymatgen_potcars()
    if available:
        click.echo('Pymatgen POTCAR functionals found:')
        for functional, folder in available.items():
            click.echo(f'- {functional}: {folder}')
    else:
        click.echo('No pymatgen POTCAR setup found.')

    source = click.prompt(
        'POTCAR source', type=click.Choice(['path', 'pymatgen'], case_sensitive=False), default='path'
    )
    if source == 'path':
        local_path = click.prompt('Path to POTCAR folder or archive', type=click.Path(exists=True))
        return _upload_potcar_family(local_path, potcar_name, potcar_description)

    if not available:
        raise click.ClickException(
            'No pymatgen POTCAR setup was found. Set PMG_VASP_PSP_DIR or choose a local POTCAR path.'
        )
    functional = click.prompt(
        'Pymatgen POTCAR functional',
        type=click.Choice(list(available), case_sensitive=True),
        default=potcar_functional if potcar_functional in available else next(iter(available)),
    )
    return _upload_potcar_family_from_pymatgen(functional, potcar_name, potcar_description)


@cmd_aiida_vasp.command('setup-local')
@click.option('--tag', required=False, help='Version or tag suffix to include in code labels, e.g. 642.')
@click.option('--vasp-std', required=False, help='Path to the local vasp_std executable.')
@click.option('--vasp-gam', required=False, help='Path to the local vasp_gam executable.')
@click.option('--vasp-ncl', required=False, help='Path to the local vasp_ncl executable.')
@click.option('--potcar-path', required=False, help='Import a POTCAR family from a local folder or archive.')
@click.option('--potcar-from-pymatgen', is_flag=True, help='Import a POTCAR family from the current pymatgen setup.')
@click.option(
    '--potcar-functional',
    type=click.Choice(FUNCTIONAL_CHOICES),
    default='PBE_64',
    show_default=True,
    help='Functional to import when using --potcar-from-pymatgen.',
)
@click.option('--potcar-name', required=False, help='Override the POTCAR family label.')
@click.option('--potcar-description', required=False, help='Override the POTCAR family description.')
@click.option('--no-potcars', is_flag=True, help='Skip POTCAR import and suppress the interactive prompt.')
def setup_local(
    tag,
    vasp_std,
    vasp_gam,
    vasp_ncl,
    potcar_path,
    potcar_from_pymatgen,
    potcar_functional,
    potcar_name,
    potcar_description,
    no_potcars,
):
    """Configure local localhost VASP codes for std, gamma-only, and non-collinear executables."""
    computer = _get_localhost()
    if tag is None:
        tag = click.prompt('VASP version/tag for code labels', default='642')
    executable_paths = _resolve_executable_paths(vasp_std, vasp_gam, vasp_ncl)
    non_interactive = bool(tag and vasp_std and vasp_gam and vasp_ncl)
    code_specs = (
        ('vasp-std', executable_paths['vasp_std']),
        ('vasp-gam', executable_paths['vasp_gam']),
        ('vasp-ncl', executable_paths['vasp_ncl']),
    )

    configured_codes = []
    for prefix, executable in code_specs:
        code, _ = _create_or_reuse_installed_code(computer, f'{prefix}-{tag}', executable, non_interactive)
        configured_codes.append(code)

    click.echo('Configured local VASP codes:')
    for code in configured_codes:
        click.echo(f'- {code.full_label} -> {code.filepath_executable}')

    family_name = _maybe_setup_potcars(
        potcar_path,
        potcar_from_pymatgen,
        potcar_functional,
        potcar_name,
        potcar_description,
        no_potcars,
    )
    if family_name:
        click.echo(f'Imported POTCAR family: {family_name}')
