"""
Test the export command.
"""

from pathlib import Path

import pytest
from aiida import orm
from aiida.cmdline.utils import echo
from click import ClickException
from click.testing import CliRunner

from aiida_vasp.commands.tools import select_calcjob_from_work, tools
from aiida_vasp.utils import export as export_mod


def run_cmd(command=None, args=None, **kwargs):
    """Run verdi data vasp.tools <command> [args]."""
    runner = CliRunner()
    params = args or []
    if command:
        params.insert(0, command)
    return runner.invoke(tools, params, **kwargs)


# TODO - add tests for other commands
# Combine export test with workflow execution test to save time
@pytest.mark.parametrize(['vasp_structure', 'vasp_kpoints'], [('str', 'mesh')], indirect=True)
def test_uploadfamily_withpath(aiida_profile_clean, tmp_path, run_vasp_process):
    """
    Test export vasp calculation
    """

    _, node = run_vasp_process(test_case='exit_codes/converged')
    result = run_cmd(
        'export',
        args=[str(node.pk), str(Path(tmp_path) / str(node.pk))],
    )
    assert result.exit_code == 0

    assert (Path(tmp_path) / f'{node.pk}/INCAR').is_file()
    assert (Path(tmp_path) / f'{node.pk}/OUTCAR').is_file()
    assert (Path(tmp_path) / f'{node.pk}/vasprun.xml').is_file()


def test_select_calcjob_from_work_errors_without_running_calc(monkeypatch):
    """Selecting from a workflow without active calcjobs should fail cleanly."""

    class FakeCalcJob:
        def __init__(self, pk, is_finished):
            self.pk = pk
            self.is_finished = is_finished

    class FakeWorkflow:
        def __init__(self, descendants):
            self.called_descendants = descendants

    monkeypatch.setattr(orm, 'CalcJobNode', FakeCalcJob)
    monkeypatch.setattr(orm, 'WorkChainNode', FakeWorkflow)
    monkeypatch.setattr(echo, 'echo_critical', lambda message: (_ for _ in ()).throw(ClickException(message)))

    wrapped = select_calcjob_from_work(lambda **kwargs: kwargs['calcjob'])

    with pytest.raises(ClickException, match='No running calculations found'):
        wrapped(calcjob=FakeWorkflow([FakeCalcJob(pk=1, is_finished=True)]), index=0)


def test_select_calcjob_from_work_returns_requested_running_calc(monkeypatch):
    """The workflow selector should pick the requested running calcjob."""

    class FakeCalcJob:
        def __init__(self, pk, is_finished):
            self.pk = pk
            self.is_finished = is_finished

    class FakeWorkflow:
        def __init__(self, descendants):
            self.called_descendants = descendants

    monkeypatch.setattr(orm, 'CalcJobNode', FakeCalcJob)
    monkeypatch.setattr(orm, 'WorkChainNode', FakeWorkflow)

    wrapped = select_calcjob_from_work(lambda **kwargs: kwargs['calcjob'])
    selected = wrapped(
        calcjob=FakeWorkflow(
            [
                FakeCalcJob(pk=1, is_finished=False),
                FakeCalcJob(pk=2, is_finished=False),
                FakeCalcJob(pk=3, is_finished=True),
            ]
        ),
        index=1,
    )

    assert selected.pk == 2


def test_export_relax_workchain_filters_and_sorts_calcjobs(monkeypatch, tmp_path):
    """Relax export should only process calcjobs and keep them in PK order."""

    exported = []
    written = []

    class FakeCalcJob:
        def __init__(self, pk):
            self.pk = pk

    class FakeParser:
        def __init__(self, data, precision):
            self.data = data

        def write(self, path):
            written.append(Path(path).name)

    class DummyRelax(export_mod.VaspRelaxWorkChain):
        pass

    fake_workchain = type(
        'FakeWorkChainNode',
        (),
        {
            'process_class': DummyRelax,
            'called_descendants': [object(), FakeCalcJob(3), FakeCalcJob(1)],
            'inputs': type('Inputs', (), {'structure': object()})(),
            'outputs': type('Outputs', (), {'relax': type('Relax', (), {'structure': object()})()})(),
            'label': 'relax',
            'description': 'desc',
            'uuid': 'uuid',
        },
    )()

    monkeypatch.setattr(export_mod, 'CalcJobNode', FakeCalcJob)
    monkeypatch.setattr(export_mod.orm, 'CalcJobNode', FakeCalcJob)
    monkeypatch.setattr(export_mod, 'PoscarParser', FakeParser)
    monkeypatch.setattr(
        export_mod,
        '_export_calculation',
        lambda node, folder, decompress=False, include_potcar=False: exported.append((node.pk, Path(folder).name)),
    )

    export_mod._export_workchain.__wrapped__.__wrapped__(fake_workchain, tmp_path)

    assert exported == [(1, 'relax_calc_000'), (3, 'relax_calc_001')]
    assert written == ['POSCAR', 'POSCAR_RELAXED']
