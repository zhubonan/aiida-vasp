"""
Recovery suggestions for terminal error codes in VASP workchains.

Provides actionable, VASP-domain-specific guidance for common failure scenarios.
Suggestions include AiiDA CLI commands for inspecting failed calculations.
"""

from __future__ import annotations

from typing import Optional

# Exit code 501 sub-scenarios (ERROR_OTHER_INTERVENTION_NEEDED)
_SUGGESTIONS_501: dict[str, str] = {
    'vasp_did_not_execute': (
        'Suggestions: Check that the VASP executable path is correct in the code configuration '
        '(`verdi code show <code_label>`). Verify the scheduler is working by submitting a test job. '
        "Ensure all required modules are loaded in the code's prepend_text. Inspect scheduler outputs: "
        '`verdi calcjob outputcat <pk> _scheduler-stdout.txt` and '
        '`verdi calcjob outputcat <pk> _scheduler-stderr.txt`.'
    ),
    'no_output_structure': (
        'Suggestions: VASP did not produce CONTCAR. This can happen if the calculation crashed early. '
        'Check the OUTCAR and stdout for error messages: `verdi calcjob outputcat <pk> OUTCAR`. '
        'Consider reducing NSW, increasing max_wallclock_seconds, or reducing POTIM.'
    ),
    'walltime': (
        'Suggestions: Increase max_wallclock_seconds in the calculation options. '
        'Consider reducing KPAR to allow more walltime per k-point batch, '
        'or reducing the number of k-points / system size.'
    ),
    'electronic_conv': (
        'Suggestions: First try reducing AMIX and BMIX (e.g. AMIX=0.1, BMIX=0.01), then switch ALGO=All. '
        'If still unconverged, check ENCUT (increase if too low), try different POTCARs (e.g. _sv or _pv), '
        'and for metallic systems adjust smearing (increase SIGMA or use ISMEAR=1). '
        'For magnetic systems, verify initial MAGMOM values. '
        'Check the last few SCF steps: `verdi calcjob outputcat <pk> OUTCAR` and look for "EDIFF".'
    ),
    'ionic_no_structure': (
        'Suggestions: The ionic relaxation crashed without producing a structure. '
        'Check OUTCAR for the last completed step: `verdi calcjob outputcat <pk> OUTCAR`. '
        'Try starting from a better geometry, reducing POTIM, or switching IBRION.'
    ),
    'energy_diff_small': (
        'Suggestions: The total energy change between consecutive restarts is below 1e-5 eV/atom '
        '- the structure is likely energy-converged but forces/stresses may still exceed EDIFFG. '
        'Consider loosening EDIFFG (e.g., -0.01 instead of -0.001), or check whether the forces '
        'and stresses are physically acceptable: `verdi calcjob outputcat <pk> OUTCAR` and search '
        'for "TOTAL-FORCE" and "TOTAL STRESS".'
    ),
    'too_few_iterations': (
        'Suggestions: Less than 5 ionic iterations completed per launch. '
        'Increase max_wallclock_seconds or reduce NCORE to allow more iterations per run.'
    ),
    'energy_increasing': (
        'Suggestions: Energy is increasing without significant volume change. '
        'Try reducing POTIM (to 0.1 or less), switching IBRION (1 to 2 or vice versa), '
        'or checking if atoms are too close together. Inspect forces: '
        '`verdi calcjob outputcat <pk> OUTCAR` and search for "TOTAL-FORCE".'
    ),
    'critical_error': (
        'Suggestions: VASP reported a critical error (e.g., EDDRMM, EDDDAV). '
        "Try reducing AMIX/BMIX, switching ALGO to 'Normal' or 'All', "
        'reducing POTIM, or using different pseudopotentials. '
        'Check the full notification output: `verdi calcjob outputcat <pk> OUTCAR`.'
    ),
}

# Direct exit code suggestions (non-501)
_SUGGESTIONS_EXIT: dict[int, str] = {
    500: (
        'Suggestions: VASP likely crashed before writing results. '
        'Check the retrieved folder for partial outputs: `verdi calcjob outputcat <pk> OUTCAR`. '
        'Verify input parameters are valid.'
    ),
    502: (
        'Suggestions: Typically a walltime issue. Increase max_wallclock_seconds, '
        'reduce KPAR, reduce NSW, or reduce system size.'
    ),
    503: (
        'Suggestions: First reduce AMIX and BMIX (e.g. AMIX=0.1, BMIX=0.01), then switch ALGO=All. '
        'If still unconverged, increase NELM (200-400), increase ENCUT, use denser k-mesh, '
        'or try different POTCARs. For metals: increase SIGMA or use ISMEAR=1. '
        'Check convergence history: `verdi calcjob outputcat <pk> OUTCAR` and search for "E0=".'
    ),
    504: (
        'Suggestions: Increase NSW, loosen EDIFFG, reduce POTIM, or switch IBRION. '
        'Check the ionic convergence history: `verdi calcjob outputcat <pk> OUTCAR` and search for "F=".'
    ),
    505: (
        'Suggestions: At least one ionic step had unconverged electronics. '
        'Increase NELM, reduce AMIX, or switch ALGO. You may set ignore_transient_nelm_breach '
        'in settings if the final step is converged.'
    ),
    600: (
        'Suggestions: The structure is still changing after max iterations. '
        'Try increasing the maximum number of relaxation cycles, loosening convergence criteria, '
        'or starting from a better initial geometry. '
        'Check the convergence trajectory: `verdi calcjob outputcat <pk> OUTCAR`.'
    ),
    601: (
        'Suggestions: The final static calculation shows higher forces than the relaxed structure. '
        'This may indicate the electronic solver converged to a different solution. '
        'Check MAGMOM values and consider tightening EDIFF. Compare forces: '
        '`verdi calcjob outputcat <pk> OUTCAR` and search for "TOTAL-FORCE".'
    ),
}


def get_error_suggestion(exit_status: int, scenario: Optional[str] = None) -> str:
    """Return an actionable recovery suggestion for a given error scenario.

    :param exit_status: The numeric exit code of the failed process.
    :param scenario: For exit code 501, a scenario key identifying the specific failure
        context (e.g. ``'electronic_conv'``, ``'walltime'``). Ignored for other exit codes.
    :returns: A human-readable suggestion string, or an empty string if no suggestion exists.
    """
    if exit_status == 501 and scenario is not None:
        suggestion = _SUGGESTIONS_501.get(scenario)
        if suggestion is not None:
            return suggestion

    suggestion = _SUGGESTIONS_EXIT.get(exit_status)
    if suggestion is not None:
        return suggestion

    return ''
