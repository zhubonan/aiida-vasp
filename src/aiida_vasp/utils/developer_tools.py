"""
Tools for developers of this plugin
"""

from aiida import orm
from aiida.common.exceptions import NotExistent
from parsevasp.stream import Stream
from tqdm import tqdm


def scan_for_vasp_errors(only_none_zero=False, print_yaml=False):
    """
    Scan uncaptured VASP errors
    :param only_none_zero: Only print errors with exit_status != 0
    :param print_yaml: Print the YAML representation of the errors
    :return: A dictionary of errors
    """

    q = orm.QueryBuilder()
    q.append(
        orm.CalcJobNode,
        filters={
            'attributes.process_label': 'VaspCalculation',
            'attributes.exit_status': {'!==': 0},
        },
        tag='calc',
        project=['*', 'attributes.exit_status'],
    )
    q.append(orm.FolderData, tag='retrieved', project=['*'])
    ntotal = q.count()
    generic_errors = {}
    for calc, exit_status, retrieved in tqdm(q.iterall(), total=ntotal):
        if only_none_zero and exit_status == 0:
            continue
        try:
            with retrieved.base.repository.open('vasp_output', 'r') as handle:
                parser = Stream(file_handler=handle)
        except NotExistent:
            print(f'No vasp output found for {calc.uuid}')
        for item in parser.entries:
            if item.shortname == 'generic_box_error':
                generic_errors[item.regex.pattern] = item
    if print_yaml:
        for item in generic_errors.values():
            print(f'{item.shortname}:')
            print(f'  kind: {item.kind}')
            print(f'  location: {item.location}')
            print(f'  regex: {item.regex.pattern}')
            print(f'  message: {item.message}')
            print(f'  recover: {item.recover}')
            print(f'  suggestion: {item.suggestion}')

    return generic_errors
