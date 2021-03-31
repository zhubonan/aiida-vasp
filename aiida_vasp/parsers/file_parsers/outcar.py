"""
OUTCAR parser.

--------------
The file parser that handles the parsing of OUTCAR files.
"""
import re
from pathlib import Path

from parsevasp.outcar import Outcar
from aiida_vasp.parsers.file_parsers.parser import BaseFileParser, SingleFile
from aiida_vasp.parsers.node_composer import NodeComposer, get_node_composer_inputs_from_file_parser

DEFAULT_OPTIONS = {'quantities_to_parse': ['elastic_moduli', 'symmetries']}


class OutcarParser(BaseFileParser):
    """
    Interface to parsevasp's OUTCAR parser.

    The quantities listed here are not yet ejected in the xml file:
    - symmetries
    - elastic moduli

    And we can thus not fully rely on the xml parser.

    No possibilities to write OUTCAR files have been implemented.

    """

    PARSABLE_ITEMS = {
        'elastic_moduli': {
            'inputs': [],
            'name': 'elastic_moduli',
            'prerequisites': []
        },
        'symmetries': {
            'inputs': [],
            'name': 'symmetries',
            'prerequisites': []
        },
        'symmetries_extended': {
            'inputs': [],
            'name': 'symmetries',
            'prerequisites': []
        },
        'magnetization': {
            'inputs': [],
            'name': 'magnetization',
            'prerequisites': []
        },
        'site_magnetization': {
            'inputs': [],
            'name': 'site_magnetization',
            'prerequisites': []
        },
        'run_stats': {
            'inputs': [],
            'name': 'run_stats',
            'prerequisites': [],
        }
    }

    def __init__(self, *args, **kwargs):
        """
        Initialize OUTCAR parser

        file_path : str
            File path.
        data : SingleFileData
            AiiDA Data class install to store a single file.
        settings : ParserSettings

        """

        super(OutcarParser, self).__init__(*args, **kwargs)
        self._outcar = None
        self._settings = kwargs.get('settings', None)
        if 'file_path' in kwargs:
            self._init_outcar(kwargs['file_path'])
        if 'data' in kwargs:
            self._init_outcar(kwargs['data'].get_file_abs_path())

    def _init_outcar(self, path):
        """Init with a filepath."""
        self._parsed_data = {}
        self._parsable_items = self.__class__.PARSABLE_ITEMS
        self._data_obj = SingleFile(path=path)

        # Since OUTCAR can be fairly large, we will parse it only
        # once and store the parsevasp Outcar object.
        try:
            self._outcar = Outcar(file_path=path, logger=self._logger)
        except SystemExit:
            self._logger.warning('Parsevasp exited abruptly. Returning None.')
            self._outcar = None

    def _parse_file(self, inputs):

        # Since all quantities will be returned by properties, we can't pass
        # inputs as a parameter, so we store them in self._parsed_data
        for key, value in inputs.items():
            self._parsed_data[key] = value

        quantities_to_parse = DEFAULT_OPTIONS.get('quantities_to_parse')
        if self._settings is not None and self._settings.quantity_names_to_parse:
            quantities_to_parse = self._settings.quantity_names_to_parse

        result = {}

        if self._outcar is None:
            # parsevasp threw an exception, which means OUTCAR could not be parsed.
            for quantity in quantities_to_parse:
                if quantity in self.parsable_items:
                    result[quantity] = None
            return result

        for quantity in quantities_to_parse:
            if quantity in self.parsable_items:
                result[quantity] = getattr(self, quantity)

        return result

    @property
    def run_stats(self):
        """Fetch the run statistics"""
        return self._outcar.get_run_stats()

    @property
    def symmetries(self):
        """Fetch the symmetries, but only the point group (if it exists)."""
        extended = self.symmetries_extended
        sym = {
            'point_group': extended['point_group'],
            'primitive_translations': extended['primitive_translations'],
            'num_space_group_operations': extended['num_space_group_operations']
        }

        return sym

    @property
    def symmetries_extended(self):
        """Fetch the symmetries, including operations etc."""
        sym = self._outcar.get_symmetry()
        # We do not want to store the site symmetry at origin
        sym = {key: value for key, value in sym.items() if key != 'site_symmetry_at_origin'}
        return sym

    @property
    def elastic_moduli(self):
        """Fetch the elastic moduli."""
        return self._outcar.get_elastic_moduli()

    @property
    def magnetization(self):
        """Fetch the full cell magnetization."""
        return self._outcar.get_magnetization()['full_cell']

    @property
    def site_magnetization(self):
        """Fetch the site dependent magnetization."""
        return self._outcar.get_magnetization()


class LegacyOutcarParser(BaseFileParser):
    """
    Parse OUTCAR into a dictionary, which is supposed to be turned into Dict later.

    For constructor params and more details check the documentation for ``aiida_vasp.parsers.file_parsers.parser`` and
    ``aiida_vasp.parsers.file_parsers.parser.BaseParser``.
    """

    FILE_NAME = 'OUTCAR'
    PARSABLE_ITEMS = {
        'outcar-volume': {
            'inputs': [],
            'name': 'volume',
            'prerequisites': []
        },
        'outcar-energies': {
            'inputs': [],
            'name': 'energies',
            'prerequisites': []
        },
        'outcar-efermi': {
            'inputs': [],
            'name': 'fermi_level',
            'prerequisites': []
        },
        'symmetries': {
            'inputs': [],
            'name': 'symmetries',
            'prerequisites': []
        }
    }

    SPACE_GROUP_OP_PATTERN = re.compile(r'Found\s*(\d+) space group operations')
    POINT_GROUP_OP_PATTERN = re.compile(r'whereof\s*(\d+) operations')
    POINT_SYMMETRY_PATTERN = re.compile(r'point symmetry (.*?)\s*\.')
    SPACE_GROUP_PATTERN = re.compile(r'space group is (.*?)\s*\.')

    def __init__(self, *args, **kwargs):
        super(LegacyOutcarParser, self).__init__(*args, **kwargs)
        self._parameter = None

    def _parse_file(self, inputs):
        """Add all quantities parsed from OUTCAR to _parsed_data."""
        result = self._read_outcar(inputs)
        return result

    @staticmethod
    def _parse_line_regex_once(line, regex, res_dict, key, convert=None):
        """
        Parse ``line`` with regular expression ``regex`` optionally converts the result and stores int in ``res_dict[key]``.

        Does not overwrite ``res_dict[key]`` if it already exists and is not None.
        """
        if res_dict.get(key, None) is None:
            regex_result = re.findall(regex, line)
            if not regex_result:
                res_dict[key] = None
            else:
                result = regex_result[0]
                if convert:
                    result = convert(result)
                res_dict[key] = result

    def _read_outcar(self, inputs):
        """Parse the OUTCAR file into a dictionary."""
        result = inputs.get('settings', {})
        result = {}
        energy_free = []
        energy_zero = []
        symmetries = {}
        with open(self._data_obj.path, 'r') as outcar_file_object:
            for line in outcar_file_object:
                # volume
                if line.rfind('volume of cell :') > -1:
                    result['outcar-volume'] = float(line.split()[-1])
                # Free energy
                if line.lower().startswith('  free  energy   toten'):
                    energy_free.append(float(line.split()[-2]))
                # Extrapolated zero point energy
                if line.startswith('  energy  without entropy'):
                    energy_zero.append(float(line.split()[-1]))
                # Fermi energy
                if line.rfind('E-fermi') > -1:
                    result['outcar-efermi'] = float(line.split()[2])
                # space group operations
                self._parse_line_regex_once(line, self.SPACE_GROUP_OP_PATTERN, symmetries, 'num_space_group_operations', int)
                # point group operations
                self._parse_line_regex_once(line, self.POINT_GROUP_OP_PATTERN, symmetries, 'num_point_group_operations', int)
                # point symmetry
                self._parse_line_regex_once(line, self.POINT_SYMMETRY_PATTERN, symmetries, 'point_symmetry')
                # space group
                self._parse_line_regex_once(line, self.SPACE_GROUP_PATTERN, symmetries, 'space_group')
        result['outcar-energies'] = {}
        result['outcar-energies']['free_energy'] = energy_free[-1]
        result['outcar-energies']['energy_without_entropy'] = energy_zero[-1]
        result['outcar-energies']['free_energy_all'] = energy_free
        result['outcar-energies']['energy_without_entropy_all'] = energy_zero
        result['symmetries'] = symmetries
        return result

    @property
    def parameter(self):
        if self._parameter is None:
            inputs = get_node_composer_inputs_from_file_parser(self, quantity_keys=DEFAULT_OPTIONS['quantities_to_parse'])
            self._parameter = NodeComposer.compose('dict', inputs)
        return self._parameter


class VtstNEBOutcarParser(OutcarParser):
    """
    Parser for processing OUTCAR generated by VASP with VTST
    """

    PARSABLE_ITEMS = {'neb_data': {'inputs': [], 'name': 'neb_data', 'prerequisites': []}, **OutcarParser.PARSABLE_ITEMS}

    def __init__(self, *args, **kwargs):
        """
        Instantiate the parser
        """
        self._neb_data = {}
        super().__init__(*args, **kwargs)

    def _init_outcar(self, path):
        """
        Init with a filepath.
        """
        super()._init_outcar(path)
        # Parse additional OUTCAR data
        self._parse_neb_outputs(path)

    def _parse_neb_outputs(self, path):  # pylint: disable=too-many-branches
        """
        Scan for NEB output in the OUTCAR content
        """
        if isinstance(path, (str, Path)):
            with open(path, 'r') as fobj:
                lines = fobj.readlines()
        # A file-like object
        elif hasattr(path, 'readlines'):
            lines = path.readlines()

        vtst_data = {'neb_converged': False}
        for idx, line in enumerate(lines):
            if 'VTST: version' in line:
                vtst_data['version'] = line.split(':')[1].strip()

            if 'NEB: Tangent' in line:
                tangents = []
                for isub in range(idx + 2, idx + 99999):
                    subline = lines[isub]
                    if subline.strip():
                        tangents.append([float(tmp) for tmp in subline.split()])
                    else:
                        break
                vtst_data['tangents'] = tangents
            elif 'NEB: forces' in line:
                forces = [float(tmp) for tmp in line.split()[-3:]]
                vtst_data['force_par_spring'] = forces[0]
                vtst_data['force_prep_real'] = forces[1]
                vtst_data['force_dneb'] = forces[2]
            elif 'stress matrix after NEB project' in line:
                stress = []
                for isub in range(idx + 1, idx + 4):
                    stress.append([float(tmp) for tmp in lines[isub].split()])
                vtst_data['stress_matrix'] = stress
            elif 'FORCES: max atom' in line:
                forces = [float(tmp) for tmp in line.split()[-2:]]
                vtst_data['force_max_atom'] = forces[0]
                vtst_data['force_rms'] = forces[1]
            elif 'FORCE total and by dimension' in line:
                forces = [float(tmp) for tmp in line.split()[-2:]]
                vtst_data['force_total'] = forces[0]
                vtst_data['force_by_dimension'] = forces[1]
            elif 'Stress total and by dimension' in line:
                forces = [float(tmp) for tmp in line.split()[-2:]]
                vtst_data['stress_total'] = forces[0]
                vtst_data['stress_by_dimension'] = forces[1]
            elif 'OPT: skip step - force has converged' in line:
                vtst_data['neb_converged'] = True
            elif 'energy(sigma->0)' in line:
                tokens = line.split()
                vtst_data['energy_extrapolated'] = float(tokens[-1])
                vtst_data['energy_without_entropy'] = float(tokens[-4])
            elif 'free  energy   TOTEN' in line:
                vtst_data['free_energy'] = float(line.split()[-2])
        self._neb_data = vtst_data

    @property
    def neb_data(self):
        """
        Parsed NEB results
        """
        return self._neb_data
