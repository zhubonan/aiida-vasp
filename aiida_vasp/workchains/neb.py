"""
VASP NEB workchain.

---------------
Contains the VaspNEBWorkChain class definition which uses the BaseRestartWorkChain.
"""
import numpy as np
from aiida.engine import while_
from aiida import orm

from aiida.common.extendeddicts import AttributeDict
from aiida.common.exceptions import NotExistent, InputValidationError
from aiida.plugins import CalculationFactory
from aiida.engine.processes.workchains.restart import BaseRestartWorkChain, process_handler, ProcessHandlerReport

from aiida_vasp.utils.aiida_utils import get_data_class, get_data_node
from aiida_vasp.utils.workchains import compose_exit_code
from aiida_vasp.assistant.parameters import ParametersMassage
from aiida_vasp.parsers.file_parsers.potcar import MultiPotcarIo


class VaspNEBWorkChain(BaseRestartWorkChain):
    """
    The NEB workchain.

    -------------------
    Error handling enriched wrapper around VaspNEBCalculation.

    Deliberately conserves most of the interface (required inputs) of the VaspNEBCalculation class, but
    makes it possible for a user to interact with a workchain and not a calculation.

    In addition, implement restarts of calculation when the calculation is net full converged for error handling.

    """
    _verbose = False
    _process_class = CalculationFactory('vasp.neb')
    _norm_disp_threshold = 1.0

    @classmethod
    def define(cls, spec):
        super(VaspNEBWorkChain, cls).define(spec)
        spec.expose_inputs(cls._process_class, exclude=('potential', 'kpoints'))
        spec.input('kpoints', valid_type=get_data_class('array.kpoints'), required=False)
        spec.input('kpoints_spacing', valid_type=get_data_class('float'), required=False)
        spec.input('potential_family', valid_type=get_data_class('str'), required=True)
        spec.input('potential_mapping', valid_type=get_data_class('dict'), required=True)
        spec.input('options', valid_type=get_data_class('dict'), required=True)
        spec.input('max_iterations',
                   valid_type=get_data_class('int'),
                   required=False,
                   default=lambda: get_data_node('int', 5),
                   help="""
            The maximum number of iterations to perform.
            """)
        spec.input('clean_workdir',
                   valid_type=get_data_class('bool'),
                   required=False,
                   default=lambda: get_data_node('bool', False),
                   help="""
            If True, clean the work dir upon the completion of a successfull calculation.
            """)
        spec.input('verbose',
                   valid_type=get_data_class('bool'),
                   required=False,
                   default=lambda: get_data_node('bool', True),
                   help="""
            If True, enable more detailed output during workchain execution.
            """)
        spec.input('dynamics.positions_dof',
                   valid_type=get_data_class('list'),
                   required=False,
                   help="""
            Site dependent flag for selective dynamics when performing relaxation
            """)
        spec.input('ldau_mapping',
                   valid_type=get_data_class('dict'),
                   required=False,
                   help="Mappings, see the doc string of 'get_ldau_keys'")
        spec.input('kpoints_spacing',
                   valid_type=get_data_class('float'),
                   required=False,
                   help='Spacing for the kpoints in units A^-1 * 2pi (CASTEP style `kpoints_mp_spacing`)')
        spec.input('kpoints_spacing_vasp',
                   valid_type=get_data_class('float'),
                   required=False,
                   help='Spacing for the kpoints in units A^-1 (VASP style)')
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                #cls.prepare_inputs,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )  # yapf: disable

        spec.expose_outputs(cls._calculation)
        spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        spec.exit_code(700, 'ERROR_NO_POTENTIAL_FAMILY_NAME', message='the user did not supply a potential family name')
        spec.exit_code(701, 'ERROR_POTENTIAL_VALUE_ERROR', message='ValueError was returned from get_potcars_from_structure')
        spec.exit_code(702, 'ERROR_POTENTIAL_DO_NOT_EXIST', message='the potential does not exist')
        spec.exit_code(703, 'ERROR_IN_PARAMETER_MASSAGER', message='the exception: {exception} was thrown while massaging the parameters')

    def setup(self):

        super().setup()

        # Setup the initial inputs
        self.ctx.inputs = self.exposed_inputs(self._process_class)

        # Stage the neb images
        self.ctx.neb_images = self.inputs.neb_images

        # Handle and convert additional inputs and store them in self.ctx.inputs
        self._setup_vasp_inputs()

        # Sanity checks
        self._check_neb_inputs()

    # def prepare_inputs(self):
    #     """
    #     Prepare the inputs stored under self.ctx.inputs
    #     """

    #     # Applied the staged NEB images
    #     self.ctx.inputs.neb_images = self.ctx.neb_images

    @process_handler(priority=50)
    def handle_unconverged(self, node):
        """
        Handle the problem where the NEB optimisation is not converged.

        Note that VASP could reach NSW before the actual convergence.
        Hence this check is necessary even for finished runs.
        """
        if 'neb_misc' not in node.outputs:
            self.report('Cannot found the `neb_misc` output containing the NEB run data')
            return None
        neb_misc = node.outputs.neb_misc

        if not neb_misc.get('neb_data'):
            self.report('Cannot found the `neb_data` dictioanry containing the NEB run data')
            return None

        neb_data = neb_misc.get('neb_data')

        converged = [tmp['neb_converged'] for tmp in neb_data.values()]
        if not all(converged):
            self.report('At least one image is not converged in the run. Restart required.')
            self._attach_output_structure(node)

            return ProcessHandlerReport()
        return None

    @process_handler(priority=100)
    def handle_unfinished(self, node):
        """
        Handle the case where the calculations is not fully finished.
        This checks the existing of the run_stats field in the parsed per-image misc output
        """
        if 'misc__image_01' not in node.outputs:
            self.report('Cannot found the `misc` output containing the parsed per-image data')
            return None

        finished = []
        for key in node.outputs:
            if key.startswith('misc__'):
                misc = node.outputs[key].get_dict()
                if 'run_stats' in misc:
                    finished.append(True)
                else:
                    finished.append(False)
        if not all(finished):
            self.report('At least one image did not reach the end of VASP execution.')

            self._attach_output_structure(node)

            # No further process handling is needed
            return ProcessHandlerReport(do_break=True)
        return None

    def _attach_output_structure(self, node):
        """
        Attached the output structure of a children node as the inputs for the
        next workchain launch.
        """
        output_images = AttributeDict()
        for key in node.outputs:
            if key.startswith('structure__'):
                output_images[key.split('__')[1]] = node.outputs[key]
        self.ctx.inputs.neb_images = output_images

    def _check_neb_inputs(self):
        """
        Perform some simple checks for the NEB inputs

        This method is called once by ``self.setup``
        """

        incar = self.inputs.parameters.get_dict()

        images = incar.get('images')

        if not images:
            raise InputValidationError('IMAGES parameters is not set in the INCAR inputs')

        nimages = len(self.ctx.inputs.neb_images)

        if nimages != images:
            raise InputValidationError('Mismatch between IMAGES and actual number supplied input structures.')

        # Check for NEB tags
        iopt = incar.get('iopt', 0)
        ibrion = incar.get('ibrion')
        potim = incar.get('potim')

        # Check the sanity of parameters
        if ibrion != 3:
            self.report('WARNING: IBRION should be set to 3 for VTST runs, proceed with caution.')
        elif potim != 0:
            self.report('WARNING: Using VTST optimisors with IBRION=3, but POTIM is not set to zero, proceed with caution.')
        if iopt == 0:
            self.report('WARNING: IOPT not set.')

        if ibrion == 2:
            raise InputValidationError('IBRION=2 should not be used for NEB optimisation!!')

        # Check the displacement of atoms between the frames
        # the hope is that this may detect simple errors such as atoms going across the PBC or
        # the order of atoms are changed between different frames

        tmp = list(self.ctx.inputs.neb_images.items())
        tmp.sort(key=lambda x: x[0])
        frames = [x[1].get_ase() for x in tmp]
        frames = [self.ctx.inputs.initial_structure.get_ase()] + frames + [self.ctx.inputs.final_structure.get_ase()]

        last_frame = frames[0]
        for iframe, frame in enumerate(frames[1:]):
            disp = frame.positions - last_frame.positions
            norm_disp = np.linalg.norm(disp, axis=1)
            sort_idx = np.argsort(norm_disp)
            if norm_disp[sort_idx[-1]] > self._norm_disp_threshold:
                raise InputValidationError('Large displacement detected for atom {} at frame {} - please check the inputs images'.format(
                    sort_idx[-1], iframe + 1))
            last_frame = frame

    def _setup_vasp_inputs(self):
        """
        Setup the inputs for VASP calculation

        This method is called once by ``self.setup``
        """

        # Set the kpoints (kpoints)
        if 'kpoints' in self.inputs:
            self.ctx.inputs.kpoints = self.inputs.kpoints
        elif 'kpoints_spacing' in self.inputs:
            kpoints = orm.KpointsData()
            kpoints.set_cell_from_structure(self.ctx.inputs.structure)
            kpoints.set_kpoints_mesh_from_density(self.inputs.kpoints_spacing.value * np.pi * 2)
            self.ctx.inputs.kpoints = kpoints
        elif 'kpoints_spacing_vasp' in self.inputs:
            kpoints = orm.KpointsData()
            kpoints.set_cell_from_structure(self.ctx.inputs.structure)
            kpoints.set_kpoints_mesh_from_density(self.inputs.kpoints_spacing.value)
            self.ctx.inputs.kpoints = kpoints
        else:
            raise InputValidationError("Must supply either 'kpoints' or 'kpoints_spacing' or 'kpoints_spacing_vasp")

        # Set settings
        unsupported_parameters = []
        if 'settings' in self.inputs:
            self.ctx.inputs.settings = self.inputs.settings
            # Also check if the user supplied additional tags that is not in the supported file.
            try:
                unsupported_parameters = self.ctx.inputs.settings.unsupported_parameters
            except AttributeError:
                pass

        # Perform inputs massage to accommodate generalization in higher lying workchains
        # and set parameters.
        unsupported_parameters += ['iopt']
        try:
            parameters_massager = ParametersMassage(self.inputs.parameters, unsupported_parameters)
        except Exception as exception:  # pylint: disable=broad-except
            return self.exit_codes.ERROR_IN_PARAMETER_MASSAGER.format(exception=exception)  # pylint: disable=no-member
        try:
            # Only set if they exists
            # Set any INCAR tags
            self.ctx.inputs.parameters = parameters_massager.parameters.incar
            # Set any dynamics input (currently only for selective dynamics, e.g. custom write to POSCAR)
            self.ctx.inputs.dynamics = parameters_massager.parameters.dynamics
            # Here we could set additional override flags, but those are not relevant for this VASP plugin
        except AttributeError:
            pass

        # Setup LDAU keys
        if 'ldau_mapping' in self.inputs:
            ldau_settings = self.inputs.ldau_mapping.get_dict()
            ldau_keys = get_ldau_keys(self.ctx.inputs.structure, **ldau_settings)
            # Directly update the raw inputs passed to VaspCalculation
            self.ctx.inputs.parameters.update(ldau_keys)

        # Set settings
        if 'settings' in self.inputs:
            self.ctx.inputs.settings = self.inputs.settings

        # Set options
        # Options is very special, not storable and should be
        # wrapped in the metadata dictionary, which is also not storable
        # and should contain an entry for options
        if 'options' in self.inputs:
            options = {}
            options.update(self.inputs.options)
            self.ctx.inputs.metadata = {}
            self.ctx.inputs.metadata['options'] = options
            # Override the parser name if it is supplied by the user.
            parser_name = self.ctx.inputs.metadata['options'].get('parser_name')
            if parser_name:
                self.ctx.inputs.metadata['options']['parser_name'] = parser_name
            # Also make sure we specify the entry point for the
            # Set MPI to True, unless the user specifies otherwise
            withmpi = self.ctx.inputs.metadata['options'].get('withmpi', True)
            self.ctx.inputs.metadata['options']['withmpi'] = withmpi

        # Utilise default input/output selections
        self.ctx.inputs.metadata['options']['input_filename'] = 'INCAR'

        # Set the CalcJobNode to have the same label as the WorkChain
        self.ctx.inputs.metadata['label'] = self.inputs.metadata.get('label', '')

        # Verify and set potentials (potcar)
        if not self.inputs.potential_family.value:
            self.report(  # pylint: disable=not-callable
                'An empty string for the potential family name was detected.')
            return self.exit_codes.ERROR_NO_POTENTIAL_FAMILY_NAME  # pylint: disable=no-member
        try:
            self.ctx.inputs.potential = get_data_class('vasp.potcar').get_potcars_from_structure(
                structure=self.inputs.initial_structure,
                family_name=self.inputs.potential_family.value,
                mapping=self.inputs.potential_mapping.get_dict())
        except ValueError as err:
            return compose_exit_code(self.exit_codes.ERROR_POTENTIAL_VALUE_ERROR.status, str(err))  # pylint: disable=no-member
        except NotExistent as err:
            return compose_exit_code(self.exit_codes.ERROR_POTENTIAL_DO_NOT_EXIST.status, str(err))  # pylint: disable=no-member

        self.ctx.verbose = bool(self.inputs.get('verbose', self._verbose))

        return None


# The code below should be moved for utility module, but I keep them here for now

FELEMS = [
    'La',
    'Ce',
    'Pr',
    'Nd',
    'Pm',
    'Sm',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'Yb',
    'Lu',
    'Ac',
    'Th',
    'Pa',
    'U',
    'Np',
    'Pu',
    'Am',
    'Cm',
    'Bk',
    'Cf',
    'Es',
    'Fm',
    'Md',
    'No',
    'Lr',
]


def get_ldau_keys(structure, mapping, utype=2, jmapping=None, felec=False):
    """
    Setup LDAU mapping. In VASP, the U for each species has to be
    defined in the order that they appear in POSCAR. This is a helper
    to make sure the values of U are associated to each specie

    Arguments:
        structure: the structure, either StructureData or ase.Atoms is fine
        mapping: a dictionary in the format of  {"Mn": [d, 4]...} for U
        utype: the type of LDA+U, default to 2, which is the one with only one parameter
        jmapping: a dictionary in the format of  {"Mn": [d, 4]...} but for J
        felec: Wether we are dealing with f electrons, will increase lmaxmix if we are.


    Returns:
        dict_update: a dictionary to be used to update the raw input parameters for VASP
    """
    if isinstance(structure, orm.StructureData):
        species = MultiPotcarIo.potentials_order(structure)
    else:
        # For ASE atoms, we keep the order of species occurrence no sorting is done
        species = []
        for symbol in structure.get_chemical_symbols():
            if symbol not in species:
                species.append(symbol)

    lsymbols = {'d': 2, 'f': 3, 'p': 1}
    if jmapping is None:
        jmapping = {}
    # Setup U array
    ldauu = []
    ldauj = []
    ldaul = []
    count = 0
    for specie in species:
        if specie in mapping:
            uvalue = mapping[specie][1]
            j = jmapping.get(specie, 0.)
            ldaul.append(lsymbols[mapping[specie][0]])
            ldauu.append(mapping[specie][1])

            j = jmapping.get(specie, 0.)
            ldauj.append(j)

            if specie in FELEMS:
                felec = True
            # Count the number of valid mappings
            if uvalue != 0. or j != 0.:
                count += 1

        else:
            ldauu.append(0.)
            ldauj.append(0.)
            ldaul.append(-1)

    if count > 0:
        # Only enable U is there is any non-zero value
        output = {'ldauu': ldauu, 'ldauj': ldauj, 'ldautype': utype, 'lmaxmix': 6 if felec else 4, 'ldaul': ldaul, 'ldau': True}
    else:
        output = {}
    return output
