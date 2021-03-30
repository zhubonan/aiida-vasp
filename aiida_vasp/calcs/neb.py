"""
Module for settings up NEB calculations
"""
from pathlib import Path

from aiida_vasp.utils.aiida_utils import get_data_class
from aiida_vasp.calcs.vasp import VaspCalculation
from aiida_vasp.parsers.file_parsers.poscar import PoscarParser


class VaspNEBCalculation(VaspCalculation):
    """
    NEB calculations using VASP

    ------------------------------------
    Calculations for performing NEB calculations.
    NEB calculations requires standard VASP inputs, but POSCAR are placed in
    folder names 00, 01, 02... N for N-1 number of images.

    Input frames should be placed under the ``neb_images`` input namespace as a dictionary like::
      {
          'image_00': structure_1,
          'image_01': structure_2
          ....
      }

    Output of individual frames are placed in the corresponding namespace under the same convention.
    """
    _VASP_OUTPUT = 'vasp_output'
    _ALWAYS_RETRIEVE_LIST = ['OUTCAR', 'vasprun.xml', _VASP_OUTPUT]
    _PER_IMAGE_ALWAYS_RETRIEVE_LIST = ['OUTCAR', 'CONTCAR']
    _query_type_string = 'vasp.neb'
    _plugin_type_string = 'vasp.neb'

    @classmethod
    def define(cls, spec):

        super(VaspNEBCalculation, cls).define(spec)

        # Define the inputs.
        # options is passed automatically.
        spec.input('parameters', valid_type=get_data_class('dict'), help='The VASP input parameters (INCAR).')
        spec.input('dynamics',
                   valid_type=get_data_class('dict'),
                   help='The VASP parameters related to ionic dynamics, e.g. flags to set the selective dynamics',
                   required=False)
        spec.input('initial_structure',
                   valid_type=(get_data_class('structure'), get_data_class('cif')),
                   help='The input structure (POSCAR) for initial image.')
        spec.input('final_structure',
                   valid_type=(get_data_class('structure'), get_data_class('cif')),
                   help='The input structure (POSCAR) for the final image.')
        spec.input_namespace('neb_images',
                             valid_type=(get_data_class('structure'), get_data_class('cif')),
                             help='Starting structure for the NEB images',
                             dynamic=True)
        # Need namespace on this as it should also accept keys that are of `kind`. These are unknown
        # until execution.
        spec.input_namespace('potential', valid_type=get_data_class('vasp.potcar'), help='The potentials (POTCAR).', dynamic=True)
        spec.input('kpoints', valid_type=get_data_class('array.kpoints'), help='The kpoints to use (KPOINTS).')
        spec.input_namespace('charge_density',
                             dynamic=True,
                             valid_type=get_data_class('vasp.chargedensity'),
                             required=False,
                             help='The charge density. (CHGCAR)')
        spec.input_namespace('wavefunctions',
                             valid_type=get_data_class('vasp.wavefun'),
                             dynamic=True,
                             required=False,
                             help='The wave function coefficients. (WAVECAR)')
        spec.input('settings', valid_type=get_data_class('dict'), required=False, help='Additional parameters not related to VASP itself.')
        spec.input('metadata.options.parser_name', default='vasp.vasp')

        # Define outputs.
        # remote_folder and retrieved are passed automatically
        spec.output_namespace('misc',
                              valid_type=get_data_class('dict'),
                              help='The output parameters containing smaller quantities that do not depend on system size.')
        spec.output_namespace('neb_images', required=True, valid_type=get_data_class('structure'), help='NEB images')
        spec.output_namespace('chgcar', valid_type=get_data_class('vasp.chargedensity'), required=False, help='The output charge density.')
        spec.output_namespace('wavecar',
                              valid_type=get_data_class('vasp.wavefun'),
                              required=False,
                              help='The output file containing the plane wave coefficients.')
        spec.output_namespace('site_magnetization',
                              valid_type=get_data_class('dict'),
                              required=False,
                              help='The output of the site magnetization')
        spec.exit_code(0, 'NO_ERROR', message='the sun is shining')
        spec.exit_code(350, 'ERROR_NO_RETRIEVED_FOLDER', message='the retrieved folder data node could not be accessed.')
        spec.exit_code(351,
                       'ERROR_NO_RETRIEVED_TEMPORARY_FOLDER',
                       message='the retrieved_temporary folder data node could not be accessed.')
        spec.exit_code(352, 'ERROR_CRITICAL_MISSING_FILE', message='a file that is marked by the parser as critical is missing.')
        spec.exit_code(333,
                       'ERROR_VASP_DID_NOT_EXECUTE',
                       message='VASP did not produce any output files and did likely not execute properly.')
        spec.exit_code(1001, 'ERROR_PARSING_FILE_FAILED', message='parsing a file has failed.')
        spec.exit_code(1002, 'ERROR_NOT_ABLE_TO_PARSE_QUANTITY', message='the parser is not able to parse the {quantity} quantity')
        spec.exit_code(
            1003,
            'ERROR_RECOVERY_PARSING_OF_XML_FAILED',
            message=
            'the vasprun.xml was truncated and recovery parsing failed to parse at least one of the requested quantities: {quantities}, '
            'very likely the VASP calculation did not run properly')

    def prepare_for_submission(self, tempfolder):
        """
        Add all files to the list of files to be retrieved.

        Notice that we here utilize both the retrieve batch of files, which are always stored after retrieval and
        the temporary retrieve list which is automatically cleared after parsing.
        """
        calcinfo = super().prepare_for_submission(self, tempfolder)

        nimages = len(self.inputs.neb_images)
        nimage_keys = sorted(list(self.inputs.neb_images.keys()))

        image_folders = []

        # Iterate though each folder that needs to be setup
        for i in range(nimages + 2):
            folder_id = f'{i:02d}'
            folder = Path(tempfolder.get_abs_path(folder_id))
            folder.mkdir()
            poscar = str(folder / 'POSCAR')
            if i == 0:
                # Write the initial image
                self.write_neb_poscar(self.inputs.initial_structure, poscar)
            elif i == nimages + 1:
                # Write the final image
                self.write_neb_poscar(self.inputs.final_structure, poscar)
            else:
                # Write NEB images
                img_key = nimage_keys[i - 1]
                self.write_neb_poscar(self.inputs.neb_images[img_key], poscar)
                image_folders.append(folder_id)

                # Link with singlefile WAVECAR/CHGCAR is needed
                if self._need_wavecar():
                    wavecar = self.inputs.wave_functions[img_key]
                    dst = folder_id + '/' + 'WAVECAR'
                    calcinfo.local_copy_list.append((wavecar.uuid, wavecar.filename, dst))

                if self._need_chgcar():
                    chgcar = self.inputs.charge_densities[img_key]
                    dst = folder_id + '/' + 'CHGCAR'
                    calcinfo.local_copy_list.append((chgcar.uuid, chgcar.filename, dst))
        try:
            store = self.inputs.settings.get_attribute('ALWAYS_STORE', default=True)
        except AttributeError:
            store = True

        try:
            additional_retrieve_list = self.inputs.settings.get_attribute('PER_IMAGE_ADDITIONAL_RETRIEVE_LIST', default=[])
        except AttributeError:
            additional_retrieve_list = []
        try:
            additional_retrieve_temp_list =\
                self.inputs.settings.get_attribute('PER_IMAGE_ADDITIONAL_RETRIEVE_TEMPORARY_LIST', default=[])  # pylint: disable=invalid-name
        except AttributeError:
            additional_retrieve_temp_list = []

        if store:
            calcinfo.retrieve_list.extend(
                list(set(image_folder_paths(image_folders, self._PER_IMAGE_ALWAYS_RETRIEVE_LIST + additional_retrieve_list))))
            calcinfo.retrieve_temporary_list.extend(image_folder_paths(image_folders, additional_retrieve_temp_list))
        else:
            calcinfo.retrieve_temporary_list.extend(
                list(set(image_folder_paths(image_folders, self._PER_IMAGE_ALWAYS_RETRIEVE_LIST + additional_retrieve_temp_list))))
            calcinfo.retrieve_list.extend(image_folder_paths(image_folders, additional_retrieve_list))

        return calcinfo

    def write_neb_poscar(self, structure, dst, positions_dof=None):  # pylint: disable=unused-argument
        """
        Write the POSCAR.

        Passes the structures node (StructureData) to the POSCAR parser for
        preparation and writes to dst.

        :param dst: absolute path of the file to write to
        """
        settings = self.inputs.get('settings')
        settings = settings.get_dict() if settings else {}
        poscar_precision = settings.get('poscar_precision', 10)
        if positions_dof is not None:
            options = {'positions_dof': positions_dof}
        else:
            options = None
        poscar_parser = PoscarParser(data=structure, precision=poscar_precision, options=options)
        poscar_parser.write(dst)


def image_folder_paths(image_folders, retrieve_names):
    """
    Return a list of folders paths to be retrieved
    """
    retrieve_list = []
    for key in retrieve_names:
        for fdname in image_folders:
            retrieve_list.append(fdname + '/' + key)
    return retrieve_list
