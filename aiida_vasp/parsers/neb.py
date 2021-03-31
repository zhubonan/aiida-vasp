"""
Parser for NEB calculations using VASP compiled with VTST
"""

import traceback
from pathlib import Path

from aiida.common.exceptions import NotExistent
from aiida_vasp.parsers.settings import ParserSettings, ParserDefinitions
from aiida_vasp.parsers.node_composer import NodeComposer, get_node_composer_inputs
from aiida_vasp.parsers.vasp import VaspParser

# pylint: disable=no-member

NEB_NODES = {
    'neb_misc': {
        'link_name': 'neb_misc',
        'type': 'dict',
        'quantities': ['neb_data']
    },
    'misc': {
        'link_name': 'misc',
        'type': 'dict',
        'quantities': [
            'notifications',
            'run_stats',
            'file_parser_warnings',
        ]
    },
    'kpoints': {
        'link_name': 'kpoints',
        'type': 'array.kpoints',
        'quantities': ['kpoints-kpoints'],
    },
    'structure': {
        'link_name': 'structure',
        'type': 'structure',
        'quantities': ['poscar-structure'],  # The output structures are parsed from the POSCAR
    },
    'chgcar': {
        'link_name': 'chgcar',
        'type': 'vasp.chargedensity',
        'quantities': ['chgcar'],
    },
    'wavecar': {
        'link_name': 'wavecar',
        'type': 'vasp.wavefun',
        'quantities': ['wavecar'],
    },
    'site_magnetization': {
        'link_name': 'site_magnetization',
        'type': 'dict',
        'quantities': ['site_magnetization'],
    },
}

DEFAULT_OPTIONS = {
    'add_bands': False,
    'add_chgcar': False,
    'add_dos': False,
    'add_kpoints': False,
    'add_misc': True,
    'add_neb_misc': True,
    'add_structure': True,
    'add_wavecar': False,
    'add_site_magnetization': False,
}


class NEBSettings(ParserSettings):
    """
    Settings for NEB calculations
    """

    NODES = NEB_NODES


class VtstNebParser(VaspParser):
    """
    Parser for parsing NEB calculations performed with VASP compiled with VTST tools.

    The major difference compared with standard VASP calculations is that the output files are placed
    in subfolders. With the only exception being `vasprun.xml` it is not clear what image this file is
    for.
    """
    COMBINED_QUANTITY = ['neb_data']
    COMBINED_NODES = ['neb_misc']

    def __init__(self, node):
        super(VtstNebParser, self).__init__(node)
        try:
            calc_settings = self.node.inputs.settings
        except NotExistent:
            calc_settings = None

        parser_settings = None
        if calc_settings:
            parser_settings = calc_settings.get_dict().get('parser_settings')

        self._settings = NEBSettings(parser_settings, default_settings=DEFAULT_OPTIONS)
        self._definitions = ParserDefinitions(file_parser_set='neb')

    def get_num_images(self):
        """
        Return the number of images
        """
        try:
            nimages = self.node.inputs.parameters['incar']['images']
        except KeyError:
            nimages = None
        return nimages

    def _setup_parsable(self):
        """Setup the parable quantities. For NEB calculations we collpase the folder structure"""
        filenames = {Path(fname).name for fname in self._retrieved_content}
        self._parsable_quantities.setup(retrieved_filenames=list(filenames),
                                        parser_definitions=self._definitions.parser_definitions,
                                        quantity_names_to_parse=self._settings.quantity_names_to_parse)

    def _parse_quantities(self):
        """
        Parse the quantities. This has to be done for each image

        Returns:
            a dictionary with keys like: '01', '02'... and values being the parsed quantities for each image
        """
        nimages = self.get_num_images()

        per_image_quantities = {}
        per_image_failed_quantities = {}

        for image_idx in range(1, nimages + 1):
            quantities, failed = self._parse_quantities_for_image(image_idx)
            per_image_quantities[f'{image_idx:02d}'] = quantities
            per_image_failed_quantities[f'{image_idx:02d}'] = failed

        return per_image_quantities, per_image_failed_quantities

    # Override super class methods
    def _parse_quantities_for_image(self, image_idx):
        """
        This method dispatch the parsing to file parsers

        :returns: A tuple of parsed quantities dictionary and a list of quantities failed to obtain due to exceptions
        """
        parsed_quantities = {}
        # A dictionary for catching instantiated file parser objects
        file_parser_instances = {}
        failed_to_parse_quantities = []
        for quantity_key in self._parsable_quantities.quantity_keys_to_parse:
            file_name = self._parsable_quantities.quantity_keys_to_filenames[quantity_key]

            # Full path of the file, including the image folder
            file_path = f'{image_idx:02d}/' + file_name
            file_parser_cls = self._definitions.parser_definitions[file_name]['parser_class']

            # If a parse object has been instantiated, use it.
            if file_parser_cls in file_parser_instances:
                parser = file_parser_instances[file_parser_cls]
            else:
                try:
                    # The next line may except for ill-formated file
                    parser = file_parser_cls(settings=self._settings, exit_codes=self.exit_codes, file_path=self._get_file(file_path))
                except Exception:  # pylint: disable=broad-except
                    parser = None
                    failed_to_parse_quantities.append(quantity_key)
                    print('Cannot instantiate {} for {}, exception {}:'.format(file_parser_cls, quantity_key, traceback.format_exc()))
                    #self.logger.warning('Cannot instantiate {}, exception {}:'.format(quantity_key, traceback.format_exc()))

                file_parser_instances[file_parser_cls] = parser

            # if the parser cannot be instantiated, add the quantity to a list of unavalaible ones
            if parser is None:
                failed_to_parse_quantities.append(quantity_key)
                parsed_quantities[quantity_key] = None
                continue

            # The next line may still except for ill-formated file - some parser load all data at
            # instantiation time, the others may not. See the `BaseFileParser.get_quantity`
            try:
                # The next line may still except for ill-formated file - some parser load all data at
                # instantiation time, the others may not
                parsed_quantity = parser.get_quantity(quantity_key)
            except Exception:  # pylint: disable=broad-except
                parsed_quantity = None
                failed_to_parse_quantities.append(quantity_key)
                #self.logger.warning('Error parsing {} from {}, exception {}:'.format(quantity_key, parser, traceback.format_exc()))
                print('Error parsing {} from {}, exception {}:'.format(quantity_key, parser, traceback.format_exc()))

            if parsed_quantity is not None:
                parsed_quantities[quantity_key] = parsed_quantity

            # Keep track of exit_code, if any
            if parser.exit_code and parser.exit_code.status != 0:
                self._file_parse_exit_codes[str(file_parser_cls)] = parser.exit_code

        return parsed_quantities, failed_to_parse_quantities

    def _compose_nodes(self, parsed_quantities):
        """
        Compose the nodes as required.

        The major difference compared to the standard calculations is that NEB
        calculations have different images and most data are parsed at a per image
        basis. For example, each image would have its own output structure, bands, and
        kpoints.

        However, some output includes the data from images, and needs to be handed separately.
        """
        # Excluded per-image quantities, they should be dispatched in one node
        exclude_list = ['neb_data']

        # Compose nodes for each image and buildthe combined quantity dictionary
        combined_quantities = {}
        for image_idx in range(1, self.get_num_images() + 1):
            quantity_dict = {}
            for key, value in parsed_quantities[f'{image_idx:02d}'].items():
                if key in self.COMBINED_QUANTITY:
                    # Combined excluded data into a single dictionary
                    if key in combined_quantities:
                        combined_quantities[key][f'{image_idx:02d}'] = value
                    else:
                        combined_quantities[key] = {f'{image_idx:02d}': value}
                else:
                    quantity_dict[key] = value

            quantity_dict = {key: value for key, value in parsed_quantities[f'{image_idx:02d}'].items() if image_idx not in exclude_list}
            self._compose_nodes_for_image(quantity_dict, image_idx)

        # Deal with the combined data
        equivalent_quantity_keys = dict(self._parsable_quantities.equivalent_quantity_keys)
        nodes_failed_to_create = []

        for node_name, node_dict in self._settings.output_nodes_dict.items():

            # Deal with only the nodes containing combined data
            if node_name not in self.COMBINED_NODES:
                continue

            inputs = get_node_composer_inputs(equivalent_quantity_keys, combined_quantities, node_dict['quantities'])
            # If the input is empty, we skip creating the node as it is bound to fail
            if not inputs:
                nodes_failed_to_create.append(node_name)
                continue

            # Guard the parsing in case of errors
            try:
                aiida_node = NodeComposer.compose(node_dict['type'], inputs)
            except Exception:  # pylint: disable=broad-except
                nodes_failed_to_create.append(node_dict['link_name'])
                aiida_node = None
                self.logger.warning('Error creating output {} with type {}, exception: {}'.format(node_dict['link_name'], node_dict['type'],
                                                                                                  traceback.format_exc()))
            if aiida_node is not None:
                # Suffix the output name with image id
                link_name = node_dict['link_name']
                # Top level outputs
                self.out(link_name, aiida_node)

    def _compose_nodes_for_image(self, parsed_quantities, image_idx):
        """
        Compose the nodes according to parsed quantities

        :returns: A list of link_names for the nodes that failed to compose
        """
        nodes_failed_to_create = []

        # Get the dictionary of equivalent quantities, and add a special quantity "parser_warnings"
        equivalent_quantity_keys = dict(self._parsable_quantities.equivalent_quantity_keys)
        equivalent_quantity_keys.update({'file_parser_warnings': ['file_parser_warnings']})

        for node_name, node_dict in self._settings.output_nodes_dict.items():
            inputs = get_node_composer_inputs(equivalent_quantity_keys, parsed_quantities, node_dict['quantities'])
            if node_name in self.COMBINED_NODES:
                continue

            # If the input is empty, we skip creating the node as it is bound to fail
            if not inputs:
                nodes_failed_to_create.append(node_name)
                continue

            # Guard the parsing in case of errors
            try:
                aiida_node = NodeComposer.compose(node_dict['type'], inputs)
            except Exception:  # pylint: disable=broad-except
                nodes_failed_to_create.append(node_dict['link_name'])
                aiida_node = None
                self.logger.warning('Error creating output {} with type {}, exception: {}'.format(node_dict['link_name'], node_dict['type'],
                                                                                                  traceback.format_exc()))

            if aiida_node is not None:
                # Suffix the output name with image id
                link_name = node_dict['link_name']
                self.out(f'{link_name}.image_{image_idx:02d}', aiida_node)

        return nodes_failed_to_create
