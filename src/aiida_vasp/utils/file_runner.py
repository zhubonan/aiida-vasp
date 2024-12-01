"""
Module to run VASP calculations via folder containing the input files.


The idea is to allow initializing a VaspCalculation with an existing folder containing the input files.
"""

from pathlib import Path
from typing import Dict, Optional

from aiida import orm
from aiida.common.extendeddicts import AttributeDict

from aiida_vasp.parsers.content_parsers.incar import IncarParser
from aiida_vasp.parsers.content_parsers.kpoints import KpointsParser
from aiida_vasp.parsers.content_parsers.poscar import PoscarParser
from aiida_vasp.parsers.content_parsers.potcar import MultiPotcarIo


class VaspCalcConverter:
    """
    Convert a VASP calculation folder to an AiiDA VaspWorkChain input.
    """

    def __init__(
        self,
        folder,
        settings: Optional[Dict] = None,
        options: Optional[Dict] = None,
        potential_mapping: Optional[Dict] = None,
        code_string: Optional[str] = None,
        potential_family: Optional[str] = None,
    ):
        """
        Initialize the VaspFolderConverter with a folder path.
        :param folder: The path to the VASP calculation folder.
        :param settings: A dictionary of VASP settings to use for the calculation.
        :param options: A dictionary of VASP options to use for the calculation.
        :param potential_mapping: A dictionary of potential mapping to use for the calculation.
        :param potential_family: The potential family to use for the calculation.
        """
        self._folder = Path(folder)
        self.inputs = AttributeDict()
        self.settings = settings
        self.options = options
        self.potential_mapping = potential_mapping
        self.potential_family = potential_family
        self.code_string = code_string
        self.build_inputs()

    def build_inputs(self):
        """Build the inputs for the VaspWorkChain from the VASP calculation folder."""

        # Parse the INCAR file
        with open(self._folder / 'INCAR', 'r') as fh:
            incar_parser = IncarParser(handler=fh)
            self.inputs.parameters = orm.Dict({'incar': incar_parser.incar})

        # Parse the POSCAR file
        with open(self._folder / 'POSCAR', 'r') as fh:
            poscar_parser = PoscarParser(handler=fh)
            structure_dict = poscar_parser.structure
            node = orm.StructureData()
            node.set_cell(structure_dict['unitcell'])
            for site in structure_dict['sites']:
                node.append_atom(position=site['position'], symbols=site['symbol'], name=site['kind_name'])
            self.inputs.structure = node
        # Parser the kpoint file
        with open(self._folder / 'KPOINTS', 'r') as fh:
            kpoints_parser = KpointsParser(handler=fh)
            kpoints_data = kpoints_parser.kpoints
            node = orm.KpointsData()
            if kpoints_data['mode'] == 'explicit':
                node.set_kpoints(
                    kpoints_data['points'], weights=kpoints_data['weights'], cartesian=kpoints_data['cartesian']
                )
            elif kpoints_data['mode'] == 'automatic':
                node.set_kpoints_mesh(kpoints_data['divisions'], offset=kpoints_data['offset'])
            else:
                raise ValueError(f'Unknown kpoints mode {kpoints_data["mode"]}')
            self.inputs.kpoints = node

        if self.potential_family is None:
            # Parse the POTCAR file - this automatically corrects inconsistent ordering of potentials
            mpotcar = MultiPotcarIo.read(self._folder / 'POTCAR')
            self.inputs.potential = {potcar.node.element: potcar.node for potcar in mpotcar.potcars}
            # Verify potentials for all kinds are present in the POTCAR
            kind_keys = [kind.symbol for kind in self.inputs.structure.kinds]
            assert set(kind_keys) == set(self.inputs.potential.keys())
            # TODO: check if these potential belong to the same family
        else:
            self.inputs.potential_family = self.potential_family
            self.inputs.potential_mapping = self.potential_mapping
            self.inputs.potential = None

    def _convert_to_aiida(self, builder, root_namespace=None):
        """
        Convert the VASP calculation folder to an AiiDA VaspCalculation input.
        """
        if root_namespace is None:
            root_namespace = builder
        builder.code = orm.load_code(self.code_string)
        root_namespace.structure = self.inputs.structure
        builder.parameters = self.inputs.parameters
        builder.kpoints = self.inputs.kpoints
        if self.inputs.potential is not None:
            builder.potential = self.inputs.potential
        if self.settings is not None:
            builder.settings = orm.Dict(dict=self.settings)
        if self.options is not None:
            builder.options = orm.Dict(dict=self.options)
        builder.code = orm.load_code(self.code_string)
        return builder

    def get_builder(self):
        """
        Convert the VASP calculation folder to an AiiDA VaspCalculation input.
        """
        from aiida_vasp.calcs.vasp import VaspCalculation

        builder = VaspCalculation.get_builder()
        self._convert_to_aiida(builder)
        return builder


class VaspWorkChainConverter(VaspCalcConverter):
    def get_builder(self):
        """
        Convert the VASP calculation folder to an AiiDA VaspCalculation input.
        """
        from aiida_vasp.workchains import VaspWorkChain

        if self.potential_family is None:
            raise ValueError('Cannot convert to VaspWorkChain without a potential family and mapping.')

        builder = VaspWorkChain.get_builder()
        self._convert_to_aiida(builder)
        builder.potential_family = orm.Str(self.potential_family)
        builder.potential_mapping = orm.Dict(self.potential_mapping)
        return builder


class VaspRelaxWorkChainConverter(VaspCalcConverter):
    def __init__(self, *args, relax_settings=None, **kwargs):
        """Initialize the VaspRelaxWorkChainConverter with a relax_settings dictionary."""
        super().__init__(*args, **kwargs)
        if self.relax_settings is None:
            self.relax_settings = {}
        self.relax_settings = relax_settings

    def get_builder(self):
        """Convert the VASP calculation folder to an AiiDA VaspRelaxWorkChain input."""
        from aiida_vasp.workchains import VaspRelaxWorkChain

        builder = VaspRelaxWorkChain.get_builder()
        self._convert_to_aiida(builder.vasp, builder)
        builder.relax_settings = self.relax_settings
        return builder
