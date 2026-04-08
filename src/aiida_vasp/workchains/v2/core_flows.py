"""Native orchestration flows built on top of the v2 VASP workchains."""

from __future__ import annotations

from copy import deepcopy

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import ProcessSpec, ToContext, WorkChain
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.plugins import WorkflowFactory

from aiida_vasp.utils.extended_dicts import update_nested_dict, update_nested_dict_node


def _dict_from_input(value: orm.Dict | dict | None) -> dict:
    """Return a plain dictionary for a Dict input or plain mapping."""
    if value is None:
        return {}
    if isinstance(value, orm.Dict):
        return value.get_dict()
    return deepcopy(value)


def _merge_dict_node(
    node: orm.Dict | None, updates: orm.Dict | dict | None, extend_list: bool = False
) -> orm.Dict | None:
    """Merge updates into a Dict node, creating one if needed."""
    update_dict = _dict_from_input(updates)
    if not update_dict:
        return node
    if node is None:
        return orm.Dict(dict=update_dict)
    return update_nested_dict_node(node, update_dict, extend_list=extend_list)


def _apply_vasp_stage_overrides(
    inputs: AttributeDict,
    *,
    parameters: orm.Dict | dict | None = None,
    settings: orm.Dict | dict | None = None,
    options: orm.Dict | dict | None = None,
) -> AttributeDict:
    """Apply stage-local overrides directly to a ``VaspWorkChain`` input namespace."""
    if parameters:
        inputs.parameters = _merge_dict_node(inputs.parameters, parameters)

    if settings:
        current = inputs.get('settings')
        inputs.settings = _merge_dict_node(current, settings, extend_list=True)

    option_updates = _dict_from_input(options)
    if option_updates:
        current_options = deepcopy(dict(inputs.calc.metadata.options))
        update_nested_dict(current_options, option_updates)
        inputs.calc.metadata.options = current_options

    return inputs


def _set_builder_namespace(namespace, values: dict) -> None:
    """Populate a builder namespace with only the explicitly populated values."""
    for key, value in values.items():
        setattr(namespace, key, value)


def _get_stage_override(inputs, stage: int, key: str):
    """Return stage override from the normalized namespace or legacy flattened ports."""
    stage_namespace = inputs.get(f'stage_{stage}')
    if stage_namespace is not None and key in stage_namespace:
        return stage_namespace.get(key)
    return inputs.get(f'stage_{stage}_{key}')


class VaspDoubleRelaxWorkChain(WorkChain):
    """Perform two chained relax-like ``VaspWorkChain`` calculations."""

    _base_workchain = WorkflowFactory('vasp.v2.vasp')
    _relax_protocol_workchain = WorkflowFactory('vasp.v2.relax')
    _relax_workchain = _relax_protocol_workchain

    @classmethod
    def define(cls, spec: ProcessSpec) -> None:
        super().define(spec)
        spec.input('structure', valid_type=(orm.StructureData, orm.CifData))
        spec.input_namespace('relax')
        spec.expose_inputs(cls._base_workchain, namespace='relax.vasp', exclude=('structure',))
        spec.input(
            'relax.relax_settings',
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
        )
        spec.input_namespace('stage_1', required=False)
        spec.input('stage_1.parameters', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1.settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1.options', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1.relax_settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input_namespace('stage_2', required=False)
        spec.input('stage_2.parameters', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2.settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2.options', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2.relax_settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1_parameters', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1_settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1_options', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_1_relax_settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2_parameters', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2_settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2_options', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.input('stage_2_relax_settings', valid_type=orm.Dict, required=False, serializer=to_aiida_type)
        spec.expose_outputs(cls._base_workchain)
        spec.output('stage_1_relax.structure', valid_type=orm.StructureData, required=False)
        spec.output('stage_2_relax.structure', valid_type=orm.StructureData, required=False)
        spec.outline(
            cls.run_stage_1,
            cls.inspect_stage_1,
            cls.run_stage_2,
            cls.inspect_stage_2,
            cls.results,
        )
        spec.exit_code(401, 'ERROR_STAGE_1_FAILED', message='The first relaxation stage failed.')
        spec.exit_code(402, 'ERROR_STAGE_2_FAILED', message='The second relaxation stage failed.')

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        protocol: str | None = None,
        overrides: dict | None = None,
        options: dict | None = None,
        stage_1_overrides: dict | None = None,
        stage_2_overrides: dict | None = None,
        **kwargs,
    ):
        """Create a builder with a shared baseline relax builder and stage-local overrides."""
        overrides = overrides or {}
        stage_1_overrides = stage_1_overrides or {}
        stage_2_overrides = stage_2_overrides or {}

        relax_builder = cls._relax_protocol_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=protocol,
            overrides=overrides,
            options=options,
            **kwargs,
        )
        relax_builder.pop('structure')

        builder = cls.get_builder()
        builder.structure = structure
        _set_builder_namespace(builder.relax.vasp, relax_builder.vasp._inputs(prune=True))
        builder.relax.relax_settings = relax_builder.relax_settings

        for key in ('parameters', 'settings', 'options', 'relax_settings'):
            value = stage_1_overrides.get(key)
            if value:
                setattr(builder.stage_1, key, orm.Dict(dict=value))
                setattr(builder, f'stage_1_{key}', orm.Dict(dict=value))
            value = stage_2_overrides.get(key)
            if value:
                setattr(builder.stage_2, key, orm.Dict(dict=value))
                setattr(builder, f'stage_2_{key}', orm.Dict(dict=value))

        return builder

    def _prepare_relax_inputs(self, structure: orm.StructureData, stage: int) -> AttributeDict:
        """Create inputs for a relax-like ``VaspWorkChain`` stage."""
        inputs = AttributeDict(self.exposed_inputs(self._base_workchain, namespace='relax.vasp', agglomerate=True))
        inputs.structure = structure
        inputs.metadata.call_link_label = f'relax_{stage}'
        inputs.metadata.label = f'{self.inputs.metadata.get("label", "")} RELAX {stage}'.strip()
        relax_settings = _dict_from_input(self.inputs.relax.get('relax_settings'))
        relax_settings.update(_dict_from_input(_get_stage_override(self.inputs, stage, 'relax_settings')))
        if relax_settings:
            inputs.parameters = _merge_dict_node(inputs.parameters, {'relax': relax_settings})

        parser_settings = {
            'parser_settings': {
                'include_node': ['structure', 'trajectory'],
                'include_quantity': ['energies'],
            }
        }
        current_settings = inputs.get('settings')
        inputs.settings = _merge_dict_node(current_settings, parser_settings, extend_list=True)

        _apply_vasp_stage_overrides(
            inputs,
            parameters=_get_stage_override(self.inputs, stage, 'parameters'),
            settings=_get_stage_override(self.inputs, stage, 'settings'),
            options=_get_stage_override(self.inputs, stage, 'options'),
        )

        if stage == 1:
            inputs.keep_last_workdir = orm.Bool(True)
        return inputs

    def run_stage_1(self) -> ToContext:
        """Run the first relaxation."""
        inputs = self._prepare_relax_inputs(self.inputs.structure, 1)
        running = self.submit(self._base_workchain, **inputs)
        return ToContext(workchain_stage_1=running)

    def inspect_stage_1(self):
        """Inspect the first relaxation."""
        workchain = self.ctx.workchain_stage_1
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_STAGE_1_FAILED
        self.out('stage_1_relax.structure', workchain.outputs.relax.structure)
        self.ctx.current_structure = workchain.outputs.relax.structure
        self.ctx.restart_folder = workchain.outputs.remote_folder if 'remote_folder' in workchain.outputs else None

    def run_stage_2(self) -> ToContext:
        """Run the second relaxation from the first-stage output."""
        inputs = self._prepare_relax_inputs(self.ctx.current_structure, 2)
        if self.ctx.restart_folder is not None and 'restart_folder' not in inputs:
            inputs.restart_folder = self.ctx.restart_folder
        running = self.submit(self._base_workchain, **inputs)
        return ToContext(workchain_stage_2=running)

    def inspect_stage_2(self):
        """Inspect the second relaxation."""
        workchain = self.ctx.workchain_stage_2
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_STAGE_2_FAILED
        self.out('stage_2_relax.structure', workchain.outputs.relax.structure)

    def results(self) -> None:
        """Expose the outputs of the second relaxation."""
        self.out_many(self.exposed_outputs(self.ctx.workchain_stage_2, self._base_workchain))


class VaspRelaxBandsWorkChain(WorkChain):
    """Run a native relaxation followed by a native band-structure workflow."""

    _relax_workchain = WorkflowFactory('vasp.v2.relax')
    _bands_workchain = WorkflowFactory('vasp.v2.bands')

    @classmethod
    def define(cls, spec: ProcessSpec) -> None:
        super().define(spec)
        spec.input('structure', valid_type=(orm.StructureData, orm.CifData))
        spec.expose_inputs(cls._relax_workchain, namespace='relax', exclude=('structure',))
        spec.expose_inputs(cls._bands_workchain, namespace='bands', exclude=('structure', 'relax'))
        spec.expose_outputs(cls._bands_workchain)
        spec.output('relax.structure', valid_type=orm.StructureData, required=False)
        spec.outline(cls.run_relax, cls.inspect_relax, cls.run_bands, cls.inspect_bands, cls.results)
        spec.exit_code(401, 'ERROR_RELAX_FAILED', message='The relaxation stage failed.')
        spec.exit_code(402, 'ERROR_BANDS_FAILED', message='The band-structure stage failed.')

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        relax_protocol: str | None = None,
        band_protocol: str | None = None,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        """Create a builder for relax followed by bands."""
        overrides = overrides or {}
        relax_overrides = overrides.get('relax', {})
        bands_overrides = overrides.get('bands', {})

        relax_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=relax_protocol,
            overrides=relax_overrides,
            options=options,
            **kwargs,
        )
        relax_builder.pop('structure')

        bands_builder = cls._bands_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol=band_protocol,
            run_relax=False,
            overrides=bands_overrides,
            options=options,
            **kwargs,
        )
        bands_builder.pop('structure')

        builder = cls.get_builder()
        builder.structure = structure
        _set_builder_namespace(builder.relax, relax_builder._inputs(prune=True))
        _set_builder_namespace(builder.bands, bands_builder._inputs(prune=True))
        return builder

    def run_relax(self) -> ToContext:
        """Run the relaxation stage."""
        inputs = AttributeDict(self.exposed_inputs(self._relax_workchain, namespace='relax', agglomerate=True))
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'relax'
        inputs.metadata.label = f'{self.inputs.metadata.get("label", "")} RELAX'.strip()
        running = self.submit(self._relax_workchain, **inputs)
        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """Inspect the relaxation stage."""
        if not self.ctx.workchain_relax.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_FAILED
        self.out('relax.structure', self.ctx.workchain_relax.outputs.relax.structure)

    def run_bands(self) -> ToContext:
        """Run the band-structure workflow from the relaxed structure."""
        inputs = AttributeDict(self.exposed_inputs(self._bands_workchain, namespace='bands', agglomerate=True))
        inputs.structure = self.ctx.workchain_relax.outputs.relax.structure
        inputs.metadata.call_link_label = 'bands'
        inputs.metadata.label = f'{self.inputs.metadata.get("label", "")} BANDS'.strip()
        running = self.submit(self._bands_workchain, **inputs)
        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """Inspect the band-structure stage."""
        if not self.ctx.workchain_bands.is_finished_ok:
            return self.exit_codes.ERROR_BANDS_FAILED

    def results(self) -> None:
        """Expose the outputs of the bands workflow."""
        self.out_many(self.exposed_outputs(self.ctx.workchain_bands, self._bands_workchain))
