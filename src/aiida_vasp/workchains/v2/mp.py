"""Native Materials Project style workflows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import ProcessSpec, ToContext, WorkChain, if_
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.plugins import WorkflowFactory

from aiida_vasp.data.potcar import PotcarData

from .core_flows import VaspDoubleRelaxWorkChain


def _recursive_merge_many(*values: dict | None) -> dict:
    """Merge many nested dictionaries."""
    output: dict[str, Any] = {}
    for value in values:
        if value:
            from aiida_vasp.protocols import recursive_merge  # noqa: PLC0415

            output = recursive_merge(output, deepcopy(value))
    return output


def _kind_symbols(structure: orm.StructureData) -> dict[str, str]:
    """Return the single-element symbol for each kind."""
    return {kind.name: kind.symbols[0] for kind in structure.kinds if len(kind.symbols) == 1}


def _select_potential_family_and_mapping(structure: orm.StructureData) -> tuple[str | None, dict[str, str] | None]:
    """Pick an installed POTCAR family and a compatible kind mapping."""
    kind_symbols = _kind_symbols(structure)
    if not kind_symbols:
        return None, None

    groups = PotcarData.get_potcar_groups(filter_elements=list(set(kind_symbols.values())))
    if not groups:
        return None, None

    preferred_labels = ('PBE.54', 'PBE', 'test_family')
    family = next(
        (group.label for label in preferred_labels for group in groups if group.label == label), groups[0].label
    )

    mapping: dict[str, str] = {}
    for kind_name, element in kind_symbols.items():
        full_names = PotcarData.get_full_names(family_name=family, element=element)
        if not full_names:
            return family, None
        if kind_name in full_names:
            mapping[kind_name] = kind_name
        elif element in full_names:
            mapping[kind_name] = element
        else:
            mapping[kind_name] = full_names[0]

    return family, mapping


def _ensure_potential_overrides(
    structure: orm.StructureData, overrides: dict | None, namespace: str | None = None
) -> dict:
    """Populate potential family and mapping if the caller did not provide them."""
    updated = deepcopy(overrides) if overrides else {}
    target = updated.setdefault(namespace, {}) if namespace else updated

    if target.get('potential_family') and target.get('potential_mapping'):
        return updated

    family, mapping = _select_potential_family_and_mapping(structure)
    if family is None or mapping is None:
        return updated

    target.setdefault('potential_family', family)
    target.setdefault('potential_mapping', mapping)
    return updated


def _get_pmg_input_overrides(
    structure: orm.StructureData,
    set_name: str,
    *,
    incar_overrides: dict | None = None,
    pmg_kwargs: dict | None = None,
) -> dict:
    """Extract native aiida-vasp input deltas from a pymatgen input set."""
    from aiida_vasp.protocols.pmg import PymatgenInputAdaptor  # noqa: PLC0415

    adaptor = PymatgenInputAdaptor(set_name, incar_overrides=incar_overrides, pmg_kwargs=pmg_kwargs)
    inputs: dict[str, Any] = {'parameters': {'incar': adaptor.get_incar_dict(structure, raw_python=True)}}

    kpoints = adaptor.get_kpoints(structure)
    if kpoints is not None:
        inputs['kpoints'] = kpoints
    else:
        spacing = adaptor.get_kpoints_spacing(structure)
        if spacing is not None:
            inputs['kpoints_spacing'] = spacing

    return inputs


def _set_builder_namespace(namespace, values: dict) -> None:
    """Populate a builder namespace with the explicit inputs from another builder."""
    for key, value in values.items():
        setattr(namespace, key, value)


def _node_to_dict(node: orm.Dict | None) -> dict | None:
    """Convert an optional Dict node to a plain dictionary."""
    if node is None:
        return None
    return node.get_dict()


class _VaspRelaxStaticWorkChain(WorkChain):
    """Base workchain for a relax-flow followed by a static calculation."""

    _relax_workchain = None
    _static_workchain = WorkflowFactory('vasp.v2.vasp')

    @classmethod
    def define(cls, spec: ProcessSpec) -> None:
        super().define(spec)
        spec.input('structure', valid_type=(orm.StructureData, orm.CifData))
        spec.expose_inputs(cls._relax_workchain, namespace='relax', exclude=('structure',))
        spec.expose_inputs(cls._static_workchain, namespace='static', exclude=('structure',))
        spec.input(
            'run_static',
            valid_type=orm.Bool,
            required=False,
            serializer=to_aiida_type,
            default=lambda: orm.Bool(True),
        )
        spec.expose_outputs(cls._relax_workchain, namespace='relax')
        spec.expose_outputs(cls._static_workchain)
        spec.outline(
            cls.run_relax,
            cls.inspect_relax,
            if_(cls.should_run_static)(cls.run_static, cls.inspect_static),
            cls.results,
        )
        spec.exit_code(401, 'ERROR_RELAX_FAILED', message='The relaxation flow failed.')
        spec.exit_code(402, 'ERROR_STATIC_FAILED', message='The final static calculation failed.')

    def should_run_static(self) -> bool:
        """Whether to launch the final static calculation."""
        return self.inputs.run_static.value

    def run_relax(self) -> ToContext:
        """Run the relaxation flow."""
        inputs = AttributeDict(self.exposed_inputs(self._relax_workchain, namespace='relax', agglomerate=True))
        inputs.structure = self.inputs.structure
        inputs.metadata.call_link_label = 'relax'
        inputs.metadata.label = f'{self.inputs.metadata.get("label", "")} RELAX'.strip()
        running = self.submit(self._relax_workchain, **inputs)
        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """Inspect the relaxation flow."""
        if not self.ctx.workchain_relax.is_finished_ok:
            return self.exit_codes.ERROR_RELAX_FAILED

    def run_static(self) -> ToContext:
        """Run the final static calculation."""
        inputs = AttributeDict(self.exposed_inputs(self._static_workchain, namespace='static', agglomerate=True))
        inputs.structure = self.ctx.workchain_relax.outputs.relax.structure
        inputs.metadata.call_link_label = 'static'
        inputs.metadata.label = f'{self.inputs.metadata.get("label", "")} STATIC'.strip()
        if 'restart_folder' not in inputs and 'remote_folder' in self.ctx.workchain_relax.outputs:
            inputs.restart_folder = self.ctx.workchain_relax.outputs.remote_folder
        running = self.submit(self._static_workchain, **inputs)
        return ToContext(workchain_static=running)

    def inspect_static(self):
        """Inspect the final static calculation."""
        if not self.ctx.workchain_static.is_finished_ok:
            return self.exit_codes.ERROR_STATIC_FAILED

    def results(self) -> None:
        """Expose outputs."""
        self.out_many(self.exposed_outputs(self.ctx.workchain_relax, self._relax_workchain, namespace='relax'))
        if self.should_run_static():
            self.out_many(self.exposed_outputs(self.ctx.workchain_static, self._static_workchain))


class VaspMPGGADoubleRelaxWorkChain(VaspDoubleRelaxWorkChain):
    """MP GGA double relax: two MP GGA relaxations."""

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        base_overrides = _recursive_merge_many(
            {
                'vasp': _get_pmg_input_overrides(
                    structure,
                    'MPRelaxSet',
                    pmg_kwargs={'force_gamma': True, 'auto_metal_kpoints': True, 'inherit_incar': False},
                )
            },
            _ensure_potential_overrides(structure, overrides, namespace='vasp'),
        )
        relax_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=base_overrides,
            options=options,
            **kwargs,
        )
        relax_builder.pop('structure')
        builder = cls.get_builder()
        builder.structure = structure
        _set_builder_namespace(builder.relax, relax_builder._inputs(prune=True))
        return builder


class VaspMPMetaGGADoubleRelaxWorkChain(VaspDoubleRelaxWorkChain):
    """MP meta-GGA double relax: PBEsol-like pre-relax followed by r2SCAN relax."""

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        stage_1_base = {
            'vasp': _get_pmg_input_overrides(
                structure,
                'MPScanRelaxSet',
                incar_overrides={
                    'ediffg': -0.05,
                    'gga': 'PS',
                    'lwave': True,
                    'lcharg': True,
                    'lelf': False,
                    'metagga': None,
                },
                pmg_kwargs={'auto_ismear': False, 'inherit_incar': False},
            )
        }
        stage_2_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_ensure_potential_overrides(
                structure,
                {
                    'vasp': _get_pmg_input_overrides(
                        structure,
                        'MPScanRelaxSet',
                        incar_overrides={
                            'gga': None,
                            'lcharg': True,
                            'lwave': True,
                            'lelf': False,
                        },
                        pmg_kwargs={'auto_ismear': False, 'inherit_incar': False},
                    )
                },
                namespace='vasp',
            ),
            options=options,
            **kwargs,
        )
        return super().get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_recursive_merge_many(
                stage_1_base,
                _ensure_potential_overrides(structure, overrides, namespace='vasp'),
            ),
            options=options,
            stage_2_overrides={
                'parameters': stage_2_builder.vasp.parameters.get_dict(),
                'settings': _node_to_dict(stage_2_builder.vasp.settings),
                'relax_settings': _node_to_dict(stage_2_builder.relax_settings),
            },
            **kwargs,
        )


class VaspMP24DoubleRelaxWorkChain(VaspDoubleRelaxWorkChain):
    """MP24 double relax: PBEsol pre-relax followed by r2SCAN relax."""

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        stage_1_base = {
            'vasp': _get_pmg_input_overrides(
                structure,
                'MP24RelaxSet',
                incar_overrides={'lwave': True},
                pmg_kwargs={'xc_functional': 'PBEsol'},
            )
        }
        stage_2_overrides = {
            'vasp': _get_pmg_input_overrides(
                structure,
                'MP24RelaxSet',
                incar_overrides={'lwave': True},
                pmg_kwargs={'xc_functional': 'r2SCAN'},
            )
        }
        stage_2_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_ensure_potential_overrides(structure, stage_2_overrides, namespace='vasp'),
            options=options,
            **kwargs,
        )
        return super().get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_recursive_merge_many(
                stage_1_base,
                _ensure_potential_overrides(structure, overrides, namespace='vasp'),
            ),
            options=options,
            stage_2_overrides={
                'parameters': stage_2_builder.vasp.parameters.get_dict(),
                'settings': _node_to_dict(stage_2_builder.vasp.settings),
                'relax_settings': _node_to_dict(stage_2_builder.relax_settings),
            },
            **kwargs,
        )


class VaspMPGGARelaxStaticWorkChain(_VaspRelaxStaticWorkChain):
    """MP GGA double relax followed by MP GGA static."""

    _relax_workchain = VaspMPGGADoubleRelaxWorkChain

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        overrides = overrides or {}
        relax_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            overrides=overrides.get('relax'),
            options=options,
            **kwargs,
        )
        relax_builder.pop('structure')
        static_builder = cls._static_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_recursive_merge_many(
                _get_pmg_input_overrides(
                    structure,
                    'MPStaticSet',
                    pmg_kwargs={'force_gamma': True, 'auto_metal_kpoints': True, 'inherit_incar': False},
                ),
                _ensure_potential_overrides(structure, overrides.get('static')),
            ),
            options=options,
            **kwargs,
        )
        static_builder.pop('structure')
        builder = cls.get_builder()
        builder.structure = structure
        _set_builder_namespace(builder.relax, relax_builder._inputs(prune=True))
        _set_builder_namespace(builder.static, static_builder._inputs(prune=True))
        return builder


class VaspMPMetaGGARelaxStaticWorkChain(_VaspRelaxStaticWorkChain):
    """MP meta-GGA double relax followed by MP meta-GGA static."""

    _relax_workchain = VaspMPMetaGGADoubleRelaxWorkChain

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        overrides = overrides or {}
        relax_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            overrides=overrides.get('relax'),
            options=options,
            **kwargs,
        )
        relax_builder.pop('structure')
        static_builder = cls._static_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_recursive_merge_many(
                _get_pmg_input_overrides(
                    structure,
                    'MPScanStaticSet',
                    incar_overrides={
                        'algo': 'FAST',
                        'gga': None,
                        'lcharg': True,
                        'lwave': False,
                        'lvhar': None,
                        'lelf': False,
                    },
                    pmg_kwargs={'auto_ismear': False, 'inherit_incar': False},
                ),
                _ensure_potential_overrides(structure, overrides.get('static')),
            ),
            options=options,
            **kwargs,
        )
        static_builder.pop('structure')
        builder = cls.get_builder()
        builder.structure = structure
        _set_builder_namespace(builder.relax, relax_builder._inputs(prune=True))
        _set_builder_namespace(builder.static, static_builder._inputs(prune=True))
        return builder


class VaspMP24RelaxStaticWorkChain(_VaspRelaxStaticWorkChain):
    """MP24 double relax followed by MP24 static."""

    _relax_workchain = VaspMP24DoubleRelaxWorkChain

    @classmethod
    def get_builder_from_protocol(
        cls,
        code: orm.AbstractCode,
        structure: orm.StructureData,
        overrides: dict | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        overrides = overrides or {}
        relax_builder = cls._relax_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            overrides=overrides.get('relax'),
            options=options,
            **kwargs,
        )
        relax_builder.pop('structure')
        static_builder = cls._static_workchain.get_builder_from_protocol(
            code=code,
            structure=structure,
            protocol='balanced',
            overrides=_recursive_merge_many(
                _get_pmg_input_overrides(
                    structure,
                    'MP24StaticSet',
                    incar_overrides={'lelf': True, 'kpar': 1},
                    pmg_kwargs={'xc_functional': 'r2SCAN'},
                ),
                _ensure_potential_overrides(structure, overrides.get('static')),
            ),
            options=options,
            **kwargs,
        )
        static_builder.pop('structure')
        builder = cls.get_builder()
        builder.structure = structure
        _set_builder_namespace(builder.relax, relax_builder._inputs(prune=True))
        _set_builder_namespace(builder.static, static_builder._inputs(prune=True))
        return builder
