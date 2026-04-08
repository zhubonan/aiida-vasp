"""
Input generators based on protocols

This module aimed at interactive post-generation update for the builder created
by `.get_builder_from_protocol` method of various workchain classes.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

from aiida import orm
from aiida.engine import run_get_node, submit
from aiida.engine.processes.builder import ProcessBuilderNamespace
from aiida.plugins import WorkflowFactory
from yaml import safe_load

from aiida_vasp.utils.dict_merge import recursive_merge

__all__ = [
    'VaspBandsInputGenerator',
    'VaspConvergenceInputGenerator',
    'VaspDoubleRelaxInputGenerator',
    'VaspHybridBandsInputGenerator',
    'VaspInputGenerator',
    'VaspMP24DoubleRelaxInputGenerator',
    'VaspMP24RelaxStaticInputGenerator',
    'VaspMPGGADoubleRelaxInputGenerator',
    'VaspMPGGARelaxStaticInputGenerator',
    'VaspMPMetaGGADoubleRelaxInputGenerator',
    'VaspMPMetaGGARelaxStaticInputGenerator',
    'VaspMatPesStaticInputGenerator',
    'VaspNscfInputGenerator',
    'VaspRelaxBandsInputGenerator',
    'VaspRelaxInputGenerator',
]


DEFAULT_PRESET = 'default'
DEFAULT_PROTOCOL = 'balanced'
CANONICAL_PRESET_DIR = 'presets'
LEGACY_PRESET_DIR = 'protocol_presets'
DEPRECATED_PRESET_FIELDS = ('protocol_overrides', 'default_relax_settings', 'default_band_settings')


def _format_schema_block(schema: dict[str, Any]) -> str:
    """Render a compact human-readable schema description."""
    lines = [schema['title']]
    summary = schema.get('summary')
    if summary:
        lines.append(summary)

    details = schema.get('details', {})
    if details:
        rendered = ', '.join(f'{key}={value}' for key, value in details.items())
        lines.append(f'Details: {rendered}')

    for key, label in (
        ('canonical_ports', 'Canonical ports'),
        ('accessors', 'Accessors'),
        ('mutators', 'Mutators'),
        ('notes', 'Notes'),
        ('examples', 'Examples'),
    ):
        values = schema.get(key) or []
        if values:
            lines.append(f'{label}:')
            lines.extend(f'  - {value}' for value in values)

    current = schema.get('current')
    if current:
        lines.append('Current:')
        lines.extend(f'  - {value}' for value in current)

    return '\n'.join(lines)


def join_namespace_path(*parts: str | None) -> str:
    """Join namespace parts using ``.`` while skipping empty values."""
    return '.'.join(part.strip('.') for part in parts if part)


def get_library_path() -> Path:
    """
    Get the path where the YAML files are stored within this package.

    :returns: Path to the library directory containing YAML configuration files
    :rtype: pathlib.Path
    """
    return Path(__file__).parent / 'presets'


def get_preset_library_paths() -> tuple[Path, ...]:
    """Return package and user paths searched for preset files."""
    return (
        get_library_path(),
        Path(f'~/.aiida-vasp/{CANONICAL_PRESET_DIR}').expanduser(),
        Path(f'~/.aiida-vasp/{LEGACY_PRESET_DIR}').expanduser(),
    )


def _warn_for_deprecated_preset_fields(data: dict[str, Any], source: str) -> None:
    """Warn when preset files still define generator fields that are no longer consumed."""
    deprecated_fields = [field for field in DEPRECATED_PRESET_FIELDS if data.get(field)]
    if deprecated_fields:
        warnings.warn(
            (
                f'Preset `{source}` defines deprecated fields {deprecated_fields}. '
                'These fields are ignored by the input-generator path and will be removed in a future release.'
            ),
            FutureWarning,
            stacklevel=2,
        )


def _iter_preset_candidate_paths(fname: str) -> list[Path]:
    """Return candidate preset paths for a named preset or explicit file path."""
    target = Path(fname).expanduser()
    if target.suffix in {'.yaml', '.yml'}:
        return [target]

    candidates: list[Path] = []
    for parent in get_preset_library_paths():
        candidates.append(parent / f'{fname}.yaml')
        candidates.append(parent / f'{fname}.yml')
    return candidates


def list_protocol_presets() -> list[Path]:
    """
    List all available presets in the package.
    """
    presets = []
    seen = set()
    for parent in get_preset_library_paths():
        files = chain(parent.glob('*.yaml'), parent.glob('*.yml'))
        for file in files:
            resolved = file.absolute()
            if resolved in seen:
                continue
            seen.add(resolved)
            presets.append(resolved)
    return presets


@dataclass
class PresetConfig:
    """Class to store the preset for inputs"""

    name: str
    default_protocol: str
    default_code: str
    code_specific: dict = field(default_factory=dict)
    default_options: dict = field(default_factory=dict)
    default_settings: dict = field(default_factory=dict)
    protocol_overrides: dict = field(default_factory=dict)
    default_relax_settings: dict = field(default_factory=dict)
    default_band_settings: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, fname: str) -> PresetConfig:
        """
        Load preset configuration from a YAML file.

        Searches for the configuration file in the package library path and user's
        home directory (`~/.aiida-vasp/presets/` or the legacy `~/.aiida-vasp/protocol_presets/`).

        :param fname: Name of the configuration file (without .yaml extension)
        :type fname: str

        :returns: ProtocolPresetConfig instance loaded from file
        :rtype: ProtocolPresetConfig

        :raises RuntimeError: If the preset definition file cannot be found
        """
        target_path = next((path for path in _iter_preset_candidate_paths(fname) if path.is_file()), None)
        if target_path is None:
            raise RuntimeError(f'Cannot find preset definition for {fname}')

        with open(target_path, encoding='utf-8', mode='r') as fhandle:
            data = safe_load(fhandle) or {}
        _warn_for_deprecated_preset_fields(data, str(target_path))
        return cls(**data)

    def get_code_specific_options(self, code: str, namespace: str) -> dict[str, Any]:
        """
        Return code-specific options for a given namespace.

        If code-specific options exist, they are merged with the default options
        for the namespace, with code-specific options taking precedence.

        :param code: Name/identifier of the computational code
        :type code: str
        :param namespace: Configuration namespace (e.g., 'options', 'settings')
        :type namespace: str

        :returns: Dictionary containing the merged options
        :rtype: dict
        """
        if code in self.code_specific:
            if namespace in self.code_specific[code]:
                code_specific = self.code_specific[code][namespace]
                default = getattr(self, f'default_{namespace}', {})
                if default is None:
                    default = {}
                default = deepcopy(default)
                default.update(code_specific)
                return default
        return deepcopy(
            getattr(
                self,
                f'default_{namespace}',
                {},
            )
        )

    def resolve_code_specific_configuration(self, code: str) -> dict[str, dict[str, Any]]:
        """Resolve all preset-derived configuration fragments for a given code."""
        return {
            'options': self.get_code_specific_options(code, 'options'),
            'settings': self.get_code_specific_options(code, 'settings'),
            'incar': self.get_code_specific_options(code, 'incar'),
        }


class BaseInputGenerator:
    """
    BaseClass for all protocol builder updaters

    The protocol updater serves two purposes:
    - Generating a builder based on a user-defined "preset", e.g. with options and overrides pre-loaded
    - Allow interactive modifications of common parameters such as incar tag's, resources and options.
    """

    WF_ENTRYPOINT = 'vasp.vasp'
    WORKFLOW_LABEL = 'VASP workflow'
    WORKFLOW_SUMMARY = 'Protocol-based builder generator.'
    CANONICAL_PORTS: tuple[str, ...] = ()
    ACCESSOR_DOCS: tuple[str, ...] = ()
    MUTATOR_DOCS: tuple[str, ...] = ()
    NOTES: tuple[str, ...] = ()
    EXAMPLES: tuple[str, ...] = ()

    def __init__(
        self,
        preset_name: str = 'default',
        protocol: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Instantiate a pipeline"""
        # Configure the builder

        assert hasattr(self, 'WF_ENTRYPOINT'), 'WF_ENTRYPOINT must be specified by the class'
        self.verbose = verbose
        # Initialise the preset
        self.preset_name = preset_name
        self.preset = PresetConfig.from_file(preset_name)
        self.protocol = protocol if protocol is not None else self.preset.default_protocol
        self.builder = None

    @staticmethod
    def _load_code_node(code):
        """Return a loaded code node from a string/PK or pass through a code instance."""
        if isinstance(code, orm.AbstractCode):
            return code
        return orm.load_code(code)

    def _resolve_build_request(self, *, code=None, protocol=None, overrides=None, options=None) -> dict[str, Any]:
        """Resolve preset-backed defaults before constructing a workflow builder."""
        resolved_code = code or self.preset.default_code
        resolved_protocol = self.protocol if protocol is None else protocol
        resolved_overrides = deepcopy(overrides or {})
        profile_code = resolved_code.full_label if isinstance(resolved_code, orm.AbstractCode) else resolved_code
        profile_config = self.preset.resolve_code_specific_configuration(profile_code)
        resolved_options = recursive_merge(profile_config['options'], options or {})
        return {
            'code': resolved_code,
            'protocol': resolved_protocol,
            'overrides': resolved_overrides,
            'options': resolved_options,
            'profile_config': profile_config,
        }

    def _finalize_builder(self, builder, *, profile_config: dict[str, dict[str, Any]]):
        """Store the builder and apply preset/profile defaults consistently."""
        self.builder = builder
        self.set_settings(profile_config['settings'])
        self.set_incar(profile_config['incar'])
        return builder

    def build(self, structure, code=None, protocol=None, overrides=None, **kwargs):
        """
        Generate builder base on a given structure and overrides (if supplied)
        """
        build_request = self._resolve_build_request(
            code=code,
            protocol=protocol,
            overrides=overrides,
            options=kwargs.pop('options', {}),
        )

        builder = WorkflowFactory(self.WF_ENTRYPOINT).get_builder_from_protocol(
            code=self._load_code_node(build_request['code']),
            structure=structure,
            protocol=build_request['protocol'],
            overrides=build_request['overrides'],
            options=build_request['options'],
            **kwargs,
        )
        return self._finalize_builder(builder, profile_config=build_request['profile_config'])

    def get_builder(self, structure, code=None, protocol=None, overrides=None, **kwargs):
        """Compatibility alias for :meth:`build`."""
        warnings.warn(
            '`get_builder(...)` is deprecated; use `build(...)` instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.build(structure=structure, code=code, protocol=protocol, overrides=overrides, **kwargs)

    @property
    def reference_structure(self):
        return self.builder.structure

    def clone(self):
        """Return a clone of the generator bound to a deep-copied builder."""
        cloned = self.__class__(preset_name=self.preset_name, protocol=self.protocol, verbose=self.verbose)
        cloned.builder = deepcopy(self.builder)
        return cloned

    def _resolve_namespace(self, namespace_path: str | None = None):
        """Resolve a builder namespace by ``.`` separated path."""
        if self.builder is None:
            raise RuntimeError('Builder has not been constructed. Call `build(...)` first.')
        if not namespace_path:
            return self.builder
        item = self.builder
        for part in namespace_path.split('.'):
            item = item.get(part)
            if item is None:
                raise AttributeError(f'Namespace `{namespace_path}` is not available on this builder.')
        return item

    def _resolve_parent_and_leaf(self, port_path: str):
        """Resolve the parent namespace and leaf name for a ``.`` separated path."""
        if not port_path:
            raise ValueError('A non-empty port path is required.')
        parts = port_path.split('.')
        parent = self._resolve_namespace('.'.join(parts[:-1])) if len(parts) > 1 else self.builder
        return parent, parts[-1]

    def _get_path_value(self, port_path: str):
        """Return the value stored at ``port_path``."""
        parent, leaf = self._resolve_parent_and_leaf(port_path)
        return parent.get(leaf)

    def _path_exists(self, port_path: str) -> bool:
        """Return whether ``port_path`` resolves on the current builder."""
        try:
            self._resolve_parent_and_leaf(port_path)
        except (AttributeError, KeyError, ValueError):
            return False
        return True

    def _set_path_value(self, port_path: str, value) -> None:
        """Set the value stored at ``port_path``."""
        parent, leaf = self._resolve_parent_and_leaf(port_path)
        setattr(parent, leaf, value)

    def _get_compatible_port_paths(self, port_path: str) -> list[str]:
        """Return compatibility-linked paths for builder ports that have legacy aliases."""
        paths = [port_path]
        if self.builder is None:
            return paths

        parent_path, leaf = port_path.rsplit('.', 1) if '.' in port_path else ('', port_path)
        if leaf not in {'band_settings', 'bs_kpoints'}:
            return paths

        if parent_path.endswith('path'):
            alias_parent = parent_path.rsplit('.', 1)[0] if '.' in parent_path else ''
        else:
            alias_parent = join_namespace_path(parent_path, 'path')
        alias_path = join_namespace_path(alias_parent, leaf)
        if alias_path != port_path and self._path_exists(alias_path):
            paths.append(alias_path)
        return paths

    def _update_compatible_dict_paths(self, port_path: str, content: dict[str, Any]):
        """Update a Dict-backed port and any compatibility aliases that point to the same setting."""
        for path in dict.fromkeys(self._get_compatible_port_paths(port_path)):
            self._update_dict_path(path, content)
        return self

    def _set_compatible_path_value(self, port_path: str, value):
        """Set a value on a port and any compatibility aliases that mirror it."""
        for path in dict.fromkeys(self._get_compatible_port_paths(port_path)):
            self._set_path_value(path, value)
        return self

    def _update_dict_path(self, port_path: str, content: dict[str, Any], namespace: str | None = None):
        """Update an ``orm.Dict`` port, creating it when absent."""
        if not content:
            return self
        node = self._get_path_value(port_path)
        if node is None:
            node = orm.Dict(dict={namespace: deepcopy(content)} if namespace else deepcopy(content))
        else:
            node = update_dict_node(node, content, namespace=namespace)
        self._set_path_value(port_path, node)
        return self

    def set_incar(self, incar_updates=None, update_all=True, ports=None, namespace='incar', **kwargs):
        """
        Set incar dictionary
        """
        if incar_updates is None and not kwargs:
            return self

        if update_all:
            ports_nodes = recursive_search_dict_with_key(self.builder, 'incar')
        else:
            ports = ports or ['parameters']
            ports_nodes = [[port, self._get_port_node(port)] for port in ports]
        updates = deepcopy(incar_updates or {})
        updates.update(kwargs)
        for port, node in ports_nodes:
            self._update_dict_node(port, updates, dict_node=node, namespace=namespace)
        return self

    def set_options(self, option_updates=None, ports=None, update_all=True, **kwargs):
        """Set the options input port"""
        if option_updates is None and not kwargs:
            return self
        if update_all:
            calc_namespaces = []
            for port, namespace in recursive_search_port_basename(self.builder, 'calc'):
                if 'metadata' in namespace and 'options' in namespace['metadata']:
                    calc_namespaces.append([port, namespace])
        else:
            ports = ports or ['calc']
            calc_namespaces = [[port, self._get_port_node(port)] for port in ports]
        updates = option_updates or {}
        # Use recursive merge so existing nested keys will not be replaced
        updates = recursive_merge(updates, kwargs)
        # Update the options
        for port, namespace in calc_namespaces:
            # Here the port is only updated if the parent namespace is not empty or it is marked as 'required'
            # This is because `options`` is a special none-db port which may exist even if 'populate_defaults' is
            # set to False for namespaces that is optional. Otherwise, these optional namespace becomes 'defined'
            # , triggering its validation and then fails (as other 'required' fields are not defined inside the
            # namespace)
            if has_content(namespace) or namespace._port_namespace._required:
                namespace['metadata']['options'] = recursive_merge(dict(namespace['metadata']['options']), updates)
        return self

    def set_resources(self, resources_updates=None, ports=None, update_all=True, **kwargs):
        """Set the options input port"""
        if resources_updates is None and not kwargs:
            return self
        if update_all:
            calc_namespaces = []
            for port, namespace in recursive_search_port_basename(self.builder, 'calc'):
                if 'metadata' in namespace and 'options' in namespace['metadata']:
                    calc_namespaces.append([port, namespace])
        else:
            ports = ports or ['calc']
            calc_namespaces = [[port, self._get_port_node(port)] for port in ports]
        # Update the resources
        updates = deepcopy(resources_updates or {})
        updates.update(kwargs)
        for port, namespace in calc_namespaces:
            # Here the port is only updated if the parent namespace is not empty or it is marked as 'required'
            # This is because `options`` is a special none-db port which may exist even if 'populate_defaults' is
            # set to False for namespaces that is optional. Otherwise, these optional namespace becomes 'defined'
            # , triggering its validation and then fails (as other 'required' fields are not defined inside the
            # namespace)
            if has_content(namespace) or namespace._port_namespace._required:
                namespace['metadata']['options']['resources'].update(updates)
        return self

    def _update_ports_by_base_name(
        self, value, port_basename, ports=None, update_all=True, merge=False, skip_empty=True
    ):
        """Update a port by basename"""
        if update_all:
            port_and_nodes = recursive_search_port_basename(self.builder, port_basename)
        else:
            ports = ports or [port_basename]
            port_and_nodes = [[port, self._get_port_node(port)] for port in ports]
        # Update the options
        for port, node in port_and_nodes:
            if merge and isinstance(node, orm.Dict):
                self._set_node_to_port(port, update_dict_node(node, value))
            else:
                self._set_node_to_port(port, value)
        return self

    def _update_dict_node(self, port, update: dict, dict_node=None, namespace=None, reuse_if_possible=True):
        """ """
        if not update:
            return
        dict_node = dict_node or self._get_port_node(port)
        updated = update_dict_node(dict_node, update, namespace=namespace, reuse_if_possible=reuse_if_possible)
        self._set_node_to_port(port, updated)

    def _set_generic_port_by_dict(self, _port_name, value=None, ports=None, update_all=True, skip_empty=True, **kwargs):
        """Set a generic port by a value or kwargs"""
        if value is None and not kwargs:
            return self
        value = value or {}
        value = deepcopy(value)
        value.update(kwargs)
        self._update_ports_by_base_name(
            value, _port_name, ports=ports, update_all=update_all, skip_empty=skip_empty, merge=True
        )

    def _get_port_node(self, port):
        """Return the node corresponds to specific port"""
        parts = port.split('.')
        item = self.builder
        for part in parts:
            item = item.get(part)
        return item

    def _set_node_to_port(self, port, node: orm.Data):
        """Set a node to a specific port of hte builder"""
        if node is None:
            return
        parts = port.split('.')
        item = self.builder
        for part in parts[:-1]:
            item = item[part]
        setattr(item, parts[-1], node)

    def __repr__(self):
        return self.describe()

    def _repr_pretty_(self, p, _=None) -> str:
        """Pretty representation for in the IPython console and notebooks."""
        p.text(self.describe())

    def set_kspacing(self, value, ports=None, update_all=True):
        """Update the kpoints spacing"""
        self._update_ports_by_base_name(value, 'kpoints_spacing', ports=ports, update_all=update_all)
        return self

    def set_kpoints_mesh(self, mesh: list[int], offset=(0.0, 0.0, 0.0), ports=None, update_all=True):
        """Set kpoints mesh"""
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(self.reference_structure)
        kpoints.set_kpoints_mesh(mesh, list(offset))
        self._update_ports_by_base_name(kpoints, 'kpoints', ports=ports, update_all=update_all)
        return self

    def set_label(self, label=None):
        """Alias to set the self.builder.metadata.label"""
        label = label or self.reference_structure.label
        self.builder.metadata.label = label
        return self

    def set_potential_family(self, value, ports=None, update_all=True):
        """Update the potential family"""
        self._update_ports_by_base_name(value, 'potential_family', ports=ports, update_all=update_all)
        return self

    def set_potential_mapping(self, value=None, ports=None, update_all=True, **kwargs):
        """Set the potential mapping"""
        self._set_generic_port_by_dict('potential_mapping', value=value, ports=ports, update_all=update_all, **kwargs)
        return self

    def set_code(self, value, ports=None, update_all=True):
        """Update the code node"""
        if isinstance(value, str):
            value = orm.load_code(value)
        self._update_ports_by_base_name(value, 'code', ports=ports, update_all=update_all)

    def set_settings(self, value, ports=None, update_all=True, **kwargs):
        """Update the `settings` port."""
        self._set_generic_port_by_dict('settings', value=value, ports=ports, update_all=update_all, **kwargs)

    def submit(self) -> orm.WorkChainNode:
        """
        Submit the workflow to the daemon and return the workchain node.

        :returns: The submitted workchain node
        :rtype: orm.WorkChainNode
        """
        return submit(self.builder)

    def run_get_node(self, verbose: bool = True) -> orm.WorkChainNode:
        """
        Run the workflow with the current python process.

        :param verbose: If True, print debugging information for failed calculations
        :type verbose: bool

        :returns: Tuple containing the workflow outputs and the workchain node
        :rtype: orm.WorkChainNode
        """
        output = run_get_node(self.builder)
        # Verbose output (for debugging)
        if not output.node.is_finished_ok and verbose:
            for node in output.node.called_descendants:
                if isinstance(node, orm.CalcJobNode):
                    stdout = node.outputs.retrieved.get_object_content('vasp_output')
                    print(node, 'STDOUT:', stdout)
                    print(node, 'Retrieved files:', node.outputs.retrieved.list_object_names())
                    script = node.base.repository.get_object_content('_aiidasubmit.sh')
                    print(node, 'Submission script:', script)
                    print(node, 'Exit_message', node.exit_message)
        return output

    def _get_help(self, namespace: str, print_to_stdout: bool = True, inout: str = 'inputs') -> str | None:
        """
        Return the help message for a given namespace.

        The `.` syntax for the namespace is supported for nested namespaces.

        :param namespace: Namespace path (e.g., 'vasp.parameters')
        :type namespace: str
        :param print_to_stdout: Whether to print help to stdout or return it
        :type print_to_stdout: bool
        :param inout: Whether to get help for 'inputs' or 'outputs'
        :type inout: str

        :returns: Help message if print_to_stdout is False, otherwise None
        :rtype: str or None
        """
        levels = namespace.split('.')
        data_dict = self.builder._process_spec.get_description()[inout]
        for key in levels:
            data_dict = data_dict[key]

        if print_to_stdout is True:
            print(data_dict.get('help', 'No help message information found'))
        else:
            return data_dict.get('help', 'No help message information found')

    def get_output_help(self, namespace: str, print_to_stdout: bool = True) -> str | None:
        """
        Return the help message for a given output namespace.

        :param namespace: Output namespace path
        :type namespace: str
        :param print_to_stdout: Whether to print help to stdout or return it
        :type print_to_stdout: bool

        :returns: Help message if print_to_stdout is False, otherwise None
        :rtype: str or None
        """
        self._get_help(namespace, print_to_stdout=print_to_stdout, inout='outputs')

    def get_input_help(self, namespace: str, print_to_stdout: bool = True) -> str | None:
        """
        Return the help message for a given input namespace.

        :param namespace: Input namespace path
        :type namespace: str
        :param print_to_stdout: Whether to print help to stdout or return it
        :type print_to_stdout: bool

        :returns: Help message if print_to_stdout is False, otherwise None
        :rtype: str or None
        """
        self._get_help(namespace, print_to_stdout=print_to_stdout, inout='inputs')

    def schema(self) -> dict[str, Any]:
        """Return a structured description of the generator interface."""
        current = ['builder constructed' if self.builder is not None else 'builder not constructed']
        if self.builder is not None:
            current.append(f'workflow entry point: {self.WF_ENTRYPOINT}')

        return {
            'title': f'{self.__class__.__name__}: {self.WORKFLOW_LABEL}',
            'summary': self.WORKFLOW_SUMMARY,
            'details': {'protocol': self.protocol, 'preset': self.preset_name},
            'canonical_ports': list(self.CANONICAL_PORTS),
            'accessors': list(self.ACCESSOR_DOCS),
            'mutators': list(self.MUTATOR_DOCS),
            'notes': list(self.NOTES),
            'examples': list(self.EXAMPLES),
            'current': current,
        }

    def describe(self, print_to_stdout: bool = False) -> str:
        """Return a human-readable description of the generator interface."""
        rendered = _format_schema_block(self.schema())
        if print_to_stdout:
            print(rendered)
        return rendered

    def accessors(self) -> dict[str, str]:
        """Return accessor descriptions keyed by accessor method name."""
        mapping = {}
        for item in self.ACCESSOR_DOCS:
            name, _, description = item.partition(': ')
            mapping[name] = description or ''
        return mapping

    def show_schema(self) -> str:
        """Print and return the generator schema."""
        return self.describe(print_to_stdout=True)


class NamespaceGenerator:
    """A typed view into a sub-namespace of a generator's builder."""

    NAMESPACE_LABEL = 'Namespace view'
    NAMESPACE_SUMMARY = 'Typed view into part of a builder.'
    ACCESSOR_DOCS: tuple[str, ...] = ()
    MUTATOR_DOCS: tuple[str, ...] = ()
    NOTES: tuple[str, ...] = ()

    def __init__(self, root: BaseInputGenerator, parent=None, namespace_path: str | None = None) -> None:
        self.root = root
        self.parent = parent if parent is not None else root
        self.namespace_path = namespace_path or ''

    @property
    def builder(self):
        return self.root.builder

    @property
    def namespace(self):
        return self.root._resolve_namespace(self.namespace_path)

    @property
    def reference_structure(self):
        return self.root.reference_structure

    def _join(self, relative_path: str | None = None) -> str:
        return join_namespace_path(self.namespace_path, relative_path)

    def __repr__(self):
        return self.describe()

    def _repr_pretty_(self, p, cycle=None) -> None:
        """Pretty representation for IPython and notebooks."""
        p.text(self.describe())

    def schema(self) -> dict[str, Any]:
        """Return a structured description of the namespace view."""
        location = self.namespace_path or '<root>'
        current = []
        if self.builder is not None:
            try:
                current.append(f'namespace path: {location}')
                current.append(f'content: {self.namespace}')
            except Exception:  # pragma: no cover - defensive for partially constructed builders
                current.append(f'namespace path: {location}')
        return {
            'title': f'{self.__class__.__name__}: {self.NAMESPACE_LABEL}',
            'summary': self.NAMESPACE_SUMMARY,
            'details': {'path': location},
            'accessors': list(self.ACCESSOR_DOCS),
            'mutators': list(self.MUTATOR_DOCS),
            'notes': list(self.NOTES),
            'current': current,
        }

    def describe(self, print_to_stdout: bool = False) -> str:
        """Return a human-readable description of the namespace interface."""
        rendered = _format_schema_block(self.schema())
        if print_to_stdout:
            print(rendered)
        return rendered

    def set_incar(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_dict_path(self._join('parameters'), updates, namespace='incar')

    def set_settings(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_dict_path(self._join('settings'), updates)

    def set_options(self, value=None, **kwargs):
        updates = recursive_merge(deepcopy(value or {}), kwargs)
        if not updates:
            return self
        path = self._join('calc.metadata.options')
        current = deepcopy(dict(self.root._get_path_value(path)))
        merged = recursive_merge(current, updates)
        self.root._set_path_value(path, merged)
        return self

    def set_resources(self, value=None, **kwargs):
        updates = recursive_merge(deepcopy(value or {}), kwargs)
        if not updates:
            return self
        path = self._join('calc.metadata.options.resources')
        current = deepcopy(dict(self.root._get_path_value(path) or {}))
        current.update(updates)
        self.root._set_path_value(path, current)
        return self

    def set_code(self, value):
        if isinstance(value, str):
            value = orm.load_code(value)
        self.root._set_path_value(self._join('code'), value)
        return self

    def set_kspacing(self, value):
        self.root._set_path_value(self._join('kpoints_spacing'), orm.Float(value))
        try:
            self.root._set_path_value(self._join('kpoints'), None)
        except Exception:  # pragma: no cover - optional path
            pass
        return self

    def set_kpoints(self, kpoints):
        self.root._set_path_value(self._join('kpoints'), kpoints)
        return self

    def set_kpoints_mesh(self, mesh: list[int], offset=(0.0, 0.0, 0.0)):
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(self.reference_structure)
        kpoints.set_kpoints_mesh(mesh, list(offset))
        return self.set_kpoints(kpoints)

    def set_potential_family(self, value):
        if not isinstance(value, orm.Str):
            value = orm.Str(value)
        self.root._set_path_value(self._join('potential_family'), value)
        return self

    def set_potential_mapping(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        if not isinstance(updates, orm.Dict):
            updates = orm.Dict(dict=updates)
        self.root._set_path_value(self._join('potential_mapping'), updates)
        return self


class VaspCalcNamespaceGenerator(NamespaceGenerator):
    """Typed generator for a VASP child namespace."""

    NAMESPACE_LABEL = 'VASP calculation branch'
    NAMESPACE_SUMMARY = 'Configure one VASP execution namespace.'
    MUTATOR_DOCS = (
        'set_incar(...): update INCAR-like parameters',
        'set_settings(...): update parser/settings Dict',
        'set_options(...): update scheduler metadata options',
        'set_resources(...): update metadata.options.resources',
        'set_code(...): replace the code node',
        'set_kspacing(...): set k-point spacing',
        'set_kpoints(...): attach explicit k-points',
        'set_kpoints_mesh(...): attach a Monkhorst-Pack mesh',
    )


class RelaxSettingsGenerator(NamespaceGenerator):
    """Generator for a relax workflow namespace exposing ``relax_settings``."""

    NAMESPACE_LABEL = 'Relax settings branch'
    NAMESPACE_SUMMARY = 'Configure workflow-level relaxation settings.'
    MUTATOR_DOCS = ('set_relax_settings(...): update relax_settings',)

    def set_relax_settings(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_dict_path(self._join('relax_settings'), updates)


class RelaxNamespaceGenerator(NamespaceGenerator):
    """Generator for a relax workflow namespace."""

    NAMESPACE_LABEL = 'Relax workflow branch'
    NAMESPACE_SUMMARY = 'Configure the child relax workflow.'
    ACCESSOR_DOCS = (
        'vasp(): access the underlying VASP calculation namespace',
        'relax(): access the workflow relax_settings namespace',
    )
    NOTES = ('Use relax().vasp() for calc inputs and relax().relax() for relax_settings.',)

    def vasp(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self.root, self, self._join('vasp'))

    def relax(self) -> RelaxSettingsGenerator:
        return RelaxSettingsGenerator(self.root, self, self.namespace_path)


class StaticNamespaceGenerator(VaspCalcNamespaceGenerator):
    """Generator for a final static calculation namespace."""

    NAMESPACE_LABEL = 'Final static branch'
    NAMESPACE_SUMMARY = 'Configure the final single-point/static calculation.'


class PathNamespaceGenerator(NamespaceGenerator):
    """Generator for path-generation controls."""

    NAMESPACE_LABEL = 'Band-path controls'
    NAMESPACE_SUMMARY = 'Compatibility view over top-level band-path controls.'
    MUTATOR_DOCS = (
        'set_band_settings(...): update top-level band_settings',
        'set_bs_kpoints(...): set explicit band-path k-points',
    )
    NOTES = ('Top-level band_settings and bs_kpoints are canonical; path() is a compatibility alias.',)

    def _sibling_path(self, leaf: str) -> str:
        if self.namespace_path.endswith('path'):
            parent = self.namespace_path.rsplit('.', 1)[0] if '.' in self.namespace_path else ''
            candidate = join_namespace_path(parent, leaf)
            try:
                self.root._get_path_value(candidate)
                return candidate
            except AttributeError:
                pass
        return self._join(leaf)

    def set_band_settings(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_compatible_dict_paths(self._sibling_path('band_settings'), updates)

    def set_bs_kpoints(self, kpoints):
        return self.root._set_compatible_path_value(self._sibling_path('bs_kpoints'), kpoints)


class ReuseNamespaceGenerator(NamespaceGenerator):
    """Generator for restart/reuse controls."""

    NAMESPACE_LABEL = 'Reuse controls'
    NAMESPACE_SUMMARY = 'Attach restart inputs for reuse-driven execution.'
    MUTATOR_DOCS = (
        'use_restart_folder(...): set reuse.restart_folder',
        'use_chgcar(...): set reuse.chgcar',
        'skip_scf(...): skip SCF using restart_folder or chgcar',
    )

    def use_restart_folder(self, folder: orm.RemoteData):
        self.root._set_path_value(self._join('restart_folder'), folder)
        return self

    def use_chgcar(self, chgcar):
        self.root._set_path_value(self._join('chgcar'), chgcar)
        return self

    def skip_scf(self, *, restart_folder: orm.RemoteData | None = None, chgcar=None):
        if restart_folder is None and chgcar is None:
            raise ValueError('skip_scf requires either a restart_folder or a chgcar.')
        if restart_folder is not None:
            self.use_restart_folder(restart_folder)
        if chgcar is not None:
            self.use_chgcar(chgcar)
        return self


class BandsChildGenerator(VaspCalcNamespaceGenerator):
    """Generator for the band-structure NSCF child namespace."""

    NAMESPACE_LABEL = 'Band-structure child branch'
    NAMESPACE_SUMMARY = 'Configure the explicit bands child of an NSCF workflow.'
    MUTATOR_DOCS = VaspCalcNamespaceGenerator.MUTATOR_DOCS + (
        'enable(): keep the bands child active',
        'disable(): disable bands by switching to DOS-only mode',
    )

    def enable(self):
        return self

    def disable(self):
        return self.parent.set_only_dos(True)


class DosChildGenerator(VaspCalcNamespaceGenerator):
    """Generator for the DOS child namespace."""

    NAMESPACE_LABEL = 'DOS child branch'
    NAMESPACE_SUMMARY = 'Configure the explicit DOS child of an NSCF workflow.'
    MUTATOR_DOCS = VaspCalcNamespaceGenerator.MUTATOR_DOCS + (
        'enable(distance=None): enable DOS and optionally set dos_kpoints_distance',
        'disable(): disable DOS',
        'set_kpoints_distance(...): update dos_kpoints_distance',
    )

    def enable(self, distance: float | None = None):
        self.parent.set_run_dos(True)
        if distance is not None:
            self.parent.set_band_settings(dos_kpoints_distance=distance)
        return self

    def disable(self):
        self.parent.set_run_dos(False)
        return self

    def set_kpoints_distance(self, value: float):
        self.parent.set_band_settings(dos_kpoints_distance=value)
        return self


class NscfNamespaceGenerator(NamespaceGenerator):
    """Generator for an NSCF execution namespace."""

    NAMESPACE_LABEL = 'NSCF execution branch'
    NAMESPACE_SUMMARY = 'Configure semilocal SCF/NSCF execution.'
    ACCESSOR_DOCS = (
        'scf(): access the SCF child namespace',
        'bands(): access the bands child namespace',
        'dos(): access the DOS child namespace',
        'reuse(): access restart/reuse inputs',
    )
    MUTATOR_DOCS = (
        'set_band_settings(...): update top-level band_settings',
        'set_bs_kpoints(...): set explicit band-path k-points',
        'enable_dos(distance=None): enable DOS and optionally set dos_kpoints_distance',
        'set_only_dos(flag=True): disable bands and keep DOS only',
        'set_run_dos(flag=True): toggle DOS execution',
        'use_restart_folder(...): attach restart_folder under reuse',
        'use_chgcar(...): attach chgcar under reuse',
        'skip_scf(...): skip SCF using restart_folder or chgcar',
    )

    def _sibling_path(self, leaf: str) -> str:
        if self.namespace_path.endswith('nscf'):
            parent = self.namespace_path.rsplit('.', 1)[0] if '.' in self.namespace_path else ''
            candidate = join_namespace_path(parent, leaf)
            try:
                self.root._get_path_value(candidate)
                return candidate
            except AttributeError:
                candidate = join_namespace_path(parent, 'path', leaf)
                try:
                    self.root._get_path_value(candidate)
                    return candidate
                except AttributeError:
                    pass
        return self._join(leaf)

    def scf(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self.root, self, self._join('scf'))

    def bands(self) -> BandsChildGenerator:
        return BandsChildGenerator(self.root, self, self._join('bands'))

    def dos(self) -> DosChildGenerator:
        return DosChildGenerator(self.root, self, self._join('dos'))

    def reuse(self) -> ReuseNamespaceGenerator:
        return ReuseNamespaceGenerator(self.root, self, self._join('reuse'))

    def set_band_settings(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_compatible_dict_paths(self._sibling_path('band_settings'), updates)

    def set_bs_kpoints(self, kpoints):
        return self.root._set_compatible_path_value(self._sibling_path('bs_kpoints'), kpoints)

    def use_restart_folder(self, folder: orm.RemoteData):
        return self.reuse().use_restart_folder(folder)

    def use_chgcar(self, chgcar):
        return self.reuse().use_chgcar(chgcar)

    def skip_scf(self, *, restart_folder: orm.RemoteData | None = None, chgcar=None):
        return self.reuse().skip_scf(restart_folder=restart_folder, chgcar=chgcar)

    def enable_dos(self, distance: float | None = None):
        self.set_run_dos(True)
        if distance is not None:
            self.set_band_settings(dos_kpoints_distance=distance)
        return self

    def set_only_dos(self, flag: bool = True):
        self.set_band_settings(only_dos=flag)
        return self

    def set_run_dos(self, flag: bool = True):
        self.set_band_settings(run_dos=flag)
        return self


class BandsWorkflowNamespaceGenerator(NamespaceGenerator):
    """Generator for a nested bands workflow namespace."""

    NAMESPACE_LABEL = 'Nested bands workflow branch'
    NAMESPACE_SUMMARY = 'Configure a child VaspBandsWorkChain inside a larger workflow.'
    ACCESSOR_DOCS = (
        'path(): access compatibility path controls',
        'nscf(): access the semilocal NSCF execution branch',
        'scf(): shortcut for bands().nscf().scf()',
        'bands(): shortcut for bands().nscf().bands()',
        'dos(): shortcut for bands().nscf().dos()',
    )
    MUTATOR_DOCS = (
        'set_band_settings(...): update the child bands workflow band_settings',
        'set_bs_kpoints(...): set explicit child band-path k-points',
        'enable_dos(distance=None): enable DOS for the child bands workflow',
    )

    def path(self) -> PathNamespaceGenerator:
        return PathNamespaceGenerator(self.root, self, self._join('path'))

    def nscf(self) -> NscfNamespaceGenerator:
        namespace_path = self._join('nscf')
        return NscfNamespaceGenerator(self.root, self, namespace_path)

    def scf(self) -> VaspCalcNamespaceGenerator:
        return self.nscf().scf()

    def bands(self) -> BandsChildGenerator:
        return self.nscf().bands()

    def dos(self) -> DosChildGenerator:
        return self.nscf().dos()

    def set_band_settings(self, value=None, **kwargs):
        return self.path().set_band_settings(value, **kwargs)

    def set_bs_kpoints(self, kpoints):
        return self.path().set_bs_kpoints(kpoints)

    def use_restart_folder(self, folder: orm.RemoteData):
        return self.nscf().use_restart_folder(folder)

    def use_chgcar(self, chgcar):
        return self.nscf().use_chgcar(chgcar)

    def enable_dos(self, distance: float | None = None):
        self.set_band_settings(run_dos=True)
        return self.dos().enable(distance=distance)


class StageGenerator(NamespaceGenerator):
    """Generator for stage-local override Dicts in staged workflows."""

    NAMESPACE_LABEL = 'Stage override branch'
    NAMESPACE_SUMMARY = 'Configure stage-local overrides for staged workflows.'
    MUTATOR_DOCS = (
        'set_incar(...): update stage-local parameters.incar',
        'set_settings(...): update stage-local settings',
        'set_options(...): update stage-local options',
        'set_relax_settings(...): update stage-local relax_settings',
    )

    def __init__(self, root: BaseInputGenerator, stage: int | str, parent=None) -> None:
        super().__init__(root, parent=parent, namespace_path=f'stage_{stage}')
        self.stage = str(stage)

    def _stage_port(self, kind: str) -> str:
        return self._join(kind)

    def set_incar(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_dict_path(self._stage_port('parameters'), updates, namespace='incar')

    def set_settings(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_dict_path(self._stage_port('settings'), updates)

    def set_options(self, value=None, **kwargs):
        updates = recursive_merge(deepcopy(value or {}), kwargs)
        return self.root._update_dict_path(self._stage_port('options'), updates)

    def set_relax_settings(self, value=None, **kwargs):
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self.root._update_dict_path(self._stage_port('relax_settings'), updates)


class VaspInputGenerator(BaseInputGenerator):
    """
    Updater for VaspWorkChain's builder
    """

    WORKFLOW_LABEL = 'Single VASP workchain'
    WORKFLOW_SUMMARY = 'Configure one vasp.v2.vasp workflow builder.'
    CANONICAL_PORTS = ('structure', 'parameters', 'settings', 'kpoints/kpoints_spacing')
    ACCESSOR_DOCS = ('vasp(): access the calculation namespace',)
    MUTATOR_DOCS = (
        'set_incar(...): update top-level parameters.incar',
        'set_settings(...): update top-level settings',
        'set_options(...): update calc.metadata.options',
    )
    EXAMPLES = ('gen.vasp().set_incar(encut=600)',)

    def vasp(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, '')


class VaspRelaxInputGenerator(BaseInputGenerator):
    """
    Updater for VaspRelaxWorkChain's builder
    """

    WF_ENTRYPOINT = 'vasp.relax'
    WORKFLOW_LABEL = 'Relaxation workchain'
    WORKFLOW_SUMMARY = 'Configure a relaxation workflow with a child VASP branch and relax_settings.'
    CANONICAL_PORTS = ('structure', 'vasp', 'relax_settings', 'static')
    ACCESSOR_DOCS = (
        'vasp(): access the main VASP calculation branch',
        'relax(): access workflow relax_settings',
        'static(): access the optional final static branch',
    )
    MUTATOR_DOCS = ('set_relax_settings(...): update top-level relax_settings',)
    EXAMPLES = (
        'gen.vasp().set_incar(encut=600)',
        'gen.relax().set_relax_settings(force_cutoff=0.02)',
    )

    def vasp(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'vasp')

    def relax(self) -> RelaxSettingsGenerator:
        return RelaxSettingsGenerator(self, self, '')

    def static(self) -> StaticNamespaceGenerator:
        return StaticNamespaceGenerator(self, self, 'static')

    def set_relax_settings(self, value=None, **kwargs):
        """Set the `relax_settings` port"""
        self._set_generic_port_by_dict('relax_settings', ports=['relax_settings'], value=value, **kwargs)
        return self

    def build(self, structure, code=None, protocol=None, overrides=None, **kwargs):
        builder = super().build(structure=structure, code=code, protocol=protocol, overrides=overrides, **kwargs)
        pdict = builder.vasp.parameters.get_dict()
        pdict['incar'].pop('nsw', None)
        pdict['incar'].pop('ibrion', None)
        pdict['incar'].pop('isif', None)
        # Case if the the parameters is stored
        if builder.vasp.parameters.is_stored:
            builder.vasp.parameters = pdict
        return builder


class VaspBandsInputGenerator(BaseInputGenerator):
    """
    Updater for VaspBandsWorkChain's builder
    """

    WF_ENTRYPOINT = 'vasp.bands'
    WORKFLOW_LABEL = 'Semilocal band-structure workchain'
    WORKFLOW_SUMMARY = 'Configure optional relax plus semilocal SCF/NSCF execution.'
    CANONICAL_PORTS = ('structure', 'relax', 'nscf', 'band_settings', 'bs_kpoints')
    ACCESSOR_DOCS = (
        'relax(): access the optional relax child workflow',
        'nscf(): access the semilocal SCF/NSCF execution branch',
        'path(): compatibility alias for top-level band_settings and bs_kpoints',
    )
    MUTATOR_DOCS = (
        'set_band_settings(...): update top-level band_settings',
        'set_bs_kpoints(...): set explicit band-path k-points',
        'use_restart_folder(...): attach restart_folder to the NSCF reuse branch',
        'use_chgcar(...): attach chgcar to the NSCF reuse branch',
    )
    NOTES = ('Canonical usage is gen.relax() and gen.nscf(); path() is a compatibility alias.',)
    EXAMPLES = (
        'gen.relax().vasp().set_incar(encut=620)',
        'gen.nscf().scf().set_incar(ismear=0)',
        'gen.nscf().dos().enable(distance=0.03)',
    )

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def nscf(self) -> NscfNamespaceGenerator:
        namespace_path = 'nscf' if self.builder is not None and self.builder.get('nscf') is not None else ''
        return NscfNamespaceGenerator(self, self, namespace_path)

    def path(self) -> PathNamespaceGenerator:
        return PathNamespaceGenerator(self, self, 'path')

    def set_band_settings(self, value=None, **kwargs):
        """Set the `band_settings` port"""
        updates = deepcopy(value or {})
        updates.update(kwargs)
        return self._update_compatible_dict_paths('band_settings', updates)

    def set_bs_kpoints(self, kpoints):
        return self._set_compatible_path_value('bs_kpoints', kpoints)

    def use_restart_folder(self, folder: orm.RemoteData):
        self.nscf().use_restart_folder(folder)
        if self.builder.get('restart_folder') is not None:
            self.builder.restart_folder = folder
        return self

    def use_chgcar(self, chgcar):
        self.nscf().use_chgcar(chgcar)
        if self.builder.get('chgcar') is not None:
            self.builder.chgcar = chgcar
        return self

    def set_settings(self, *args, **kwargs):
        """Set the settings port"""
        ports = ['scf.settings']
        if self.builder is not None and self.builder.get('nscf') is not None:
            ports = ['nscf.scf.settings', 'scf.settings']
        return super().set_settings(*args, ports=ports, update_all=False, **kwargs)

    def build(self, structure, code=None, protocol=None, overrides=None, run_relax=True, **kwargs):
        """
        Generate builder base on a given structure and overrides (if supplied)
        """
        build_request = self._resolve_build_request(
            code=code,
            protocol=protocol,
            overrides=overrides,
            options=kwargs.pop('options', {}),
        )

        builder = WorkflowFactory(self.WF_ENTRYPOINT).get_builder_from_protocol(
            code=self._load_code_node(build_request['code']),
            structure=structure,
            protocol=build_request['protocol'],
            overrides=build_request['overrides'],
            options=build_request['options'],
            run_relax=run_relax,
            **kwargs,
        )
        return self._finalize_builder(builder, profile_config=build_request['profile_config'])


class VaspConvergenceInputGenerator(BaseInputGenerator):
    """Updater for VaspConvergenceWorkChain"""

    WF_ENTRYPOINT = 'vasp.converge'
    WORKFLOW_LABEL = 'Convergence workchain'
    WORKFLOW_SUMMARY = 'Configure cutoff and k-point convergence tests around one VASP child branch.'
    CANONICAL_PORTS = ('structure', 'vasp', 'conv_settings')
    ACCESSOR_DOCS = ('vasp(): access the child VASP calculation branch',)
    MUTATOR_DOCS = ('set_conv_settings(...): update conv_settings',)

    def vasp(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'vasp')

    def set_conv_settings(self, value=None, **kwargs):
        """Set the `conv_settings` port"""
        self._set_generic_port_by_dict('conv_settings', ports=['conv_settings'], value=value, **kwargs)
        return self


class VaspHybridBandsInputGenerator(VaspBandsInputGenerator):
    """Input generator for ``VaspHybridBandsWorkChain``."""

    WF_ENTRYPOINT = 'vasp.hybrid_bands'
    WORKFLOW_LABEL = 'Hybrid band-structure workchain'
    WORKFLOW_SUMMARY = 'Configure optional relax plus hybrid split-path SCF execution.'
    CANONICAL_PORTS = ('structure', 'relax', 'scf', 'band_settings', 'bs_kpoints')
    ACCESSOR_DOCS = (
        'relax(): access the optional relax child workflow',
        'scf(): access the hybrid SCF branch used for split-path runs',
    )
    MUTATOR_DOCS = (
        'set_band_settings(...): update top-level band_settings',
        'set_bs_kpoints(...): set explicit band-path k-points',
    )
    NOTES = ('Hybrid bands uses scf(), not nscf(). It does not expose reuse or DOS branches.',)
    EXAMPLES = (
        'gen.relax().vasp().set_incar(encut=620)',
        'gen.scf().set_incar(ismear=0)',
        'gen.set_band_settings(kpoints_per_split=150)',
    )

    def scf(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'scf')

    def nscf(self) -> NscfNamespaceGenerator:  # pragma: no cover - defensive API guard
        raise AttributeError('VaspHybridBandsInputGenerator uses `scf()`, not `nscf()`.')

    def use_restart_folder(self, folder: orm.RemoteData):  # pragma: no cover - defensive API guard
        raise AttributeError('VaspHybridBandsInputGenerator does not expose restart-folder reuse inputs.')

    def use_chgcar(self, chgcar):  # pragma: no cover - defensive API guard
        raise AttributeError('VaspHybridBandsInputGenerator does not expose CHGCAR reuse inputs.')

    def set_settings(self, *args, **kwargs):
        """Set the settings port for the hybrid SCF branch."""
        return super(VaspBandsInputGenerator, self).set_settings(
            *args, ports=['scf.settings'], update_all=False, **kwargs
        )


class VaspNscfInputGenerator(VaspBandsInputGenerator):
    """Updater for ``VaspNscfWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.nscf'
    WORKFLOW_LABEL = 'Standalone NSCF workchain'
    WORKFLOW_SUMMARY = 'Configure semilocal SCF/NSCF execution without relaxation or path generation.'
    CANONICAL_PORTS = ('structure', 'scf', 'bands', 'dos', 'reuse', 'band_settings', 'bs_kpoints')
    ACCESSOR_DOCS = (
        'scf(): access the SCF child branch',
        'bands(): access the bands child branch',
        'dos(): access the DOS child branch',
        'reuse(): access restart/reuse inputs',
    )
    MUTATOR_DOCS = (
        'set_band_settings(...): update top-level band_settings',
        'set_bs_kpoints(...): set explicit band-path k-points',
        'skip_scf(...): skip SCF using restart data',
        'enable_dos(...): enable DOS and optionally set dos_kpoints_distance',
    )

    def scf(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'scf')

    def bands(self) -> BandsChildGenerator:
        return BandsChildGenerator(self, self, 'bands')

    def dos(self) -> DosChildGenerator:
        return DosChildGenerator(self, self, 'dos')

    def reuse(self) -> ReuseNamespaceGenerator:
        return ReuseNamespaceGenerator(self, self, 'reuse')

    def skip_scf(self, *, restart_folder: orm.RemoteData | None = None, chgcar=None):
        return self.nscf().skip_scf(restart_folder=restart_folder, chgcar=chgcar)

    def enable_dos(self, distance: float | None = None):
        return self.nscf().enable_dos(distance=distance)

    def set_only_dos(self, flag: bool = True):
        return self.nscf().set_only_dos(flag)

    def set_run_dos(self, flag: bool = True):
        return self.nscf().set_run_dos(flag)


class VaspDoubleRelaxInputGenerator(VaspRelaxInputGenerator):
    """Input generator for ``VaspDoubleRelaxWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.double_relax'
    WORKFLOW_LABEL = 'Double-relax workchain'
    WORKFLOW_SUMMARY = 'Configure a shared relax branch plus stage-local overrides for two relax stages.'
    CANONICAL_PORTS = ('structure', 'relax', 'stage_1', 'stage_2')
    ACCESSOR_DOCS = (
        'relax(): access shared relax inputs',
        'stage_1(): access first-stage overrides',
        'stage_2(): access second-stage overrides',
    )
    EXAMPLES = (
        'gen.relax().vasp().set_incar(encut=520)',
        'gen.stage_2().set_relax_settings(force_cutoff=0.02)',
    )

    def build(self, structure, code=None, protocol=None, overrides=None, **kwargs):
        """Build the staged double-relax workflow without assuming a top-level ``vasp`` namespace."""
        return BaseInputGenerator.build(
            self, structure=structure, code=code, protocol=protocol, overrides=overrides, **kwargs
        )

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def stage_1(self) -> StageGenerator:
        return StageGenerator(self, 1, parent=self)

    def stage_2(self) -> StageGenerator:
        return StageGenerator(self, 2, parent=self)


class VaspRelaxBandsInputGenerator(VaspBandsInputGenerator):
    """Input generator for ``VaspRelaxBandsWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.relax_bands'
    WORKFLOW_LABEL = 'Relax-plus-bands workchain'
    WORKFLOW_SUMMARY = 'Configure a mandatory relax stage followed by a nested bands workflow.'
    CANONICAL_PORTS = ('structure', 'relax', 'bands')
    ACCESSOR_DOCS = (
        'relax(): access the top-level relax workflow',
        'bands(): access the nested bands workflow',
    )
    EXAMPLES = (
        'gen.relax().relax().set_relax_settings(force_cutoff=0.02)',
        'gen.bands().scf().set_incar(ismear=0)',
        'gen.bands().set_band_settings(run_dos=True)',
    )

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def bands(self) -> BandsWorkflowNamespaceGenerator:
        return BandsWorkflowNamespaceGenerator(self, self, 'bands')

    def set_settings(self, *args, **kwargs):
        """Set settings across the nested relax+bands workflow namespaces."""
        return BaseInputGenerator.set_settings(self, *args, update_all=True, **kwargs)

    def build(
        self,
        structure,
        code=None,
        protocol=None,
        overrides=None,
        relax_protocol=None,
        band_protocol=None,
        **kwargs,
    ):
        """Generate a builder for the native relax-plus-bands workflow."""
        build_request = self._resolve_build_request(
            code=code,
            protocol=protocol,
            overrides=overrides,
            options=kwargs.pop('options', {}),
        )
        relax_protocol = build_request['protocol'] if relax_protocol is None else relax_protocol
        band_protocol = build_request['protocol'] if band_protocol is None else band_protocol

        builder = WorkflowFactory(self.WF_ENTRYPOINT).get_builder_from_protocol(
            code=self._load_code_node(build_request['code']),
            structure=structure,
            relax_protocol=relax_protocol,
            band_protocol=band_protocol,
            overrides=build_request['overrides'],
            options=build_request['options'],
            **kwargs,
        )
        return self._finalize_builder(builder, profile_config=build_request['profile_config'])


class VaspMPGGADoubleRelaxInputGenerator(BaseInputGenerator):
    """Input generator for ``VaspMPGGADoubleRelaxWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.mp_gga_double_relax'

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def stage_1(self) -> StageGenerator:
        return StageGenerator(self, 1, parent=self)

    def stage_2(self) -> StageGenerator:
        return StageGenerator(self, 2, parent=self)


class VaspMPGGARelaxStaticInputGenerator(BaseInputGenerator):
    """Input generator for ``VaspMPGGARelaxStaticWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.mp_gga_relax_static'

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def static(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'static')


class VaspMPMetaGGADoubleRelaxInputGenerator(BaseInputGenerator):
    """Input generator for ``VaspMPMetaGGADoubleRelaxWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.mp_meta_gga_double_relax'

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def stage_1(self) -> StageGenerator:
        return StageGenerator(self, 1, parent=self)

    def stage_2(self) -> StageGenerator:
        return StageGenerator(self, 2, parent=self)


class VaspMPMetaGGARelaxStaticInputGenerator(BaseInputGenerator):
    """Input generator for ``VaspMPMetaGGARelaxStaticWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.mp_meta_gga_relax_static'

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def static(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'static')


class VaspMP24DoubleRelaxInputGenerator(BaseInputGenerator):
    """Input generator for ``VaspMP24DoubleRelaxWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.mp24_double_relax'

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def stage_1(self) -> StageGenerator:
        return StageGenerator(self, 1, parent=self)

    def stage_2(self) -> StageGenerator:
        return StageGenerator(self, 2, parent=self)


class VaspMP24RelaxStaticInputGenerator(BaseInputGenerator):
    """Input generator for ``VaspMP24RelaxStaticWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.mp24_relax_static'

    def relax(self) -> RelaxNamespaceGenerator:
        return RelaxNamespaceGenerator(self, self, 'relax')

    def static(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'static')


class VaspMatPesStaticInputGenerator(BaseInputGenerator):
    """Input generator for ``MatPesStaticWorkChain``."""

    WF_ENTRYPOINT = 'vasp.v2.matpes_static'
    WORKFLOW_LABEL = 'MatPES static flow workchain'
    WORKFLOW_SUMMARY = 'Configure PBE static followed by r2SCAN static with WAVECAR reuse.'
    CANONICAL_PORTS = ('structure', 'static1', 'static2')
    ACCESSOR_DOCS = (
        'static1(): access the first PBE static stage',
        'static2(): access the second r2SCAN static stage',
    )
    EXAMPLES = (
        'gen.static1().set_incar(encut=600)',
        'gen.static2().set_incar(ismear=0)',
    )

    def static1(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'static1')

    def static2(self) -> VaspCalcNamespaceGenerator:
        return VaspCalcNamespaceGenerator(self, self, 'static2')


def update_dict_node(
    node: orm.Dict,
    content: dict[str, Any],
    namespace: str | None = None,
    reuse_if_possible: bool = True,
) -> orm.Dict:
    """
    Update a Dict node with new content.

    Optionally updates a specific namespace within the Dict node.
    If the node is stored and immutable, creates a new node with updated content.

    :param node: The Dict node to update
    :type node: orm.Dict
    :param content: Dictionary content to merge into the node
    :type content: dict
    :param namespace: Optional namespace key within the Dict to update
    :type namespace: str or None
    :param reuse_if_possible: Whether to reuse the existing node if content is unchanged
    :type reuse_if_possible: bool

    :returns: Updated Dict node (may be the same or a new node)
    :rtype: orm.Dict
    """
    # Get pure-python dictionary
    dtmp = node.get_dict()
    dtmp_backup = None
    if reuse_if_possible and node.is_stored:
        dtmp_backup = deepcopy(dtmp)
    if namespace:
        left = dtmp.get(namespace, {})
    else:
        left = dtmp
    left = recursive_merge(left, content)
    # If namepsace is supplied, only update the target namespace inside the dict
    if namespace:
        dtmp[namespace] = left
    else:
        dtmp = left
    if node.is_stored:
        # There is no need to update the node if the content is the same as before
        if reuse_if_possible and dtmp == dtmp_backup:
            return node
        # The content is different, but the node is immutable, so we create a new node
        return orm.Dict(dict=dtmp)
    node.set_dict(dtmp)
    return node


def incar_dict_to_relax_settings(incar_in: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Convert INCAR tags to relax_settings and remove them from INCAR.

    Extracts relaxation-specific INCAR parameters (NSW, IBRION, EDIFFG) and
    converts them to equivalent relax_settings options.

    :param incar_in: Input dictionary containing INCAR parameters
    :type incar_in: dict

    :returns: Tuple of (updated_incar_dict, relax_settings_dict)
    :rtype: tuple
    """
    # Convert INCAR tags to relax_settings
    updated = {}
    incar_out = dict(incar_in)
    nsw = incar_out['incar'].pop('nsw', None)
    if nsw is not None:
        updated['steps'] = nsw
    # Convert ibrion
    ibrion = incar_out['incar'].pop('ibrion', None)
    if ibrion == 1:
        updated['algo'] = 'rd'
    if ibrion == 2:
        updated['algo'] = 'cg'
    # Convert ediffg
    ediffg = incar_out['incar'].pop('ediffg', None)
    if ediffg is not None:
        updated['force_cutoff'] = ediffg
    return incar_out, updated


def recursive_search_dict_with_key(namespace, search_key):
    """Recursively search for Dict node with certain key"""
    ports = []
    for port_key in namespace._valid_fields:
        value = namespace.get(port_key)
        if isinstance(value, orm.Dict):
            if search_key in value.get_dict():
                ports.append([port_key, value])
        if isinstance(value, ProcessBuilderNamespace):
            ports.extend(
                [
                    [port_key + '.' + sub_key, sub_value]
                    for sub_key, sub_value in recursive_search_dict_with_key(value, search_key)
                ]
            )
    return ports


def recursive_search_port_basename(namespace, basename):
    """Recursively search for Dict node with certain key"""
    ports = []
    for port_key in namespace._valid_fields:
        value = namespace.get(port_key)
        if port_key == basename:
            ports.append([port_key, value])
        if isinstance(value, ProcessBuilderNamespace):
            ports.extend(
                [
                    [port_key + '.' + sub_key, sub_value]
                    for sub_key, sub_value in recursive_search_port_basename(value, basename)
                ]
            )
    return ports


def has_content(mapping):
    """Check if a dictionary is all empty"""
    for key, value in mapping.items():
        if hasattr(value, 'items'):
            if has_content(value) is True:
                return True
        else:
            return True
    return False
