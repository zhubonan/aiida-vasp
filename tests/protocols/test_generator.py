"""Test the generator.py utility functions."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiida import orm
from aiida.engine.processes.builder import ProcessBuilderNamespace

from aiida_vasp.protocols.generator import (
    PresetConfig,
    VaspBandsInputGenerator,
    VaspHybridBandsInputGenerator,
    VaspInputGenerator,
    VaspNscfInputGenerator,
    VaspRelaxInputGenerator,
    get_library_path,
    has_content,
    incar_dict_to_relax_settings,
    list_protocol_presets,
    recursive_search_dict_with_key,
    recursive_search_port_basename,
    update_dict_node,
)


class TestGetLibraryPath:
    """Test the get_library_path function."""

    def test_get_library_path(self):
        """Test that get_library_path returns the correct path."""
        path = get_library_path()
        assert isinstance(path, Path)
        assert path.name == 'presets'
        assert 'protocols' in str(path)


class TestListProtocolPresets:
    """Test the list_protocol_presets function."""

    def test_list_protocol_presets_empty_directories(self):
        """Test list_protocol_presets with no preset files."""
        with patch('aiida_vasp.protocols.generator.get_library_path') as mock_get_path:
            # Create temporary directories with no files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                mock_get_path.return_value = temp_path

                with patch('pathlib.Path.expanduser') as mock_expanduser:
                    mock_expanduser.return_value = temp_path
                    presets = list_protocol_presets()
                    assert isinstance(presets, list)
                    assert len(presets) == 0

    def test_list_protocol_presets_with_files(self):
        """Test list_protocol_presets with yaml files."""
        with patch('aiida_vasp.protocols.generator.get_library_path') as mock_get_path:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                mock_get_path.return_value = temp_path

                # Create test yaml files
                (temp_path / 'test1.yaml').touch()
                (temp_path / 'test2.yml').touch()
                (temp_path / 'not_yaml.txt').touch()

                with patch('pathlib.Path.expanduser') as mock_expanduser:
                    mock_expanduser.return_value = Path(temp_dir) / 'nonexistent'
                    presets = list_protocol_presets()

                    assert isinstance(presets, list)
                    assert len(presets) == 2
                    assert all(isinstance(p, Path) for p in presets)
                    assert any('test1.yaml' in str(p) for p in presets)
                    assert any('test2.yml' in str(p) for p in presets)


class TestPresetConfig:
    """Tests for preset file discovery and loading."""

    def test_from_file_supports_yml(self, tmp_path, monkeypatch):
        """Named preset loading should support ``.yml`` files as well as ``.yaml``."""
        preset_path = tmp_path / 'custom.yml'
        preset_path.write_text('name: custom\ndefault_protocol: balanced\ndefault_code: mock-vasp@localhost\n')
        monkeypatch.setattr('aiida_vasp.protocols.generator.get_preset_library_paths', lambda: (tmp_path,))

        preset = PresetConfig.from_file('custom')

        assert preset.name == 'custom'
        assert preset.default_code == 'mock-vasp@localhost'

    def test_from_file_warns_for_deprecated_fields(self, tmp_path, monkeypatch):
        """Deprecated preset keys should warn instead of silently suggesting active behavior."""
        preset_path = tmp_path / 'deprecated.yaml'
        preset_path.write_text(
            '\n'.join(
                [
                    'name: deprecated',
                    'default_protocol: balanced',
                    'default_code: mock-vasp@localhost',
                    'protocol_overrides:',
                    '  foo: bar',
                ]
            )
        )
        monkeypatch.setattr('aiida_vasp.protocols.generator.get_preset_library_paths', lambda: (tmp_path,))

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            preset = PresetConfig.from_file('deprecated')

        assert preset.name == 'deprecated'
        assert any('deprecated fields' in str(record.message) for record in records)


class TestUpdateDictNode:
    """Test the update_dict_node function."""

    def test_update_dict_node_empty_content(self, aiida_profile):
        """Test update_dict_node with empty update content."""
        # Create an unstored Dict node
        original_dict = {'incar': {'nsw': 50, 'ibrion': 2}}
        node = orm.Dict(dict=original_dict)

        # Update with empty content
        update_content = {}

        # Update the node
        updated_node = update_dict_node(node, update_content, namespace='incar')

        # Should return the same node, content unchanged due to empty update
        assert updated_node is node
        expected = {'incar': {'nsw': 50, 'ibrion': 2}}
        assert updated_node.get_dict() == expected

    def test_update_dict_node_unstored_node(self, aiida_profile):
        """Test update_dict_node with unstored node."""
        # Create an unstored Dict node
        original_dict = {'incar': {'nsw': 50, 'ibrion': 2}}
        node = orm.Dict(dict=original_dict)

        # Update content
        update_content = {'nsw': 100, 'ediffg': -0.01}

        # Update the node
        updated_node = update_dict_node(node, update_content, namespace='incar')

        # Should return the same node (modified in place)
        assert updated_node is node
        expected = {'incar': {'nsw': 100, 'ibrion': 2, 'ediffg': -0.01}}
        assert updated_node.get_dict() == expected

    def test_update_dict_node_stored_node_different_content(self, aiida_profile):
        """Test update_dict_node with stored node and different content."""
        # Create and store a Dict node
        original_dict = {'incar': {'nsw': 50, 'ibrion': 2}}
        node = orm.Dict(dict=original_dict)
        node.store()

        # Update with different content
        update_content = {'nsw': 100, 'ediffg': -0.01}

        # Update the node
        updated_node = update_dict_node(node, update_content, namespace='incar')
        expected = {'incar': {'nsw': 100, 'ibrion': 2, 'ediffg': -0.01}}

        assert updated_node is not node
        assert updated_node.get_dict() == expected

    def test_update_dict_node_stored_node_same_content(self, aiida_profile):
        """Test update_dict_node with stored node and same content (reuse_if_possible=True)."""
        # Create and store a Dict node
        original_dict = {'incar': {'nsw': 50, 'ibrion': 2}}
        node = orm.Dict(dict=original_dict)
        node.store()

        # Update with same content
        update_content = {}

        # Update the node with reuse_if_possible=True
        updated_node = update_dict_node(node, update_content, namespace='incar', reuse_if_possible=True)

        # Should return the same node
        assert updated_node is node

    def test_update_dict_node_stored_node_no_reuse(self, aiida_profile):
        """Test update_dict_node with stored node and reuse_if_possible=False."""
        # Create and store a Dict node
        original_dict = {'incar': {'nsw': 50, 'ibrion': 2}}
        node = orm.Dict(dict=original_dict)
        node.store()

        # Update with content
        update_content = {'nsw': 100}

        # Update the node with reuse_if_possible=False
        updated_node = update_dict_node(node, update_content, namespace='incar', reuse_if_possible=False)

        # Should return a new node even if content might be similar
        assert updated_node is not node
        assert isinstance(updated_node, orm.Dict)

    def test_update_dict_node_no_namespace(self, aiida_profile):
        """Test update_dict_node without namespace."""
        # Create an unstored Dict node
        original_dict = {'nsw': 50, 'ibrion': 2}
        node = orm.Dict(dict=original_dict)

        # Update content without namespace
        update_content = {'nsw': 100, 'ediffg': -0.01}

        # Update the node
        updated_node = update_dict_node(node, update_content)

        # Should return the same node (modified in place)
        assert updated_node is node
        assert updated_node.get_dict()['nsw'] == 100
        assert updated_node.get_dict()['ibrion'] == 2
        assert updated_node.get_dict()['ediffg'] == -0.01

    def test_update_dict_node_nonexistent_namespace(self, aiida_profile):
        """Test update_dict_node with non-existent namespace."""
        # Create an unstored Dict node
        original_dict = {'incar': {'nsw': 50}}
        node = orm.Dict(dict=original_dict)

        # Update content with non-existent namespace
        update_content = {'new_param': 123}

        # Update the node
        updated_node = update_dict_node(node, update_content, namespace='kpoints')

        # Should return the same node (modified in place)
        assert updated_node is node
        # The non-existent namespace gets created
        expected = {'incar': {'nsw': 50}, 'kpoints': {'new_param': 123}}
        assert updated_node.get_dict() == expected


class TestIncarDictToRelaxSettings:
    """Test the incar_dict_to_relax_settings function."""

    def test_incar_dict_to_relax_settings_all_params(self):
        """Test converting all relaxation parameters."""
        incar_dict = {
            'incar': {
                'nsw': 100,
                'ibrion': 2,
                'ediffg': -0.01,
                'encut': 400,  # Should remain in incar
            }
        }

        updated_incar, relax_settings = incar_dict_to_relax_settings(incar_dict)

        # Check relax_settings
        assert relax_settings['steps'] == 100
        assert relax_settings['algo'] == 'cg'
        assert relax_settings['force_cutoff'] == -0.01

        # Check updated incar (relaxation params removed)
        assert 'nsw' not in updated_incar['incar']


class TestComposableInputGenerators:
    """Tests for the refactored composable input generator API."""

    @staticmethod
    def _band_overrides(potcar_family_name):
        return {
            'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
            'relax': {'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        }

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_build_and_clone(self, aiida_profile, mock_vasp, potcar_family_name, upload_potcar, vasp_structure):
        gen = VaspInputGenerator()
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides={'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}},
        )
        original_encut = gen.builder.parameters['incar']['encut']
        clone = gen.clone()
        clone.vasp().set_incar(encut=600)

        assert gen.builder.structure == vasp_structure
        assert clone.builder.parameters['incar']['encut'] == 600
        assert gen.builder.parameters['incar']['encut'] == original_encut

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_relax_child_generators(self, aiida_profile, mock_vasp, potcar_family_name, upload_potcar, vasp_structure):
        gen = VaspRelaxInputGenerator()
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides={'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        )
        gen.vasp().set_incar(encut=650)
        gen.relax().set_relax_settings(force_cutoff=0.02)

        assert gen.builder.vasp.parameters['incar']['encut'] == 650
        assert gen.builder.relax_settings['force_cutoff'] == 0.02
        assert gen.builder.static_overrides.get('code') is None
        assert gen.builder.static_overrides.get('parameters') is None

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_relax_static_generator_lazily_initializes_namespace(
        self, aiida_profile, mock_vasp, potcar_family_name, upload_potcar, vasp_structure
    ):
        gen = VaspRelaxInputGenerator()
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides={'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        )

        assert gen.builder.static_overrides.get('code') is None
        assert gen.builder.static_overrides.get('parameters') is None

        gen.static().set_incar(ismear=-5)

        assert gen.builder.static_overrides.get('parameters')['incar']['ismear'] == -5

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_relax_build_does_not_populate_static_settings(
        self, aiida_profile, mock_vasp, potcar_family_name, upload_potcar, vasp_structure
    ):
        gen = VaspRelaxInputGenerator(protocol='balanced')
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides={'vasp': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        )

        assert gen.builder.vasp.settings is not None
        assert gen.builder.static_overrides.get('settings') is None

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_nscf_child_generators(
        self, aiida_profile, localhost, mock_vasp, potcar_family_name, upload_potcar, vasp_structure
    ):
        gen = VaspNscfInputGenerator()
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides={'scf': {'potential_family': potcar_family_name, 'potential_mapping': {'In_d': 'In_d'}}},
        )
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(vasp_structure)
        kpoints.set_kpoints([[0.0, 0.0, 0.0]])
        restart = orm.RemoteData(computer=localhost, remote_path='/tmp')

        gen.scf().set_incar(ismear=0)
        gen.set_bs_kpoints(kpoints)
        gen.enable_dos(distance=0.05)
        gen.set_only_dos(True)
        gen.skip_scf(restart_folder=restart)
        gen.bands().set_incar(ismear=-5)
        gen.dos().set_incar(ismear=1)

        assert gen.builder.scf.parameters['incar']['ismear'] == 0
        assert gen.builder.bands_overrides.get('parameters')['incar']['ismear'] == -5
        assert gen.builder.dos_overrides.get('parameters')['incar']['ismear'] == 1
        assert gen.builder.bs_kpoints == kpoints
        assert gen.builder.band_settings['run_dos'] is True
        assert gen.builder.band_settings['only_dos'] is True
        assert gen.builder.band_settings['dos_kpoints_distance'] == 0.05
        assert gen.builder.reuse.restart_folder == restart

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_bands_generator_views(self, aiida_profile, mock_vasp, potcar_family_name, upload_potcar, vasp_structure):
        gen = VaspBandsInputGenerator()
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides=self._band_overrides(potcar_family_name),
            run_relax=True,
        )
        gen.relax().vasp().set_incar(encut=620)
        gen.nscf().scf().set_incar(ismear=0)
        gen.set_band_settings(run_dos=True)
        gen.nscf().dos().enable(distance=0.04)

        assert gen.builder.relax.vasp.parameters['incar']['encut'] == 620
        assert gen.builder.nscf.scf.parameters['incar']['ismear'] == 0
        assert gen.builder.band_settings['run_dos'] is True
        assert gen.builder.band_settings['dos_kpoints_distance'] == 0.04
        assert gen.builder.path.band_settings['run_dos'] is True
        assert gen.builder.path.band_settings['dos_kpoints_distance'] == 0.04
        assert 'Semilocal band-structure workchain' in repr(gen)
        assert 'nscf(): access the semilocal SCF/NSCF execution branch' in repr(gen)
        assert 'NSCF execution branch' in repr(gen.nscf())
        assert 'scf(): access the SCF child namespace' in repr(gen.nscf())

    @pytest.mark.parametrize(['vasp_structure'], [('str',)], indirect=True)
    def test_hybrid_bands_generator_views(
        self, aiida_profile, mock_vasp, potcar_family_name, upload_potcar, vasp_structure
    ):
        gen = VaspHybridBandsInputGenerator()
        gen.build(
            structure=vasp_structure,
            code='mock-vasp-loose@localhost',
            overrides=self._band_overrides(potcar_family_name),
            run_relax=True,
        )
        gen.relax().vasp().set_incar(encut=620)
        gen.scf().set_incar(ismear=0)
        gen.set_band_settings(kpoints_per_split=150)

        assert gen.builder.relax.vasp.parameters['incar']['encut'] == 620
        assert gen.builder.scf.parameters['incar']['ismear'] == 0
        assert gen.builder.band_settings['kpoints_per_split'] == 150
        assert 'Hybrid band-structure workchain' in repr(gen)
        assert 'scf(): access the hybrid SCF branch used for split-path runs' in repr(gen)
        assert 'uses scf(), not nscf()' in repr(gen)
        assert 'VASP calculation branch' in repr(gen.scf())

        with pytest.raises(AttributeError, match='uses `scf\\(\\)`, not `nscf\\(\\)`'):
            gen.nscf()

    def test_incar_dict_to_relax_settings_ibrion_rd(self):
        """Test converting ibrion=1 to RMM-DIIS algorithm."""
        incar_dict = {
            'incar': {
                'ibrion': 1,
                'encut': 400,
            }
        }

        _, relax_settings = incar_dict_to_relax_settings(incar_dict)

        # Check relax_settings
        assert relax_settings['algo'] == 'rd'
        assert 'steps' not in relax_settings
        assert 'force_cutoff' not in relax_settings

    def test_incar_dict_to_relax_settings_partial_params(self):
        """Test converting with only some relaxation parameters present."""
        incar_dict = {
            'incar': {
                'nsw': 50,
                'encut': 400,
            }
        }

        _, relax_settings = incar_dict_to_relax_settings(incar_dict)

        # Check relax_settings
        assert relax_settings['steps'] == 50
        assert 'algo' not in relax_settings
        assert 'force_cutoff' not in relax_settings

    def test_incar_dict_to_relax_settings_no_relax_params(self):
        """Test with no relaxation parameters."""
        incar_dict = {
            'incar': {
                'encut': 400,
                'ismear': 0,
            }
        }

        updated_incar, relax_settings = incar_dict_to_relax_settings(incar_dict)

        # Check relax_settings (should be empty)
        assert relax_settings == {}

        # Check updated incar (should be unchanged)
        assert updated_incar['incar']['encut'] == 400
        assert updated_incar['incar']['ismear'] == 0

    def test_incar_dict_to_relax_settings_ibrion_other_values(self):
        """Test with ibrion values other than 1 or 2."""
        incar_dict = {
            'incar': {
                'ibrion': 3,  # Not 1 or 2
                'nsw': 100,
            }
        }

        _, relax_settings = incar_dict_to_relax_settings(incar_dict)

        # Check relax_settings
        assert relax_settings['steps'] == 100
        assert 'algo' not in relax_settings  # Should not be set for ibrion != 1 or 2


class TestRecursiveSearchDictWithKey:
    """Test the recursive_search_dict_with_key function."""

    def test_recursive_search_dict_with_key_found(self, aiida_profile):
        """Test finding Dict nodes with specific key."""
        # Create mock namespace with Dict nodes
        namespace = MagicMock(spec=ProcessBuilderNamespace)

        # Create Dict nodes
        dict_with_key = orm.Dict(dict={'incar': {'nsw': 50}})
        dict_without_key = orm.Dict(dict={'other': {'value': 100}})

        # Mock the namespace structure
        namespace._valid_fields = ['parameters', 'settings', 'other']
        namespace.get.side_effect = lambda key: {
            'parameters': dict_with_key,
            'settings': dict_without_key,
            'other': 'not_a_dict',
        }[key]

        results = recursive_search_dict_with_key(namespace, 'incar')

        assert len(results) == 1
        assert results[0][0] == 'parameters'
        assert results[0][1] is dict_with_key

    def test_recursive_search_dict_with_key_nested(self, aiida_profile):
        """Test recursive search in nested namespaces."""
        # Create nested namespace structure
        sub_namespace = MagicMock(spec=ProcessBuilderNamespace)
        main_namespace = MagicMock(spec=ProcessBuilderNamespace)

        # Create Dict nodes
        dict_with_key = orm.Dict(dict={'incar': {'nsw': 50}})

        # Set up sub namespace
        sub_namespace._valid_fields = ['parameters']
        sub_namespace.get.side_effect = lambda key: {'parameters': dict_with_key}[key]

        # Set up main namespace
        main_namespace._valid_fields = ['calc']
        main_namespace.get.side_effect = lambda key: {'calc': sub_namespace}[key]

        results = recursive_search_dict_with_key(main_namespace, 'incar')

        assert len(results) == 1
        assert results[0][0] == 'calc.parameters'
        assert results[0][1] is dict_with_key

    def test_recursive_search_dict_with_key_no_matches(self, aiida_profile):
        """Test when no Dict nodes contain the search key."""
        namespace = MagicMock(spec=ProcessBuilderNamespace)

        dict_without_key = orm.Dict(dict={'other': {'value': 100}})

        namespace._valid_fields = ['settings']
        namespace.get.side_effect = lambda key: {'settings': dict_without_key}[key]

        results = recursive_search_dict_with_key(namespace, 'incar')

        assert len(results) == 0


class TestRecursiveSearchPortBasename:
    """Test the recursive_search_port_basename function."""

    def test_recursive_search_port_basename_direct_match(self):
        """Test finding ports with matching basename."""
        namespace = MagicMock(spec=ProcessBuilderNamespace)

        test_value = 'test_calc'

        namespace._valid_fields = ['calc', 'settings', 'other']
        namespace.get.side_effect = lambda key: {
            'calc': test_value,
            'settings': 'other_value',
            'other': 'another_value',
        }[key]

        results = recursive_search_port_basename(namespace, 'calc')

        assert len(results) == 1
        assert results[0][0] == 'calc'
        assert results[0][1] == test_value

    def test_recursive_search_port_basename_nested(self):
        """Test recursive search for ports in nested namespaces."""
        sub_namespace = MagicMock(spec=ProcessBuilderNamespace)
        main_namespace = MagicMock(spec=ProcessBuilderNamespace)

        test_value = 'nested_calc'

        # Set up sub namespace
        sub_namespace._valid_fields = ['calc']
        sub_namespace.get.side_effect = lambda key: {'calc': test_value}[key]

        # Set up main namespace
        main_namespace._valid_fields = ['vasp']
        main_namespace.get.side_effect = lambda key: {'vasp': sub_namespace}[key]

        results = recursive_search_port_basename(main_namespace, 'calc')

        assert len(results) == 1
        assert results[0][0] == 'vasp.calc'
        assert results[0][1] == test_value

    def test_recursive_search_port_basename_multiple_matches(self):
        """Test finding multiple ports with same basename."""
        sub_namespace1 = MagicMock(spec=ProcessBuilderNamespace)
        sub_namespace2 = MagicMock(spec=ProcessBuilderNamespace)
        main_namespace = MagicMock(spec=ProcessBuilderNamespace)

        test_value1 = 'calc1'
        test_value2 = 'calc2'

        # Set up sub namespaces
        sub_namespace1._valid_fields = ['calc']
        sub_namespace1.get.side_effect = lambda key: {'calc': test_value1}[key]

        sub_namespace2._valid_fields = ['calc']
        sub_namespace2.get.side_effect = lambda key: {'calc': test_value2}[key]

        # Set up main namespace
        main_namespace._valid_fields = ['scf', 'bands']
        main_namespace.get.side_effect = lambda key: {'scf': sub_namespace1, 'bands': sub_namespace2}[key]

        results = recursive_search_port_basename(main_namespace, 'calc')

        assert len(results) == 2
        # Results could be in any order
        result_keys = [r[0] for r in results]
        result_values = [r[1] for r in results]
        assert 'scf.calc' in result_keys
        assert 'bands.calc' in result_keys
        assert test_value1 in result_values
        assert test_value2 in result_values

    def test_recursive_search_port_basename_no_matches(self):
        """Test when no ports match the basename."""
        namespace = MagicMock(spec=ProcessBuilderNamespace)

        namespace._valid_fields = ['settings', 'parameters']
        namespace.get.side_effect = lambda key: {'settings': 'some_value', 'parameters': 'other_value'}[key]

        results = recursive_search_port_basename(namespace, 'calc')

        assert len(results) == 0


class TestHasContent:
    """Test the has_content function."""

    def test_has_content_empty_dict(self):
        """Test has_content with empty dictionary."""
        mapping = {}
        assert has_content(mapping) is False

    def test_has_content_dict(self):
        """Test has_content with simple non-empty dictionary."""
        mapping = {'key': 'value'}
        assert has_content(mapping) is True

        # Test has_content with nested empty dictionaries.
        mapping = {'level1': {'level2': {}}}
        assert has_content(mapping) is False

        # Test has_content with nested dictionaries containing content.
        mapping = {'level1': {'level2': {'key': 'value'}}}
        assert has_content(mapping) is True

    def test_has_content_mixed_empty_and_content(self):
        """Test has_content with mix of empty and content dictionaries."""
        mapping = {'empty': {}, 'with_content': {'key': 'value'}}
        assert has_content(mapping) is True

    def test_has_content_deep_nested(self):
        """Test has_content with deeply nested structure."""
        mapping = {'level1': {'level2': {'level3': {'level4': {'key': 'value'}}}}}
        assert has_content(mapping) is True

    def test_has_content_all_nested_empty(self):
        """Test has_content with all nested dictionaries empty."""
        mapping = {'level1': {'level2a': {}, 'level2b': {'level3': {}}}, 'other': {}}
        assert has_content(mapping) is False

    def test_has_content_non_dict_values(self):
        """Test has_content with non-dictionary values."""
        mapping = {'string': 'value', 'number': 42, 'list': [1, 2, 3], 'none': None}
        # Should return True on first non-dict value found
        assert has_content(mapping) is True
