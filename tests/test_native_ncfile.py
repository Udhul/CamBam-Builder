"""Native NCFile references, plain fields and targetless document ownership."""

import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from cambam_builder import CBProject
from cambam_builder.native.cam import NCFileMop
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def native_document(fields: str = '', *, enabled: str = 'true') -> bytes:
    # Independent of the writer: installed MOPNCFile XmlSerializer establishes
    # these element names, plain enum fields and omission defaults.
    return f'''<CADFile Version="0.9.8.0" Name="external program" units="Millimeters">
      <layers/><parts><part Name="Part"><machineops>
      <ncfile Enabled="{enabled}"><Name>Imported</Name>{fields}</ncfile>
      </machineops></part></parts></CADFile>'''.encode()


def encoded_mop(project: CBProject) -> ET.Element:
    return ET.fromstring(serialize_cambam_bytes(project)).find('./parts/part/machineops/ncfile')


class NativeNCFileTests(unittest.TestCase):
    def make_project(self):
        project = CBProject('mixed native job')
        layer = project.add_layer('Geometry')
        target = project.add_rect(layer, identifier='outline', width=10, height=8)
        part = project.add_part('Part')
        profile = project.add_profile_mop(part, [target], identifier='cutout')
        return project, part, target, profile

    def test_mixed_job_roundtrip_identity_notes_order_and_clone(self):
        project, part, target, profile = self.make_project()
        source = r'programs\input program.nc'
        mop = project.add_ncfile_mop(
            part, source, identifier='supplied', name='Supplied program',
            target_mop=profile, place_last=False, native_tag='Operator supplied',
            distance_mode='Incremental', arc_center_mode='IncrementalFromP1')
        self.assertIsInstance(mop, NCFileMop)
        self.assertEqual(project.get_mop_targets(mop), [])
        self.assertIs(project.get_part_of_mop(mop), part)
        for _ in range(2):
            element = encoded_mop(project)
            self.assertIsNone(element.find('primitive'))
            self.assertEqual(element.find('SourceFile').attrib, {})
            self.assertEqual(element.find('DistanceMode').attrib, {})
            project = read_cambam_bytes(serialize_cambam_bytes(project))
            reopened = project.get_mop('supplied')
            self.assertEqual(reopened.internal_id, mop.internal_id)
            self.assertEqual(reopened.source_file, source)
            self.assertEqual(reopened.name, 'Supplied program')
            self.assertEqual(reopened.native_tag, 'Operator supplied')
            self.assertEqual((reopened.distance_mode, reopened.arc_center_mode),
                             ('Incremental', 'IncrementalFromP1'))
            self.assertEqual([m.user_identifier for m in project.get_mops_in_part('Part')],
                             ['supplied', 'cutout'])
            self.assertEqual(project.get_mop_targets(reopened), [])
        clone = project.clone()
        clone.get_mop('supplied').source_file = 'replacement.nc'
        self.assertEqual(project.get_mop('supplied').source_file, source)
        copied = CBProject('copied geometry')
        project.copy_primitive_tree('outline', copied, preserve_ids=False)
        self.assertIsNone(copied.get_mop('supplied'))
        self.assertIsNotNone(copied.get_mop('cutout'))

    def test_fresh_defaults_and_explicit_common_metadata(self):
        project = CBProject('minimal')
        part = project.add_part('Part', default_spindle_speed=10000)
        mop = project.add_ncfile_mop(part, 'external.nc', identifier='external')
        self.assertEqual([node.tag for node in encoded_mop(project)], ['Name', 'Tag', 'SourceFile'])
        self.assertEqual(mop.source_file, 'external.nc')
        # Explicitly supplied values remain explicit even when equal to the
        # inherited dataclass defaults. No native toolpath effect is inferred.
        explicit = project.add_ncfile_mop(part, 'other.nc', identifier='explicit',
                                         optimisation_mode='Standard', tool_number=0)
        element = explicit.to_xml_element(project, [])
        self.assertEqual(element.findtext('OptimisationMode'), 'Standard')
        self.assertEqual(element.find('OptimisationMode').attrib, {'state': 'Value'})
        self.assertEqual(element.findtext('ToolNumber'), '0')
        mop.optimisation_mode = 'Standard'
        self.assertEqual(mop.to_xml_element(project, []).findtext('OptimisationMode'), 'Standard')
        for name in ('source_file', 'distance_mode', 'arc_center_mode'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'Unsupported MOP parameter'):
                    mop.set_parameter_state(name, 'Default')

    def test_native_defaults_empty_source_and_unknown_payload_survive(self):
        project = read_cambam_bytes(native_document('''
          <CustomMOPHeader state="Default" vendor="keep">(header)</CustomMOPHeader>
          <FutureOption state="Native">preserve</FutureOption>'''))
        mop = project.list_mops()[0]
        self.assertEqual((mop.source_file, mop.distance_mode, mop.arc_center_mode),
                         ('', 'Absolute', 'Default'))
        mop.name = 'Renamed'
        element = encoded_mop(project)
        for tag in ('SourceFile', 'DistanceMode', 'ArcCenterMode', 'primitive'):
            self.assertIsNone(element.find(tag))
        self.assertEqual(element.findtext('FutureOption'), 'preserve')
        self.assertEqual(element.find('CustomMOPHeader').attrib,
                         {'state': 'Default', 'vendor': 'keep'})
        mop.source_file = 'new.nc'
        mop.distance_mode = 'Incremental'
        mop.arc_center_mode = 'Absolute'
        mop.custom_mop_header = '(edited)'
        element = encoded_mop(project)
        for tag in ('SourceFile', 'DistanceMode', 'ArcCenterMode'):
            self.assertEqual(element.find(tag).attrib, {})
        self.assertEqual(element.findtext('SourceFile'), 'new.nc')
        self.assertEqual(element.findtext('DistanceMode'), 'Incremental')
        self.assertEqual(element.findtext('ArcCenterMode'), 'Absolute')
        self.assertEqual(element.find('CustomMOPHeader').attrib,
                         {'state': 'Value', 'vendor': 'keep'})
        encoded_mop(read_cambam_bytes(serialize_cambam_bytes(project)))
        disabled = read_cambam_bytes(native_document(enabled='false'))
        disabled_mop = disabled.list_mops()[0]
        disabled_mop.enabled = True
        with self.assertRaisesRegex(ValueError, 'without a source_file'):
            disabled_mop.to_xml_element(disabled, [])
        disabled_mop.source_file = 'repaired.nc'
        self.assertEqual(encoded_mop(disabled).findtext('SourceFile'), 'repaired.nc')

    def test_all_observed_modes_and_same_value_edits_roundtrip(self):
        for distance in ('Absolute', 'Incremental'):
            for arc in ('Default', 'Absolute', 'IncrementalFromP1', 'IncrementalToP1',
                        'IncrementalFromP2', 'IncrementalToP2'):
                with self.subTest(distance=distance, arc=arc):
                    project = read_cambam_bytes(native_document(
                        f'<SourceFile>x.nc</SourceFile><DistanceMode>{distance}</DistanceMode>'
                        f'<ArcCenterMode>{arc}</ArcCenterMode>'))
                    mop = project.list_mops()[0]
                    mop.distance_mode = distance
                    mop.arc_center_mode = arc
                    element = encoded_mop(project)
                    self.assertEqual(element.findtext('DistanceMode'), distance)
                    self.assertEqual(element.findtext('ArcCenterMode'), arc)
                    self.assertEqual(element.find('DistanceMode').attrib, {})

    def test_unsupported_imported_modes_preserved_but_not_activated_or_switched(self):
        for field, tag in (('distance_mode', 'DistanceMode'), ('arc_center_mode', 'ArcCenterMode')):
            fields = f'<SourceFile>x.nc</SourceFile><{tag}>FutureMode</{tag}>'
            project = read_cambam_bytes(native_document(fields, enabled='false'))
            mop = project.list_mops()[0]
            mop.source_file = 'edited.nc'
            self.assertEqual(encoded_mop(project).findtext(tag), 'FutureMode')
            for value in ('FutureMode', 'Absolute'):
                with self.subTest(field=field, value=value):
                    project = read_cambam_bytes(native_document(fields))
                    mop = project.list_mops()[0]
                    setattr(mop, field, value)
                    with self.assertRaisesRegex(ValueError, 'imported unsupported NCFile'):
                        mop.to_xml_element(project, [])
            project = read_cambam_bytes(native_document(fields, enabled='false'))
            mop = project.list_mops()[0]
            mop.enabled = True
            with self.assertRaisesRegex(ValueError, 'activate'):
                mop.to_xml_element(project, [])

    def test_invalid_authoring_is_atomic_and_mutated_export_rejects(self):
        project = CBProject('invalid')
        part = project.add_part('Part')
        for name, value in (('source_file', ''), ('source_file', '  '),
                            ('source_file', None), ('source_file', 'x\n.nc'),
                            ('distance_mode', 'Unknown'), ('arc_center_mode', 'Unknown'),
                            ('arc_center_mode', [])):
            with self.subTest(name=name, value=value):
                kwargs = {'source_file': 'x.nc', name: value}
                with self.assertLogs('cambam_builder.native.project', level='ERROR'):
                    self.assertIsNone(project.add_ncfile_mop(part, identifier='invalid', **kwargs))
                self.assertEqual(project.list_mops(), [])
                self.assertIsNone(project.get_entity('invalid'))
                mop = NCFileMop(**kwargs)
                with self.assertRaises(ValueError):
                    mop.to_xml_element(project, [])
        mop = project.add_ncfile_mop(part, 'valid.nc')
        mop.source_file = ''
        with self.assertRaisesRegex(ValueError, 'source_file'):
            mop.to_xml_element(project, [])

    def test_targets_rejected_at_each_boundary(self):
        project, part, target, profile = self.make_project()
        for kwargs in ({'targets': [target]}, {'targets': []}, {'target_group': 'future'}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, 'cannot target'):
                    project.add_ncfile_mop(part, 'x.nc', **kwargs)
        mop = project.add_ncfile_mop(part, 'x.nc')
        with self.assertRaisesRegex(ValueError, 'cannot target'):
            project.set_mop_targets(mop, [target])
        self.assertEqual(project.get_mop_targets(mop), [])
        project.set_mop_targets(mop, [])
        with self.assertRaisesRegex(ValueError, 'cannot target'):
            project.set_mop_target_group(mop, 'empty group')
        with self.assertRaisesRegex(ValueError, 'cannot target'):
            mop.to_xml_element(project, [1])
        project._mop_targets[mop.internal_id] = 'future group'
        with self.assertRaisesRegex(ValueError, 'cannot target'):
            mop.to_xml_element(project, [])

    def test_malformed_native_fields_and_targets_reject(self):
        for fields in (
                '<SourceFile>a.nc</SourceFile><SourceFile>b.nc</SourceFile>',
                '<DistanceMode>Absolute</DistanceMode><DistanceMode>Incremental</DistanceMode>',
                '<ArcCenterMode>Default</ArcCenterMode><ArcCenterMode>Absolute</ArcCenterMode>',
                '<SourceFile state="Value">x.nc</SourceFile>',
                '<DistanceMode state="Default">Absolute</DistanceMode>',
                '<ArcCenterMode><nested/></ArcCenterMode>', '<DistanceMode/>',
                '<primitive><prim>1</prim></primitive>', '<primitive>1</primitive>',
                '<primitive/><primitive/>', '<primitive vendor="unknown"/>'):
            with self.subTest(fields=fields):
                with self.assertLogs('cambam_builder.native.reader', level='ERROR'):
                    with self.assertRaises(ValueError):
                        read_cambam_bytes(native_document(fields))

    def test_reference_import_export_performs_no_external_io(self):
        payload = native_document('<SourceFile>does-not-exist/secret.nc</SourceFile>')
        with patch('builtins.open', side_effect=AssertionError('Unexpected external IO')):
            project = read_cambam_bytes(payload)
            data = serialize_cambam_bytes(project)
            reopened = read_cambam_bytes(data)
        self.assertEqual(reopened.list_mops()[0].source_file, 'does-not-exist/secret.nc')


if __name__ == '__main__':
    unittest.main()
