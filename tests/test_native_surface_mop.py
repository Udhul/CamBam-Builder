"""Native 3D Profile schema, bounded edits and state preservation."""

import unittest
import xml.etree.ElementTree as ET

from cambam_builder import CBProject
from cambam_builder.native.cam import (
    Surface3DMop, EngraveMop, MOP_SURFACE3D_FIELD_POLICIES,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def native_document(parameters: str = '') -> bytes:
    # Installed MOP3DSurface XmlSerializer and Surface output establish these
    # native names. This input is deliberately independent of the Python writer.
    return f'''<CADFile Version="0.9.8.0" Name="native" units="Millimeters">
      <layers><layer name="Geometry"><objects>
      <surface id="7"><verts><v>0,0,-1</v><v>2,0,-1</v><v>0,2,-.5</v></verts>
      <faces><f>0,1,2</f></faces></surface></objects></layer></layers>
      <parts><part Name="Part"><machineops><surface3d Enabled="true">
      <primitive><prim>7</prim></primitive><Name>Finish</Name>{parameters}
      </surface3d></machineops></part></parts></CADFile>'''.encode()


def encoded_mop(project):
    return ET.fromstring(serialize_cambam_bytes(project)).find('./parts/part/machineops/surface3d')


class NativeSurfaceMopTests(unittest.TestCase):
    def test_native_stock_zero_z_abbreviation_round_trip(self):
        data = native_document().replace(
            b'<part Name="Part">',
            b'<part Name="Part"><Stock><PMin>-1,0.5,-1.2</PMin>'
            b'<PMax>2,1.5</PMax></Stock>')
        project = read_cambam_bytes(data)
        part = project.get_part('Part')
        self.assertEqual((part.stock_width, part.stock_height, part.stock_thickness), (3, 1, 1.2))
        self.assertEqual(part.stock_offset, (-1, .5))
        self.assertEqual(part.stock_surface, 0)
        reopened = read_cambam_bytes(serialize_cambam_bytes(project))
        actual = reopened.get_part('Part')
        self.assertEqual((actual.stock_width, actual.stock_height, actual.stock_thickness), (3, 1, 1.2))
        self.assertEqual(actual.stock_offset, (-1, .5))
        for value in ('2', '2,1.5,0,4', 'nan,1.5', '2,inf', '2,bad', ''):
            with self.subTest(value=value):
                malformed = data.replace(b'<PMax>2,1.5</PMax>', f'<PMax>{value}</PMax>'.encode())
                with self.assertLogs('cambam_builder.native.reader', level='ERROR'):
                    with self.assertRaisesRegex(ValueError, 'Stock PMin/PMax'):
                        read_cambam_bytes(malformed)

    def test_native_integer_fields_emit_integer_lexical_forms(self):
        project = CBProject('integer')
        layer = project.add_layer('geometry')
        target = project.add_rect(layer, identifier='target')
        part = project.add_part('part', default_spindle_speed=10000.0)
        mop = project.add_engrave_mop(part, targets=[target], tool_number=2.0)
        element = mop.to_xml_element(project, [1])
        self.assertEqual(element.findtext('SpindleSpeed'), '10000')
        self.assertEqual(element.findtext('ToolNumber'), '2')
        mop.spindle_speed = 12000.0
        mop.set_parameter_state('spindle_speed', 'Default')
        element = mop.to_xml_element(project, [1])
        self.assertEqual(element.findtext('SpindleSpeed'), '12000')
        self.assertEqual(element.find('SpindleSpeed').get('state'), 'Default')
        for name in ('spindle_speed', 'tool_number'):
            for value in (1.5, float('inf'), float('nan'), True, 2 ** 31):
                with self.subTest(name=name, value=value):
                    bad = EngraveMop(user_identifier='bad', **{name: value})
                    with self.assertRaisesRegex(ValueError, 'integer'):
                        bad.to_xml_element(project, [])

    def test_imported_integer_cache_untouched_preserved_and_edit_normalized(self):
        project = read_cambam_bytes(native_document('''
          <SpindleSpeed state="Default" vendor="keep">10000.0</SpindleSpeed>
          <ToolNumber state="Value">2</ToolNumber>'''))
        mop = project.list_mops()[0]
        self.assertEqual(encoded_mop(project).findtext('SpindleSpeed'), '10000.0')
        mop.spindle_speed = 12000.0
        element = encoded_mop(project)
        self.assertEqual(element.findtext('SpindleSpeed'), '12000')
        self.assertEqual(element.find('SpindleSpeed').get('vendor'), 'keep')
        mop.set_parameter_state('tool_number', 'Default')
        self.assertEqual(encoded_mop(project).find('ToolNumber').get('state'), 'Default')
        mop.tool_number = 1.5
        with self.assertRaisesRegex(ValueError, 'integer'):
            with self.assertLogs('cambam_builder.native.writer', level='ERROR'):
                encoded_mop(project)

    def test_native_omission_is_preserved_and_targets_resolve(self):
        project = read_cambam_bytes(native_document())
        mop = project.list_mops()[0]
        self.assertIsInstance(mop, Surface3DMop)
        self.assertEqual(len(project.get_mop_targets(mop)), 1)
        element = encoded_mop(project)
        for policy in MOP_SURFACE3D_FIELD_POLICIES.values():
            self.assertIsNone(element.find(policy.xml_path[0]))
        reopened = read_cambam_bytes(serialize_cambam_bytes(project))
        self.assertEqual(reopened.list_mops()[0].internal_id, mop.internal_id)
        self.assertEqual(reopened.get_mop_targets(reopened.list_mops()[0]), project.get_mop_targets(mop))

    def test_native_default_cached_and_unknown_values_survive_edits(self):
        project = read_cambam_bytes(native_document('''
          <StepOver state="Default" vendor="keep">0.37</StepOver>
          <Resolution state="Value">0.04</Resolution>
          <Profile3DMethod state="Value">Horizontal</Profile3DMethod>
          <ToolProfile state="Value">BallNose</ToolProfile>
          <LeadOutMove state="Default"><NativeOption enabled="yes">keep</NativeOption></LeadOutMove>
          <FutureSurfaceMode state="Auto">vendor</FutureSurfaceMode>'''))
        mop = project.list_mops()[0]
        self.assertEqual(mop.stepover, .37)
        mop.resolution = .02
        first = encoded_mop(project)
        self.assertEqual(first.find('StepOver').attrib, {'state': 'Default', 'vendor': 'keep'})
        self.assertEqual(first.findtext('StepOver'), '0.37')
        self.assertEqual(first.findtext('Resolution'), '0.02')
        self.assertEqual(first.findtext('LeadOutMove/NativeOption'), 'keep')
        self.assertEqual(first.find('FutureSurfaceMode').attrib, {'state': 'Auto'})
        mop.stepover = .37  # Intentional same-value assignment opts into Value.
        second = encoded_mop(project)
        self.assertEqual(second.find('StepOver').attrib, {'state': 'Value', 'vendor': 'keep'})
        mop.set_parameter_state('resolution', 'Default')
        third = encoded_mop(project)
        self.assertEqual(third.find('Resolution').get('state'), 'Default')
        round_trip = encoded_mop(read_cambam_bytes(serialize_cambam_bytes(project)))
        self.assertEqual(ET.tostring(third), ET.tostring(round_trip))

    def test_unsupported_imported_mode_is_preserved_not_authorable(self):
        project = read_cambam_bytes(native_document('''
          <Profile3DMethod state="Value">WaterlineRough</Profile3DMethod>
          <Mold state="Value">True</Mold><ToolProfile state="Value">EndMill</ToolProfile>'''))
        mop = project.list_mops()[0]
        self.assertEqual(encoded_mop(project).findtext('Profile3DMethod'), 'WaterlineRough')
        mop.profile_3d_method = 'WaterlineRough'
        with self.assertRaisesRegex(ValueError, 'profile_3d_method'):
            with self.assertLogs('cambam_builder.native.writer', level='ERROR'):
                encoded_mop(project)
        mop.profile_3d_method = 'Horizontal'
        with self.assertRaisesRegex(ValueError, 'imported unsupported'):
            with self.assertLogs('cambam_builder.native.writer', level='ERROR'):
                encoded_mop(project)
        project = read_cambam_bytes(native_document(
            '<Profile3DMethod state="Default">WaterlineRough</Profile3DMethod>'))
        project.list_mops()[0].set_parameter_state('profile_3d_method', 'Value')
        with self.assertRaisesRegex(ValueError, 'imported unsupported'):
            with self.assertLogs('cambam_builder.native.writer', level='ERROR'):
                encoded_mop(project)

    def test_fresh_authoring_policy_and_vertical_edit(self):
        project = CBProject('fresh')
        mop = Surface3DMop(user_identifier='finish', tool_diameter=.5,
                           resolution=.04, target_depth=-1, cut_feedrate=240)
        element = mop.to_xml_element(project, [7])
        self.assertEqual(element.tag, 'surface3d')
        self.assertEqual(element.findtext('ToolProfile'), 'BallNose')
        self.assertEqual(element.findtext('RoughingFinishing'), 'Finishing')
        self.assertEqual(element.findtext('Resolution'), '0.04')
        for policy in MOP_SURFACE3D_FIELD_POLICIES.values():
            self.assertEqual(element.find(policy.xml_path[0]).get('state'), 'Value')
        mop.profile_3d_method = 'Vertical'
        mop.set_parameter_state('stepover', 'Default')
        element = mop.to_xml_element(project, [7])
        self.assertEqual(element.findtext('Profile3DMethod'), 'Vertical')
        self.assertEqual(element.find('StepOver').get('state'), 'Default')

    def test_invalid_authored_or_mutated_values_reject_at_export(self):
        for name, value in (
            ('stepover', 0), ('stepover', 1.1), ('resolution', float('nan')),
            ('resolution', -.2), ('boundary_margin', float('inf')),
            ('tool_diameter', 0), ('cut_feedrate', -1), ('target_depth', float('nan')),
            ('profile_3d_method', 'WaterlineFinish'), ('roughing_finishing', 'Roughing'),
            ('tool_profile', 'EndMill'), ('mold', True), ('backface_culling', 'true'),
        ):
            with self.subTest(name=name, value=value):
                project = read_cambam_bytes(native_document())
                setattr(project.list_mops()[0], name, value)
                with self.assertRaises(ValueError):
                    with self.assertLogs('cambam_builder.native.writer', level='ERROR'):
                        encoded_mop(project)


if __name__ == '__main__':
    unittest.main()
