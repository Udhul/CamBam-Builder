"""PointList placement uses owned identity, never retained native numeric IDs."""

import unittest
import uuid
import xml.etree.ElementTree as ET

from cambam_builder import CBProject
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import build_xml_tree


def native_source(reference="20"):
    return f'''<CADFile><layers><layer name="Geometry"><objects>
      <rect id="10" p="0,0,0" w="20" h="10"/>
      <points id="20"><pts><p>3,4,0</p><p>30,4,0</p></pts></points>
    </objects></layer></layers><parts><part Name="Board"><Nesting retained="yes">
      <NestMethod>PointList</NestMethod><PointListID>{reference}</PointListID>
      <GCodeOrder>Auto</GCodeOrder></Nesting><machineops/></part></parts></CADFile>'''.encode()


class PointListNestingTests(unittest.TestCase):
    def test_nonconsecutive_native_ids_rebind_and_preserve_payload(self):
        project = read_cambam_bytes(native_source(), strict=True)
        root = build_xml_tree(project).getroot()
        nesting = root.find("./parts/part/Nesting")
        self.assertEqual(root.find(".//points").get("id"), nesting.findtext("PointListID"))
        self.assertEqual("yes", nesting.get("retained"))
        self.assertEqual("Auto", nesting.findtext("GCodeOrder"))
        loaded = read_cambam_bytes(ET.tostring(root), strict=True)
        self.assertEqual(
            project.get_part_nesting_points("Board").internal_id,
            loaded.get_part_nesting_points("Board").internal_id,
        )

    def test_missing_unresolved_and_wrong_kind_native_references_reject(self):
        for value in ("", "999", "10", "20.5", "abc"):
            with self.subTest(value=value), self.assertLogs("cambam_builder.native.reader", "ERROR"), self.assertRaises(ValueError):
                read_cambam_bytes(native_source(value), strict=True)
        reference = b"<PointListID>20</PointListID>"
        for replacement in (b"", reference * 2):
            source = native_source().replace(reference, replacement)
            with self.subTest(replacement=replacement):
                with self.assertLogs("cambam_builder.native.reader", "ERROR"), self.assertRaises(ValueError):
                    read_cambam_bytes(source, strict=True)

    def test_imported_layout_rebind_and_copy_preserve_native_payload(self):
        source = read_cambam_bytes(native_source(), strict=True)
        part = source.get_part("Board")
        outline = next(p for p in source.list_primitives() if p is not source.get_part_nesting_points(part))
        replacement = source.add_points("Geometry", identifier="replacement", points=[(7, 9)])
        source.set_part_nesting_points(part, replacement)
        source.link_primitive_parent(replacement, outline)
        source.add_profile_mop(part, [outline], identifier="cutout")
        target = CBProject("copied")
        mapping = source.copy_primitive_tree(outline, target, preserve_ids=False)
        root = build_xml_tree(target).getroot()
        nesting = root.find("./parts/part/Nesting")
        self.assertEqual(root.find(".//points").get("id"), nesting.findtext("PointListID"))
        self.assertEqual("yes", nesting.get("retained"))
        self.assertEqual("Auto", nesting.findtext("GCodeOrder"))
        self.assertEqual(mapping[replacement.internal_id], target.get_part_nesting_points(mapping[part.internal_id]).internal_id)

    def make_board(self):
        project = CBProject("board")
        layer = project.add_layer("Geometry")
        outline = project.add_rect(layer, identifier="outline")
        points = project.add_points(layer, identifier="layout", points=[(3, 4), (30, 4)])
        project.link_primitive_parent(points, outline)
        part = project.add_part("Board", nesting_method="PointList", nesting_point_list=points)
        project.add_profile_mop(part, [outline], identifier="cutout")
        return project, outline, points, part

    def assert_reference(self, project, part, points):
        self.assertIs(project.get_part_nesting_points(part), points)
        root = build_xml_tree(project).getroot()
        nesting = root.find("./parts/part/Nesting")
        self.assertEqual(root.find(".//points").get("id"), nesting.findtext("PointListID"))

    def test_direct_authoring_clone_and_earlier_geometry_ids(self):
        project, outline, points, part = self.make_board()
        before = build_xml_tree(project).getroot().findtext("./parts/part/Nesting/PointListID")
        extra = project.add_rect(project.get_layer("Geometry"), identifier="earlier")
        old_id = extra.internal_id
        project._primitives.pop(old_id)
        # Force a lower UUID so the writer must assign a different numeric ID
        # to the existing layout regardless of randomly generated identities.
        extra.internal_id = uuid.UUID(int=1)
        project._primitives[extra.internal_id] = extra
        project._identifier_registry["earlier"] = extra.internal_id
        project._primitive_layer_assignment[extra.internal_id] = project._primitive_layer_assignment.pop(old_id)
        members = project._layer_primitive_membership[project.get_layer("Geometry").internal_id]
        members.remove(old_id)
        members.add(extra.internal_id)
        self.assert_reference(project, part, points)
        after = build_xml_tree(project).getroot().findtext("./parts/part/Nesting/PointListID")
        self.assertEqual(int(before) + 1, int(after))
        clone = project.clone()
        self.assert_reference(clone, clone.get_part("Board"), clone.get_primitive("layout"))

    def test_copy_remaps_and_transfer_clears_source_relationship(self):
        for transfer in (False, True):
            with self.subTest(transfer=transfer):
                source, outline, points, part = self.make_board()
                target = CBProject("destination")
                operation = source.transfer_primitive_tree if transfer else source.copy_primitive_tree
                mapping = operation(outline, target, preserve_ids=False)
                self.assertNotEqual(points.internal_id, mapping[points.internal_id])
                self.assert_reference(target, target.get_part(mapping[part.internal_id]),
                                      target.get_primitive(mapping[points.internal_id]))
                if transfer:
                    self.assertIsNone(source.get_part_nesting_points(part))
                    with self.assertRaisesRegex(ValueError, "PointList"):
                        build_xml_tree(source)
                else:
                    self.assert_reference(source, part, points)

    def test_copy_rejects_external_layout_without_destination_mutation(self):
        source, outline, points, part = self.make_board()
        source.link_primitive_parent(points, None)
        target = CBProject("destination")
        with self.assertRaisesRegex(ValueError, "nesting.*outside"):
            source.copy_primitive_tree(outline, target)
        self.assertEqual([], target.list_parts())
        self.assertEqual([], target.list_primitives())

    def test_bad_authoring_deleted_and_unexported_layout_reject(self):
        project, outline, points, part = self.make_board()
        for invalid in (outline, "absent", uuid.uuid4()):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                project.set_part_nesting_points(part, invalid)
        self.assertIs(project.get_part_nesting_points(part), points)
        project.remove_layer("Geometry")
        with self.assertRaisesRegex(ValueError, "PointList"):
            build_xml_tree(project)
        project.remove_primitive(points)
        self.assertIsNone(project.get_part_nesting_points(part))
        with self.assertRaisesRegex(ValueError, "PointList"):
            build_xml_tree(project)

    def test_missing_and_corrupted_owned_references_cannot_export(self):
        project, outline, points, part = self.make_board()
        for invalid in (outline.internal_id, uuid.uuid4()):
            project._part_nesting_points[part.internal_id] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "PointList"):
                build_xml_tree(project)
        project.set_part_nesting_points(part, None)
        with self.assertRaisesRegex(ValueError, "PointList"):
            build_xml_tree(project)
        project.set_part_nesting_points(part, points)
        project.add_part("Board", nesting_method="None")
        self.assertIsNone(project.get_part_nesting_points(part))
        self.assertIsNone(build_xml_tree(project).getroot().find("./parts/part/Nesting/PointListID"))


if __name__ == "__main__":
    unittest.main()
