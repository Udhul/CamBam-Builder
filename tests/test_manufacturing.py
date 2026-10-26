"""Reusable native setup contracts, independent of consumer design recipes."""

from dataclasses import replace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from cambam_builder import CBProject
from cambam_builder.native.manufacturing import (
    ComponentPlacement, SetupFrame, SetupStock, compose_setup,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import build_xml_tree, serialize_cambam_bytes


class ManufacturingTests(unittest.TestCase):
    def setUp(self):
        self.source = CBProject("component")
        layer = self.source.add_layer("geometry")
        self.parent = self.source.add_rect(layer, width=100, height=50, identifier="parent")
        self.root = self.source.add_rect(layer, width=30, height=20,
                                         identifier="outline", parent=self.parent, groups=["component"])
        self.feature = self.source.add_rect(layer, corner=(3, 4), width=6, height=2,
                                            identifier="groove", parent=self.root, groups=["cuts"])
        self.source.translate_primitive(self.parent, 10, 20)
        self.source.translate_primitive(self.root, 2, 1)
        self.source.translate_primitive_z(self.root, 0.5)
        self.front = self.source.add_part("front", stock_thickness=8, stock_material="Birch",
                                          default_spindle_speed=12000, default_tool_diameter=3)
        self.back = self.source.add_part("back", stock_thickness=8, stock_material="Birch")
        self.source.add_pocket_mop(self.front, target_group="cuts", identifier="pocket",
                                   target_depth=-2, tool_number=2, tool_diameter=3)
        self.source.add_profile_mop(self.front, [self.root], identifier="cutout",
                                    target_depth=-8, tool_number=1, tool_diameter=4)
        self.source.add_pocket_mop(self.back, [self.feature], identifier="relief",
                                   target_depth=-1, tool_number=3, tool_diameter=2)
        self.stock = SetupStock(200, 100, 8, "Birch", origin=(5, 7))
        self.placement = ComponentPlacement(self.source, self.root, "one", (self.front,))

    def compose(self, *placements):
        return compose_setup("setup", self.stock, placements or (self.placement,))

    def test_repeated_components_keep_independent_identity_groups_and_native_values(self):
        before = serialize_cambam_bytes(self.source)
        result = self.compose(self.placement, replace(self.placement, name="two", frame=SetupFrame(x=50)))
        self.assertEqual(before, serialize_cambam_bytes(self.source))
        self.assertEqual(["one/front", "two/front"], [p.user_identifier for p in result.list_parts()])
        self.assertEqual(["one/pocket", "one/cutout", "two/pocket", "two/cutout"],
                         [m.user_identifier for m in result.list_mops()])
        self.assertIsNone(result.get_primitive("one/parent"))
        for name in ("one", "two"):
            root, groove = result.get_primitive(f"{name}/outline"), result.get_primitive(f"{name}/groove")
            self.assertIs(result.get_parent_of_primitive(groove), root)
            self.assertIsNone(result.get_parent_of_primitive(root))
            self.assertEqual(result.get_mop_target_group(f"{name}/pocket"), f"{name}/cuts")
            self.assertEqual(result.get_mop_targets(f"{name}/pocket"), [groove.internal_id])
            self.assertNotEqual(root.internal_id, self.root.internal_id)
            part = result.get_part(f"{name}/front")
            self.assertEqual((part.stock_width, part.stock_height, part.stock_thickness), (200, 100, 8))
            self.assertEqual(part.stock_drawing_origin, (5, 7, 0))
            self.assertEqual(part.default_spindle_speed, 12000)
            self.assertEqual(part.default_tool_diameter, 3)
        result.get_mop("one/pocket").target_depth = -3
        result.get_primitive("one/groove").width = 9
        self.assertEqual(result.get_mop("two/pocket").target_depth, -2)
        self.assertEqual(self.feature.width, 6)

    def test_front_back_frame_world_witnesses_and_z_are_preserved_over_two_cycles(self):
        front = self.compose()
        back = self.compose(replace(self.placement, parts=(self.back,), frame=SetupFrame(x=100, reflect_x=True)))
        # Source inherited offsets: (10,20)+(2,1)+(3,4)=(15,25).
        np.testing.assert_allclose(front.get_primitive("one/groove").get_absolute_coordinates()[0], (15, 25))
        np.testing.assert_allclose(back.get_primitive("one/groove").get_absolute_coordinates()[0], (85, 25))
        self.assertEqual([m.user_identifier for m in back.list_mops()], ["one/relief"])
        self.assertEqual(back.get_primitive("one/groove").get_total_z_offset(), 0.5)
        ids = {p.user_identifier: p.internal_id for p in back.list_primitives()}
        for _ in range(2):
            back = read_cambam_bytes(serialize_cambam_bytes(back))
            groove = back.get_primitive("one/groove")
            # Reflected Rect is natively serialized as a Pline; its first vertex
            # and parent binding still have the same physical drawing meaning.
            np.testing.assert_allclose(groove.get_absolute_coordinates()[0], (85, 25))
            self.assertEqual(groove.get_total_z_offset(), 0.5)
            self.assertEqual(back.get_mop_targets("one/relief"), [groove.internal_id])
            self.assertEqual(back.get_mop("one/relief").target_depth, -1)
            self.assertEqual(back.get_part("one/back").stock_drawing_origin, (5, 7, 0))
            self.assertEqual(ids, {p.user_identifier: p.internal_id for p in back.list_primitives()})

    def test_part_selection_order_is_explicit_and_source_order_is_unchanged(self):
        result = self.compose(replace(self.placement, parts=(self.back, self.front)))
        self.assertEqual([m.user_identifier for m in result.list_mops()],
                         ["one/relief", "one/pocket", "one/cutout"])
        self.assertEqual([p.user_identifier for p in self.source.list_parts()], ["front", "back"])

    def test_rotated_frame_and_nonzero_stock_surface_do_not_change_mop_coordinates(self):
        self.front.stock_surface = 2
        self.stock = replace(self.stock, surface=2)
        mop = self.source.get_mop("pocket")
        mop.stock_surface = 2
        result = self.compose(replace(self.placement, frame=SetupFrame(90, 10, 90)))
        np.testing.assert_allclose(result.get_primitive("one/groove").get_absolute_coordinates()[0], (65, 25))
        self.assertEqual(result.get_mop("one/pocket").stock_surface, 2)
        self.assertEqual(result.get_mop("one/pocket").target_depth, -2)
        self.assertEqual(result.get_part("one/front").stock_drawing_origin, (5, 7, 2))

    def test_live_groups_are_scoped_until_native_interchange_snapshots_targets(self):
        result = self.compose(self.placement, replace(self.placement, name="two"))
        extra = result.add_rect(result.list_layers()[0], identifier="extra", groups=["one/cuts"])
        self.assertIn(extra.internal_id, result.get_mop_targets("one/pocket"))
        self.assertNotIn(extra.internal_id, result.get_mop_targets("two/pocket"))
        reopened = read_cambam_bytes(serialize_cambam_bytes(result))
        self.assertEqual(set(reopened.get_mop_targets("one/pocket")), set(result.get_mop_targets("one/pocket")))
        self.assertIsNone(reopened.get_mop_target_group("one/pocket"))

    def test_units_are_explicit_without_numeric_conversion(self):
        self.source.units = "Inches"
        with self.assertRaisesRegex(ValueError, "units"):
            self.compose()
        result = compose_setup("inch", self.stock, [self.placement], units="Inches")
        self.assertEqual(result.units, "Inches")
        self.assertEqual(result.get_part("one/front").stock_thickness, 8)
        self.source.units = None
        with self.assertRaisesRegex(ValueError, "units"):
            compose_setup("unset", self.stock, [self.placement], units=None)

    def test_incompatible_stock_and_source_setup_context_are_rejected(self):
        cases = {"stock_material": "Aluminum", "stock_thickness": 10, "stock_surface": 1,
                 "stock_present": False, "machining_origin": (1, 0), "nesting_method": "Grid"}
        for field, invalid in cases.items():
            original = getattr(self.front, field)
            with self.subTest(field=field):
                setattr(self.front, field, invalid)
                with self.assertRaises(ValueError):
                    self.compose()
                setattr(self.front, field, original)

    def test_bad_part_selections_fail_instead_of_silently_dropping_operations(self):
        self.source.add_part("empty", stock_thickness=8, stock_material="Birch")
        for parts in (("missing",), (self.front, self.front), ("empty",)):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                self.compose(replace(self.placement, parts=parts))
        self.source.add_profile_mop(self.front, [self.parent], identifier="outside")
        with self.assertRaisesRegex(ValueError, "subtree"):
            self.compose()
        # An unrelated/unselected face may target outside without blocking this one.
        self.compose(replace(self.placement, parts=(self.back,)))

    def test_empty_group_selection_rejects(self):
        self.source.set_primitive_groups(self.feature, [])
        with self.assertRaisesRegex(ValueError, "subtree"):
            self.compose()

    def test_failed_later_placement_leaves_all_sources_unchanged(self):
        before = serialize_cambam_bytes(self.source)
        with self.assertRaises(ValueError):
            self.compose(self.placement, replace(self.placement, name="bad", root="absent"))
        self.assertEqual(before, serialize_cambam_bytes(self.source))
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.compose(self.placement, self.placement)

    def test_conflicting_project_tool_defaults_reject(self):
        different = self.source.clone()
        different.default_tool_diameter = 12
        with self.assertRaisesRegex(ValueError, "default tool"):
            self.compose(self.placement, replace(self.placement, source=different, name="two"))

    def test_imported_machining_context_and_inherited_values_are_preserved(self):
        xml = build_xml_tree(self.source).getroot()
        options = xml.find("MachiningOptions")
        ET.SubElement(options, "Style").text = "Global style"
        ET.SubElement(options, "StyleLibrary").text = "Example-mm"
        ET.SubElement(options, "ClearancePlane", state="Value").text = "7.5"
        operations = xml.findall("./parts/part/machineops/*")
        self.assertEqual(len(operations), 3)
        for operation in operations:
            operation.find("ClearancePlane").set("state", "Default")
        imported = read_cambam_bytes(ET.tostring(xml))
        placement = replace(self.placement, source=imported, parts=("front",))
        result = self.compose(placement, replace(placement, name="two"))
        for _ in range(2):
            output = build_xml_tree(result).getroot()
            self.assertEqual(ET.tostring(options), ET.tostring(output.find("MachiningOptions")))
            operations = output.findall("./parts/part/machineops/*")
            self.assertEqual(len(operations), 4)
            for operation in operations:
                self.assertEqual(operation.find("ClearancePlane").get("state"), "Default")
            result = read_cambam_bytes(ET.tostring(output))
        result._xml_machining_options.find("Style").text = "Changed output only"
        self.assertEqual(imported._xml_machining_options.findtext("Style"), "Global style")
        with self.assertRaisesRegex(ValueError, "MachiningOptions"):
            self.compose(placement, replace(self.placement, name="fresh"))

    def test_manual_tab_intent_is_rejected_before_creating_unexportable_placement(self):
        self.source.get_mop("cutout").tab_method = "Manual"
        with self.assertRaisesRegex(ValueError, "manual holding tabs"):
            self.compose(replace(self.placement, frame=SetupFrame(x=10)))

    def test_finite_rigid_frames_and_positive_stock_declarations(self):
        for kwargs in ({"x": float("nan")}, {"y": float("inf")}, {"rotation_degrees": "90"},
                       {"reflect_x": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SetupFrame(**kwargs)
        for kwargs in ({"width": 0}, {"height": -1}, {"thickness": float("inf")},
                       {"origin": (1,)}, {"origin": (float("nan"), 0)}, {"material": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.stock, **kwargs)
        with self.assertRaises(ValueError):
            replace(self.placement, parts=())
        with self.assertRaises(ValueError):
            compose_setup("empty", self.stock, [])


if __name__ == "__main__":
    unittest.main()
