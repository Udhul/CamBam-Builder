"""Cross shape integration coverage for Region and elevation parity.

The detailed geometry tests belong with each entity.  This module checks that
the shared project, XML, transform, persistence, and transfer boundaries keep
the declared XYZ data together across all supported shapes.
"""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cambam_entities import Pline
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file
from cambam_builder.cad_transformations import (
    from_cambam_matrix_str,
    rotation_matrix_deg,
    to_cambam_matrix_str,
    translation_matrix,
)


class ShapeParityIntegrationTests(unittest.TestCase):
    """Verify elevation survives the project boundaries as one contract."""

    @staticmethod
    def _assert_nested_close(testcase, expected, actual, *, atol=1e-8):
        if isinstance(expected, np.ndarray):
            np.testing.assert_allclose(expected, actual, rtol=0, atol=atol)
        elif isinstance(expected, dict):
            testcase.assertEqual(set(expected), set(actual))
            for key in expected:
                ShapeParityIntegrationTests._assert_nested_close(
                    testcase, expected[key], actual[key], atol=atol
                )
        elif isinstance(expected, (list, tuple)):
            testcase.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                ShapeParityIntegrationTests._assert_nested_close(
                    testcase, left, right, atol=atol
                )
        elif isinstance(expected, (float, int, np.floating, np.integer)):
            testcase.assertAlmostEqual(float(expected), float(actual), delta=atol)
        else:
            testcase.assertEqual(expected, actual)

    @staticmethod
    def _xyz_projection(value):
        """Drop a Pline bulge when comparing XYZ geometry after conversion."""
        if isinstance(value, dict):
            return {key: ShapeParityIntegrationTests._xyz_projection(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            if value and all(not isinstance(item, (list, tuple, dict, np.ndarray))
                             for item in value) and len(value) >= 3:
                return tuple(value[:3])
            return type(value)(ShapeParityIntegrationTests._xyz_projection(item)
                               for item in value)
        return value

    def _assert_xyz_projection_close(self, expected, actual, *, atol=1e-8):
        self._assert_nested_close(
            self,
            self._xyz_projection(expected),
            self._xyz_projection(actual),
            atol=atol,
        )

    @staticmethod
    def _primitive_tag_data(element):
        return json.loads(element.findtext("Tag"))

    @staticmethod
    def _primitive_elements(tree):
        return tree.getroot().findall("./layers/layer/objects/*")

    @classmethod
    def _elements_by_identifier(cls, tree):
        return {
            cls._primitive_tag_data(element)["user_id"]: element
            for element in cls._primitive_elements(tree)
        }

    @staticmethod
    def _contour(user_identifier, points, elevations, *, closed=True):
        return Pline(
            user_identifier=user_identifier,
            relative_points=points,
            vertex_z=elevations,
            closed=closed,
        )

    def make_all_shape_project(self):
        project = CBProject("shape-parity-integration")
        # Two successive matrix decompositions amplify rounded rotation values;
        # retain enough digits for the declared 1e-8 world-geometry tolerance.
        project.output_decimals = 15
        layer = project.add_layer("Geometry")
        part = project.add_part("Machining")

        # The root contributes both a world XY pose and a parent Z offset.
        root = project.add_pline(
            layer,
            [(0.0, 0.0), (3.0, 0.0)],
            vertex_z=[8.0, 8.0],
            identifier="root",
            local_z_offset=3.5,
        )
        root.effective_transform = translation_matrix(20.0, -7.0) @ rotation_matrix_deg(17.0)

        pline = project.add_pline(
            layer,
            [(0.0, 0.0, 0.75), (2.0, 0.0, -0.25), (2.0, 1.0, 0.0)],
            vertex_z=[4.25, 4.25, 4.25],
            identifier="bulged-pline",
            parent=root,
            local_z_offset=-1.25,
        )
        pline.effective_transform = translation_matrix(2.0, 3.0)

        circle = project.add_circle(
            layer, center=(1.0, 2.0), diameter=4.5, elevation=-2.75,
            identifier="circle", parent=root, local_z_offset=0.5,
        )
        arc = project.add_arc(
            layer, center=(-2.0, 1.0), radius=2.25, start_angle=25.0,
            extent_angle=210.0, elevation=6.125, identifier="arc", parent=root,
            local_z_offset=-0.75,
        )
        rect = project.add_rect(
            layer, corner=(-3.0, -1.0), width=5.0, height=2.0, elevation=1.875,
            identifier="rect", parent=root, local_z_offset=0.25,
        )
        points = project.add_points(
            layer, [(0.0, 0.0), (1.0, 3.0), (-2.0, 1.0)],
            vertex_z=[-1.5, 0.0, 2.75], identifier="points", parent=root,
            local_z_offset=1.125,
        )
        text = project.add_text(
            layer, "Z text", position=(4.0, -2.0), height=3.0,
            elevation=-4.5, baseline_elevation=-3.25,
            baseline_position=(6.0, -2.0), identifier="text", parent=root,
            local_z_offset=0.875,
        )

        outer = self._contour(
            "outer", [(0.0, 0.0, 0.5), (10.0, 0.0, 0.0), (10.0, 8.0, -0.5), (0.0, 8.0, 0.0)],
            [2.25, 2.25, 2.25, 2.25],
        )
        hole = self._contour(
            "hole", [(2.0, 2.0, 0.25), (4.0, 2.0, 0.0), (4.0, 4.0, 0.0), (2.0, 4.0, 0.0)],
            [2.25, 2.25, 2.25, 2.25],
        )
        region = project.add_region(
            layer, outer, hole_curves=[hole], identifier="region", parent=root,
            local_z_offset=-0.375,
        )

        mop = project.add_profile_mop(
            part, targets=[region], identifier="region-profile", name="Region profile",
            target_depth=-2.0,
        )
        self.assertIsNotNone(mop)
        return project, {
            "root": root,
            "bulged-pline": pline,
            "circle": circle,
            "arc": arc,
            "rect": rect,
            "points": points,
            "text": text,
            "region": region,
        }

    def test_all_shapes_keep_xyz_parent_pose_identity_and_mop_across_two_xml_roundtrips(self):
        source, shapes = self.make_all_shape_project()
        names = tuple(shapes)
        expected = {name: shapes[name].get_absolute_coordinates_xyz() for name in names}

        with tempfile.TemporaryDirectory(prefix="shape-parity-") as directory:
            first = Path(directory) / "first.cb"
            second = Path(directory) / "second.cb"
            save_cambam_file(source, str(first))

            tree = ET.parse(first)
            by_name = self._elements_by_identifier(tree)
            # Bulge and vertex Z occupy independent XML fields.
            pline_points = by_name["bulged-pline"].find("./pts").findall("p")
            self.assertAlmostEqual(float(pline_points[0].get("b")), 0.75)
            self.assertAlmostEqual(float(pline_points[0].text.split(",")[2]), 4.25)
            self.assertAlmostEqual(float(pline_points[1].get("b")), -0.25)
            self.assertAlmostEqual(float(pline_points[1].text.split(",")[2]), 4.25)
            self.assertAlmostEqual(float(by_name["circle"].get("c").split(",")[2]), -2.75)
            self.assertAlmostEqual(float(by_name["arc"].get("p").split(",")[2]), 6.125)
            self.assertAlmostEqual(float(by_name["rect"].get("p").split(",")[2]), 1.875)
            point_nodes = by_name["points"].find("./pts").findall("p")
            self.assertEqual([-1.5, 0.0, 2.75], [float(p.text.split(",")[2]) for p in point_nodes])
            self.assertAlmostEqual(float(by_name["text"].get("p1").split(",")[2]), -4.5)
            self.assertAlmostEqual(float(by_name["text"].get("p2").split(",")[2]), -3.25)

            loaded_once = read_cambam_file(str(first))
            self.assertIsNotNone(loaded_once)
            self.assertEqual(
                {name: shapes[name].internal_id for name in names},
                {name: loaded_once.get_primitive(name).internal_id for name in names},
            )
            for name in names:
                self._assert_nested_close(
                    self, expected[name], loaded_once.get_primitive(name).get_absolute_coordinates_xyz()
                )
            self.assertEqual(
                [loaded_once.get_primitive("region").internal_id],
                loaded_once.get_mop_targets("region-profile"),
            )

            # Reader-created projects use the default precision; retain the
            # fixture precision for the second interchange boundary.
            loaded_once.output_decimals = source.output_decimals
            save_cambam_file(loaded_once, str(second))
            loaded_twice = read_cambam_file(str(second))
            self.assertIsNotNone(loaded_twice)
            for name in names:
                self._assert_nested_close(
                    self, expected[name], loaded_twice.get_primitive(name).get_absolute_coordinates_xyz()
                )
            self.assertEqual(
                [loaded_twice.get_primitive("region").internal_id],
                loaded_twice.get_mop_targets("region-profile"),
            )

    def test_typed_region_xml_object_is_imported_as_one_mop_target(self):
        project, shapes = self.make_all_shape_project()
        with tempfile.TemporaryDirectory(prefix="shape-parity-region-") as directory:
            path = Path(directory) / "region.cb"
            save_cambam_file(project, str(path))
            tree = ET.parse(path)
            region = self._elements_by_identifier(tree)["region"]
            region.set("{http://www.w3.org/2001/XMLSchema-instance}type", "Region")
            tree.write(path, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            imported = loaded.get_primitive("region")
            self.assertEqual(shapes["region"].internal_id, imported.internal_id)
            self.assertEqual([imported.internal_id], loaded.get_mop_targets("region-profile"))
            self.assertEqual(8, len(loaded.list_primitives()))  # root + six shapes + one Region

    def test_translate_z_both_modes_and_xy_rotation_preserve_shape_contract(self):
        project, shapes = self.make_all_shape_project()
        root = shapes["root"]
        before = {name: shapes[name].get_absolute_coordinates_xyz() for name in shapes}

        self.assertTrue(project.translate_primitive_z(root, 2.5, bake=False))
        after_offset = {name: shapes[name].get_absolute_coordinates_xyz() for name in shapes}
        for name in shapes:
            self._assert_nested_z_delta(self, before[name], after_offset[name], 2.5)

        self.assertTrue(project.translate_primitive_z(root, -1.0, bake=True))
        after_baked = {name: shapes[name].get_absolute_coordinates_xyz() for name in shapes}
        for name in shapes:
            self._assert_nested_z_delta(self, before[name], after_baked[name], 1.5)

        xy_before = root.get_absolute_coordinates_xyz()
        self.assertTrue(project.rotate_primitive_deg(root, 90.0, cx=0.0, cy=0.0))
        xy_after = root.get_absolute_coordinates_xyz()
        for old, new in zip(xy_before, xy_after):
            self.assertAlmostEqual(new[2], xy_before[xy_before.index(old)][2])
            self.assertAlmostEqual(new[0], -old[1], delta=1e-8)
            self.assertAlmostEqual(new[1], old[0], delta=1e-8)

    @staticmethod
    def _assert_nested_z_delta(testcase, before, after, delta):
        if isinstance(before, dict):
            testcase.assertEqual(set(before), set(after))
            for key in before:
                if isinstance(before[key], (list, tuple, dict)):
                    ShapeParityIntegrationTests._assert_nested_z_delta(
                        testcase, before[key], after[key], delta
                    )
        elif isinstance(before, (list, tuple)):
            if before and all(not isinstance(item, (list, tuple, dict)) for item in before):
                # XYZ values are always the first three entries; Pline's fourth is bulge.
                if len(before) >= 3:
                    testcase.assertAlmostEqual(float(after[2]), float(before[2]) + delta, delta=1e-8)
                return
            testcase.assertEqual(len(before), len(after))
            for left, right in zip(before, after):
                ShapeParityIntegrationTests._assert_nested_z_delta(testcase, left, right, delta)

    def test_full_bake_recursive_and_nonrecursive_preserve_world_xyz(self):
        def make_hierarchy():
            project = CBProject("z-bake")
            layer = project.add_layer("Geometry")
            root = project.add_pline(layer, [(0, 0), (2, 0)], vertex_z=[1.0, 1.0],
                                     identifier="root", local_z_offset=2.0)
            child = project.add_points(layer, [(1, 0), (2, 1)], vertex_z=[3.0, 4.0],
                                       identifier="child", parent=root, local_z_offset=1.5)
            leaf = project.add_rect(layer, (0, 0), 2, 1, elevation=-2.0,
                                    identifier="leaf", parent=child, local_z_offset=-0.5)
            root.effective_transform = translation_matrix(12.0, 8.0)
            child.effective_transform = translation_matrix(-3.0, 1.0)
            leaf.effective_transform = translation_matrix(2.0, 3.0)
            return project, (root, child, leaf)

        for recursive in (False, True):
            with self.subTest(recursive=recursive):
                project, nodes = make_hierarchy()
                expected = {node.user_identifier: node.get_absolute_coordinates_xyz() for node in nodes}
                self.assertTrue(project.bake_primitive_transform("root", recursive=recursive))
                for node in nodes:
                    self._assert_nested_close(
                        self, expected[node.user_identifier], node.get_absolute_coordinates_xyz()
                    )
                if recursive:
                    for node in nodes:
                        np.testing.assert_allclose(node.effective_transform, np.eye(3))
                else:
                    np.testing.assert_allclose(nodes[0].effective_transform, np.eye(3))
                    np.testing.assert_allclose(
                        nodes[1].effective_transform,
                        translation_matrix(12.0, 8.0)
                        @ translation_matrix(-3.0, 1.0),
                    )

    def test_rect_to_pline_conversion_carries_elevation_to_each_corner(self):
        project = CBProject("rect-conversion-z")
        layer = project.add_layer("Geometry")
        rect = project.add_rect(layer, corner=(2.0, -1.0), width=3.0, height=4.0,
                                elevation=6.25, identifier="rect")
        rect.effective_transform = translation_matrix(5.0, 2.0) @ rotation_matrix_deg(30.0)
        converted = rect.to_pline_representation()
        self.assertIsInstance(converted, Pline)
        self.assertEqual(4, len(converted.relative_points))
        self.assertEqual([6.25] * 4, list(converted.vertex_z))
        self._assert_xyz_projection_close(
            rect.get_absolute_coordinates_xyz(), converted.get_absolute_coordinates_xyz()
        )

    def test_pickle_and_detached_copy_transfer_keep_z_pose_links_and_mop_sources(self):
        source = CBProject("z-transfer")
        layer = source.add_layer("Geometry")
        part = source.add_part("Machining")
        root = source.add_pline(layer, [(0, 0), (1, 0)], vertex_z=[5, 5],
                                identifier="root", local_z_offset=3.0)
        child = source.add_points(layer, [(1, 1), (2, 2)], vertex_z=[-1, 2],
                                  identifier="child", parent=root, local_z_offset=1.5)
        leaf = source.add_circle(layer, (4, 4), 2.0, elevation=7.0,
                                 identifier="leaf", parent=child, local_z_offset=-2.0)
        root.effective_transform = translation_matrix(11.0, -2.0)
        child.effective_transform = rotation_matrix_deg(25.0)
        mop = source.add_profile_mop(part, targets=[child, leaf], identifier="source-mop")
        self.assertIsNotNone(mop)
        expected_child_xyz = child.get_absolute_coordinates_xyz()
        expected_leaf_xyz = leaf.get_absolute_coordinates_xyz()

        with tempfile.TemporaryDirectory(prefix="shape-parity-state-") as directory:
            state = Path(directory) / "project.pkl"
            source.save_state(str(state))
            restored = CBProject.load_state(str(state))
            self.assertIsNotNone(restored)
            self._assert_nested_close(
                self, expected_child_xyz, restored.get_primitive("child").get_absolute_coordinates_xyz()
            )
            self.assertEqual("root", restored.get_parent_of_primitive("child").user_identifier)
            self.assertEqual("child", restored.get_parent_of_primitive("leaf").user_identifier)
            self.assertEqual(
                {restored.get_primitive("child").internal_id, restored.get_primitive("leaf").internal_id},
                set(restored.get_mop_targets("source-mop")),
            )

        target = CBProject("copy-z")
        mapping = source.copy_primitive_tree(child, target)
        copied_child = target.get_primitive("child")
        copied_leaf = target.get_primitive("leaf")
        self.assertIsNone(target.get_parent_of_primitive(copied_child))
        self._assert_nested_close(self, expected_child_xyz, copied_child.get_absolute_coordinates_xyz())
        self._assert_nested_close(self, expected_leaf_xyz, copied_leaf.get_absolute_coordinates_xyz())
        self.assertEqual("child", target.get_parent_of_primitive("leaf").user_identifier)
        self.assertEqual(
            {copied_child.internal_id, copied_leaf.internal_id},
            set(target.get_mop_targets("source-mop")),
        )

        transfer_target = CBProject("transfer-z")
        source.transfer_primitive_tree(child, transfer_target)
        self.assertIsNone(source.get_primitive("child"))
        self.assertIsNone(source.get_primitive("leaf"))
        transferred_child = transfer_target.get_primitive("child")
        transferred_leaf = transfer_target.get_primitive("leaf")
        self._assert_nested_close(self, expected_child_xyz, transferred_child.get_absolute_coordinates_xyz())
        self._assert_nested_close(self, expected_leaf_xyz, transferred_leaf.get_absolute_coordinates_xyz())
        self.assertIsNone(transfer_target.get_parent_of_primitive(transferred_child))
        self.assertEqual("child", transfer_target.get_parent_of_primitive("leaf").user_identifier)
        self.assertEqual(
            {transferred_child.internal_id, transferred_leaf.internal_id},
            set(transfer_target.get_mop_targets("source-mop")),
        )

    def test_matrix_z_roundtrip_and_unsupported_xyz_mixing_fail_at_reader_boundary(self):
        matrix = translation_matrix(3.5, -2.25)
        encoded = to_cambam_matrix_str(matrix, z_offset=7.75)
        decoded, z_offset = from_cambam_matrix_str(encoded, return_z=True)
        np.testing.assert_allclose(matrix, decoded)
        self.assertAlmostEqual(7.75, z_offset)

        project = CBProject("invalid-z-matrix")
        layer = project.add_layer("Geometry")
        project.add_pline(layer, [(0, 0), (1, 0)], identifier="shape")
        with tempfile.TemporaryDirectory(prefix="shape-parity-invalid-") as directory:
            path = Path(directory) / "invalid.cb"
            save_cambam_file(project, str(path))
            tree = ET.parse(path)
            matrix_values = np.eye(4)
            matrix_values[2, 0] = 0.5  # A tilted XY/Z basis is unsupported.
            matrix_text = " ".join(
                str(matrix_values[row, column])
                for column in range(4)
                for row in range(4)
            )
            tree.getroot().find("./layers/layer/objects/*/mat").set("m", matrix_text)
            tree.write(path, encoding="utf-8", xml_declaration=True)
            self.assertIsNone(read_cambam_file(str(path)))

    def test_invalid_bulged_varying_z_and_failed_import_never_publish_partial_project(self):
        with self.assertRaises(ValueError):
            Pline(
                user_identifier="invalid-bulge-z",
                relative_points=[(0.0, 0.0, 0.75), (2.0, 0.0, 0.0)],
                vertex_z=[1.0, 2.0],
            )

        project = CBProject("duplicate-xml-id")
        layer = project.add_layer("Geometry")
        project.add_pline(layer, [(0, 0), (1, 0)], identifier="one")
        project.add_pline(layer, [(2, 0), (3, 0)], identifier="two")
        with tempfile.TemporaryDirectory(prefix="shape-parity-failed-import-") as directory:
            path = Path(directory) / "duplicate.cb"
            save_cambam_file(project, str(path))
            tree = ET.parse(path)
            elements = self._primitive_elements(tree)
            elements[1].set("id", elements[0].get("id"))
            tree.write(path, encoding="utf-8", xml_declaration=True)
            self.assertIsNone(read_cambam_file(str(path)))


if __name__ == "__main__":
    unittest.main()
