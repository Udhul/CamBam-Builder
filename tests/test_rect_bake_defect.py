"""Regression coverage for exact Rect outline baking and representation changes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import (
    apply_transform,
    identity_matrix,
    mirror_y_matrix,
    rotation_matrix_deg,
    scale_matrix,
    skew_matrix,
    translation_matrix,
)
from cambam_builder.cambam_entities import Pline, Rect
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class RectBakeDefectTests(unittest.TestCase):
    """Synthetic geometry and relationship cases for the Rect bake contract."""

    BASE_CORNERS = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]

    @staticmethod
    def xy(points):
        return np.asarray([(point[0], point[1]) for point in points], dtype=float)

    @classmethod
    def transformed_corners(cls, matrix):
        return apply_transform(cls.BASE_CORNERS, matrix)

    @staticmethod
    def polygon_area(points):
        values = RectBakeDefectTests.xy(points)
        return 0.5 * abs(
            np.dot(values[:, 0], np.roll(values[:, 1], -1))
            - np.dot(values[:, 1], np.roll(values[:, 0], -1))
        )

    @classmethod
    def assert_outline(cls, test_case, expected, actual, atol):
        """Compare ordered closed edges, allowing cyclic order or winding reversal."""
        expected = cls.xy(expected)
        actual = cls.xy(actual)
        test_case.assertEqual(expected.shape, (4, 2))
        test_case.assertEqual(actual.shape, (4, 2))
        candidates = []
        for source in (actual, actual[::-1]):
            candidates.extend(np.roll(source, shift, axis=0) for shift in range(4))
        test_case.assertTrue(
            any(np.allclose(expected, candidate, rtol=0, atol=atol) for candidate in candidates),
            msg=f"expected outline {expected.tolist()}, got {actual.tolist()}",
        )
        test_case.assertAlmostEqual(
            cls.polygon_area(expected), cls.polygon_area(actual), delta=atol
        )

    @staticmethod
    def save(project, directory, name):
        path = Path(directory) / f"{name}.cb"
        save_cambam_file(project, str(path))
        return path

    @staticmethod
    def make_rect(matrix=None, *, description="", groups=None):
        project = CBProject("rect-bake-regression")
        layer = project.add_layer("Geometry")
        rect = project.add_rect(
            layer,
            corner=(0.0, 0.0),
            width=4.0,
            height=2.0,
            identifier="rect",
            description=description,
            groups=groups,
        )
        if matrix is not None:
            rect.effective_transform = np.asarray(matrix, dtype=float)
        return project, rect

    def test_full_bake_preserves_rotation_shear_and_axis_aligned_outlines(self):
        cases = (
            ("rotation", rotation_matrix_deg(45.0), Pline),
            ("shear", skew_matrix(angle_y_deg=45.0), Pline),
            ("quarter-turn", rotation_matrix_deg(90.0), Rect),
            ("reflection", mirror_y_matrix(), Rect),
            ("translation", translation_matrix(3.0, -2.0), Rect),
            ("diagonal-scale", scale_matrix(2.0, 3.0), Rect),
        )
        for name, matrix, expected_type in cases:
            with self.subTest(transform=name):
                project, rect = self.make_rect(matrix)
                expected = self.transformed_corners(matrix)
                object_id = id(rect)
                self.assertTrue(project.bake_primitive_transform(rect))
                self.assertIs(project.get_primitive("rect"), rect)
                self.assertEqual(id(rect), object_id)
                self.assertIsInstance(rect, expected_type)
                self.assert_outline(self, expected, rect.get_absolute_coordinates(), 1e-10)
                np.testing.assert_allclose(rect.effective_transform, identity_matrix(), rtol=0, atol=0)
                if isinstance(rect, Pline):
                    self.assertTrue(rect.closed)
                    self.assertEqual(len(rect.relative_points), 4)
                    self.assertTrue(all(len(point) == 3 and point[2] == 0.0 for point in rect.relative_points))
                self.assertTrue(project.bake_primitive_transform(rect))
                self.assert_outline(self, expected, rect.get_absolute_coordinates(), 1e-10)

    def make_hierarchy(self):
        project = CBProject("rect-bake-hierarchy")
        layer = project.add_layer("Geometry")
        ancestor = project.add_pline(layer, [(20.0, 1.0), (21.0, 4.0)], identifier="ancestor")
        root = project.add_rect(layer, width=4.0, height=2.0, identifier="root", parent=ancestor, groups=["profiles"])
        child = project.add_rect(
            layer, corner=(1.0, 1.0), width=2.0, height=1.0,
            identifier="child", parent=root, groups=["profiles"],
        )
        sibling = project.add_pline(layer, [(8.0, 1.0), (9.0, 4.0)], identifier="sibling")
        ancestor.effective_transform = translation_matrix(12.0, 7.0) @ scale_matrix(1.5, 0.75)
        root.effective_transform = rotation_matrix_deg(45.0)
        child.effective_transform = translation_matrix(6.0, -3.0) @ skew_matrix(angle_x_deg=13.0)
        sibling.effective_transform = translation_matrix(-4.0, 5.0)
        return project, ancestor, root, child, sibling

    def test_full_bake_recursive_and_nonrecursive_preserve_hierarchy_world_geometry(self):
        for recursive in (False, True):
            with self.subTest(recursive=recursive):
                project, ancestor, root, child, sibling = self.make_hierarchy()
                before = {
                    node.user_identifier: self.xy(node.get_absolute_coordinates())
                    for node in (ancestor, root, child, sibling)
                }
                parent_links = dict(project._primitive_parent_link)
                child_links = {key: set(value) for key, value in project._primitive_children_link.items()}
                root_matrix = root.effective_transform.copy()
                child_matrix = child.effective_transform.copy()

                self.assertTrue(project.bake_primitive_transform(root, recursive=recursive))
                for node in (ancestor, root, child, sibling):
                    np.testing.assert_allclose(
                        self.xy(node.get_absolute_coordinates()), before[node.user_identifier], rtol=0, atol=1e-10
                    )
                self.assertEqual(project._primitive_parent_link, parent_links)
                self.assertEqual(
                    {key: set(value) for key, value in project._primitive_children_link.items()}, child_links
                )
                np.testing.assert_allclose(root.effective_transform, identity_matrix(), rtol=0, atol=0)
                np.testing.assert_allclose(ancestor.effective_transform, translation_matrix(12.0, 7.0) @ scale_matrix(1.5, 0.75), rtol=0, atol=0)
                if recursive:
                    np.testing.assert_allclose(child.effective_transform, identity_matrix(), rtol=0, atol=0)
                    self.assertIsInstance(child, Pline)
                else:
                    np.testing.assert_allclose(child.effective_transform, root_matrix @ child_matrix, rtol=0, atol=1e-10)
                np.testing.assert_allclose(sibling.effective_transform, translation_matrix(-4.0, 5.0), rtol=0, atol=0)

    def test_explicit_matrix_bake_preserves_effective_matrix_and_exact_outline(self):
        project, rect = self.make_rect(translation_matrix(11.0, -4.0))
        explicit = rotation_matrix_deg(45.0) @ skew_matrix(angle_y_deg=20.0)
        expected_local = self.transformed_corners(explicit)
        expected_world = apply_transform(expected_local, rect.effective_transform)
        effective_before = rect.effective_transform.copy()

        self.assertTrue(project.bake_primitive_transform(rect, transform_to_bake=explicit))
        self.assertIs(project.get_primitive("rect"), rect)
        self.assertIsInstance(rect, Pline)
        self.assert_outline(self, expected_world, rect.get_absolute_coordinates(), 1e-10)
        np.testing.assert_allclose(rect.effective_transform, effective_before, rtol=0, atol=0)

    def test_rotation_component_bake_preserves_exact_outline(self):
        matrix = rotation_matrix_deg(45.0)
        project, rect = self.make_rect(matrix)
        expected = self.transformed_corners(matrix)

        self.assertTrue(project.bake_primitive_transform_component(rect, "rotation"))
        self.assertIsInstance(rect, Pline)
        self.assert_outline(self, expected, rect.get_absolute_coordinates(), 1e-10)
        np.testing.assert_allclose(rect.effective_transform, identity_matrix(), rtol=0, atol=0)

    def test_unbaked_rotation_and_shear_roundtrip_controls_preserve_outline(self):
        for name, matrix, expected_type in (
            ("rotation", rotation_matrix_deg(45.0), Rect),
            ("shear", skew_matrix(angle_y_deg=45.0), Pline),
        ):
            with self.subTest(transform=name), tempfile.TemporaryDirectory() as directory:
                project, rect = self.make_rect(matrix)
                expected = self.transformed_corners(matrix)
                first = self.save(project, directory, "unbaked-first")
                loaded_once = read_cambam_file(str(first))
                second = self.save(loaded_once, directory, "unbaked-second")
                loaded_twice = read_cambam_file(str(second))
                for loaded in (loaded_once, loaded_twice):
                    result = loaded.get_primitive("rect")
                    self.assertIsInstance(result, expected_type)
                    self.assert_outline(self, expected, result.get_absolute_coordinates(), 1e-8)
                    expected_matrix = matrix if expected_type is Rect else identity_matrix()
                    np.testing.assert_allclose(result.effective_transform, expected_matrix, rtol=0, atol=1e-8)

    def test_global_bake_preserves_effective_matrix_and_moves_only_target_subtree(self):
        project = CBProject("rect-global-bake")
        layer = project.add_layer("Geometry")
        ancestor = project.add_pline(layer, [(20.0, 1.0), (21.0, 4.0)], identifier="ancestor")
        ancestor.effective_transform = translation_matrix(12.0, 7.0) @ rotation_matrix_deg(19.0)
        rect = project.add_rect(layer, width=4.0, height=2.0, identifier="rect", parent=ancestor)
        rect.effective_transform = translation_matrix(5.0, 2.0)
        child = project.add_rect(layer, width=2.0, height=1.0, identifier="child", parent=rect)
        child.effective_transform = translation_matrix(3.0, -2.0)
        child_before = self.xy(child.get_absolute_coordinates())
        child_matrix = child.effective_transform.copy()
        sibling = project.add_pline(layer, [(8.0, 1.0), (9.0, 4.0)], identifier="sibling")
        sibling_before = self.xy(sibling.get_absolute_coordinates())
        before_world = self.xy(rect.get_absolute_coordinates())
        effective_before = rect.effective_transform.copy()
        global_matrix = rotation_matrix_deg(31.0) @ skew_matrix(angle_y_deg=17.0)

        self.assertTrue(project.transform_primitive(rect, global_matrix, bake=True))
        self.assertIsInstance(rect, Pline)
        expected_world = apply_transform(before_world.tolist(), global_matrix)
        self.assert_outline(self, expected_world, rect.get_absolute_coordinates(), 1e-10)
        self.assert_outline(self, apply_transform(child_before.tolist(), global_matrix), child.get_absolute_coordinates(), 1e-10)
        np.testing.assert_array_equal(child.effective_transform, child_matrix)
        np.testing.assert_allclose(rect.effective_transform, effective_before, rtol=0, atol=0)
        np.testing.assert_allclose(ancestor.effective_transform, translation_matrix(12.0, 7.0) @ rotation_matrix_deg(19.0), rtol=0, atol=0)
        np.testing.assert_allclose(self.xy(sibling.get_absolute_coordinates()), sibling_before, rtol=0, atol=1e-10)

    def test_small_shear_is_not_skipped_as_identity(self):
        matrix = np.array([[1., 1e-9, 0.], [0., 1., 0.], [0., 0., 1.]])
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                project, rect = self.make_rect(None if explicit else matrix)
                self.assertTrue(project.bake_primitive_transform(
                    rect, transform_to_bake=matrix if explicit else None
                ))
                self.assertIsInstance(rect, Pline)
                self.assert_outline(self, self.transformed_corners(matrix), rect.get_absolute_coordinates(), 1e-10)

    def test_invalid_explicit_matrix_fails_without_mutation(self):
        project, rect = self.make_rect(translation_matrix(3., 4.))
        before = self.xy(rect.get_absolute_coordinates())
        effective = rect.effective_transform.copy()
        for matrix in (np.zeros((2, 2)), np.full((3, 3), np.nan), np.diag([1., 1., 2.])):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    rect.bake_geometry(matrix)
                with self.assertLogs("cambam_builder.cambam_project", level="ERROR"):
                    self.assertFalse(project.bake_primitive_transform(rect, transform_to_bake=matrix))
                self.assertIsInstance(rect, Rect)
                self.assertIs(project.get_primitive("rect"), rect)
                np.testing.assert_array_equal(rect.effective_transform, effective)
                np.testing.assert_array_equal(self.xy(rect.get_absolute_coordinates()), before)

    def test_metadata_relationships_and_target_resolution_survive_bake_and_two_xml_roundtrips(self):
        project, ancestor, root, child, sibling = self.make_hierarchy()
        root.description = "keep this description"
        part = project.add_part("Part")
        mop = project.add_profile_mop(part, targets=[root], name="Profile", identifier="profile", target_depth=-1.0)
        expected = {node.user_identifier: self.xy(node.get_absolute_coordinates()) for node in (ancestor, root, child, sibling)}
        ancestor_id = ancestor.internal_id
        root_id = root.internal_id
        child_id = child.internal_id
        layer_id = project.get_layer_of_primitive(root).internal_id

        self.assertTrue(project.bake_primitive_transform(root, recursive=True))
        self.assertIs(project.get_primitive(root_id), root)
        self.assertEqual(root.internal_id, root_id)
        self.assertEqual(root.user_identifier, "root")
        self.assertEqual(root.description, "keep this description")
        self.assertEqual(project.get_groups_of_primitive(root), ["profiles"])
        self.assertEqual(project.get_layer_of_primitive(root).internal_id, layer_id)
        self.assertEqual(project.get_parent_of_primitive(child), root)
        self.assertEqual(project.get_children_of_primitive(root)[0].internal_id, child_id)
        self.assertEqual(project.get_mop_targets(mop), [root_id])
        self.assertIsNone(project.get_mop_target_group(mop))

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "baked-state.pkl"
            project.save_state(str(state_path))
            restored = CBProject.load_state(str(state_path))
            self.assertIsInstance(restored.get_primitive(root_id), Pline)
            self.assertEqual(restored.get_primitive(root_id).internal_id, root_id)
            self.assertEqual(restored.get_parent_of_primitive(root_id).internal_id, ancestor_id)
            self.assertEqual(restored.get_parent_of_primitive(child_id).internal_id, root_id)
            self.assertEqual(restored.get_mop_targets(restored.get_mop("profile")), [root_id])
            self.assert_outline(self, expected["root"], restored.get_primitive(root_id).get_absolute_coordinates(), 1e-10)

            first = self.save(project, directory, "first")
            loaded_once = read_cambam_file(str(first))
            second = self.save(loaded_once, directory, "second")
            loaded_twice = read_cambam_file(str(second))
            for loaded in (loaded_once, loaded_twice):
                loaded_root = loaded.get_primitive(root_id)
                loaded_child = loaded.get_primitive(child_id)
                self.assertEqual(len(loaded.list_primitives()), 4)
                self.assertIsInstance(loaded_root, Pline)
                self.assertTrue(loaded_root.closed)
                self.assertEqual(len(loaded_root.relative_points), 4)
                self.assertTrue(all(len(point) == 3 and point[2] == 0.0 for point in loaded_root.relative_points))
                self.assertEqual(loaded_root.internal_id, root_id)
                self.assertEqual(loaded.get_parent_of_primitive(loaded_root).internal_id, ancestor_id)
                self.assertEqual(loaded_root.description, "keep this description")
                self.assertEqual(loaded.get_groups_of_primitive(loaded_root), ["profiles"])
                self.assertEqual(loaded.get_layer_of_primitive(loaded_root).user_identifier, "Geometry")
                self.assertEqual(loaded.get_parent_of_primitive(loaded_child).internal_id, root_id)
                self.assert_outline(self, expected["root"], loaded_root.get_absolute_coordinates(), 1e-8)
                for name in ("child", "sibling"):
                    np.testing.assert_allclose(self.xy(loaded.get_primitive(name).get_absolute_coordinates()), expected[name], rtol=0, atol=1e-8)
                loaded_mop = loaded.get_mop("profile")
                self.assertEqual(loaded.get_mop_targets(loaded_mop), [root_id])
                self.assertIsNone(loaded.get_mop_target_group(loaded_mop))
                self.assertEqual(loaded.get_part_of_mop(loaded_mop).user_identifier, "Part")


if __name__ == "__main__":
    unittest.main()
