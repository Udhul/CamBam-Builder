"""Regression coverage for global transform ordering and preflight validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import (
    apply_transform,
    rotation_matrix_deg,
    scale_matrix,
    skew_matrix,
    translation_matrix,
)
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class GlobalTransformTests(unittest.TestCase):
    """Use noncommuting affine transforms on a small primitive hierarchy."""

    def make_project(self):
        project = CBProject("global-transform-regression")
        root_layer = project.add_layer("Root")
        child_layer = project.add_layer("Child")
        root = project.add_pline(root_layer, [(1.0, 2.0), (4.0, -1.0)], identifier="root")
        child = project.add_pline(
            child_layer, [(-2.0, 1.0), (0.0, 3.0)], identifier="child", parent=root
        )
        grandchild = project.add_pline(
            child_layer, [(2.0, -1.0), (5.0, 2.0)], identifier="grandchild", parent=child
        )
        sibling = project.add_pline(
            root_layer, [(8.0, 1.0), (9.0, 4.0)], identifier="sibling"
        )

        root.effective_transform = (
            translation_matrix(11.0, -3.0)
            @ rotation_matrix_deg(27.0)
            @ scale_matrix(1.5, 0.75)
        )
        child.effective_transform = (
            translation_matrix(-4.0, 6.0)
            @ skew_matrix(angle_x_deg=13.0)
            @ rotation_matrix_deg(-19.0)
        )
        grandchild.effective_transform = (
            translation_matrix(3.0, -5.0)
            @ scale_matrix(0.8, 1.7)
            @ rotation_matrix_deg(11.0)
        )
        sibling.effective_transform = translation_matrix(-7.0, 4.0) @ rotation_matrix_deg(8.0)
        return project, {p.user_identifier: p for p in (root, child, grandchild, sibling)}

    @staticmethod
    def world_matrix(node):
        return node.get_total_transform().copy()

    @staticmethod
    def xy(points):
        return np.asarray([(point[0], point[1]) for point in points], dtype=float)

    @classmethod
    def transformed_xy(cls, points, matrix):
        coordinates = [
            (point.x, point.y) if hasattr(point, "x") else tuple(point)
            for point in points
        ]
        return cls.xy(apply_transform(coordinates, matrix))

    def snapshot(self, project, nodes):
        return {
            "points": {name: self.xy(node.get_absolute_coordinates()) for name, node in nodes.items()},
            "relative": {name: list(node.vertices) for name, node in nodes.items()},
            "matrices": {name: node.effective_transform.copy() for name, node in nodes.items()},
            "parents": dict(project._primitive_parent_link),
            "children": {key: set(value) for key, value in project._primitive_children_link.items()},
            "layers": dict(project._primitive_layer_assignment),
        }

    def assert_registries_unchanged(self, project, state):
        self.assertEqual(project._primitive_parent_link, state["parents"])
        self.assertEqual(
            {key: set(value) for key, value in project._primitive_children_link.items()},
            state["children"],
        )
        self.assertEqual(project._primitive_layer_assignment, state["layers"])

    def assert_world_move(self, before, nodes, moved, matrix):
        for name, old_points in before["points"].items():
            expected = self.transformed_xy(old_points, matrix) if name in moved else old_points
            np.testing.assert_allclose(self.xy(nodes[name].get_absolute_coordinates()), expected, rtol=0, atol=1e-10)

    def test_unbaked_global_ordering_uses_parent_frame_and_moves_subtree_once(self):
        for target, moved in (("root", {"root", "child", "grandchild"}),
                              ("child", {"child", "grandchild"})):
            with self.subTest(target=target):
                project, nodes = self.make_project()
                before = self.snapshot(project, nodes)
                matrix = translation_matrix(17.0, -9.0) @ rotation_matrix_deg(31.0)
                parent = nodes[target].get_project().get_parent_of_primitive(nodes[target])
                parent_world = parent.get_total_transform() if parent else np.eye(3)
                expected_local = np.linalg.solve(
                    parent_world, matrix @ parent_world @ before["matrices"][target]
                )

                self.assertTrue(project.transform_primitive(target, matrix))
                np.testing.assert_allclose(nodes[target].effective_transform, expected_local, rtol=0, atol=1e-10)
                for name in nodes:
                    self.assertEqual(nodes[name].vertices, before["relative"][name])
                    if name != target:
                        np.testing.assert_array_equal(nodes[name].effective_transform, before["matrices"][name])
                self.assert_world_move(before, nodes, moved, matrix)
                self.assert_registries_unchanged(project, before)

    def test_unbaked_target_singular_is_allowed_but_singular_parent_rejects_atomically(self):
        project, nodes = self.make_project()
        nodes["child"].effective_transform = scale_matrix(0.0, 1.0) @ translation_matrix(2.0, 3.0)
        before = self.snapshot(project, nodes)
        matrix = translation_matrix(4.0, 5.0) @ rotation_matrix_deg(18.0)
        self.assertTrue(project.transform_primitive("child", matrix))
        self.assert_world_move(before, nodes, {"child", "grandchild"}, matrix)

        project, nodes = self.make_project()
        nodes["root"].effective_transform = scale_matrix(0.0, 1.0) @ translation_matrix(2.0, 3.0)
        before = self.snapshot(project, nodes)
        self.assertFalse(project.transform_primitive("child", matrix))
        after = self.snapshot(project, nodes)
        for name in nodes:
            np.testing.assert_array_equal(after["matrices"][name], before["matrices"][name])
            np.testing.assert_array_equal(after["relative"][name], before["relative"][name])
        self.assert_registries_unchanged(project, before)

    def test_baked_global_transform_uses_each_node_world_frame_and_preserves_local_state(self):
        project, nodes = self.make_project()
        before = self.snapshot(project, nodes)
        worlds = {name: self.world_matrix(node) for name, node in nodes.items()}
        matrix = translation_matrix(-12.0, 7.0) @ rotation_matrix_deg(-23.0) @ skew_matrix(angle_y_deg=9.0)
        self.assertTrue(project.transform_primitive("root", matrix, bake=True))

        for name in ("root", "child", "grandchild"):
            q = np.linalg.solve(worlds[name], matrix @ worlds[name])
            expected_relative = self.transformed_xy(before["relative"][name], q)
            np.testing.assert_allclose(
                [(vertex.x, vertex.y) for vertex in nodes[name].vertices],
                expected_relative, rtol=0, atol=1e-10,
            )
            np.testing.assert_array_equal(nodes[name].effective_transform, before["matrices"][name])
        self.assertEqual(nodes["sibling"].vertices, before["relative"]["sibling"])
        np.testing.assert_array_equal(nodes["sibling"].effective_transform, before["matrices"]["sibling"])
        self.assert_world_move(before, nodes, {"root", "child", "grandchild"}, matrix)
        self.assert_registries_unchanged(project, before)

    def test_baked_singular_world_anywhere_rejects_before_mutation(self):
        project, nodes = self.make_project()
        nodes["grandchild"].effective_transform = scale_matrix(0.0, 1.0)
        before = self.snapshot(project, nodes)
        matrix = translation_matrix(3.0, 4.0)
        self.assertFalse(project.transform_primitive("root", matrix, bake=True))
        after = self.snapshot(project, nodes)
        for name in nodes:
            np.testing.assert_array_equal(after["matrices"][name], before["matrices"][name])
            np.testing.assert_array_equal(after["relative"][name], before["relative"][name])
        self.assert_registries_unchanged(project, before)

    def test_wrappers_apply_global_translation_and_rotation_centers(self):
        project, nodes = self.make_project()
        before = self.snapshot(project, nodes)
        matrix = translation_matrix(6.0, -2.5)
        self.assertTrue(project.translate_primitive("child", 6.0, -2.5))
        self.assert_world_move(before, nodes, {"child", "grandchild"}, matrix)

        project, nodes = self.make_project()
        before = self.snapshot(project, nodes)
        explicit = rotation_matrix_deg(37.0, 2.0, -3.0)
        self.assertTrue(project.rotate_primitive_deg("child", 37.0, 2.0, -3.0))
        self.assert_world_move(before, nodes, {"child", "grandchild"}, explicit)

        project, nodes = self.make_project()
        before = self.snapshot(project, nodes)
        cx, cy = nodes["child"].get_geometric_center()
        default = rotation_matrix_deg(-29.0, cx, cy)
        self.assertTrue(project.rotate_primitive_deg("child", -29.0))
        self.assert_world_move(before, nodes, {"child", "grandchild"}, default)

    def test_global_transform_round_trips_twice_in_both_modes(self):
        matrix = translation_matrix(9.0, -8.0) @ rotation_matrix_deg(16.0)
        for bake in (False, True):
            with self.subTest(bake=bake), tempfile.TemporaryDirectory() as directory:
                project, nodes = self.make_project()
                before = self.snapshot(project, nodes)
                self.assertTrue(project.transform_primitive("child", matrix, bake=bake))
                expected = {name: self.transformed_xy(points, matrix) if name in {"child", "grandchild"} else points
                            for name, points in before["points"].items()}
                first = Path(directory) / "first.cb"
                second = Path(directory) / "second.cb"
                save_cambam_file(project, str(first))
                loaded = read_cambam_file(str(first))
                self.assertIsNotNone(loaded)
                save_cambam_file(loaded, str(second))
                loaded_twice = read_cambam_file(str(second))
                self.assertIsNotNone(loaded_twice)
                for result in (loaded, loaded_twice):
                    self.assertEqual(len(result.list_primitives()), len(nodes))
                    self.assertEqual(result._primitive_parent_link, before["parents"])
                    for name, node in nodes.items():
                        self.assertEqual(result.get_primitive(name).internal_id, node.internal_id)
                        self.assertEqual(
                            result.get_layer_of_primitive(name).user_identifier,
                            project.get_layer_of_primitive(name).user_identifier,
                        )
                for name, expected_points in expected.items():
                    actual = self.xy(loaded_twice.get_primitive(name).get_absolute_coordinates())
                    np.testing.assert_allclose(actual, expected_points, rtol=0, atol=2e-8)

    def test_invalid_shape_finite_and_affine_matrices_reject_without_mutation(self):
        invalid = (
            [[1, 0, 10**1000], [0, 1, 0], [0, 0, 1]],
            np.ones((2, 2)),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
            np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, np.inf], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        for matrix in invalid:
            with self.subTest(matrix=matrix):
                project, nodes = self.make_project()
                before = self.snapshot(project, nodes)
                self.assertFalse(project.transform_primitive("root", matrix))
                after = self.snapshot(project, nodes)
                for name in nodes:
                    np.testing.assert_array_equal(after["matrices"][name], before["matrices"][name])
                    np.testing.assert_array_equal(after["relative"][name], before["relative"][name])
                self.assert_registries_unchanged(project, before)

    def test_baked_small_translation_is_not_discarded_as_identity(self):
        project = CBProject("small-translation")
        node = project.add_pline(project.add_layer("Geometry"), [(0., 0.), (1., 0.)])
        self.assertTrue(project.translate_primitive(node, 1e-9, 0., bake=True))
        np.testing.assert_allclose(self.xy(node.get_absolute_coordinates()),
                                   [[1e-9, 0.], [1. + 1e-9, 0.]], rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
