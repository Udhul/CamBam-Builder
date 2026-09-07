"""Full local-transform baking must preserve hierarchy world geometry."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class FullBakeTests(unittest.TestCase):
    def make_hierarchy(self, singular=False):
        project = CBProject("full-bake")
        layer = project.add_layer("Geometry")
        nodes = {}
        for name, parent in (("root", None), ("child", "root"), ("leaf", "child")):
            nodes[name] = project.add_pline(
                layer, [(1., 2.), (4., -1.)], identifier=name,
                parent=nodes.get(parent), groups=["fixture"],
            )
        nodes["root"].effective_transform = np.array([[0., -2., 20.], [3., 0., 10.], [0., 0., 1.]])
        nodes["child"].effective_transform = np.array([[1., 0.5, 7.], [0., -1., 4.], [0., 0., 1.]])
        nodes["leaf"].effective_transform = np.array([[2., 0., -2.], [0., 1., 8.], [0., 0., 1.]])
        if singular:
            nodes["root"].effective_transform[0, :2] = 0
        return project, nodes

    def check_bake(self, target, recursive, singular=False):
        project, nodes = self.make_hierarchy(singular)
        coordinates = {name: node.get_absolute_coordinates() for name, node in nodes.items()}
        local_points = {name: list(node.relative_points) for name, node in nodes.items()}
        matrices = {name: node.effective_transform.copy() for name, node in nodes.items()}
        edges = dict(project._primitive_parent_link)
        children = {key: set(value) for key, value in project._primitive_children_link.items()}
        layers = dict(project._primitive_layer_assignment)

        self.assertTrue(project.bake_primitive_transform(target, recursive=recursive))
        baked = {target}
        if recursive:
            baked |= {"child", "leaf"} if target == "root" else {"leaf"}
        for name, node in nodes.items():
            np.testing.assert_allclose(node.get_absolute_coordinates(), coordinates[name], rtol=0, atol=1e-10)
            if name in baked:
                np.testing.assert_array_equal(node.effective_transform, np.eye(3))
            else:
                self.assertEqual(node.relative_points, local_points[name])
        if not recursive:
            child_name = "child" if target == "root" else "leaf"
            np.testing.assert_allclose(nodes[child_name].effective_transform,
                                       matrices[target] @ matrices[child_name], rtol=0, atol=1e-10)
        self.assertEqual(project._primitive_parent_link, edges)
        self.assertEqual(project._primitive_children_link, children)
        self.assertEqual(project._primitive_layer_assignment, layers)
        for name, node in nodes.items():
            self.assertIs(project.get_primitive(name), node)
            self.assertEqual(project.get_groups_of_primitive(node), ["fixture"])

        baked_points = {name: list(node.relative_points) for name, node in nodes.items()}
        baked_matrices = {name: node.effective_transform.copy() for name, node in nodes.items()}
        self.assertTrue(project.bake_primitive_transform(target, recursive=recursive))
        for name, node in nodes.items():
            self.assertEqual(node.relative_points, baked_points[name])
            np.testing.assert_array_equal(node.effective_transform, baked_matrices[name])

        # A singular unbaked parent cannot be reconstructed by the reader;
        # singular cases below bake the whole hierarchy before serialization.
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                path = Path(directory) / f"baked-{index}.cb"
                save_cambam_file(project, str(path))
                project = read_cambam_file(str(path))
                self.assertIsNotNone(project)
                self.assertEqual(len(project.list_primitives()), 3)
                self.assertEqual(project._primitive_parent_link, edges)
                for name, node in nodes.items():
                    loaded = project.get_primitive(name)
                    self.assertEqual(loaded.internal_id, node.internal_id)
                    np.testing.assert_allclose(loaded.get_absolute_coordinates(), coordinates[name], rtol=0, atol=1e-8)

    def test_root_full_bake_preserves_descendants_without_baking_them(self):
        self.check_bake("root", recursive=False)

    def test_child_full_bake_preserves_leaf_under_transformed_parent(self):
        self.check_bake("child", recursive=False)

    def test_recursive_root_bake_preserves_world_geometry(self):
        self.check_bake("root", recursive=True)

    def test_recursive_child_bake_preserves_world_geometry(self):
        self.check_bake("child", recursive=True)

    def test_recursive_singular_bake_requires_no_inverse(self):
        self.check_bake("root", recursive=True, singular=True)


if __name__ == "__main__":
    unittest.main()
