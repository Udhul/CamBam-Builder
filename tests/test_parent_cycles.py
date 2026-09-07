"""Regression coverage for atomic rejection of primitive parent cycles."""

import unittest

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import apply_transform, rotation_matrix_deg, translation_matrix


class ParentCycleTests(unittest.TestCase):
    """Exercise parent links while preserving project and transform state."""

    def make_hierarchy(self):
        project = CBProject("parent-cycle-regression")
        root_layer = project.add_layer("Root", color="Red")
        child_layer = project.add_layer("Child", color="Blue")
        root = project.add_pline(
            root_layer, [(0.0, 0.0), (2.0, 0.0)], identifier="root",
            groups=["hierarchy", "root"],
        )
        child = project.add_pline(
            child_layer, [(0.0, 0.0), (0.0, 1.0)], identifier="child",
            groups=["hierarchy"], parent=root,
        )
        grandchild = project.add_pline(
            child_layer, [(0.0, 0.0), (1.0, 1.0)], identifier="grandchild",
            groups=["hierarchy", "leaf"], parent=child,
        )
        root.effective_transform = translation_matrix(20.0, 3.0) @ rotation_matrix_deg(25.0)
        child.effective_transform = translation_matrix(-4.0, 5.0)
        grandchild.effective_transform = rotation_matrix_deg(-15.0)
        return project, {"root": root, "child": child, "grandchild": grandchild}

    @staticmethod
    def snapshot(project, primitives):
        """Capture relationship, entity, classification, and transform state."""
        registries = {}
        for name in ("_primitives", "_layers", "_parts", "_mops"):
            registry = getattr(project, name)
            registries[name] = {key: id(value) for key, value in registry.items()}

        relationships = {
            "_primitive_parent_link": dict(project._primitive_parent_link),
            "_primitive_children_link": {
                key: set(value) for key, value in project._primitive_children_link.items()
            },
            "_primitive_layer_assignment": dict(project._primitive_layer_assignment),
            "_layer_primitive_membership": {
                key: set(value) for key, value in project._layer_primitive_membership.items()
            },
            "_primitive_groups": {
                key: set(value) for key, value in project._primitive_groups.items()
            },
            "_primitive_group_membership": {
                key: set(value) for key, value in project._primitive_group_membership.items()
            },
            "_identifier_registry": dict(project._identifier_registry),
        }
        matrices = {
            name: {
                "effective": primitive.effective_transform.copy(),
                "total": primitive.get_total_transform().copy(),
            }
            for name, primitive in primitives.items()
        }
        return registries, relationships, matrices

    def assert_snapshot_equal(self, project, primitives, snapshot):
        registries, relationships, matrices = snapshot
        for name, expected in registries.items():
            registry = getattr(project, name)
            self.assertEqual({key: id(value) for key, value in registry.items()}, expected, name)
        self.assertEqual(dict(project._primitive_parent_link), relationships["_primitive_parent_link"])
        self.assertEqual(
            {key: set(value) for key, value in project._primitive_children_link.items()},
            relationships["_primitive_children_link"],
        )
        for name in (
            "_primitive_layer_assignment", "_layer_primitive_membership",
            "_primitive_groups", "_primitive_group_membership", "_identifier_registry",
        ):
            actual = getattr(project, name)
            if name.endswith("membership") or name in ("_primitive_groups", "_primitive_group_membership"):
                actual = {key: set(value) for key, value in actual.items()}
            self.assertEqual(actual, relationships[name], name)
        for name, expected in matrices.items():
            primitive = primitives[name]
            self.assertTrue(np.array_equal(primitive.effective_transform, expected["effective"]), name)
            self.assertTrue(np.allclose(primitive.get_total_transform(), expected["total"], rtol=0, atol=0), name)

    def assert_world_poses_usable(self, primitives, expected):
        for name, primitive in primitives.items():
            self.assertTrue(np.allclose(primitive.get_total_transform(), expected[name]["total"], rtol=0, atol=0), name)
            self.assertTrue(np.allclose(primitive.get_absolute_coordinates(), expected[name]["geometry"], rtol=0, atol=0), name)

    def test_self_parent_rejection_is_atomic(self):
        project, primitives = self.make_hierarchy()
        before = self.snapshot(project, primitives)

        self.assertFalse(project.link_primitive_parent("child", "child"))
        self.assert_snapshot_equal(project, primitives, before)

    def test_two_node_cycle_rejection_preserves_existing_parent_and_traversal(self):
        project, primitives = self.make_hierarchy()
        expected = {
            name: {"total": primitive.get_total_transform().copy(),
                   "geometry": primitive.get_absolute_coordinates().copy()}
            for name, primitive in primitives.items()
        }
        before = self.snapshot(project, primitives)

        # The child already has root as its parent; linking it below its own
        # descendant would create a two-node cycle and must preserve that edge.
        self.assertFalse(project.link_primitive_parent("child", "grandchild"))
        self.assert_snapshot_equal(project, primitives, before)
        self.assert_world_poses_usable(primitives, expected)

    def test_longer_cycle_rejection_and_valid_reparent_detach_are_supported(self):
        project, primitives = self.make_hierarchy()
        before = self.snapshot(project, primitives)

        self.assertFalse(project.link_primitive_parent("root", "grandchild"))
        self.assert_snapshot_equal(project, primitives, before)

        self.assertTrue(project.link_primitive_parent("grandchild", "root"))
        self.assertIs(project.get_parent_of_primitive("grandchild"), primitives["root"])
        self.assertTrue(project.link_primitive_parent("grandchild", "root"))
        self.assertEqual(
            {primitive.internal_id for primitive in project.get_children_of_primitive("root")},
            {primitives["child"].internal_id, primitives["grandchild"].internal_id},
        )
        self.assertTrue(project.link_primitive_parent("grandchild", None))
        self.assertIsNone(project.get_parent_of_primitive("grandchild"))
        self.assertNotIn(primitives["grandchild"], project.get_children_of_primitive("root"))
        self.assertTrue(np.allclose(primitives["grandchild"].get_total_transform(), primitives["grandchild"].effective_transform, rtol=0, atol=0))
        expected_geometry = apply_transform(
            [(0.0, 0.0), (1.0, 1.0)], primitives["grandchild"].effective_transform
        )
        expected_geometry = [(*point, 0.0) for point in expected_geometry]
        self.assertTrue(np.allclose(
            primitives["grandchild"].get_absolute_coordinates(), expected_geometry,
            rtol=0, atol=0,
        ))


if __name__ == "__main__":
    unittest.main()
