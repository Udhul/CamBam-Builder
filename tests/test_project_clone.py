"""Regression tests for the public project clone transaction primitive."""

import unittest

from cambam_builder import CBProject


class ProjectCloneTests(unittest.TestCase):
    def test_clone_copies_complete_graph_and_rebinds_primitives(self):
        source = CBProject("clone-source")
        layer = source.add_layer("Geometry")
        root = source.add_rect(layer, width=20, height=10, identifier="root")
        child = source.add_circle(
            layer, center=(2, 3), diameter=4, identifier="child", parent=root
        )
        source.add_primitive_to_group(root, "cut")
        part = source.add_part("Part")
        mop = source.add_profile_mop(
            part,
            targets=[root],
            identifier="profile",
            name="profile",
            target_depth=-1,
            depth_increment=0.5,
            tool_diameter=3,
            cut_feedrate=300,
            plunge_feedrate=100,
            spindle_speed=12000,
        )

        clone = source.clone()

        self.assertIsNot(clone, source)
        self.assertEqual(source.project_name, clone.project_name)
        self.assertEqual(
            {p.internal_id for p in source.list_primitives()},
            {p.internal_id for p in clone.list_primitives()},
        )
        self.assertEqual(
            {m.internal_id for m in source.list_mops()},
            {m.internal_id for m in clone.list_mops()},
        )
        self.assertIsNot(clone.get_primitive(root.internal_id), root)
        self.assertIsNot(clone.get_mop(mop.internal_id), mop)
        self.assertIs(clone.get_primitive(root.internal_id).get_project(), clone)
        self.assertIs(clone.get_primitive(child.internal_id).get_project(), clone)
        self.assertEqual(
            clone.get_parent_of_primitive(child.internal_id).internal_id,
            root.internal_id,
        )
        self.assertEqual(
            clone.get_children_of_primitive(root.internal_id)[0].internal_id,
            child.internal_id,
        )
        self.assertEqual(
            clone.get_layer_of_primitive(root.internal_id).user_identifier,
            "Geometry",
        )
        self.assertEqual(clone.get_groups_of_primitive(root.internal_id), ["cut"])
        self.assertEqual(clone.get_mop_targets(mop.internal_id), [root.internal_id])
        self.assertIs(clone.get_part_of_mop(mop.internal_id), clone.get_part("Part"))

        clone.get_primitive(root.internal_id).relative_corner = (99, 99)
        clone.get_layer("Geometry").color = "Blue"
        clone.get_mop(mop.internal_id).target_depth = -2
        self.assertEqual(source.get_primitive(root.internal_id).relative_corner, (0.0, 0.0))
        self.assertEqual(source.get_layer("Geometry").color, "Green")
        self.assertEqual(source.get_mop(mop.internal_id).target_depth, -1)


if __name__ == "__main__":
    unittest.main()
