"""Regression coverage for primitive subtree copy and transfer operations."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cambam_entities import Layer
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file
from cambam_builder.cad_transformations import rotation_matrix_deg, scale_matrix, translation_matrix


class _ExplodingCopy:
    def __deepcopy__(self, memo):
        raise RuntimeError("injected deepcopy failure")


class CopyTransferTests(unittest.TestCase):
    def make_source(self, invalid_mop=False):
        source = CBProject("copy-source")
        root_layer = source.add_layer("SourceRoot")
        child_layer = source.add_layer("SourceChild")
        outside_layer = source.add_layer("OutsideLayer")
        cut_part = source.add_part("CutPart", default_tool_diameter=3.0,
                                   default_spindle_speed=12000)
        other_part = source.add_part("OtherPart")

        root = source.add_pline(root_layer, [(0, 0), (2, 0)], identifier="root",
                                groups=["copied", "shared"], description="root shape")
        root.effective_transform = np.array([[1.2, .25, 11.0], [-.15, .8, -4.0], [0, 0, 1]])
        child = source.add_rect(child_layer, corner=(1, 2), width=3, height=2,
                                identifier="child", groups=["copied", "live"], parent=root)
        child.effective_transform = translation_matrix(-3, 5) @ rotation_matrix_deg(23)
        leaf = source.add_circle(child_layer, center=(2, 1), diameter=1.5,
                                 identifier="leaf", groups=["copied", "live"], parent=child)
        leaf.effective_transform = scale_matrix(1.4, .7)
        outside = source.add_points(outside_layer, [(40, 2), (41, 3)], identifier="outside",
                                    groups=["outside"])

        inside = source.add_profile_mop(
            cut_part, targets=[child, leaf], identifier="inside", name="Inside",
            target_depth=-1.25, depth_increment=.5, tool_diameter=2.0,
            spindle_speed=9000, cut_feedrate=240.0,
        )
        live = source.add_pocket_mop(
            cut_part, target_group="live", identifier="live", name="Live",
            target_depth=-2.0, depth_increment=.25, tool_diameter=2.5,
            spindle_speed=8000, cut_feedrate=180.0,
        )
        unrelated = source.add_drill_mop(other_part, targets=[outside], identifier="unrelated",
                                         name="Unrelated", target_depth=-3.0)
        empty = source.add_engrave_mop(cut_part, target_group="empty", identifier="empty",
                                        name="Empty")
        mixed = None
        if invalid_mop:
            mixed = source.add_profile_mop(cut_part, targets=[child, outside],
                                           identifier="mixed", name="Mixed")
        entities = {
            "root": root, "child": child, "leaf": leaf, "outside": outside,
            "root_layer": root_layer, "child_layer": child_layer,
            "outside_layer": outside_layer, "cut_part": cut_part, "other_part": other_part,
            "inside": inside, "live": live, "unrelated": unrelated, "empty": empty,
        }
        if mixed is not None:
            entities["mixed"] = mixed
        return source, entities

    @staticmethod
    def included_entities(e):
        return [e[k] for k in ("root", "child", "leaf", "root_layer", "child_layer",
                               "cut_part", "inside", "live")]

    def test_copy_clones_subtree_relationships_and_appends_order(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        existing_layer = target.add_layer("ExistingLayer")
        existing_part = target.add_part("ExistingPart")
        existing = target.add_rect(existing_layer, identifier="existing")
        target.add_profile_mop(existing_part, targets=[existing], identifier="existing-mop")

        mapping = source.copy_primitive_tree(e["root"], target)
        included = self.included_entities(e)
        self.assertEqual({x.internal_id for x in included}, set(mapping))
        self.assertEqual({x.internal_id for x in included}, set(mapping.values()))
        self.assertEqual(["ExistingLayer", "SourceRoot", "SourceChild"],
                         [x.user_identifier for x in target.list_layers()])
        self.assertEqual(["ExistingPart", "CutPart"],
                         [x.user_identifier for x in target.list_parts()])
        # Primitive listing is UUID-sorted by the existing public API; layer,
        # part, and per-part MOP order are the append-order contracts.
        self.assertEqual({"existing", "root", "child", "leaf"},
                         {x.user_identifier for x in target.list_primitives()})
        self.assertEqual(["existing-mop", "inside", "live"],
                         [x.user_identifier for x in target.list_mops()])
        for name in ("root", "child", "leaf"):
            clone = target.get_primitive(name)
            self.assertIsNotNone(clone)
            self.assertEqual(mapping[e[name].internal_id], clone.internal_id)
        self.assertEqual("root", target.get_parent_of_primitive("child").user_identifier)
        self.assertEqual("child", target.get_parent_of_primitive("leaf").user_identifier)
        self.assertEqual({target.get_primitive("child").internal_id,
                          target.get_primitive("leaf").internal_id},
                         set(target.get_mop_targets("inside")))
        self.assertEqual("live", target.get_mop_target_group("live"))

    def test_copy_preserves_ids_by_default_and_remaps_all_names_groups_and_ids(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        included = self.included_entities(e)
        names = {x.user_identifier: x.user_identifier + "-new" for x in included}
        groups = {"copied": "copied-new", "shared": "shared-new", "live": "live-new"}
        mapping = source.copy_primitive_tree(e["root"], target, preserve_ids=False,
                                             identifier_map=names, group_map=groups)
        self.assertEqual(len(mapping), len(included))
        self.assertTrue(all(mapping[x.internal_id] != x.internal_id for x in included))
        self.assertEqual({"root-new", "child-new", "leaf-new"},
                         {x.user_identifier for x in target.list_primitives()})
        self.assertEqual(["copied-new", "shared-new"], target.get_groups_of_primitive("root-new"))
        self.assertEqual(["copied-new", "live-new"], sorted(target.get_groups_of_primitive("child-new")))
        self.assertEqual("live-new", target.get_mop_target_group("live-new"))
        self.assertEqual("Inside", target.get_mop("inside-new").name)

    def test_selected_descendant_is_detached_at_source_world_pose_and_keeps_child_locals(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        mapping = source.copy_primitive_tree(e["child"], target)
        copied_child = target.get_primitive("child")
        copied_leaf = target.get_primitive("leaf")
        self.assertIsNone(target.get_parent_of_primitive(copied_child))
        np.testing.assert_allclose(copied_child.get_total_transform(), e["child"].get_total_transform())
        np.testing.assert_allclose(copied_leaf.effective_transform, e["leaf"].effective_transform)
        np.testing.assert_allclose(copied_leaf.get_total_transform(), e["leaf"].get_total_transform())
        self.assertEqual({e["child"].internal_id, e["leaf"].internal_id,
                          e["child_layer"].internal_id, e["cut_part"].internal_id,
                          e["inside"].internal_id, e["live"].internal_id}, set(mapping))

    def test_copy_is_deep_and_does_not_mutate_source(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        source.copy_primitive_tree(e["root"], target)
        clone = target.get_primitive("child")
        clone.relative_corner = (99, 88)
        target.get_mop("inside").target_depth = -99
        target.add_primitive_to_group(clone, "only-target")
        self.assertEqual((1, 2), e["child"].relative_corner)
        self.assertEqual(-1.25, e["inside"].target_depth)
        self.assertNotIn("only-target", e["child"].groups)

    def test_transfer_removes_primitives_and_cleans_retained_mop_targets_without_mops(self):
        source, e = self.make_source()
        target = CBProject("transfer-target")
        source.transfer_primitive_tree(e["root"], target, include_mops=False)
        self.assertEqual(["outside"], [x.user_identifier for x in source.list_primitives()])
        self.assertEqual(["inside", "live", "empty", "unrelated"],
                         [x.user_identifier for x in source.list_mops()])
        self.assertEqual([], source.get_mop_targets("inside"))
        self.assertEqual([], source.get_mop_targets("live"))
        self.assertEqual([e["outside"].internal_id], source.get_mop_targets("unrelated"))
        self.assertEqual("empty", source.get_mop_target_group("empty"))
        self.assertEqual(["SourceRoot", "SourceChild", "OutsideLayer"],
                         [x.user_identifier for x in source.list_layers()])
        self.assertEqual(["CutPart", "OtherPart"], [x.user_identifier for x in source.list_parts()])
        self.assertEqual({"root", "child", "leaf"},
                         {x.user_identifier for x in target.list_primitives()})
        self.assertEqual([], target.list_mops())
        self.assertIsNone(e["root"].get_project())

    def test_transfer_removes_included_mops_and_retains_unrelated_entity_references(self):
        source, e = self.make_source()
        target = CBProject("transfer-target")
        source.transfer_primitive_tree(e["root"], target)
        self.assertEqual(["outside"], [x.user_identifier for x in source.list_primitives()])
        self.assertEqual(["empty", "unrelated"], [x.user_identifier for x in source.list_mops()])
        self.assertEqual([e["outside"].internal_id], source.get_mop_targets("unrelated"))
        self.assertEqual({"root", "child", "leaf"},
                         {x.user_identifier for x in target.list_primitives()})
        self.assertEqual(["inside", "live"], [x.user_identifier for x in target.list_mops()])
        self.assertTrue(all(x.get_project() is target for x in target.list_primitives()))

    def test_mixed_mop_target_rejects_atomically(self):
        source, e = self.make_source(invalid_mop=True)
        target = CBProject("copy-target")
        before_source = ([x.internal_id for x in source.list_primitives()],
                         [x.internal_id for x in source.list_mops()])
        with self.assertRaises(ValueError):
            source.copy_primitive_tree(e["root"], target)
        self.assertEqual([], target.list_primitives())
        self.assertEqual(before_source[0], [x.internal_id for x in source.list_primitives()])
        self.assertEqual(before_source[1], [x.internal_id for x in source.list_mops()])

    def test_identifier_layer_part_mop_and_group_collisions_reject(self):
        source, e = self.make_source()
        cases = (
            ("primitive", lambda t: t.add_rect(t.add_layer("T"), identifier="root")),
            ("layer", lambda t: t.add_layer("SourceRoot")),
            ("part", lambda t: t.add_part("CutPart")),
            ("mop", lambda t: t.add_profile_mop(t.add_part("P"), identifier="inside")),
            ("group", lambda t: t.add_rect(t.add_layer("T"), identifier="g", groups=["copied"])),
            # A live selector reserves its group name even when currently empty.
            ("empty-live-group", lambda t: t.add_profile_mop(
                t.add_part("P"), target_group="copied", identifier="reserved")),
        )
        for kind, seed in cases:
            with self.subTest(kind=kind):
                target = CBProject("copy-target")
                seed(target)
                with self.assertRaises(ValueError):
                    source.copy_primitive_tree(e["root"], target)

    def test_uuid_collision_across_entity_types_rejects_atomically(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        collision = Layer(user_identifier="uuid-collision")
        collision.internal_id = e["root"].internal_id
        self.assertTrue(target._register_entity(collision, target._layers))
        target._layer_order.append(collision.internal_id)
        target._layer_primitive_membership[collision.internal_id] = set()
        with self.assertRaises(ValueError):
            source.copy_primitive_tree(e["root"], target)
        self.assertEqual(["uuid-collision"], [x.user_identifier for x in target.list_layers()])
        self.assertEqual([], target.list_primitives())

    def test_group_map_collision_rejects_but_group_and_entity_namespaces_are_separate(self):
        source, e = self.make_source()
        with self.assertRaises(ValueError):
            source.copy_primitive_tree(e["root"], CBProject("copy-target"),
                                       group_map={"copied": "shared"})

        target = CBProject("copy-target")
        layer = target.add_layer("TargetLayer")
        target.add_rect(layer, identifier="copied")
        mapping = source.copy_primitive_tree(e["root"], target)
        self.assertEqual(e["root"].internal_id, mapping[e["root"].internal_id])
        self.assertEqual({"copied", "shared"}, set(target.get_groups_of_primitive("root")))

    def test_invalid_affine_transform_rejects_atomically(self):
        source, e = self.make_source()
        original = e["root"].effective_transform.copy()
        e["root"].effective_transform = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]])
        target = CBProject("copy-target")
        with self.assertRaises(ValueError):
            source.copy_primitive_tree(e["root"], target)
        self.assertEqual([], target.list_primitives())
        self.assertEqual([], target.list_layers())
        e["root"].effective_transform = original

    def test_live_group_with_outside_member_rejects_atomically(self):
        source, e = self.make_source()
        self.assertTrue(source.add_primitive_to_group(e["outside"], "live"))
        target = CBProject("copy-target")
        source_before = [x.internal_id for x in source.list_primitives()]
        with self.assertRaises(ValueError):
            source.copy_primitive_tree(e["root"], target)
        self.assertEqual([], target.list_primitives())
        self.assertEqual(source_before, [x.internal_id for x in source.list_primitives()])

    def test_invalid_root_options_and_mapping_keys_are_rejected(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        for call in (
            lambda: source.copy_primitive_tree("missing", target),
            lambda: source.copy_primitive_tree(e["root"], target, include_mops=1),
            lambda: source.copy_primitive_tree(e["root"], target, identifier_map={"unknown": "x"}),
            lambda: source.copy_primitive_tree(e["root"], target, identifier_map={"root": ""}),
            lambda: source.copy_primitive_tree(e["root"], target,
                                               identifier_map={"root": "same", "child": "same"}),
            lambda: source.copy_primitive_tree(e["root"], target, group_map={"unknown": "x"}),
            lambda: source.copy_primitive_tree(e["root"], target, group_map={"copied": "shared"}),
        ):
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError, KeyError)):
                call()

    def test_same_project_copy_with_full_remap_succeeds_but_transfer_rejects(self):
        source, e = self.make_source()
        included = self.included_entities(e)
        identifier_map = {x.user_identifier: x.user_identifier + "-same" for x in included}
        group_map = {"copied": "copied-same", "shared": "shared-same", "live": "live-same"}
        mapping = source.copy_primitive_tree(e["root"], source, preserve_ids=False,
                                             identifier_map=identifier_map, group_map=group_map)
        self.assertEqual(len(included), len(mapping))
        self.assertEqual(7, len(source.list_primitives()))
        with self.assertRaises(ValueError):
            source.transfer_primitive_tree(e["root"], source)

    def test_late_deepcopy_failure_leaves_both_projects_unchanged(self):
        for operation in ("copy", "transfer"):
            with self.subTest(operation=operation):
                source, e = self.make_source()
                e["child"].injected_failure = _ExplodingCopy()
                target = CBProject("copy-target")
                target_layer = target.add_layer("Existing")
                target.add_rect(target_layer, identifier="existing")
                source_before = ([x.internal_id for x in source.list_primitives()],
                                 [x.user_identifier for x in source.list_layers()])
                target_before = ([x.internal_id for x in target.list_primitives()],
                                 [x.user_identifier for x in target.list_layers()])
                with self.assertRaises(RuntimeError):
                    getattr(source, operation + "_primitive_tree")(e["root"], target)
                self.assertEqual(source_before, ([x.internal_id for x in source.list_primitives()],
                                                 [x.user_identifier for x in source.list_layers()]))
                self.assertEqual(target_before, ([x.internal_id for x in target.list_primitives()],
                                                 [x.user_identifier for x in target.list_layers()]))

    def test_two_xml_round_trips_preserve_clones_references_world_pose_and_parameters(self):
        source, e = self.make_source()
        target = CBProject("copy-target")
        mapping = source.copy_primitive_tree(e["root"], target)
        output = Path("output").resolve()
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="copy-transfer-checks-", dir=output) as directory:
            first = Path(directory) / "first.cb"
            second = Path(directory) / "second.cb"
            save_cambam_file(target, str(first))
            loaded_once = read_cambam_file(str(first))
            save_cambam_file(loaded_once, str(second))
            loaded_twice = read_cambam_file(str(second))
        expected_geometry = {
            name: target.get_primitive(name).get_absolute_coordinates()
            for name in ("root", "child", "leaf")
        }
        expected_matrices = {
            name: target.get_primitive(name).get_total_transform()
            for name in ("root", "leaf")
        }
        expected_targets = {
            name: set(target.get_mop_targets(name)) for name in ("inside", "live")
        }
        for loaded in (loaded_once, loaded_twice):
            self.assertEqual(3, len(loaded.list_primitives()))
            self.assertEqual(2, len(loaded.list_layers()))
            self.assertEqual(1, len(loaded.list_parts()))
            self.assertEqual(["inside", "live"], [m.user_identifier for m in loaded.list_mops()])
            for name in ("root", "child", "leaf"):
                actual = loaded.get_primitive(name)
                self.assertEqual(mapping[e[name].internal_id], actual.internal_id)
                actual_geometry = actual.get_absolute_coordinates()
                expected = expected_geometry[name]
                if isinstance(expected, dict):
                    np.testing.assert_allclose(actual_geometry["center"], expected["center"],
                                               rtol=0, atol=2e-8)
                    self.assertAlmostEqual(actual_geometry["diameter"], expected["diameter"],
                                           delta=2e-8)
                elif name == "child":
                    # The sheared Rect is emitted as a Pline with its original
                    # corners and the world matrix carried separately.
                    np.testing.assert_allclose(np.asarray(actual_geometry)[:, :2],
                                               np.asarray(expected), rtol=0, atol=2e-8)
                else:
                    np.testing.assert_allclose(actual_geometry, expected, rtol=0, atol=2e-8)
            for name, matrix in expected_matrices.items():
                np.testing.assert_allclose(loaded.get_primitive(name).get_total_transform(), matrix,
                                           rtol=0, atol=2e-8)
            self.assertIsNone(loaded.get_parent_of_primitive("root"))
            self.assertEqual("root", loaded.get_parent_of_primitive("child").user_identifier)
            self.assertEqual("child", loaded.get_parent_of_primitive("leaf").user_identifier)
            for name, layer_name in (("root", "SourceRoot"), ("child", "SourceChild"),
                                     ("leaf", "SourceChild")):
                self.assertEqual(layer_name, loaded.get_layer_of_primitive(name).user_identifier)
            for name in ("inside", "live"):
                mop = loaded.get_mop(name)
                self.assertEqual(mapping[e[name].internal_id], mop.internal_id)
                self.assertEqual(target.get_mop(name).target_depth, mop.target_depth)
                self.assertEqual(expected_targets[name], set(loaded.get_mop_targets(name)))
            # XML stores resolved primitive references; live group selectors are
            # runtime relationships and are intentionally not serialized.
            self.assertIsNone(loaded.get_mop_target_group("live"))


if __name__ == "__main__":
    unittest.main()
