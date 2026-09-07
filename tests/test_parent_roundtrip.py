"""Regression coverage for parent metadata and world-pose XML round trips."""

import json
import logging
import tempfile
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cambam_builder import CBProject
from cambam_builder.cad_transformations import (
    from_cambam_matrix_str,
    rotation_matrix_deg,
    scale_matrix,
    translation_matrix,
)
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class ParentRoundTripTests(unittest.TestCase):
    """Small synthetic hierarchy used by all tests in this module."""

    def make_project(self, with_child=True):
        project = CBProject("parent-regression")
        root_layer = project.add_layer("RootLayer")
        child_layer = project.add_layer("ChildLayer")
        root = project.add_pline(
            root_layer, [(0.0, 0.0), (2.0, 0.0)], identifier="root",
            groups=["hierarchy"], description="root primitive",
        )
        root.effective_transform = translation_matrix(20.0, 3.0) @ rotation_matrix_deg(25.0)
        if not with_child:
            return project, {"root": root}
        child = project.add_pline(
            child_layer, [(0.0, 0.0), (0.0, 1.0)], identifier="child",
            parent=root,
        )
        grandchild = project.add_pline(
            child_layer, [(0.0, 0.0), (1.0, 1.0)], identifier="grandchild",
            parent=child,
        )
        child.effective_transform = translation_matrix(-4.0, 5.0) @ scale_matrix(2.0, 0.5)
        grandchild.effective_transform = rotation_matrix_deg(-15.0) @ translation_matrix(3.0, 2.0)
        return project, {"root": root, "child": child, "grandchild": grandchild}

    def save(self, project, directory, name="case"):
        path = Path(directory) / (name + ".cb")
        save_cambam_file(project, str(path))
        return path

    @staticmethod
    def primitive_elements(tree):
        return list(tree.getroot().findall("./layers/layer/objects/*"))

    def rewrite_parent_tags(self, path, values, reverse=False):
        tree = ET.parse(path)
        elements = self.primitive_elements(tree)
        by_name = {}
        by_xml_id = {}
        for element in elements:
            tag = json.loads(element.findtext("Tag"))
            by_name[tag["user_id"]] = (element, tag)
            by_xml_id[tag["user_id"]] = int(element.get("id"))
        for name, value in values.items():
            element, tag = by_name[name]
            tag["parent"] = by_xml_id[value] if reverse and value in by_xml_id else value
            element.find("Tag").text = json.dumps(tag, separators=(",", ":"))
        tree.write(path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def reverse_xml_order(path):
        tree = ET.parse(path)
        layers = tree.getroot().find("layers")
        layer_elements = list(layers)
        for layer in layer_elements:
            objects = layer.find("objects")
            if objects is not None:
                objects[:] = list(reversed(list(objects)))
        layers[:] = list(reversed(layer_elements))
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def assert_world_pose(self, expected, actual):
        self.assertEqual(len(expected), len(actual.list_primitives()))
        for name, source in expected.items():
            loaded = actual.get_primitive(name)
            self.assertIsNotNone(loaded, name)
            self.assertTrue(np.allclose(source.get_total_transform(), loaded.get_total_transform(), rtol=0, atol=2e-8), name)
            self.assertTrue(np.allclose(source.get_absolute_coordinates(), loaded.get_absolute_coordinates(), rtol=0, atol=2e-8), name)

    def assert_hierarchy(self, source, loaded):
        self.assertEqual(len(source.list_primitives()), len(loaded.list_primitives()))
        self.assertEqual(len(source.list_layers()), len(loaded.list_layers()))
        self.assertEqual({x.user_identifier for x in source.list_layers()}, {x.user_identifier for x in loaded.list_layers()})
        for name in ("root", "child", "grandchild"):
            original = source.get_primitive(name)
            actual = loaded.get_primitive(name)
            self.assertIsNotNone(actual)
            self.assertEqual(original.internal_id, actual.internal_id)
            self.assertEqual(original.groups, actual.groups)
            self.assertEqual(original.description, actual.description)
            self.assertEqual(original.get_project().get_layer_of_primitive(original).user_identifier,
                             loaded.get_layer_of_primitive(actual).user_identifier)
        for parent, children in (("root", {"child"}), ("child", {"grandchild"}), ("grandchild", set())):
            self.assertEqual({p.user_identifier for p in loaded.get_children_of_primitive(parent)}, children)
        self.assertIsNone(loaded.get_parent_of_primitive("root"))
        self.assertEqual(loaded.get_parent_of_primitive("child").user_identifier, "root")
        self.assertEqual(loaded.get_parent_of_primitive("grandchild").user_identifier, "child")

    def test_writer_emits_parent_uuid_and_two_round_trips_preserve_world_pose(self):
        source, expected = self.make_project()
        with tempfile.TemporaryDirectory() as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            elements = self.primitive_elements(tree)
            self.assertEqual(len(elements), 3)
            self.assertEqual(len({element.get("id") for element in elements}), 3)
            for element in elements:
                name = json.loads(element.findtext("Tag"))["user_id"]
                np.testing.assert_allclose(
                    from_cambam_matrix_str(element.find("mat").get("m")),
                    expected[name].get_total_transform(), rtol=0, atol=2e-8,
                )
            tags = {
                json.loads(element.findtext("Tag"))["user_id"]: json.loads(element.findtext("Tag"))
                for element in self.primitive_elements(tree)
            }
            self.assertEqual(tags["child"]["parent"], str(expected["root"].internal_id))
            self.assertEqual(tags["grandchild"]["parent"], str(expected["child"].internal_id))
            loaded_once = read_cambam_file(str(first))
            self.assertIsNotNone(loaded_once)
            self.assert_hierarchy(source, loaded_once)
            self.assert_world_pose(expected, loaded_once)
            second = self.save(loaded_once, directory, "second")
            self.reverse_xml_order(second)
            loaded_twice = read_cambam_file(str(second))
            self.assertIsNotNone(loaded_twice)
            self.assert_hierarchy(source, loaded_twice)
            self.assert_world_pose(expected, loaded_twice)

    def test_cyclic_parent_metadata_rejects_closing_edge_and_preserves_world_pose(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as directory:
                source, expected = self.make_project()
                path = self.save(source, directory)
                self.rewrite_parent_tags(path, {"root": str(expected["grandchild"].internal_id)})
                if reverse:
                    self.reverse_xml_order(path)
                loaded = read_cambam_file(str(path))
                self.assertIsNotNone(loaded)
                self.assertEqual(len(loaded._primitive_parent_link), 2)
                for primitive in loaded.list_primitives():
                    original = expected[primitive.user_identifier]
                    self.assertEqual(primitive.internal_id, original.internal_id)
                    self.assertEqual(loaded.get_layer_of_primitive(primitive).user_identifier,
                                     source.get_layer_of_primitive(original).user_identifier)
                    visited = set()
                    current = primitive
                    while current is not None:
                        self.assertNotIn(current.internal_id, visited)
                        visited.add(current.internal_id)
                        current = loaded.get_parent_of_primitive(current)
                    parent = loaded.get_parent_of_primitive(primitive)
                    if parent is not None:
                        self.assertIn(primitive.internal_id, {
                            child.internal_id for child in loaded.get_children_of_primitive(parent)
                        })
                self.assert_world_pose(expected, loaded)
                second = self.save(loaded, directory, "second")
                reloaded = read_cambam_file(str(second))
                self.assertIsNotNone(reloaded)
                self.assertEqual(loaded._primitive_parent_link, reloaded._primitive_parent_link)
                self.assert_world_pose(expected, reloaded)

    def test_reversed_layer_and_object_xml_order_preserves_hierarchy(self):
        source, expected = self.make_project()
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory)
            self.reverse_xml_order(path)
            # Correct UUID metadata makes this an ordering-focused reader test.
            self.rewrite_parent_tags(path, {"child": str(expected["root"].internal_id),
                                            "grandchild": str(expected["child"].internal_id)})
            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            self.assert_hierarchy(source, loaded)
            self.assert_world_pose(expected, loaded)

    def test_correct_uuid_parent_reconstructs_world_matrix_without_doubling(self):
        source, expected = self.make_project()
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory)
            self.rewrite_parent_tags(path, {"child": str(expected["root"].internal_id),
                                            "grandchild": str(expected["child"].internal_id)})
            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            self.assert_world_pose(expected, loaded)

    def test_integer_xml_parent_reference_is_supported(self):
        source, expected = self.make_project()
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory)
            self.rewrite_parent_tags(path, {"child": "root"}, reverse=True)
            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.get_parent_of_primitive("child").user_identifier, "root")
            self.assert_world_pose(expected, loaded)

    def test_missing_and_invalid_parent_keep_exported_world_pose_parentless(self):
        source, expected = self.make_project()
        for value in (None, "", str(uuid.uuid4()), "not-a-parent", "999999", 999999, [], {}, True, 1.5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = self.save(source, directory)
                self.rewrite_parent_tags(path, {"child": value})
                loaded = read_cambam_file(str(path))
                self.assertIsNotNone(loaded)
                child = loaded.get_primitive("child")
                self.assertIsNone(loaded.get_parent_of_primitive(child))
                self.assertTrue(np.allclose(expected["child"].get_total_transform(), child.get_total_transform(), rtol=0, atol=2e-8))
                self.assertTrue(np.allclose(expected["child"].get_absolute_coordinates(), child.get_absolute_coordinates(), rtol=0, atol=2e-8))

    def test_self_parent_is_rejected_without_changing_world_pose(self):
        source, expected = self.make_project()
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory)
            self.rewrite_parent_tags(path, {"child": str(expected["child"].internal_id)})
            loaded = read_cambam_file(str(path))
            self.assertIsNotNone(loaded)
            self.assertIsNone(loaded.get_parent_of_primitive("child"))
            self.assert_world_pose(expected, loaded)

    def test_singular_parent_transform_is_rejected_when_child_exists(self):
        source, expected = self.make_project()
        expected["root"].effective_transform = np.array([[0.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        with tempfile.TemporaryDirectory() as directory:
            path = self.save(source, directory)
            self.rewrite_parent_tags(path, {"child": str(expected["root"].internal_id),
                                            "grandchild": str(expected["child"].internal_id)})
            with self.assertLogs("cambam_builder.cambam_reader", level=logging.ERROR) as logs:
                self.assertIsNone(read_cambam_file(str(path)))
            self.assertIn("singular world transform", " ".join(logs.output))
            # UUID-sorted export may encounter either singular ancestor first.
            message = " ".join(logs.output)
            self.assertTrue(any(
                f"child {expected[child].internal_id}: parent {expected[parent].internal_id}" in message
                for child, parent in (("child", "root"), ("grandchild", "child"))
            ))

    def test_singular_root_without_children_loads(self):
        source, expected = self.make_project(with_child=False)
        expected["root"].effective_transform = np.array([[0.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        with tempfile.TemporaryDirectory() as directory:
            loaded = read_cambam_file(str(self.save(source, directory)))
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.list_primitives()), 1)
            self.assert_world_pose(expected, loaded)

    def test_rect_world_geometry_survives_nonuniform_scale_round_trip(self):
        project = CBProject("rect-regression")
        layer = project.add_layer("Geometry")
        rect = project.add_rect(layer, corner=(1.0, 2.0), width=3.0, height=4.0, identifier="rect")
        rect.effective_transform = translation_matrix(7.0, -3.0) @ scale_matrix(2.0, 3.0)
        with tempfile.TemporaryDirectory() as directory:
            loaded = read_cambam_file(str(self.save(project, directory)))
            self.assertIsNotNone(loaded)
            actual = loaded.get_primitive("rect")
            self.assertTrue(np.allclose(rect.get_total_transform(), actual.get_total_transform(), rtol=0, atol=2e-8))
            self.assertTrue(np.allclose(rect.get_absolute_coordinates(), actual.get_absolute_coordinates(), rtol=0, atol=2e-8))


if __name__ == "__main__":
    unittest.main()
