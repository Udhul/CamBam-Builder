"""Focused contract tests for canonical Pline/Points vertex records."""

import copy
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject, Vertex
from cambam_builder.cambam_entities import Pline, Points
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class VertexRecordTests(unittest.TestCase):
    def test_vertex_constructor_defaults_and_keyword_only_bulge(self):
        self.assertEqual(Vertex(1, 2), Vertex(1.0, 2.0, 0.0, bulge=0.0))
        self.assertEqual(Vertex(1, 2, 3), Vertex(1.0, 2.0, 3.0, bulge=0.0))
        self.assertEqual(Vertex(1, 2, bulge=.5), Vertex(1.0, 2.0, 0.0, bulge=.5))
        self.assertEqual(Vertex(1, 2, 3, bulge=.5), Vertex(1.0, 2.0, 3.0, bulge=.5))
        with self.assertRaises(TypeError):
            Vertex(1, 2, 3, .5)

    def test_tuple_shorthand_is_only_xy_or_xyz(self):
        pline = Pline(vertices=[(1, 2), (3, 4, .5)])
        self.assertEqual(
            pline.vertices,
            [Vertex(1, 2), Vertex(3, 4, .5)],
        )
        self.assertEqual(
            pline.get_absolute_coordinates(),
            [(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)],
        )
        with self.assertRaises(ValueError):
            Pline(vertices=[(1, 2, 3, .5)])

    def test_points_reject_nonzero_bulge_and_malformed_values(self):
        with self.assertRaisesRegex(ValueError, "zero bulge"):
            Points(vertices=[Vertex(1, 2, bulge=.5)])
        for values in ([Vertex(0, 0), "bad"], [[1, 2]], [(1, 2, math.inf)]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Points(vertices=values)

    def test_collection_edits_keep_each_vertex_record_together(self):
        points = Points(vertices=[(0, 0, 1), (2, 0, 3)])
        points.vertices.insert(1, Vertex(1, 0, 2))
        points.vertices[:] = [points.vertices[2], points.vertices[0], points.vertices[1]]
        self.assertEqual(
            points.get_absolute_coordinates_xyz(),
            [(2.0, 0.0, 3.0), (0.0, 0.0, 1.0), (1.0, 0.0, 2.0)],
        )

        pline = Pline(vertices=[Vertex(0, 0, 4, bulge=.25), (2, 0, 4)])
        cloned = copy.deepcopy(pline)
        pline.vertices.reverse()
        self.assertEqual(pline.vertices[1], Vertex(0, 0, 4, bulge=.25))
        self.assertEqual(pline.get_absolute_coordinates_xyz()[1], (0.0, 0.0, 4.0, .25))
        self.assertEqual(cloned.vertices[0], Vertex(0, 0, 4, bulge=.25))

        points.vertices.append((4, 0, 5))
        with self.assertRaisesRegex(ValueError, "Vertex record"):
            points.get_absolute_coordinates_xyz()

    def test_project_adders_normalize_records_and_reject_invalid_points(self):
        project = CBProject("vertex-adders")
        layer = project.add_layer("Geometry")
        pline = project.add_pline(layer, [(0, 0), Vertex(1, 0, 2, bulge=.5)])
        points = project.add_points(layer, [(0, 0), (1, 2, 3)])
        self.assertTrue(all(isinstance(vertex, Vertex) for vertex in pline.vertices))
        self.assertEqual(points.vertices[1].z, 3.0)
        self.assertIsNone(project.add_points(layer, [Vertex(0, 0, bulge=.5)]))

    def test_two_xml_round_trips_preserve_records_and_points_reject_xml_bulge(self):
        project = CBProject("vertex-xml")
        project.output_decimals = 12
        layer = project.add_layer("Geometry")
        expected_pline = [
            Vertex(0, 0, 4, bulge=.5), Vertex(2, 0, 4), Vertex(2, 2, 7)
        ]
        expected_points = [Vertex(3, 4, -2), Vertex(5, 6, 8)]
        project.add_pline(layer, expected_pline, identifier="curve")
        project.add_points(layer, expected_points, identifier="drills")

        with tempfile.TemporaryDirectory(prefix="vertex-records-") as directory:
            first = Path(directory) / "first.cb"
            second = Path(directory) / "second.cb"
            save_cambam_file(project, str(first))
            loaded = read_cambam_file(str(first))
            self.assertIsNotNone(loaded)
            save_cambam_file(loaded, str(second))
            loaded_twice = read_cambam_file(str(second))
            self.assertEqual(loaded_twice.get_primitive("curve").vertices, expected_pline)
            self.assertEqual(loaded_twice.get_primitive("drills").vertices, expected_points)

            tree = ET.parse(second)
            point = tree.getroot().find("./layers/layer/objects/points/pts/p")
            point.set("b", "0.25")
            tree.write(second, encoding="utf-8", xml_declaration=True)
            self.assertIsNone(read_cambam_file(str(second)))


if __name__ == "__main__":
    unittest.main()
