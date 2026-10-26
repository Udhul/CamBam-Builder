"""Independent synthetic Default-post inputs for bounded native tab evidence.

No generated program in this module is represented as actual CamBam output.
"""

from dataclasses import replace
import unittest

from cambam_builder.integrations.cambam.native_tabs import (
    DEFAULT_POINTS, audit_post, create_document,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def synthetic_post(style="Triangle", *, left=(25,), bottom=40):
    """Hand-constructed offset rectangle; round corners have radius1.5 mm."""
    lines = ["( Made using CamBam - http://www.cambam.co.uk )",
             "( synthetic native-tabs regression )", "( Post processor: Default )",
             "G21 G90 G17 G61", "G0 Z5", "( manual-tab-fixture )", "T1 M6",
             "S12000 M3", "G0 X10 Y8.5"]
    corners = ("G3 X71.5 Y10 I0 J1.5", "G3 X70 Y41.5 I-1.5 J0",
               "G3 X8.5 Y40 I0 J-1.5", "G3 X10 Y8.5 I1.5 J0")
    # Components use actual drawing coordinates and explicit direction, rather
    # than importing the integration's tab/depth/section calculations.
    edges = (("X", 70, (bottom,), 1), ("Y", 40, (25,), 1),
             ("X", 10, (40,), -1), ("Y", 10, tuple(sorted(left, reverse=True)), -1))
    for depth in (1, 2, 3):
        lines.append(f"G1 F80 Z{-depth}")
        for edge, corner in zip(edges, corners):
            axis, end, centers, direction = edge
            if depth < 3:
                lines.append(f"G1 F250 {axis}{end}")
            else:
                for center in centers:
                    lines.append(f"G1 F250 {axis}{center - direction * 4.5}")
                    if style == "Triangle":
                        lines.extend((f"G1 {axis}{center - direction * 1.5} Z-2",
                                      f"G1 {axis}{center + direction * 1.5}",
                                      f"G1 {axis}{center + direction * 4.5} Z-3"))
                    else:
                        lines.extend(("G0 Z-2", f"G1 F250 {axis}{center + direction * 4.5}",
                                      "G1 F80 Z-3"))
                lines.append(f"G1 F250 {axis}{end}")
            lines.append(corner)
    lines.extend(("G0 Z5", "M5", "M30"))
    return ("\n".join(lines) + "\n").encode("ascii")


class NativeTabDocumentTests(unittest.TestCase):
    def test_both_styles_and_native_parameter_roundtrip(self):
        for style in ("Triangle", "Square"):
            with self.subTest(style=style):
                document = create_document(style=style)
                project = read_cambam_bytes(document.xml_bytes, strict=True)
                mop = project.get_mop("manual-tab-fixture")
                self.assertEqual((mop.tab_style, mop.tab_method, mop.depth_increment),
                                 (style, "Manual", 1))
                inspected = mop.inspected_manual_tab_points(project)
                self.assertIsNotNone(inspected)
                for got, expected in zip(inspected, DEFAULT_POINTS):
                    self.assertAlmostEqual(got[0], expected[0], places=9)
                    self.assertAlmostEqual(got[1], expected[1], places=9)
                self.assertEqual(document.check_document(serialize_cambam_bytes(project))["status"], "pass")

    def test_binding_rejects_source_geometry_process_and_tab_edits(self):
        document = create_document()
        for old, new in ((b"70.0,40.0,0.0", b"71.0,40.0,0.0"),
                         (b"<PMax>80.0,50.0,0.0", b"<PMax>81.0,50.0,0.0"),
                         (b">12000<", b">11000<"),
                         (b">Triangle<", b">Square<"),
                         (b"<Width>6<", b"<Width>7<"),
                         (b">0.166666666666667<", b">0.15<")):
            with self.subTest(old=old):
                self.assertIn(old, document.xml_bytes)
                changed = document.xml_bytes.replace(old, new, 1)
                with self.assertRaises(ValueError):
                    document.check_document(changed)
                with self.assertRaises(ValueError):
                    replace(document, xml_bytes=changed).check_document()

    def test_unsupported_tab_sources_fail_closed(self):
        for options in ({"style": "Skip"}, {"points": DEFAULT_POINTS[:3]},
                        {"points": DEFAULT_POINTS[:-1] + ((10, 39),)},
                        {"points": DEFAULT_POINTS[:-1] + ((float("nan"), 25),)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_document(**options)


class NativeTabPostTests(unittest.TestCase):
    def test_default_post_precision_is_explicit_at_authoring_boundary(self):
        points = ((40.1234, 10),) + DEFAULT_POINTS[1:]
        self.assertEqual(audit_post(create_document(points=points),
                                   synthetic_post(bottom=40.1234))["status"], "pass")
        with self.assertRaisesRegex(ValueError, r"0\.0001 mm"):
            create_document(points=((40.12345, 10),) + DEFAULT_POINTS[1:])

    def test_native_edge_projection_tolerance_reaches_posted_audit(self):
        for distance in (1e-9, 0.5e-6, 1e-6):
            with self.subTest(distance=distance):
                points = DEFAULT_POINTS[:-1] + ((10 + distance, 25),)
                document = create_document(points=points)
                self.assertEqual(audit_post(document, synthetic_post())["status"], "pass")
        with self.assertRaises(ValueError):
            create_document(points=DEFAULT_POINTS[:-1] + ((10 + 1.1e-6, 25),))

    def test_triangle_sections_clip_ramps_and_square_retains_full_width(self):
        for style, expected in (("Triangle", (1.5, 3, 4.5, 6)),
                                ("Square", (6, 6, 6, 6))):
            with self.subTest(style=style):
                report = audit_post(create_document(style=style), synthetic_post(style))
                self.assertEqual(report["status"], "pass")
                self.assertEqual(report["native_ramp_moves"], 8 if style == "Triangle" else 0)
                self.assertTrue(report["retained_part_connected_at_bottom"])
                for row, width in zip(report["sections"], expected):
                    self.assertEqual(row["ideal_minimum_bridge_width_mm"], width)
                    self.assertAlmostEqual(row["minimum_bridge_width_mm_lower"], width, delta=0.01)
                    self.assertEqual(row["retained_part_connected"], width > 0)
                    self.assertLess(row["protected_part_overcut_mm2_upper"], 0.2)

    def test_moved_and_added_manual_tabs_require_matching_posts(self):
        for left in ((33,), (33, 17)):
            with self.subTest(left=left):
                points = DEFAULT_POINTS[:3] + tuple((10, y) for y in left)
                document = create_document(points=points)
                report = audit_post(document, synthetic_post(left=left))
                self.assertEqual(report["tab_count"], 3 + len(left))
                self.assertTrue(report["retained_part_connected_at_bottom"])
                with self.assertRaises(ValueError):
                    audit_post(document, synthetic_post())

    def test_wrong_style_missing_moved_or_flattened_ramps_reject(self):
        document, post = create_document(), synthetic_post()
        for replacement in (b"G1 X44.5 Z-3", b"G1 X40 Z-2.1\nG1 X44.5 Z-3",
                            b"G1 X41 Z-2\nG1 X45.5 Z-3",
                            b"G1 X40 Z-2\nG1 X44.5 Z-3",
                            b"G0 Z-2\nG1 X44.5\nG1 F80 Z-3"):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                audit_post(document, post.replace(b"G1 X38.5 Z-2\nG1 X41.5\nG1 X44.5 Z-3", replacement, 1))
        with self.assertRaises(ValueError):
            audit_post(document, synthetic_post("Square"))

    def test_complete_motion_process_and_execution_guards(self):
        document, post = create_document(), synthetic_post()
        changes = ((b"S12000 M3", b"S11000 M3"), (b"T1 M6", b"T2 M6"),
                   (b"F250", b"F251"), (b"G61", b"G64"),
                   (b"G61", b"G40"), (b"G90", b"G91"),
                   (b"G0 X10 Y8.5", b"G0 Z-1\nG0 X10 Y8.5"),
                   (b"G1 X38.5 Z-2", b"G0 X38.5 Z-2"),
                   (b"G3 X71.5 Y10 I0 J1.5", b"G3 X71.5 Y10 I0 J1.4"),
                   (b"G3 X71.5 Y10 I0 J1.5", b"G2 X71.5 Y10 I0 J1.5"),
                   (b"G1 F250 X70\n", b"G1 F250 X69\n"),
                   (b"G1 F250 X70\n", b"G1 F250 X70\nG1 X60\nG1 X70\n"),
                   (b"G1 F80 Z-2\n", b"G1 F80 Z-3\n"),
                   (b"M5\n", b""), (b"M30\n", b""),
                   (b"( manual-tab-fixture )", b"( unrelated )"))
        for old, new in changes:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.assertIn(old, post)
                audit_post(document, post.replace(old, new, 1))

    def test_every_byte_and_candidate_revision_are_bound(self):
        document, post = create_document(), synthetic_post()
        report = audit_post(document, post)
        # Harmless whitespace changes are a different complete-byte observation.
        changed = audit_post(document, post + b"\n")
        self.assertNotEqual(report["certificate"], changed["certificate"])
        self.assertNotEqual(report["post_sha256"], changed["post_sha256"])
        altered = document.xml_bytes.replace(b"<Height>1<", b"<Height>0.9<")
        with self.assertRaises(ValueError):
            audit_post(document, post, candidate_bytes=altered)


if __name__ == "__main__":
    unittest.main()
