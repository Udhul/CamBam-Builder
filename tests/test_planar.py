"""Detached nominal runtime acceptance; no stock or machining certification."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import importlib.util
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

from cambam_builder.planar import (
    Circle, ErrorBudget, PlanarFrame, Polygon, Rectangle, RegionSet,
    difference, feasible_centers, nominal_area_erosion, normalize, union,
)
from cambam_builder import _planar_shapely as adapter


FRAME = PlanarFrame("mm", "fixture", (0, 0), 0)
BUDGET = ErrorBudget(0.001, 0.01)
HAS_BACKEND = importlib.util.find_spec("shapely") is not None


def square(x, y, size):
    return ((x, y), (x+size, y), (x+size, y+size), (x, y+size), (x, y))


def area(value):
    def ring_area(ring):
        return abs(sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(ring, ring[1:]))) / 2
    return sum(ring_area(c.shell) - sum(ring_area(h) for h in c.holes) for c in value.components)


class AnalyticTests(unittest.TestCase):
    def centers(self, shape, radius=1, frame=FRAME, units="mm", budget=BUDGET):
        result = feasible_centers(shape, frame, radius, units, budget)
        self.assertEqual(result.status, "ok", result.diagnostics)
        self.assertFalse(result.budget_certified)
        return result.value

    def test_exact_and_nextafter_rectangle_dimensions(self):
        for width, expected in ((math.nextafter(2, 0), set()), (2, {1}),
                                (math.nextafter(2, math.inf), {2})):
            value = self.centers(Rectangle((0, 0), width, 10))
            self.assertEqual(value.dimensions, expected)
        line = self.centers(Rectangle((0, 0), 2, 10))
        self.assertEqual(line.segments, (((0, -4), (0, 4)),))
        self.assertEqual(self.centers(Rectangle((0, 0), 2, 2)).points, ((0, 0),))

    def test_exact_and_nextafter_circle_dimensions(self):
        for radius, dimensions in ((math.nextafter(1, 0), set()), (1, {0}),
                                   (math.nextafter(1, math.inf), {2})):
            self.assertEqual(self.centers(Circle((3, 4), radius)).dimensions, dimensions)

    def test_exact_mixed_units_and_rigid_placement(self):
        frame = PlanarFrame("inch", "fixture", (100, -200), 0)
        value = self.centers(Rectangle((101, -199), 10, 20, math.pi/3),
                             radius=127, frame=frame)
        self.assertEqual(value.dimensions, {1})
        self.assertEqual(value.center_mm, (Fraction(127, 5), Fraction(127, 5)))
        self.assertEqual(value.segments, (((0, -127), (0, 127)),))
        self.assertEqual(value.angle_radians, math.pi/3)

    def test_zero_tool_and_unsupported_general_centers(self):
        self.assertEqual(self.centers(Circle((0, 0), 2), 0).areas, (("disk", (2,)),))
        result = feasible_centers(Polygon(square(0, 0, 10)), FRAME, 1, "mm", BUDGET)
        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.value)

    def test_invalid_and_certified_requests(self):
        for shape in (Circle((0, 0), -1), Rectangle((0, 0), 0, 2), Circle((math.nan, 0), 1)):
            result = feasible_centers(shape, FRAME, 1, "mm", BUDGET)
            self.assertEqual(result.status, "invalid_input")
        result = feasible_centers(Circle((0, 0), 1), FRAME, 1, "mm", replace(BUDGET, require_certified=True))
        self.assertEqual(result.status, "unresolved")

    def test_optional_import_and_analytic_without_shapely(self):
        script = '''
import sys
from cambam_builder.planar import *
assert 'shapely' not in sys.modules
sys.modules['shapely'] = None
f = PlanarFrame('mm', 'test', (0, 0))
b = ErrorBudget(.001, .01)
assert feasible_centers(Circle((0, 0), 1), f, 1, 'mm', b).status == 'ok'
assert normalize(RegionSet(f, (Circle((0, 0), 1),)), b).status == 'unsupported'
'''
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_input_ownership(self):
        points = [list(p) for p in square(0, 0, 10)]
        components = [Polygon(points)]
        region = RegionSet(FRAME, components)
        points[0][0] = 900
        components.clear()
        self.assertEqual(region.components[0].shell[0], (0, 0))
        with self.assertRaises(FrozenInstanceError):
            region.frame = FRAME


@unittest.skipUnless(HAS_BACKEND, "optional planar extra unavailable")
class NominalTests(unittest.TestCase):
    def normalized(self, *components, frame=FRAME, budget=BUDGET):
        result = normalize(RegionSet(frame, components), budget)
        self.assertEqual(result.status, "ok", result.diagnostics)
        return result

    def test_end_to_end_overlay_erosion_and_provenance(self):
        a = self.normalized(Polygon(square(0, 0, 10), source="opening"))
        b = self.normalized(Polygon(square(5, 0, 10), source="extension"))
        joined = union(a.value, b.value, BUDGET)
        self.assertEqual(joined.status, "ok", joined.diagnostics)
        self.assertEqual(area(joined.value), 150)
        cut = difference(joined.value, b.value, BUDGET)
        self.assertEqual(area(cut.value), 50)
        eroded = nominal_area_erosion(cut.value, 1, BUDGET)
        self.assertEqual(eroded.status, "ok", eroded.diagnostics)
        self.assertEqual(area(eroded.value), 24)
        self.assertEqual({s.source for s in eroded.value.source_spans}, {"opening", "extension"})
        self.assertEqual(eroded.provenance.input_fingerprints, (cut.value.fingerprint,))
        self.assertEqual(eroded.provenance.backend_versions, adapter.versions())
        self.assertTrue(eroded.topology_events)
        self.assertTrue(any(t.kind == "unknown" for t in eroded.error_ledger))
        self.assertFalse(eroded.budget_certified)
        self.assertEqual(area(a.value), 100)

    def test_hole_and_split_topology(self):
        outer = self.normalized(Polygon(square(0, 0, 10)))
        inner = self.normalized(Polygon(square(2, 2, 6)))
        cut = difference(outer.value, inner.value, BUDGET)
        self.assertEqual(area(cut.value), 64)
        self.assertEqual(len(cut.value.components[0].holes), 1)
        slot = self.normalized(Rectangle((5, 5), 2, 20))
        split = difference(outer.value, slot.value, BUDGET)
        self.assertEqual(len(split.value.components), 2)
        self.assertEqual(area(split.value), 80)

    def test_strict_input_admission(self):
        invalid = [
            Polygon(((0, 0), (1, 0), (1, 1))),
            Polygon(((0, 0), (1, 1), (0, 1), (1, 0), (0, 0))),
            Polygon(((0, 0), (1, 0), (1, 0), (0, 1), (0, 0))),
            Polygon(square(0, 0, 10), (square(10, 0, 1),)),
            Polygon(square(0, 0, 10), (square(2, 2, 6), square(3, 3, 1))),
            Polygon(square(0, 0, 10), (square(2, 2, 2), square(4, 2, 2))),
            Polygon(((0, 0), (math.inf, 0), (1, 1), (0, 0))),
        ]
        for shape in invalid:
            with self.subTest(shape=shape):
                result = normalize(RegionSet(FRAME, (shape,)), BUDGET)
                self.assertEqual(result.status, "invalid_input", result)
                self.assertIsNone(result.value)
        result = normalize(RegionSet(FRAME, (Polygon(square(0, 0, 10)), Polygon(square(5, 5, 10)))), BUDGET)
        self.assertEqual(result.status, "invalid_input")

    def test_contact_and_unrepresentable_output(self):
        contact = self.normalized(Polygon(square(0, 0, 1)), Polygon(square(1, 1, 1)))
        self.assertTrue(any("point" in event for event in contact.topology_events))
        outer = self.normalized(Polygon(square(0, 0, 10)))
        diamond = self.normalized(Polygon(((0, 5), (2, 3), (4, 5), (2, 7), (0, 5))))
        result = difference(outer.value, diamond.value, BUDGET)
        self.assertEqual(result.status, "unsupported", result)
        self.assertIsNone(result.value)
        self.assertIn("contact", result.diagnostics[0])

    def test_canonical_key_and_source_mapping(self):
        ring = square(0, 0, 10)
        reordered = ring[2:-1] + ring[:3]
        a = self.normalized(Polygon(ring, source="a"))
        b = self.normalized(Polygon(reordered[::-1], source="b"))
        self.assertEqual(a.value.fingerprint, b.value.fingerprint)
        self.assertEqual(a.value.components, b.value.components)
        self.assertNotEqual(a.value.source_spans, b.value.source_spans)
        numeric_variant = self.normalized(Polygon(tuple(tuple(float(v) for v in p) for p in ring)))
        self.assertEqual(a.value.fingerprint, numeric_variant.value.fingerprint)
        other_policy = self.normalized(Polygon(ring), budget=ErrorBudget(.002, .02))
        self.assertNotEqual(a.value.fingerprint, other_policy.value.fingerprint)
        for span in b.value.source_spans:
            self.assertIn(span.start_mm, b.value.components[0].shell)
            self.assertIn(span.end_mm, b.value.components[0].shell)

    def test_conversion_loss_and_owned_value_admission(self):
        x1, x2 = 1.5000000000000002, 1.5000000000000004
        result = normalize(RegionSet(PlanarFrame("inch", "fixture", (0, 0)),
                                    (Polygon(((x1, 0), (x2, 0), (x2, 1), (x1, 1), (x1, 0))),)),
                           ErrorBudget(1, 1))
        self.assertEqual(result.status, "unresolved", result)
        a = self.normalized(Rectangle((0, 0), 10, 10)).value
        components = list(a.components)
        copied = replace(a, components=components)
        components.clear()
        self.assertEqual(copied.components, a.components)
        malformed = replace(a, components=(Circle((0, 0), 1),))
        self.assertEqual(union(a, malformed, BUDGET).status, "invalid_input")

    def test_erosion_around_hole_has_nominal_round_corners(self):
        value = self.normalized(Polygon(square(0, 0, 10), (square(4, 4, 2),))).value
        result = nominal_area_erosion(value, 1, BUDGET)
        self.assertEqual(result.status, "ok", result)
        # Outer 8x8 less expanded 2x2 hole: 4 + perimeter*1 + pi*1^2.
        self.assertAlmostEqual(area(result.value), 64 - (12 + math.pi), delta=.01)
        self.assertTrue(any(t.stage == "operation_approximation" and t.area_mm2 is None
                            for t in result.error_ledger))

    def test_frames_units_and_large_origin(self):
        a = self.normalized(Rectangle((0, 0), 254, 254))
        b = self.normalized(Rectangle((0, 0), 10, 10), frame=PlanarFrame("inch", "fixture", (0, 0), 0))
        self.assertEqual(a.value.components, b.value.components)
        self.assertEqual(union(a.value, b.value, BUDGET).status, "ok")
        for frame in (replace(FRAME, frame_id="other"), replace(FRAME, section_z=1)):
            c = self.normalized(Rectangle((0, 0), 10, 10), frame=frame)
            self.assertEqual(union(a.value, c.value, BUDGET).status, "invalid_input")
        shifted = self.normalized(Rectangle((1e9, -1e9), 254, 254),
                                  frame=replace(FRAME, origin=(1e9, -1e9)))
        self.assertEqual(a.value.components, shifted.value.components)
        result = normalize(RegionSet(replace(FRAME, origin=(1e15, 0)), (Rectangle((1e15, 0), 254, 254),)), BUDGET)
        self.assertEqual(result.status, "unresolved")

    def test_circle_budget_and_area_reference(self):
        result = self.normalized(Circle((0, 0), 10, "circle"))
        term = result.error_ledger[0]
        self.assertLessEqual(term.boundary_mm, BUDGET.boundary_mm)
        self.assertLessEqual(term.area_mm2, BUDGET.area_mm2)
        self.assertAlmostEqual(math.pi*100 - area(result.value), term.area_mm2, delta=1e-10)
        self.assertEqual(result.value.source_spans[0].parameter_start, 0)
        self.assertEqual(result.value.source_spans[-1].parameter_end, 1)
        limited = normalize(RegionSet(FRAME, (Circle((0, 0), 10),)), replace(BUDGET, max_segments=4))
        self.assertEqual(limited.status, "unresolved")
        composite = normalize(RegionSet(FRAME, (Circle((0, 0), 10), Circle((30, 0), 10))), BUDGET)
        self.assertEqual(composite.status, "unsupported")

    def test_empty_area_is_not_general_feasibility(self):
        rectangle = Rectangle((0, 0), 2, 10)
        value = self.normalized(rectangle).value
        eroded = nominal_area_erosion(value, 1, BUDGET)
        self.assertEqual(eroded.status, "ok")
        self.assertEqual(eroded.value.components, ())
        self.assertEqual(eroded.operation, "nominal_area_erosion")
        centers = feasible_centers(rectangle, FRAME, 1, "mm", BUDGET)
        self.assertEqual(centers.value.dimensions, {1})
        empty = self.normalized().value
        self.assertEqual(union(value, empty, BUDGET).value.components, value.components)
        self.assertEqual(difference(value, value, BUDGET).value.components, ())

    def test_budget_certification_and_backend_failure_atomicity(self):
        a = self.normalized(Rectangle((0, 0), 10, 10)).value
        result = nominal_area_erosion(a, 1, replace(BUDGET, require_certified=True))
        self.assertEqual(result.status, "unresolved")
        with patch.object(adapter, "operate", side_effect=adapter.AdapterError("backend_failure", "injected")):
            result = difference(a, a, BUDGET)
        self.assertEqual(result.status, "backend_failure")
        self.assertIsNone(result.value)
        self.assertEqual(area(a), 100)
        with patch.object(adapter, "_backend", side_effect=RuntimeError("injected native failure")):
            result = normalize(RegionSet(FRAME, (Rectangle((0, 0), 10, 10),)), BUDGET)
        self.assertEqual(result.status, "backend_failure")


if __name__ == "__main__":
    unittest.main()
