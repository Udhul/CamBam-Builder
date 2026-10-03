"""Generated BO01 cylinder paths retain design, axial and output contracts."""
from dataclasses import replace
import math
import unittest

try:
    from shapely.geometry import LineString
    from cambam_builder.cam_core import ordered_job, planar_rest, replay, v_region as v
    from cambam_builder.cam_core.occupancy import (
        Box, OccupancySetup, ToolBand, ToolBody,
    )
    from cambam_builder.integrations.ordered_output import emit
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _target():
    return v.VTarget.polygon("generated-cylinder-island",
        ((0, 0), (12, 0), (12, 10), (0, 10)),
        (((5, 4), (7, 4), (7, 6), (5, 6)),), 1,
        design_angle_degrees=90)


def _generate(target=None, tool=None, **kwargs):
    return planar_rest.generate_cylindrical(target or _target(),
        tool or replay.ToolProfile("T1", "cylinder", .5, 3),
        **dict(dict(stepover_mm=2, xy_step_mm=3, safe_z=3,
                    max_stepdown_mm=.5, initial_tip=(-2, -2, 3)), **kwargs))


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class PlanarCylinderTests(unittest.TestCase):
    def test_protected_island_full_height_and_exact_axial_retraces(self):
        target = _target()
        fingerprint = target.fingerprint
        trace = _generate(target)
        stock = replay.replay(trace, expected_source=target.source_id)
        self.assertEqual(target.fingerprint, fingerprint)
        self.assertEqual(trace.frame, target.frame)
        self.assertEqual(trace.operations[0].target.depth, target.cap_depth)
        self.assertEqual(trace.operations[0].target.region_holes,
            tuple(tuple(ring.coords)[:-1] for ring in target.safe.interiors))
        self.assertEqual(v.VComposition(target, (trace,)).target, target)
        self.assertFalse(stock.removed_contains(6, 5, .5))
        grouped = {}
        for item in trace.items:
            if type(item) is not replay.Motion:
                continue
            self.assertNotEqual(item.start, item.end)
            if item.role == "rapid":
                self.assertGreaterEqual(min(item.start[2], item.end[2]), 3)
            elif item.role == "entry":
                self.assertEqual(item.start[:2], item.end[:2])
                self.assertEqual(item.start[2], 3)
            elif item.role == "retract":
                self.assertEqual(item.start[:2], item.end[:2])
                self.assertEqual(item.end[2], 3)
            elif item.role == "cut":
                self.assertEqual(item.start[2], item.end[2])
                depth = -item.end[2]
                grouped.setdefault(depth, []).append((item.start[:2], item.end[:2]))
                line = LineString((item.start[:2], item.end[:2]))
                # Independent necessary full-height clearance against source:
                # a cylinder at depth d needs radius + d*tan(design/2).
                self.assertTrue(target.safe.covers(line))
                self.assertGreaterEqual(line.distance(target.safe.boundary)+1e-8,
                    trace.operations[0].tool.radius+depth*target.tangent)
        self.assertEqual(set(grouped), {.5, 1})
        self.assertEqual(grouped[.5], grouped[1])
        self.assertGreater(len(grouped[1]), 0)
        self.assertGreater(sum(math.dist(a, b) for a, b in grouped[1]), 10)
        # Source-region replay alone is weaker than the original V envelope.
        with self.assertRaisesRegex(ValueError, "finish target"):
            v.VComposition(replace(target, design_angle_degrees=120), (trace,))

    def test_offset_second_motif_and_short_flute_preserve_original_cap(self):
        target = v.VTarget.polygon("second-cylinder-design",
            ((0, 0), (10, 0), (10, 4), (6, 4), (6, 8), (0, 8)), (), 1.2,
            design_angle_degrees=120, frame="fixture-local")
        tool = replay.ToolProfile("T7", "cylinder", .3, .8)
        trace = _generate(target, tool, fill_pattern="offset", stepover_mm=1)
        self.assertEqual(trace.frame, "fixture-local")
        self.assertEqual(trace.operations[0].target.depth, 1.2)
        self.assertEqual(trace.operations[0].tool, tool)
        depths = {-m.end[2] for m in trace.items
                  if type(m) is replay.Motion and m.role == "cut"}
        self.assertEqual(depths, {.4, .8})
        stock = replay.replay(trace, expected_source=target.source_id)
        self.assertTrue(stock.cuts)
        self.assertFalse(stock.removed_contains(3, 3, 1))
        self.assertEqual(v.VComposition(target, (trace,)).target.cap_depth, 1.2)

    def test_empty_centers_and_tiny_invalid_or_exhausted_controls(self):
        self.assertIsNone(_generate(tool=replay.ToolProfile("T2", "cylinder", 10, 3)))
        for controls in ({"stepover_mm": 0}, {"stepover_mm": 1e-300},
                {"xy_step_mm": 1e-300}, {"xy_step_mm": float("nan")},
                {"margin_mm": 1e-6}, {"safe_z": float("inf")},
                {"max_stepdown_mm": 1e-300}, {"max_stepdown_mm": True},
                {"max_paths": True}, {"max_paths": 0}, {"max_paths": 10001},
                {"max_paths": 1}, {"max_paths": 2},
                {"xy_step_mm": 1e-6}, {"stepover_mm": 1e-6},
                {"fill_pattern": "spiral"}, {"initial_tip": (0, 0, 0)},
                {"initial_tip": (0, float("inf"), 3)}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                _generate(**controls)
        legacy = replace(_target(), design_angle_degrees=None)
        with self.assertRaises(ValueError):
            _generate(legacy)
        with self.assertRaises(ValueError):
            _generate(tool=replay.ToolProfile("T3", "pointed_cone", 3, 3))
        with self.assertRaises(ValueError):
            _generate(tool=replay.ToolProfile("rough", "cylinder", .5, 3))

    def test_generated_trace_passes_decoded_axial_and_occupancy_gates(self):
        target = _target()
        trace = _generate(target)
        operation = trace.operations[0]
        moves = tuple(ordered_job.JobMove(item.role, item.start, item.end,
            0 if item.role == "rapid" else 120)
            for item in trace.items if type(item) is replay.Motion)
        stage = ordered_job.Stage(operation.name, operation.tool.name, moves, 12000,
            operation=operation, source_revision=trace.motion_fingerprint,
            axial_limits=ordered_job.AxialLimits(.5, 1))
        setup = OccupancySetup("program", Box("virgin-stock", (0, 0, -1, 12, 10, 0)),
            (), (ToolBody("T1", (ToolBand("cutter", 0, 3, .5),
                ToolBand("shank", 3, 4, .5), ToolBand("holder", 4, 6, 2))),))
        job = ordered_job.Job(target.source_id, (stage,), trace.initial_position,
                             occupancy_setup=setup)
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                files, report = emit(job, dialect,
                    coordinate_decimals=6 if dialect == "uccnc" else 4)
                self.assertTrue(files)
                self.assertEqual(report["motion_equivalence"]["status"], "pass")
                self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
                self.assertEqual(report["axial_process_limits"]["status"], "pass")
                self.assertAlmostEqual(report["axial_process_limits"]["stages"][0]
                                       ["maximum_observed_advance_mm"], .5)


if __name__ == "__main__":
    unittest.main()
