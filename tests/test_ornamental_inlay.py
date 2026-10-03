"""Independent prismatic assembly witnesses and replayed ornamental stocks."""
from dataclasses import replace
import math
import hashlib
import unittest

try:
    from shapely.geometry import LineString, Point, Polygon, box
    from cambam_builder.cam_core import ornamental_inlay as oi, replay
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def design(**changes):
    # Asymmetric ring and a smaller lobe joined by a 1 mm bridge.
    motif = box(0, 0, 6, 6).union(box(6, 2, 9, 3)).union(box(9, 1, 12, 4))
    motif = motif.difference(box(1.5, 1.5, 3.5, 4.5))
    values = dict(revision="ornament-1", motif=motif, stock_xy=box(-1, -1, 13, 7),
        seating_mm=1.2, bottom_gap_mm=.3, surface_gap_mm=.4, backing_mm=1,
        receiver_thickness_mm=3, side_fit_mm=.15, fit_limit_mm=.3,
        minimum_web_mm=.4, edge_access_mm=.5)
    values.update(changes)
    return oi.Design(**values)


def clearing_trace(q, side, radius=.12):
    """Synthetic independent contour + raster consumer of public target Regions."""
    tool = replay.ToolProfile(side + "-endmill", "cylinder", radius, 4)
    high = (0., 0., 3.)
    at = high
    items = [replay.Event("tool_change", tool.name, at),
             replay.Event("spindle_start", tool.name, at)]
    operations = []
    for i, target in enumerate(q.targets(side)):
        op = replay.Operation(f"{side}-{i}", tool, target)
        operations.append(op)
        shape = Polygon(target.region_shell, target.region_holes)
        feasible = shape.buffer(-radius-.0005, quad_segs=128)
        paths = []
        if not feasible.is_empty:
            for p in oi._parts(feasible):
                paths.extend(tuple(r.coords) for r in (p.exterior, *p.interiors))
            x0, y0, x1, y1 = feasible.bounds
            count = math.ceil((y1-y0) / radius)
            for j in range(count + 1):
                y = y0 + (y1-y0) * j / count
                line = feasible.intersection(LineString(((x0-1, y), (x1+1, y))))
                segments = (line,) if line.geom_type == "LineString" else (
                    tuple(line.geoms) if hasattr(line, "geoms") else ())
                paths.extend(tuple(s.coords) for s in segments
                             if s.geom_type == "LineString" and s.length > 1e-10)
        for path in paths:
            top = (*path[0], 3.)
            if at != top:
                items.append(replay.Motion("rapid", tool.name, op.name, at, top))
            at = (*path[0], -target.depth)
            items.append(replay.Motion("entry", tool.name, op.name, top, at))
            for xy in path[1:]:
                end = (*xy, -target.depth)
                if at != end:
                    items.append(replay.Motion("cut", tool.name, op.name, at, end))
                at = end
            top = (*at[:2], 3.)
            items.append(replay.Motion("retract", tool.name, op.name, at, top))
            at = top
    items.append(replay.Event("spindle_stop", tool.name, at))
    return replay.Trace(q.source(side), q.frame(side), high, tuple(operations), tuple(items))


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class OrnamentalInlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = design()
        cls.receiver = clearing_trace(cls.q, "receiver")
        cls.plug = clearing_trace(cls.q, "plug", .1)

    def verify(self, q=None, receiver=None, plug=None, **kwargs):
        r, p = receiver or self.receiver, plug or self.plug
        return oi.verify_pair(q or self.q, r, p,
            expected_receiver_motion=kwargs.pop("expected_receiver_motion", r.motion_fingerprint),
            expected_plug_motion=kwargs.pop("expected_plug_motion", p.motion_fingerprint),
            **kwargs)

    def test_asymmetric_hole_bridge_actual_stock_continuous_insertion(self):
        result = self.verify()
        self.assertEqual(result.status, "pass", result)
        self.assertEqual(result.collision_volume_upper_mm3, 0)
        self.assertEqual(result.minimum_bottom_gap_mm, .3)
        self.assertEqual(result.minimum_surface_gap_mm, .4)
        self.assertEqual(len(self.q.targets("plug")), 2)
        self.assertAlmostEqual(self.q.motif.area, 42)
        self.assertEqual(len(self.q.plug_xy.interiors), 1)
        self.assertNotEqual(oi._flip(self.q.motif).wkb, self.q.motif.wkb)
        # Independent frame landmark: the receiver island at (2,3) becomes
        # a plug machining hole pocket at (-2,3), not (+2,3).
        holes = [Polygon(t.region_shell, t.region_holes) for t in self.q.targets("plug")]
        self.assertTrue(any(p.covers(Point(-2, 3)) for p in holes))
        self.assertFalse(any(p.covers(Point(2, 3)) for p in holes))

    def test_independent_allowances_and_tool_independent_targets(self):
        q = self.q
        self.assertAlmostEqual(q.receiver_depth, 1.5)
        self.assertAlmostEqual(q.plug_depth, 1.6)
        self.assertEqual(replace(q, backing_mm=2).targets("plug"), q.targets("plug"))
        self.assertEqual(replace(q, bottom_gap_mm=.5).targets("plug"), q.targets("plug"))
        self.assertEqual(replace(q, surface_gap_mm=.7).targets("receiver"), q.targets("receiver"))
        self.assertNotEqual(replace(q, backing_mm=2).fingerprint, q.fingerprint)
        other = clearing_trace(q, "plug", .08)
        self.assertEqual(self.verify(plug=other).status, "pass")

    def test_missing_hole_pocket_collides_with_receiver_island(self):
        p = self.plug
        # Keep one complete exterior-clearing operation, leaving the hole solid.
        name = p.operations[0].name
        moves = tuple(m for m in p.items if type(m) is replay.Motion and m.operation == name)
        items = p.items[:2] + moves + (replay.Event("spindle_stop", p.operations[0].tool.name,
                                                  moves[-1].end),)
        p = replace(p, operations=p.operations[:1], items=items)
        result = self.verify(plug=p)
        self.assertEqual(result.status, "collision")
        self.assertGreater(result.collision_volume_lower_mm3, 5)

    def test_registration_collision_and_outside_blank_are_not_passes(self):
        self.assertEqual(self.verify(registration_xy=(.4, 0)).status, "collision")
        self.assertNotEqual(self.verify(registration_xy=(100, 0)).status, "pass")
        self.assertEqual(self.verify(registration_xy=(.01, 0)).registration_xy_mm, (.01, 0))

    def test_incomplete_glue_and_surface_clearances_do_not_pass(self):
        def shallower(trace, depth):
            def point(xyz):
                return (*xyz[:2], -depth if xyz[2] < 0 else xyz[2])
            return replace(trace, items=tuple(
                replace(m, start=point(m.start), end=point(m.end))
                if type(m) is replay.Motion else m for m in trace.items))
        shoulder = shallower(self.plug, self.q.seating_mm + .1)
        result = self.verify(plug=shoulder)
        self.assertEqual(result.collision_volume_upper_mm3, 0)
        self.assertNotEqual(result.status, "pass")
        self.assertEqual(result.minimum_surface_gap_mm, 0)
        floor = shallower(self.receiver, self.q.seating_mm + .1)
        result = self.verify(receiver=floor)
        self.assertEqual(result.collision_volume_upper_mm3, 0)
        self.assertNotEqual(result.status, "pass")
        self.assertEqual(result.minimum_bottom_gap_mm, 0)
        # Only the top 0.05 mm collides: a single mid-depth section misses it.
        shoulder = shallower(self.plug, self.q.seating_mm - .05)
        result = self.verify(plug=shoulder)
        self.assertEqual(result.status, "collision")
        self.assertAlmostEqual(result.sections[0][1], .05)
        self.assertGreater(result.sections[0][2], 0)
        self.assertEqual(result.sections[-1][3], 0)

    def test_rectangular_side_fit_has_independent_area_oracle(self):
        for fit in (-.1, 0, .1):
            q = design(motif=box(0, 0, 4, 3), side_fit_mm=fit)
            expected = ((4-2*fit)*(3-2*fit) if fit >= 0 else
                        12 + 14*(-fit) + math.pi*fit**2)
            self.assertAlmostEqual(q.plug_xy.area, expected, delta=1e-6)
            self.assertAlmostEqual(q.motif.area * q.receiver_depth, 18)

    def test_revision_frame_tool_and_motion_binding(self):
        with self.assertRaisesRegex(ValueError, "source"):
            self.verify(q=replace(self.q, revision="new"))
        with self.assertRaisesRegex(ValueError, "frame"):
            self.verify(plug=replace(self.plug, frame=self.receiver.frame))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.verify(expected_plug_motion="old-motion")
        changed = replace(self.plug, operations=tuple(replace(o,
            tool=replace(o.tool, radius=.2)) for o in self.plug.operations))
        self.assertNotEqual(changed.motion_fingerprint, self.plug.motion_fingerprint)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.verify(plug=changed, expected_plug_motion=self.plug.motion_fingerprint)

    def test_signed_interference_and_fragile_or_collapsed_features(self):
        q = design(side_fit_mm=-.1)
        result = self.verify(q, clearing_trace(q, "receiver"), clearing_trace(q, "plug", .1))
        self.assertEqual(result.status, "collision")
        self.assertGreater(result.collision_volume_lower_mm3, .5)
        for changes in ({"side_fit_mm": .31}, {"minimum_web_mm": .8},
                        {"bottom_gap_mm": -1}, {"backing_mm": 0},
                        {"surface_gap_mm": float("nan")}, {"seating_mm": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                design(**changes)

    def test_unreachable_large_tool_cannot_certify_nominal_target(self):
        p = clearing_trace(self.q, "plug", 1.1)
        self.assertNotEqual(self.verify(plug=p).status, "pass")

    def test_finish_envelope_requires_process_and_retained_thickness(self):
        kwargs = dict(expected_receiver_motion=self.receiver.motion_fingerprint,
            expected_plug_motion=self.plug.motion_fingerprint, removal_mm=.2,
            minimum_retained_mm=.8, assembly_token="assembled", cure_token="cured",
            renewed_setup_token="reset")
        report = oi.finish_envelope(self.q, self.receiver, self.plug, **kwargs)
        self.assertEqual(report["nominal_core_minimum_retained_mm"], 1)
        self.assertTrue(report["section_enclosure_topology_matches"], report)
        self.assertLess(report["motif_missing_upper_mm2"], 1e-6)
        self.assertLess(report["motif_excess_upper_mm2"], .1)
        for changes in ({"cure_token": ""}, {"removal_mm": .5}, {"removal_mm": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                oi.finish_envelope(self.q, self.receiver, self.plug, **(kwargs | changes))

    def test_complete_decoded_outputs_bind_each_independent_part(self):
        from cambam_builder.cam_core.ordered_job import Job, JobMove, Stage, Transition
        from cambam_builder.integrations import inlay_output, ordered_output
        q = design(motif=box(0, 0, 4, 3).difference(box(1, 1, 2, 2)),
                   stock_xy=box(-1, -1, 5, 4))
        base_jobs = []
        for side in ("receiver", "plug"):
            trace = clearing_trace(q, side)
            stages = []
            for i, op in enumerate(trace.operations):
                op = replace(op, tool=replace(op.tool, name="T1"))
                moves = tuple(JobMove(m.role, m.start, m.end,
                              0 if m.role == "rapid" else 100)
                              for m in trace.items if type(m) is replay.Motion
                              and m.operation == op.name)
                stages.append(Stage(op.name, "T1", moves, 10000, operation=op,
                              source_revision=q.source(side), transition=None if i == 0
                              else Transition("operator", "split", "T1", moves[0].start,
                                              "same-part-setup")))
            base_jobs.append(Job(q.source(side), tuple(stages), trace.initial_position,
                                 program_frame=q.frame(side)))
        self.assertEqual(len(base_jobs[1].stages), 2)
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                jobs = tuple(replace(j, stages=tuple(replace(s, transition=(
                    replace(s.transition, boundary="pause") if dialect == "grbl"
                    and s.transition else s.transition)) for s in j.stages)) for j in base_jobs)
                files = tuple(ordered_output.emit(j, dialect)[0] for j in jobs)
                hashes = tuple(tuple(hashlib.sha256(f).hexdigest() for f in fs) for fs in files)
                kwargs = dict(expected_receiver_job=jobs[0].fingerprint,
                    expected_plug_job=jobs[1].fingerprint,
                    expected_receiver_hashes=hashes[0], expected_plug_hashes=hashes[1])
                report = inlay_output.audit_pair(q, *jobs, dialect, *files, **kwargs)
                self.assertEqual(report["assembly"]["status"], "pass")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    inlay_output.audit_pair(q, *jobs, dialect,
                        (files[0][0] + b"\n",), files[1], **kwargs)
                with self.assertRaisesRegex(ValueError, "binding"):
                    inlay_output.audit_pair(q, *jobs, dialect, *files,
                        **(kwargs | {"expected_plug_job": "stale"}))


if __name__ == "__main__":
    unittest.main()
