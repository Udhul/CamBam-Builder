"""Composite facing with independent stock, fault and decoded-byte witnesses."""
from dataclasses import replace
import hashlib
import unittest

try:
    from shapely.geometry import Point, box
    from cambam_builder.cam_core import composite_inlay as ci, replay
    from cambam_builder.integrations import inlay_output as io, ordered_output
    from tests.test_ornamental_inlay import design, clearing_trace
    from tests.test_tapered_inlay import tapered, stock, outputs
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def assembly(q, r=None, p=None):
    r = r or clearing_trace(q, "receiver")
    p = p or clearing_trace(q, "plug", .1)
    return ci.Assembly(q, r, p, r.motion_fingerprint, p.motion_fingerprint)


def finish(a, **changes):
    values = dict(assembly=a, removal_mm=.2, plane_tolerance_mm=.001,
        motif_tolerance_mm=.02, minimum_plug_core_mm=.8,
        minimum_receiver_floor_mm=1, edge_access_mm=1,
        assembly_token="assembled-1", cure_token="cured-1", renewed_setup_token="setup-2")
    values.update(changes)
    return ci.Finish(**values)


def facing(f):
    return ci.generate(f, replay.ToolProfile("T1", "cylinder", .6, 4),
                       stepover_mm=.7, stepdown_mm=.5, clearance_mm=2)


def output(job, dialect):
    files = ordered_output.emit(job, dialect)[0]
    return io.ComponentOutput(job, files, job.fingerprint,
        tuple(hashlib.sha256(f).hexdigest() for f in files))


def straight_output(q, side, dialect):
    from cambam_builder.cam_core import ordered_job as oj
    trace = clearing_trace(q, side)
    stages = []
    for i, operation in enumerate(trace.operations):
        op = replace(operation, tool=replace(operation.tool, name="T1"))
        moves = tuple(oj.JobMove(m.role, m.start, m.end, 0 if m.role == "rapid" else 100)
                      for m in trace.items if type(m) is replay.Motion and m.operation == op.name)
        stages.append(oj.Stage(op.name, "T1", moves, 10000, operation=op,
            source_revision=q.source(side), transition=None if i == 0 else
            oj.Transition("operator", "pause" if dialect == "grbl" else "split",
                          "T1", moves[0].start, "same-part-setup")))
    return (output(oj.Job(q.source(side), tuple(stages), trace.initial_position,
                         program_frame=q.frame(side)), dialect),)


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class CompositeInlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = assembly(design())
        cls.f = finish(cls.a)
        cls.trace = facing(cls.f)

    def verify(self, f=None, trace=None, **kwargs):
        t = trace or self.trace
        return ci.verify(f or self.f, t,
            expected_motion=kwargs.pop("expected_motion", t.motion_fingerprint), **kwargs)

    def test_asymmetric_holed_bridge_facing_plane_and_thickness(self):
        result = self.verify()
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["unfaced_upper_mm2"], 0)
        self.assertAlmostEqual(result["plane_depth_interval_mm"][1], .2)
        self.assertAlmostEqual(result["nominal_core_minimum_retained_mm"], 1)
        self.assertAlmostEqual(result["minimum_receiver_floor_mm"], 1.5)
        self.assertTrue(result["backing_removed"])
        self.assertTrue(result["section_enclosure_topology_matches"])
        self.assertLess(result["motif_excess_upper_mm2"], .1)
        # Independent physical landmarks: solid backing above the surface,
        # receiver island in the hole, and plug in the asymmetric right lobe.
        receiver, plug = self.a.section(-.8)
        self.assertTrue(receiver[1].is_empty)
        self.assertAlmostEqual(plug[0].area, 112)
        receiver, plug = self.a.section(.2)
        self.assertTrue(receiver[0].covers(Point(2, 3)))
        self.assertFalse(plug[1].covers(Point(2, 3)))
        self.assertTrue(plug[0].covers(Point(10, 2)))
        self.assertTrue(all(s.is_empty for pair in self.a.section(-2) for s in pair))

    def test_rectangular_consumer_has_independent_finish_outline_oracle(self):
        q = design(motif=box(0, 0, 4, 3), stock_xy=box(-1, -1, 5, 4))
        f = finish(assembly(q))
        self.assertAlmostEqual(f.nominal_motif.area, 3.7*2.7)
        result = self.verify(f, facing(f))
        self.assertEqual(result["status"], "pass", result)
        self.assertAlmostEqual(result["nominal_core_minimum_retained_mm"], 1)

    def test_small_registration_keeps_own_core_but_checks_fixed_motif(self):
        f = finish(replace(self.a, registration_xy=(.01, 0)), motif_tolerance_mm=.03)
        result = self.verify(f, facing(f))
        self.assertEqual(result["status"], "pass", result)
        self.assertAlmostEqual(result["nominal_core_minimum_retained_mm"], 1)

    def test_omitted_final_pass_and_row_remain_partial(self):
        t = self.trace
        # A complete shallower facing program is safe, but misses the final plane.
        def raise_depth(p):
            return (*p[:2], max(p[2], -1.3))
        shallow = replace(t, items=tuple(replace(m, start=raise_depth(m.start),
            end=raise_depth(m.end)) if type(m) is replay.Motion else m for m in t.items))
        result = self.verify(trace=shallow)
        self.assertEqual(result["status"], "unresolved")
        self.assertFalse(result["plane_ok"])
        self.assertGreater(result["unfaced_upper_mm2"], 100)
        self.assertIsNone(result["plane_depth_interval_mm"])
        self.assertIsNone(result["motif_excess_upper_mm2"])
        self.assertFalse(result["motif_ok"])
        # Drop a complete row at every depth and reconnect only above stock.
        items = list(t.items[:2])
        at = t.initial_position
        for m in t.items[2:-1]:
            if type(m) is replay.Motion and m.start[1] == 3 and m.end[1] == 3:
                continue
            if m.start != at:
                if m.start[2] != 2:
                    continue
                items.append(replay.Motion("rapid", "T1", "face", at, m.start))
            items.append(m)
            at = m.end
        items.append(replay.Event("spindle_stop", "T1", at))
        missing = replace(t, items=tuple(items))
        # Rows are spaced by 2/3 mm; removing the y=3 row opens a stripe.
        result = self.verify(trace=missing)
        self.assertFalse(result["plane_ok"], result)
        self.assertGreater(result["unfaced_upper_mm2"], 0)

    def test_independent_motif_core_and_receiver_floor_gates(self):
        for changes, gate in (({"motif_tolerance_mm": 0}, "motif_ok"),
                              ({"minimum_plug_core_mm": 1.01}, "thickness_ok"),
                              ({"minimum_receiver_floor_mm": 1.51}, "thickness_ok")):
            f = replace(self.f, **changes)
            result = self.verify(f, facing(f))
            self.assertEqual(result["status"], "unresolved", result)
            self.assertFalse(result[gate])

    def test_overcut_stale_revision_process_tool_and_frame_fail(self):
        for key in ("assembly_token", "cure_token", "renewed_setup_token"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(self.f, **{key: ""})
        for f in (replace(self.f, cure_token="new-cure"),
                  replace(self.f, renewed_setup_token="new-setup")):
            with self.assertRaises(ValueError):
                self.verify(f)
        for t in (replace(self.trace, frame="receiver"),
                  replace(self.trace, operations=tuple(replace(o,
                    tool=replace(o.tool, radius=.61)) for o in self.trace.operations))):
            with self.assertRaises(ValueError):
                self.verify(trace=t, expected_motion=self.trace.motion_fingerprint)
        def deeper(p):
            return (*p[:2], -1.7 if p[2] < -1.5 else p[2])
        deep = replace(self.trace, items=tuple(replace(m, start=deeper(m.start), end=deeper(m.end))
            if type(m) is replay.Motion else m for m in self.trace.items))
        with self.assertRaisesRegex(ValueError, "depth"):
            self.verify(trace=deep)
        with self.assertRaises(ValueError):
            replace(self.a, design=replace(self.a.design, revision="stale"))
        with self.assertRaises(ValueError):
            replace(self.a, registration_xy=(.4, 0))

    def test_generator_controls_and_top_reference(self):
        tool = replay.ToolProfile("T1", "cylinder", .6, 4)
        args = dict(stepover_mm=.7, stepdown_mm=.5, clearance_mm=2)
        for changes in ({"stepover_mm": 1.2}, {"stepdown_mm": 0},
                        {"clearance_mm": 0}, {"stepover_mm": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ci.generate(self.f, tool, **(args | changes))
        for bad in (replace(tool, cutting_length=1), replace(tool, radius=1.1)):
            with self.assertRaises(ValueError):
                ci.generate(self.f, bad, **args)
        depths = sorted({-m.end[2] for m in self.trace.items
                         if type(m) is replay.Motion and m.role == "cut"})
        self.assertAlmostEqual(depths[-1], 1.6)  # .4 gap + 1 backing + .2 finish
        self.assertLessEqual(max(b-a for a, b in zip([0]+depths, depths)), .5)

    def test_complete_straight_decoded_pair_and_face_both_dialects(self):
        q = design(motif=box(0, 0, 4, 3).difference(box(1, 1, 2, 2)),
                   stock_xy=box(-1, -1, 5, 4))
        for dialect in ("uccnc", "grbl"):
            with self.subTest(dialect=dialect):
                a, parts = io.assemble_outputs(q, straight_output(q, "receiver", dialect),
                    straight_output(q, "plug", dialect), dialect)
                f = finish(a)
                o = output(io.facing_job(f, facing(f), feed_mm_min=100, rpm=10000), dialect)
                result = io.audit_facing(f, o, dialect)
                self.assertEqual(parts["assembly"]["status"], "pass")
                self.assertEqual(result["finishing"]["status"], "pass", result["finishing"])
                for bad in (replace(o, expected_job="stale"),
                            replace(o, expected_hashes=()),
                            replace(o, files=(o.files[0]+b"\n",))):
                    with self.assertRaises(ValueError):
                        io.audit_facing(f, bad, dialect)

    def test_tapered_flat_rounded_parts_and_decoded_composite_finish(self):
        q = tapered()
        r, p = stock(q, "receiver", "flat"), stock(q, "plug", "rounded")
        a = ci.Assembly(q, r, p, r.fingerprint, p.fingerprint, slabs=8)
        f = finish(a, removal_mm=.05, minimum_plug_core_mm=.25,
                   motif_tolerance_mm=.2)
        result = self.verify(f, facing(f))
        self.assertEqual(result["status"], "pass", result)
        self.assertAlmostEqual(result["nominal_core_minimum_retained_mm"], .3)
        self.assertAlmostEqual(result["minimum_receiver_floor_mm"], 2.25)
        for dialect in ("uccnc", "grbl"):
            a, _ = io.assemble_outputs(q, outputs(q, "receiver", r, dialect),
                outputs(q, "plug", p, dialect), dialect, slabs=8)
            f = replace(f, assembly=a)
            o = output(io.facing_job(f, facing(f), feed_mm_min=100, rpm=10000), dialect)
            result = io.audit_facing(f, o, dialect)["finishing"]
            self.assertEqual(result["status"], "pass", result)


if __name__ == "__main__":
    unittest.main()
