"""Synthetic independent plans, analytic sections and decoded tapered assembly."""
from dataclasses import replace
import hashlib
import math
import unittest

try:
    from shapely.geometry import LineString, box
    from cambam_builder.cam_core import tapered_inlay as ti, v_region as v
    from tests.test_ornamental_inlay import design
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def tapered(**changes):
    values = dict(seating_mm=.35, bottom_gap_mm=.05, surface_gap_mm=.05,
                  side_fit_mm=.18, edge_access_mm=1)
    values.update(changes)
    return ti.Design(design(**values), 30, .35, .35)


def supplied_plan(target, tool, pitch=.12):
    """Independent constant-depth contour/raster caller of fixed-V targets."""
    radius = v.contact_radius(target, tool, target.cap_depth)
    centers = target.safe.buffer(-radius-.001, quad_segs=128)
    paths = []
    if not centers.is_empty:
        lines = []
        for poly in ti.straight._parts(centers):
            lines.extend(LineString(r.coords).simplify(1e-8)
                         for r in (poly.exterior, *poly.interiors))
        x0, y0, x1, y1 = centers.bounds
        count = max(1, math.ceil((y1-y0)/pitch))
        for i in range(count+1):
            y = y0+(y1-y0)*i/count
            cut = centers.intersection(LineString(((x0-1,y), (x1+1,y))))
            lines.extend((cut,) if cut.geom_type == "LineString" else
                         tuple(cut.geoms) if hasattr(cut, "geoms") else ())
        for line in lines:
            if line.geom_type == "LineString" and line.length > 1e-8:
                points = tuple((x,y,target.cap_depth) for x,y in line.coords)
                paths.append(v.VPath("fill", points))
    paths = tuple(paths)
    plan = v.VPlan(target, tool, paths, v._motions(paths, 3), 3, .001,
                   pitch, "partial" if paths else "infeasible", "synthetic supplied plan")
    v.verify(plan)
    return plan


def stock(q, side, kind="pointed", tip=.03, angle=30):
    tool = v.VProfile(kind, angle, 0 if kind == "pointed" else tip, max(2,tip+2), 2)
    return ti.PartStock(tuple(v.VComposition(target, (supplied_plan(target, tool),))
                             for target in q.targets(side)))


def outputs(q, side, part, dialect, *, include_rough=False):
    from cambam_builder.cam_core import ordered_job as oj
    from cambam_builder.integrations import inlay_output, ordered_output
    results = []
    for component in part.components:
        plan = component.stages[0]
        start = (-2, -2, 3)
        motions = tuple(oj.JobMove(m.role, m.start, m.end,
                        0 if m.role == "rapid" else 100)
                        for m in v.complete_motion(plan, start))
        stage = oj.Stage("tapered", "T1", motions, 10000, v_plan=plan,
                         source_revision=plan.fingerprint)
        stages = (stage,)
        if include_rough:
            from tests.test_mixed_v_composition import _cylinder
            rough = _cylinder(component.target,"rough","T2",.05,
                              (.45,.5),(.75,.5),depths=(.2,))
            stage = replace(stage,transition=oj.Transition("operator",
                "pause" if dialect == "grbl" else "split","T1",start,"install"))
            stages = (rough,stage)
        job = oj.Job(q.source(side), stages, start, program_frame=q.frame(side))
        files = ordered_output.emit(job, dialect)[0]
        hashes = tuple(hashlib.sha256(f).hexdigest() for f in files)
        results.append(inlay_output.ComponentOutput(job, files, job.fingerprint, hashes))
    return tuple(results)


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class TaperedInlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = tapered()
        cls.receiver = stock(cls.q, "receiver", "flat")
        cls.plug = stock(cls.q, "plug", "rounded")

    def verify(self, q=None, receiver=None, plug=None, **kwargs):
        r, p = receiver or self.receiver, plug or self.plug
        return ti.verify_pair(q or self.q, r, p,
            expected_receiver_stock=kwargs.pop("expected_receiver_stock", r.fingerprint),
            expected_plug_stock=kwargs.pop("expected_plug_stock", p.fingerprint),
            **kwargs)

    def test_holed_bridge_profile_pair_and_continuous_insertion(self):
        result = self.verify(slabs=8)
        self.assertEqual(result.status, "pass", result)
        self.assertEqual(result.collision_volume_upper_mm3, 0)
        self.assertEqual(result.minimum_bottom_gap_mm, .05)
        self.assertEqual(result.minimum_surface_gap_mm, .05)
        self.assertEqual(len(self.q.targets("plug")), 2)
        self.assertEqual(len(self.q.plug_tip.interiors), 1)

    def test_pointed_tool_does_not_redefine_targets(self):
        r = stock(self.q, "receiver")
        self.assertEqual(r.components[0].target.fingerprint,
                         self.receiver.components[0].target.fingerprint)
        self.assertEqual(self.verify(receiver=r, slabs=8).status, "pass")

    def test_rectangle_has_independent_offset_area_and_profile_oracles(self):
        q = tapered(motif=box(0, 0, 4, 3))
        inset = .35*math.tan(math.pi/12)+.18
        self.assertAlmostEqual(q.plug_tip.area, (4-2*inset)*(3-2*inset))
        t = q.targets("receiver")[0]
        for z in (0, .2, .4):
            d = z*math.tan(math.pi/12)
            expected = (4-2*d)*(3-2*d)
            self.assertLessEqual(t.section(z).area, expected+1e-12)
            self.assertGreaterEqual(t.section(z, outer=True).area, expected-1e-12)
        # One length-1 capsule: area = 2*L*r + pi*r^2. The rounded
        # oracle covers both its spherical tip and tangent cone independently.
        target = v.VTarget("analytic", box(-5,-5,5,5), box(-5,-5,5,5), 1,
                           design_angle_degrees=30)
        for kind in ("pointed", "flat", "rounded"):
            tool = v.VProfile(kind, 30, 0 if kind == "pointed" else .1, 2, 2)
            paths = (v.VPath("fill", ((0,0,.8),(1,0,.8))),)
            plan = v.VPlan(target, tool, paths, v._motions(paths,3),
                           3,.001,.1,"partial","analytic")
            part = ti.PartStock((v.VComposition(target,(plan,)),))
            for h in (.02,.2):
                r = h*math.tan(math.pi/12)
                if kind == "flat":
                    r += .1
                elif kind == "rounded":
                    join = .1*(1-math.sin(math.pi/12))
                    r = (math.sqrt(.2*h-h*h) if h <= join else
                         .1*math.cos(math.pi/12)+(h-join)*math.tan(math.pi/12))
                expected = 2*r+math.pi*r*r
                inner, outer = ti.section(part,.8-h)
                self.assertLessEqual(inner.area,expected)
                self.assertGreaterEqual(outer.area,expected)
                self.assertLess(outer.area-inner.area,.001)
                self.assertTrue(outer.covers(inner))

    def test_missing_hole_remains_solid_and_registration_collides(self):
        p = ti.PartStock(self.plug.components[:1])
        result = self.verify(plug=p, slabs=2)
        self.assertEqual(result.status,"collision")
        self.assertGreater(result.collision_volume_lower_mm3,1)
        result = self.verify(registration_xy=(.6,0),slabs=2)
        self.assertEqual(result.status,"collision")
        self.assertGreater(result.collision_volume_lower_mm3,0)

    def test_stale_revision_frame_motion_and_tool_reject_before_stock(self):
        for changes in ({"expected_receiver_stock":"stale"},
                        {"slabs":True}, {"registration_xy":(float("nan"),0)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.verify(**changes)
        with self.assertRaisesRegex(ValueError,"revision"):
            self.verify(q=replace(self.q,allowances=replace(self.q.allowances,revision="new")))
        component = self.receiver.components[0]
        plan = component.stages[0]
        changed = replace(plan,tool=replace(plan.tool,tip_radius=.02))
        r = ti.PartStock((v.VComposition(component.target,(changed,)),))
        with self.assertRaisesRegex(ValueError,"stale"):
            self.verify(receiver=r,expected_receiver_stock=self.receiver.fingerprint)
        wrong = replace(plan,target=replace(plan.target,frame="other"))
        r = ti.PartStock((v.VComposition(wrong.target,(wrong,)),))
        with self.assertRaisesRegex(ValueError,"frame"):
            self.verify(receiver=r)

    def test_invalid_taper_web_stock_and_overtravel(self):
        for changes in ({"angle_degrees":0}, {"angle_degrees":120},
                        {"receiver_overtravel_mm":3}, {"plug_overtravel_mm":1},
                        {"plug_overtravel_mm":-1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.q,**changes)
        with self.assertRaisesRegex(ValueError,"rim"):
            tapered(edge_access_mm=.01)

    def test_signed_interference_and_insufficient_floor_do_not_pass(self):
        q = tapered(motif=box(0,0,3,3),stock_xy=box(-1,-1,4,4),side_fit_mm=-.1)
        r,p = stock(q,"receiver","flat"),stock(q,"plug","flat")
        self.assertEqual(self.verify(q,r,p,slabs=4).status,"collision")
        q = replace(q,allowances=replace(q.allowances,side_fit_mm=.18),
                    receiver_overtravel_mm=0)
        r,p = stock(q,"receiver"),stock(q,"plug","flat")
        result = self.verify(q,r,p,slabs=4)
        self.assertNotEqual(result.status,"pass")
        self.assertEqual(result.minimum_bottom_gap_mm,0)

    def test_slab_refinement_bounds_same_continuous_stock(self):
        q = tapered(motif=box(0,0,3,3),stock_xy=box(-1,-1,4,4))
        r,p = stock(q,"receiver","flat"),stock(q,"plug","rounded")
        coarse = self.verify(q,r,p,slabs=1)
        fine = self.verify(q,r,p,slabs=4)
        self.assertLessEqual(fine.collision_volume_upper_mm3,
                             coarse.collision_volume_upper_mm3+1e-10)
        self.assertGreaterEqual(fine.collision_volume_lower_mm3,
                                coarse.collision_volume_lower_mm3-1e-10)
        self.assertEqual(fine.status,"pass")

    def test_complete_decoded_component_outputs(self):
        from cambam_builder.integrations import inlay_output
        q = tapered(motif=box(0,0,4,3).difference(box(1,1,2,2)),
                    stock_xy=box(-1,-1,5,4),minimum_web_mm=.2)
        r,p = stock(q,"receiver","flat"),stock(q,"plug","rounded")
        for dialect in ("uccnc","grbl"):
            with self.subTest(dialect=dialect):
                ro = outputs(q,"receiver",r,dialect,include_rough=(dialect == "grbl"))
                po = outputs(q,"plug",p,dialect)
                result = inlay_output.audit_tapered_pair(q,ro,po,dialect,slabs=4)
                self.assertEqual(result["assembly"]["status"],"pass",result["assembly"])
                self.assertEqual(len(result["parts"][1]),2)
                with self.assertRaisesRegex(ValueError,"bytes changed"):
                    inlay_output.audit_tapered_pair(q,(replace(ro[0],
                        files=(ro[0].files[0]+b"\n",)),),po,dialect)
                with self.assertRaisesRegex(ValueError,"binding"):
                    inlay_output.audit_tapered_pair(q,ro,(replace(po[0],
                        expected_job="stale"),),dialect)

    def test_different_angles_varying_depth_and_unreachable_tool(self):
        q = tapered(motif=box(0,0,3,3),stock_xy=box(-1,-1,4,4))
        r,p = stock(q,"receiver","flat"),stock(q,"plug","rounded",angle=20)
        original = r.components[0].stages[0]
        paths = tuple(replace(path,points=tuple((x,y,d-.005*x)
                        for x,y,d in path.points))
                      for path in original.paths)
        variable = replace(original,paths=paths,motions=v._motions(paths,3))
        r = ti.PartStock((v.VComposition(original.target,(variable,)),))
        self.assertEqual(self.verify(q,r,p,slabs=4).status,"pass")
        # A flat tip larger than the component opening cannot clear it.
        blocked = stock(q,"plug","flat",tip=20)
        self.assertNotEqual(self.verify(q,r,blocked,slabs=1).status,"pass")


if __name__ == "__main__":
    unittest.main()
