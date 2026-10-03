"""BO01-B generated independent parts, insertion, facing and compound costs."""
from dataclasses import replace
import math
import unittest

try:
    from shapely.geometry import box
    from cambam_builder.cam_core import ordered_job as oj, ornamental_inlay as oi, replay
    from cambam_builder.cam_core import tapered_inlay as ti, v_region as v
    from cambam_builder.cam_core.occupancy import Box, OccupancySetup, ToolBand, ToolBody
    from cambam_builder.cam_extensions import bundle_search as b, inlay_search as s
    from cambam_builder.integrations import inlay_output as io, ordered_dialects
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != 'shapely':
        raise
    HAS_PLANAR = False


def _design(motif=None):
    return ti.Design(oi.Design('BO01-generated-pair', box(0, 0, 2, 2) if motif is None else motif,
        box(-.5, -.5, 2.5, 2.5), .35, .05, .05, 1, 3, .18, .3, .2, .5), 30, .35, .35)


def _tools(kind='flat'):
    return (b.Tool('T1', v.VProfile(kind, 30, .03, 2, 2), oj.AxialLimits(.75, 2),
                   10000, 200, 100, 600),
            b.Tool('T2', replay.ToolProfile('T2', 'cylinder', .2, 2),
                   oj.AxialLimits(.75, 2), 10000, 1200, 200, 600))


def _body(tool):
    p = tool.profile
    radius = p.radius(p.cutting_length) if type(p) is v.VProfile else p.radius
    return ToolBody(tool.tool_id, (ToolBand('cutter', 0, p.cutting_length, radius),
        ToolBand('shank', p.cutting_length, p.cutting_length+1, radius),
        ToolBand('holder', p.cutting_length+1, p.cutting_length+3, radius+.2)))


def _setup(target, tools):
    x0, y0, x1, y1 = target.safe.bounds
    return OccupancySetup(target.frame or 'program',
        Box('part-envelope', (x0, y0, -target.cap_depth, x1, y1, 0)), (),
        tuple(_body(t) for t in tools))


def _requests(q, tools=None, families=None):
    tools = _tools() if tools is None else tools
    families = (b.Family('offset', 'offset', .12, .15, margin_mm=.002),) if families is None else families
    return tuple(tuple(s.PartRequest(tools, families, b.Constraints(100, 100),
        _setup(t, tools), (-2, -2, 3), 3, min(2, len(tools)*len(families)))
        for t in q.targets(side)) for side in ('receiver', 'plug'))


def _facing():
    tool = b.Tool('T3', replay.ToolProfile('T3', 'cylinder', .6, 4),
                  oj.AxialLimits(.5, 2), 10000, 800, 200, 600)
    setup = OccupancySetup('assembled-backing-top',
        Box('assembled-envelope', (-1.5, -1.5, -1.101, 3.5, 3.5, 0)), (), (_body(tool),))
    return s.Facing(tool, setup, .7, 3, 200, .05, .001, .2, .25, 1, 1,
                    'assembled', 'cured', 'renewed', 10, 20)


def _search(q, receiver=None, plug=None, **kwargs):
    if receiver is None:
        receiver, plug = _requests(q)
    options = dict(facing=_facing(), constraints=b.Constraints(200, 200),
        cost_model=b.CostModel(3000, 2, 3, 5), max_evaluations=1, slabs=8)
    options.update(kwargs)
    return s.search_pair(q, receiver, plug, **options)


@unittest.skipUnless(HAS_PLANAR, 'optional planar backend is absent')
class InlaySearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = _design()
        cls.result = _search(cls.q)

    def test_generated_pair_passes_decoded_insertion_and_final_facing(self):
        result = self.result
        self.assertEqual(result.status, 'selected', [(a.actions, a.reasons) for a in result.assessments])
        self.assertFalse(result.enumeration_complete)
        self.assertEqual((result.total_orders, result.evaluated_orders), (16, 1))
        self.assertIn('unevaluated', result.quality)
        bundle = result.chosen.bundle
        self.assertEqual(bundle.assembly_report['assembly']['status'], 'pass')
        self.assertEqual(bundle.facing_report['finishing']['status'], 'pass')
        self.assertEqual(bundle.facing_report['finishing']['unfaced_upper_mm2'], 0)
        self.assertEqual(bundle.facing_report['finishing']['minimum_receiver_floor_mm'], 2.25)
        report = io.audit_facing(bundle.finish, bundle.facing, 'uccnc')
        self.assertEqual(report['finishing']['status'], 'pass')
        split = len(self.q.targets('receiver'))
        outputs = tuple(io.ComponentOutput(x.job, x.files, x.job.fingerprint, x.program_sha256)
                        for x in bundle.parts)
        report = io.audit_tapered_pair(self.q, outputs[:split], outputs[split:], 'uccnc', slabs=8)
        self.assertEqual(report['assembly']['collision_volume_upper_mm3'], 0)
        self.assertEqual({x.job.program_frame for x in bundle.parts},
                         {self.q.frame('receiver'), self.q.frame('plug')})

    def test_compound_time_changes_and_boundaries_match_decoded_motion(self):
        bundle = self.result.chosen.bundle
        jobs = tuple((x.job, x.files) for x in bundle.parts)+((bundle.facing.job, bundle.facing.files),)
        time = 10+20+len(jobs)*2
        ids = []
        for job, files in jobs:
            decoded = ordered_dialects.decode(files, 'uccnc', initial_work_tip=job.initial_tip)
            for stage in decoded.stages:
                ids.append(stage.tool_id)
                time += sum(60*math.dist(m.start, m.end)/(3000 if m.g == 0 else m.feed)
                            for m in stage.moves)
        boundaries = len(ids)-1
        changes = sum(a != z for a, z in zip(ids, ids[1:]))
        time += boundaries*3+changes*5
        self.assertAlmostEqual(bundle.cost.estimated_time_s, time)
        self.assertEqual(bundle.cost.setup_boundaries, boundaries)
        self.assertEqual(bundle.cost.tool_changes, changes)

    def test_manual_budget_part_limits_and_registration_remain_explicit(self):
        order = self.result.chosen.actions
        selected = _search(self.q, manual_orders=order, max_evaluations=1)
        self.assertEqual(selected.chosen.actions, order)
        self.assertFalse(selected.enumeration_complete)
        blocked = _search(self.q, manual_orders=order, max_evaluations=1,
                          constraints=b.Constraints(200, 200, max_tool_changes=0))
        self.assertEqual(blocked.status, 'infeasible')
        self.assertIn('tool changes exceeds declared limit', blocked.assessments[0].reasons)
        collision = _search(self.q, manual_orders=order, max_evaluations=1, registration_xy=(1, 0))
        self.assertIsNone(collision.chosen)
        self.assertEqual(collision.assessments[0].status, 'constrained')
        self.assertIn('IN01 insertion/fit', collision.assessments[0].reasons[0])
        receiver, plug = _requests(self.q)
        receiver = tuple(replace(r, constraints=b.Constraints(0, 0)) for r in receiver)
        partial = _search(self.q, receiver, plug, manual_orders=order, max_evaluations=1)
        self.assertEqual(partial.assessments[0].status, 'constrained')
        self.assertTrue(all('receiver component' in reason for reason in partial.assessments[0].reasons))
        self.assertIsNotNone(partial.assessments[0].part_assessments[0].bundle)

    def test_facing_limits_and_unsupported_setup_cannot_win(self):
        order = self.result.chosen.actions
        failed = _search(self.q, manual_orders=order, max_evaluations=1,
                         facing=replace(_facing(), minimum_plug_core_mm=.31))
        self.assertEqual(failed.assessments[0].status, 'constrained')
        self.assertIn('thickness_ok', failed.assessments[0].reasons[0])
        policy = _facing()
        fixture = Box('clamp', (-1, -1, .1, 4, 4, 2))
        failed = _search(self.q, manual_orders=order, max_evaluations=1,
            facing=replace(policy, setup=replace(policy.setup, fixtures=(fixture,))))
        self.assertEqual(failed.assessments[0].status, 'rejected')
        self.assertIn('fixture', failed.assessments[0].reasons[0])
        with self.assertRaises(ValueError):
            _search(self.q, constraints=b.Constraints(200, 200, max_floor_cusp_mm=.1))

    def test_mixed_parts_and_cylinder_only_binding_limit(self):
        mixed_order = ((('T2', 'offset'), ('T1', 'offset')),)*2
        mixed = _search(self.q, manual_orders=mixed_order)
        self.assertEqual(mixed.status, 'selected', mixed.assessments[0].reasons)
        self.assertTrue(all(x.job.stages[0].operation is not None for x in mixed.chosen.bundle.parts))
        self.assertEqual(mixed.chosen.assembly_report['assembly']['status'], 'pass')
        unsupported = _search(self.q, manual_orders=((('T2', 'offset'),),)*2)
        self.assertEqual(unsupported.status, 'unresolved')
        self.assertEqual(unsupported.unresolved_orders, 1)
        self.assertIn('matching V design', unsupported.assessments[0].reasons[0])
        self.assertIn('subset', unsupported.quality)

    def test_other_profile_grbl_and_compound_ranking(self):
        slow = _tools()[0]
        fast = replace(slow, tool_id='T4', profile=replace(slow.profile, kind='rounded'),
                       cut_feed_mm_min=900)
        receiver, plug = _requests(self.q, tools=(slow, fast))
        # One tool action per component gives the complete four-pair space.
        receiver = tuple(replace(r, max_operations=1) for r in receiver)
        plug = tuple(replace(r, max_operations=1) for r in plug)
        result = _search(self.q, receiver, plug, max_evaluations=4, dialect='grbl')
        self.assertEqual(result.status, 'selected', [(a.actions, a.reasons) for a in result.assessments])
        self.assertTrue(result.enumeration_complete)
        feasible = [a for a in result.assessments if a.status == 'feasible']
        self.assertGreaterEqual(len(feasible), 2)
        self.assertEqual(result.chosen.bundle.cost.estimated_time_s,
                         min(a.bundle.cost.estimated_time_s for a in feasible))
        self.assertTrue(any('T4' in {stage.tool_id for part in a.bundle.parts for stage in part.job.stages}
                            for a in feasible))
        self.assertEqual(io.audit_facing(result.chosen.bundle.finish, result.chosen.bundle.facing,
                                        'grbl')['finishing']['status'], 'pass')

    def test_input_bindings_and_all_component_requests_are_required(self):
        receiver, plug = _requests(self.q)
        for kwargs in ({'max_evaluations': True}, {'slabs': 0}, {'objective': 'global'},
                       {'registration_xy': (float('nan'), 0)}, {'manual_orders': (None, None)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _search(self.q, **kwargs)
        holed = _design(box(0, 0, 2, 2).difference(box(.9, .9, 1.1, 1.1)))
        r, p = _requests(holed)
        self.assertEqual(len(p), 2)
        with self.assertRaisesRegex(ValueError, 'every independent design component'):
            _search(holed, r, p[:1])
        changed = replace(plug[0], tools=(replace(plug[0].tools[0],
            profile=replace(plug[0].tools[0].profile, tip_radius=.04)), plug[0].tools[1]))
        with self.assertRaisesRegex(ValueError, 'conflicting physical profiles'):
            _search(self.q, receiver, (changed,))


if __name__ == '__main__':
    unittest.main()
