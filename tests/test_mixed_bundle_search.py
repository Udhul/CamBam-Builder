"""BO01-B generated mixed orders preserve original-target and safety gates."""
from dataclasses import replace
from itertools import permutations
import math
import unittest

try:
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    from cambam_builder.cam_core import ordered_job as oj, replay, v_region as v
    from cambam_builder.cam_core.occupancy import Box
    from cambam_builder.cam_extensions import bundle_search as b
    from cambam_builder.integrations import ordered_dialects
    from cambam_builder.integrations.ordered_output import audit_files
    from tests.test_inlay_search import _setup
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != 'shapely':
        raise
    HAS_PLANAR = False


def _tools(angle=90, tip=.2, radius=.5):
    return (b.Tool('T1', v.VProfile('flat', angle, tip, 4, 2),
                   oj.AxialLimits(.75, 2), 10000, 100, 120, 600),
            b.Tool('T2', replay.ToolProfile('T2', 'cylinder', radius, 2),
                   oj.AxialLimits(.75, 2), 10000, 1800, 120, 600))


def _search(target, tools, families, **kwargs):
    options = dict(constraints=b.Constraints(100, 100),
        cost_model=b.CostModel(3000, 0, 0, 1), setup=_setup(target, tools),
        initial_tip=(-1, -1, 3), safe_z=3, max_operations=2, max_evaluations=1)
    options.update(kwargs)
    return b.search(target, tools, families, **options)


def _second_target():
    shape = box(0, 0, 4, 3).difference(box(1.5, 1, 2, 1.5))
    return v.VTarget('BO01-B-second-motif', shape, shape, .4,
                     design_angle_degrees=60, frame='program')


@unittest.skipUnless(HAS_PLANAR, 'optional planar backend is absent')
class MixedBundleSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shape = unary_union((box(0, 0, 8, 5), box(2, 5, 4, 6), box(5, 5, 6, 7)))
        shape = shape.difference(box(4, 2, 5, 3))
        cls.target = v.VTarget('BO01-B-mixed-island', shape, shape, .6,
                              design_angle_degrees=90, frame='program')
        cls.tools = _tools()
        cls.families = (b.Family('raster', 'raster', .6, 1),
                        b.Family('feature', 'feature', .6, 1, max_cusp_mm=.3))
        cls.constraints = b.Constraints(.8, 2.2)
        cls.mixed = _search(cls.target, cls.tools, cls.families,
            constraints=cls.constraints, objective='time',
            manual_order=(('T2', 'raster'), ('T1', 'feature')))
        cls.baseline = _search(cls.target, cls.tools[:1], cls.families,
            constraints=cls.constraints, objective='time',
            manual_order=(('T1', 'raster'), ('T1', 'feature')))

    def test_generated_mixed_ornament_beats_same_budget_all_v_baseline(self):
        self.assertEqual(self.mixed.status, 'selected', self.mixed.assessments[0].reasons)
        self.assertEqual(self.mixed.chosen.executed_actions,
                         (('T2', 'raster'), ('T1', 'feature')))
        mixed = self.mixed.chosen.bundle
        baseline = self.baseline.assessments[0]
        self.assertEqual(baseline.status, 'constrained')
        self.assertIn('residual area exceeds declared limit', baseline.reasons)
        self.assertIsNone(self.baseline.chosen)
        self.assertLess(mixed.area_mm2[1], .8)
        self.assertLess(mixed.volume_mm3[1], 2.2)
        self.assertLess(mixed.area_mm2[1], baseline.bundle.area_mm2[0])
        self.assertLess(mixed.cost.estimated_time_s, baseline.bundle.cost.estimated_time_s)
        self.assertEqual(mixed.cost.tool_changes, 1)
        self.assertEqual(mixed.cost.setup_boundaries, 1)
        self.assertFalse(self.mixed.enumeration_complete)
        self.assertEqual((self.mixed.total_orders, self.mixed.evaluated_orders), (16, 1))

    def test_decoded_mixed_stock_preserves_island_and_original_design(self):
        bundle = self.mixed.chosen.bundle
        report = audit_files(bundle.job, bundle.dialect, bundle.files)
        for gate in ('motion_equivalence', 'stock_access_residual',
                     'tool_fixture_occupancy', 'axial_process_limits'):
            self.assertEqual(report[gate]['status'], 'pass')
        composition, decoded = b._composition(bundle.job, bundle.files, bundle.dialect, self.target)
        self.assertEqual(composition.target.fingerprint, self.target.fingerprint)
        self.assertEqual(len(composition.stages), 2)
        for depth in (0, .3, .6):
            self.assertFalse(v.section_evidence(composition, depth).known_free_outer
                             .covers(Point(4.5, 2.5)))
        self.assertEqual(bundle.area_mm2, tuple(v.section_report(composition, .6)))
        self.assertEqual(bundle.volume_mm3, tuple(v.volume_bounds(composition)))
        seconds = 1 + sum(60*math.dist(m.start, m.end)/(3000 if m.g == 0 else m.feed)
                          for stage in decoded.stages for m in stage.moves)
        self.assertAlmostEqual(bundle.cost.estimated_time_s, seconds)

    def test_second_profile_grbl_complete_space_and_cylinder_only_scoring(self):
        target = _second_target()
        tools = _tools(angle=60, tip=.15, radius=.3)
        families = (b.Family('raster', 'raster', .8, 2),)
        result = _search(target, tools, families, max_evaluations=4, dialect='grbl',
                         objective='time')
        actions = (('T1', 'raster'), ('T2', 'raster'))
        expected = {(a,) for a in actions} | set(permutations(actions, 2))
        self.assertEqual({a.actions for a in result.assessments}, expected)
        self.assertEqual((result.total_orders, result.evaluated_orders), (4, 4))
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(result.unresolved_orders, 0)
        self.assertTrue(all(a.status == 'feasible' for a in result.assessments))
        self.assertEqual(result.chosen.executed_actions, (('T2', 'raster'),))
        cylinder = next(a.bundle for a in result.assessments if a.actions == (actions[1],))
        self.assertTrue(all(stage.v_plan is None for stage in cylinder.job.stages))
        composition, _ = b._composition(cylinder.job, cylinder.files, 'grbl', target)
        self.assertEqual(cylinder.area_mm2, tuple(v.section_report(composition, target.cap_depth)))
        self.assertEqual(cylinder.volume_mm3, tuple(v.volume_bounds(composition)))
        # The cylindrical operation's vertical target alone cannot supply the
        # residual of the original tapered finish envelope.
        self.assertGreater(cylinder.volume_mm3[0], 0)
        self.assertFalse(v.section_evidence(composition, .2).known_free_outer
                         .covers(Point(1.75, 1.25)))
        mixed = next(a.bundle for a in result.assessments if a.actions == actions)
        decoded = ordered_dialects.decode(mixed.files, 'grbl', initial_work_tip=mixed.job.initial_tip)
        self.assertTrue(decoded.stages[0].pause_after)
        self.assertEqual([stage.tool_id for stage in decoded.stages], ['T1', 'T2'])

    def test_v_cylinder_v_keeps_decoded_predecessor_and_entry_order(self):
        target = _second_target()
        tools = _tools(angle=60, tip=.15, radius=.3)
        families = (b.Family('raster', 'raster', .8, 2),
                    b.Family('feature', 'feature', .8, 2, max_cusp_mm=.3))
        order = (('T1', 'raster'), ('T2', 'raster'), ('T1', 'feature'))
        result = _search(target, tools, families, max_operations=3, manual_order=order)
        self.assertEqual(result.status, 'selected', result.assessments[0].reasons)
        self.assertEqual(result.chosen.executed_actions, order)
        bundle = result.chosen.bundle
        self.assertEqual([stage.tool_id for stage in bundle.job.stages], ['T1', 'T2', 'T1'])
        self.assertEqual(bundle.cost.tool_changes, 2)
        self.assertEqual(bundle.cost.setup_boundaries, 2)
        self.assertEqual(audit_files(bundle.job, 'uccnc', bundle.files)
                         ['stock_access_residual']['status'], 'pass')

    def test_manual_cylinder_safety_and_unsupported_family_cannot_win(self):
        target, tools = _second_target(), _tools(angle=60, tip=.15, radius=.3)[1:]
        families = (b.Family('raster', 'raster', .8, 2),)
        order = (('T2', 'raster'),)
        setup = replace(_setup(target, tools), fixtures=(Box('clamp', (0, 0, .1, 4, 3, 2)),))
        fixture = _search(target, tools, families, max_operations=1,
                          manual_order=order, setup=setup)
        self.assertIsNone(fixture.chosen)
        self.assertEqual(fixture.assessments[0].status, 'rejected')
        self.assertIn('fixture', ' '.join(fixture.assessments[0].reasons))
        restricted = (replace(tools[0], axial_limits=oj.AxialLimits(.75, .1)),)
        entry = _search(target, restricted, families, max_operations=1, manual_order=order)
        self.assertIsNone(entry.chosen)
        self.assertEqual(entry.assessments[0].status, 'rejected')
        self.assertIn('entry', ' '.join(entry.assessments[0].reasons))
        unsupported = _search(target, tools, (b.Family('feature', 'feature', .8, 2),),
                              max_operations=1, manual_order=(('T2', 'feature'),))
        self.assertEqual(unsupported.status, 'unresolved')
        self.assertEqual(unsupported.unresolved_orders, 1)
        self.assertIn('V profiles only', unsupported.assessments[0].reasons[0])


if __name__ == '__main__':
    unittest.main()
