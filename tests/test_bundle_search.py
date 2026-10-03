"""BO01 finite-search oracles and real generated/decoded consumers."""
from dataclasses import replace
from itertools import permutations
import math
import re
import unittest
from unittest.mock import patch

try:
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    from cambam_builder.cam_core import ordered_job as oj, v_region as v
    from cambam_builder.cam_core.occupancy import Box, OccupancySetup, ToolBand, ToolBody
    from cambam_builder.cam_extensions import bundle_search as b
    from cambam_builder.integrations import ordered_dialects
    from cambam_builder.integrations.ordered_output import audit_files
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != 'shapely':
        raise
    HAS_PLANAR = False


def _tool(name, *, feed=200, angle=90, tip=0, step=.75):
    return b.Tool(name, v.VProfile('flat' if tip else 'pointed', angle, tip, 4, 2),
                  oj.AxialLimits(step, 2), 10000, feed, 120, 600)


def _setup(target, tools):
    x0, y0, x1, y1 = target.safe.bounds
    bodies = tuple(ToolBody(t.tool_id, (
        ToolBand('cutter', 0, 2, t.profile.radius(2)),
        ToolBand('shank', 2, 3, 2), ToolBand('holder', 3, 5, 3))) for t in tools)
    return OccupancySetup('program', Box('stock', (x0, y0, -target.cap_depth, x1, y1, 0)), (), bodies)


def _search(target, tools, families, **kwargs):
    options = dict(constraints=b.Constraints(1000, 1000),
        cost_model=b.CostModel(3000, 0, 0, 1), setup=_setup(target, tools),
        initial_tip=(-1, -1, 3), safe_z=3, max_operations=2, max_evaluations=100)
    options.update(kwargs)
    return b.search(target, tools, families, **options)


def _rectangle():
    return v.VTarget.polygon('BO01-oracle', ((0, 0), (10, 0), (10, 8), (0, 8)), (),
                            1.5, design_angle_degrees=90, frame='program')


def _oracle_plan(prior, tool, family, safe_z):
    segments = {'T1': ((2.5, 4, 1.5), (7.5, 4, 1.5)),
                'T2': ((2.5, 4, 1.5), (4.5, 4, 1.5)),
                'T3': ((5.5, 5, 1.5), (7.5, 5, 1.5))}
    paths = (v.VPath('fill', segments[tool.tool_id]),)
    return v.VPlan(prior.target, tool.profile, paths, v._motions(paths, safe_z),
                   safe_z, .02, 1, 'partial', 'analytic capsule oracle')


@unittest.skipUnless(HAS_PLANAR, 'optional planar backend is absent')
class BundleOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = _rectangle()
        cls.tools = (_tool('T1', feed=20, step=1.5),
                     _tool('T2', feed=1200, step=1.5),
                     _tool('T3', feed=1200, step=1.5))
        cls.families = (b.Family('analytic', 'raster', 1, 1),)
        cls.results = {}
        for objective in ('residual', 'time', 'tool_changes'):
            with patch.object(b, '_plan', _oracle_plan):
                cls.results[objective] = _search(cls.target, cls.tools, cls.families,
                    constraints=b.Constraints(43, 100), objective=objective)

    def test_exhaustive_orders_and_three_different_optima(self):
        actions = tuple((t.tool_id, 'analytic') for t in self.tools)
        expected = set((a,) for a in actions) | set(permutations(actions, 2))
        for result in self.results.values():
            self.assertEqual({a.actions for a in result.assessments}, expected)
            self.assertEqual((result.total_orders, result.evaluated_orders), (9, 9))
            self.assertTrue(result.enumeration_complete)
            self.assertEqual(result.unresolved_orders, 0)
            self.assertEqual(result.quality, 'complete declared enumeration')
        # Analytic area at Z=-1: rectangle 8*6, each section radius .5.
        # Slow alone removes length-five capsule; the two fast capsules are
        # disjoint at this height. Slow+right has the least residual; fast+fast
        # meets the same area limit sooner; only slow meets it with no changes.
        self.assertEqual({x[0] for x in self.results['residual'].chosen.executed_actions},
                         {'T1', 'T3'})
        self.assertEqual({x[0] for x in self.results['time'].chosen.executed_actions},
                         {'T2', 'T3'})
        self.assertEqual(self.results['tool_changes'].chosen.executed_actions,
                         (('T1', 'analytic'),))
        for result in self.results.values():
            self.assertNotIn(result.chosen.actions, result.dominated_actions)

    def test_independent_capsule_area_and_decoded_constant_feed_cost(self):
        result = self.results['residual']
        for assessment in result.assessments:
            bundle = assessment.bundle
            names = {action[0] for action in assessment.executed_actions}
            length = 5 if 'T1' in names else 2 if 'T2' in names else 0
            removed = length + math.pi*.25 if length else 0
            if 'T3' in names:
                removed += 2 + math.pi*.25
            exact = 48-removed
            self.assertLessEqual(bundle.area_mm2[0], exact)
            self.assertGreaterEqual(bundle.area_mm2[1], exact)
            self.assertLess(bundle.area_mm2[1]-bundle.area_mm2[0], .02)
            decoded = ordered_dialects.decode(bundle.files, bundle.dialect,
                                               initial_work_tip=bundle.job.initial_tip)
            duration = (len(bundle.job.stages)-1)*1.0
            for stage in decoded.stages:
                for move in stage.moves:
                    duration += 60*math.dist(move.start, move.end)/(3000 if move.feed == 0 else move.feed)
            self.assertAlmostEqual(bundle.cost.estimated_time_s, duration)
            self.assertEqual(bundle.cost.tool_changes, len(bundle.job.stages)-1)
            self.assertIn('no runtime observation', bundle.cost.confidence)

    def test_manual_choice_and_hard_limits_cannot_be_bypassed(self):
        order = (('T1', 'analytic'), ('T2', 'analytic'))
        with patch.object(b, '_plan', _oracle_plan):
            selected = _search(self.target, self.tools, self.families, manual_order=order)
            blocked = _search(self.target, self.tools, self.families, manual_order=order,
                              constraints=b.Constraints(0, 0, max_time_s=0))
        self.assertEqual(selected.chosen.actions, order)
        self.assertIsNone(blocked.chosen)
        self.assertEqual(blocked.status, 'infeasible')
        self.assertEqual(blocked.assessments[0].status, 'constrained')

    def test_budget_preserves_unevaluated_quality_and_manual_order_priority(self):
        order = (('T3', 'analytic'), ('T2', 'analytic'))
        with patch.object(b, '_plan', _oracle_plan):
            result = _search(self.target, self.tools, self.families,
                             max_evaluations=1, manual_order=order)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual((result.total_orders, result.evaluated_orders), (9, 1))
        self.assertEqual(result.chosen.actions, order)
        self.assertIn('unevaluated', result.quality)

    def test_exact_retrace_omission_and_real_change_count(self):
        tools = (_tool('T1', feed=20, step=.75),)
        families = (b.Family('one', 'raster', 1, 1), b.Family('duplicate', 'raster', 1, 1))
        with patch.object(b, '_plan', _oracle_plan):
            result = _search(self.target, tools, families,
                manual_order=(('T1', 'one'), ('T1', 'duplicate')))
        selected = result.chosen
        self.assertEqual(selected.omitted_actions, (('T1', 'duplicate'),))
        self.assertEqual(len(selected.bundle.job.stages), 2)
        self.assertEqual(selected.bundle.cost.tool_changes, 0)
        self.assertEqual(selected.bundle.cost.setup_boundaries, 1)
        with patch.object(b, '_plan', _oracle_plan):
            blocked = _search(self.target, tools, families, constraints=b.Constraints(
                1000, 1000, max_setup_boundaries=0))
        self.assertEqual(blocked.status, 'infeasible')

    def test_grbl_multi_tool_resume_costs_and_unresolved_budget(self):
        order = (('T2', 'analytic'), ('T3', 'analytic'))
        with patch.object(b, '_plan', _oracle_plan):
            result = _search(self.target, self.tools, self.families, dialect='grbl',
                manual_order=order, max_evaluations=1,
                constraints=b.Constraints(43, 100, max_tool_changes=1),
                cost_model=b.CostModel(3000, 7, 3, 11))
            blocked = _search(self.target, self.tools, self.families, dialect='grbl',
                manual_order=order, max_evaluations=1,
                constraints=b.Constraints(43, 100, max_tool_changes=0))
            unfinished = _search(self.target, self.tools, self.families,
                max_evaluations=1, constraints=b.Constraints(0, 0))
        bundle = result.chosen.bundle
        self.assertEqual(len(bundle.files), 1)
        decoded = ordered_dialects.decode(bundle.files, 'grbl', initial_work_tip=bundle.job.initial_tip)
        self.assertTrue(decoded.stages[0].pause_after)
        self.assertEqual(len(decoded.stages), 2)
        motion_s = sum(60*math.dist(m.start, m.end)/(3000 if m.g == 0 else m.feed)
                       for stage in decoded.stages for m in stage.moves)
        self.assertAlmostEqual(bundle.cost.estimated_time_s, motion_s+7+3+11)
        self.assertEqual(bundle.cost.tool_changes, 1)
        self.assertEqual(bundle.cost.setup_boundaries, 1)
        self.assertIsNone(blocked.chosen)
        self.assertIn('tool changes exceeds declared limit', blocked.assessments[0].reasons)
        self.assertEqual(unfinished.status, 'unresolved')
        self.assertEqual(unfinished.unresolved_orders, 0)
        self.assertFalse(unfinished.enumeration_complete)

    def test_dominance_requires_separated_residual_bounds(self):
        sample = self.results['residual'].chosen
        left = replace(sample, bundle=replace(sample.bundle, area_mm2=(2, 5), volume_mm3=(2, 5),
            cost=replace(sample.bundle.cost, estimated_time_s=1)))
        overlapping = replace(sample, bundle=replace(sample.bundle, area_mm2=(1, 6), volume_mm3=(1, 6),
            cost=replace(sample.bundle.cost, estimated_time_s=2)))
        separated = replace(overlapping, bundle=replace(overlapping.bundle,
            area_mm2=(6, 7), volume_mm3=(6, 7)))
        self.assertFalse(b._dominates(left, overlapping))
        self.assertFalse(b._dominates(overlapping, left))
        self.assertTrue(b._dominates(left, separated))

    def test_failed_generation_is_not_an_optimum_claim(self):
        with patch.object(b, '_plan', side_effect=ValueError('declared guide budget exceeded')):
            result = _search(self.target, self.tools, self.families)
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(result.unresolved_orders, 9)
        self.assertEqual(result.status, 'unresolved')
        self.assertIn('unresolved', result.quality)


@unittest.skipUnless(HAS_PLANAR, 'optional planar backend is absent')
class GeneratedBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shape = unary_union((box(0, 0, 8, 5), box(2, 5, 4, 6), box(5, 5, 6, 7)))
        shape = shape.difference(box(4, 2, 5, 3))
        cls.target = v.VTarget('BO01-island-ornament', shape, shape, .6,
                              design_angle_degrees=90, frame='program')
        cls.tools = (_tool('T4', tip=.2), _tool('T5', angle=120))
        cls.families = (b.Family('raster', 'raster', 1.2, 2),
                        b.Family('feature', 'feature', 1.2, 2, max_cusp_mm=.3))
        cls.result = _search(cls.target, cls.tools, cls.families, max_evaluations=16)

    def test_actual_families_multi_tool_orders_and_protected_island(self):
        result = self.result
        self.assertEqual(result.total_orders, 16)
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(result.unresolved_orders, 0)
        self.assertEqual(result.status, 'selected')
        self.assertTrue(any(len({x[0] for x in a.executed_actions}) == 2 for a in result.assessments))
        self.assertTrue(all(a.status == 'feasible' for a in result.assessments))
        # Every route is audited by search. Re-audit representative returned
        # bundles to challenge downstream use without duplicating all 16 gates.
        representatives = (result.chosen, result.assessments[0], result.assessments[-1])
        for assessment in representatives:
            bundle = assessment.bundle
            report = audit_files(bundle.job, bundle.dialect, bundle.files)
            for gate in ('motion_equivalence', 'stock_access_residual', 'tool_fixture_occupancy',
                         'axial_process_limits'):
                self.assertEqual(report[gate]['status'], 'pass')
            prior, _ = b._composition(bundle.job, bundle.files, bundle.dialect)
            for depth in (0, .3, .6):
                self.assertFalse(v.section_evidence(prior, depth).known_free_outer.covers(Point(4.5, 2.5)))
            self.assertEqual(bundle.area_mm2, tuple(report['stock_access_residual']['section_1_mm2']))

    def test_other_dialect_offset_and_changed_design_profile(self):
        target = v.VTarget.polygon('BO01-other-target', ((0, 0), (7, 0), (7, 5), (0, 5)), (),
                                   .6, design_angle_degrees=60, frame='program')
        tools = (_tool('T6', angle=60),)
        families = (b.Family('offset', 'offset', 1.2, 2),)
        result = _search(target, tools, families, max_operations=1, dialect='grbl')
        self.assertEqual(result.status, 'selected')
        bundle = result.chosen.bundle
        self.assertEqual(audit_files(bundle.job, 'grbl', bundle.files)['stock_access_residual']['status'], 'pass')
        changed, count = re.subn(rb'(G0[^\r\n]*\bZ)3(?=\r?$)', rb'\g<1>-0.1', bundle.files[0],
                               count=1, flags=re.MULTILINE)
        self.assertEqual(count, 1)
        with self.assertRaises(ValueError):
            audit_files(bundle.job, 'grbl', (changed,))

    def test_fixture_and_entry_failures_remain_rejected_even_manual(self):
        tools, families = self.tools[:1], self.families[:1]
        setup = _setup(self.target, tools)
        setup = replace(setup, fixtures=(Box('obstacle', (0, 0, .1, 8, 5, 2)),))
        result = _search(self.target, tools, families, setup=setup, max_operations=1,
                         manual_order=(('T4', 'raster'),))
        self.assertEqual(result.assessments[0].status, 'rejected')
        self.assertIsNone(result.chosen)
        self.assertIn('fixture', ' '.join(result.assessments[0].reasons))
        restricted = (replace(tools[0], axial_limits=oj.AxialLimits(.75, .1)),)
        result = _search(self.target, restricted, families, max_operations=1)
        self.assertEqual(result.assessments[0].status, 'rejected')
        self.assertIn('entry', ' '.join(result.assessments[0].reasons))

    def test_floor_cusp_is_checked_not_inferred_from_requested_pitch(self):
        result = _search(self.target, self.tools[:1], self.families[:1], max_operations=1,
            constraints=b.Constraints(1000, 1000, max_floor_cusp_mm=.0001))
        self.assertEqual(result.assessments[0].status, 'constrained')
        self.assertGreater(result.assessments[0].bundle.unproved_floor_area_mm2, 0)
        self.assertIsNone(result.chosen)

    def test_extreme_sampling_control_is_rejected_without_aborting_search(self):
        families = (b.Family('tiny', 'raster', 1, 5e-324),)
        result = _search(self.target, self.tools[:1], families, max_operations=1)
        self.assertEqual(result.status, 'unresolved')
        self.assertEqual(result.unresolved_orders, 1)
        self.assertEqual(result.assessments[0].status, 'rejected')

    def test_input_contract_rejects_invalid_domains(self):
        for changes in ({'max_evaluations': True}, {'max_operations': 0},
                        {'safe_z': float('nan')}, {'objective': 'global'},
                        {'manual_order': (('absent', 'raster'),)},
                        {'initial_tip': (-1, -1, 0)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _search(self.target, self.tools, self.families, **changes)
        for factory in (lambda: b.Constraints(None, 1), lambda: b.Constraints(1, -1),
                        lambda: b.CostModel(0, 0, 0, 0), lambda: b.Family('bad', 'raster', 0, 1)):
            with self.assertRaises(ValueError):
                factory()


if __name__ == '__main__':
    unittest.main()
