import math
import unittest
from dataclasses import FrozenInstanceError

from cambam_builder import (
    ApplicableRange,
    MachineCapabilities,
    MaterialProfile,
    MillingConstraints,
    ProfileRecommendationStrategy,
    Recommendation,
    RecommendationContext,
    RecommendationProvenance,
    ToolProfile,
    plan_depth_passes,
    plan_milling,
)


class MillingPlanningTests(unittest.TestCase):
    def setUp(self):
        self.source = RecommendationProvenance(
            "measured_shop_policy", "router trials", "setup 42", "2026-09-21")
        self.user_source = RecommendationProvenance(
            "user_override", "operator", "confirmed job values", "2026-09-21")
        self.context = RecommendationContext(
            ToolProfile("tool-6", "mm", 6, 2, "flat end mill",
                        substrate="carbide", entry_modes=("ramp",)),
            MaterialProfile("al-6082-t6", "aluminium 6082", "T6"),
            MachineCapabilities(
                "router", "mm", max_spindle_speed=6000, max_feed_rate=400,
                max_cutting_power=0.05, max_torque=0.05),
            "slotting",
        )

    def recommendation(self, field, value, units, *, fixed=False,
                       entry_mode=None, rule=None):
        return Recommendation(
            field, value, units,
            self.user_source if fixed else self.source,
            (ApplicableRange("cutter_diameter", "mm", 3, 8),),
            fixed=fixed, entry_mode=entry_mode, rule=rule,
        )

    def full_strategy(self):
        return ProfileRecommendationStrategy("validated shop profile", (
            self.recommendation("surface_speed", 150, "m/min"),
            self.recommendation("chip_load", 0.035, "mm/tooth"),
            self.recommendation("axial_depth", 3, "mm"),
            self.recommendation("radial_engagement", 2, "mm"),
            self.recommendation("specific_cutting_force", 1800, "N/mm^2"),
            self.recommendation(
                "ramp_feed_rate", 150, "mm/min", entry_mode="ramp",
                rule="measured three-degree ramp limit"),
        ))

    def test_depth_plan_preserves_existing_balancing_contract(self):
        plan = plan_depth_passes(
            units="mm", stock_thickness=9, cut_through=0.5,
            max_depth_increment=3)
        self.assertEqual(plan.pass_count, 4)
        self.assertEqual(plan.depth_increment, 2.4)
        self.assertEqual(plan.pass_depths, (2.4, 4.8, 7.2, 9.5))
        self.assertAlmostEqual(plan.final_pass_stock, 1.8)
        self.assertTrue(plan.recommendation_met)
        with self.assertRaises(FrozenInstanceError):
            plan.depth_increment = 1

    def test_depth_plan_metric_imperial_equivalence_and_invalid_inputs(self):
        metric = plan_depth_passes(
            units="mm", stock_thickness=9, cut_through=0.5,
            max_depth_increment=3, rounding_increment=0.0254)
        imperial = plan_depth_passes(
            units="in", stock_thickness=9 / 25.4, cut_through=0.5 / 25.4,
            max_depth_increment=3 / 25.4, rounding_increment=0.001)
        self.assertEqual(metric.pass_count, imperial.pass_count)
        self.assertAlmostEqual(metric.depth_increment / 25.4,
                               imperial.depth_increment, places=12)
        self.assertAlmostEqual(metric.final_stock_fraction,
                               imperial.final_stock_fraction, places=12)
        for arguments in (
            {"units": "mm", "stock_thickness": 9, "cut_through": 0.5},
            {"units": "mm", "stock_thickness": 9, "cut_through": 0.5,
             "pass_count": 3, "max_depth_increment": 3},
            {"units": "mm", "stock_thickness": 9, "cut_through": 0.5,
             "max_depth_increment": 0.0001},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                plan_depth_passes(**arguments)

    def test_integrated_candidate_caps_machine_settings_and_uses_actual_stepdown(self):
        plan = plan_milling(
            self.context, (self.full_strategy(),),
            stock_thickness=9, cut_through=0.5)
        self.assertEqual(plan.safe_axial_depth, 3)
        self.assertEqual(plan.depth_plan.depth_increment, 2.4)
        self.assertEqual(plan.solution.axial_depth, 2.4)
        self.assertEqual(plan.solution.spindle_speed, 6000)
        self.assertEqual(plan.solution.feed_rate, 400)
        self.assertAlmostEqual(plan.solution.surface_speed,
                               math.pi * 6 * 6000 / 1000)
        self.assertAlmostEqual(plan.solution.chip_load, 400 / 12000)
        self.assertAlmostEqual(plan.solution.material_removal_rate, 1.92)
        self.assertEqual(plan.stepover, 2)
        self.assertAlmostEqual(plan.stepover_fraction, 1 / 3)
        self.assertEqual(plan.ramp_feed_rate, 150)
        self.assertEqual(
            plan.recommendations.get("axial_depth").provenance.reference,
            "setup 42")
        self.assertEqual([item.field for item in plan.active_constraints],
                         ["spindle_speed", "feed_rate"])
        codes = {item.code for item in plan.diagnostics}
        self.assertIn("MACHINE_CUTTING_POWER_EXCEEDED", codes)
        self.assertIn("MACHINE_TORQUE_EXCEEDED", codes)
        self.assertIn("Starting recommendation only", plan.safety_notice)

    def test_explicit_operating_values_win_over_related_nonfixed_targets(self):
        plan = plan_milling(
            self.context, (self.full_strategy(),),
            fixed_values=MillingConstraints(
                "mm", spindle_speed=5000, feed_rate=350),
        )
        self.assertEqual(plan.solution.spindle_speed, 5000)
        self.assertEqual(plan.solution.feed_rate, 350)
        self.assertAlmostEqual(plan.solution.surface_speed,
                               math.pi * 6 * 5000 / 1000)
        self.assertAlmostEqual(plan.solution.chip_load, 350 / 10000)
        self.assertFalse(plan.active_constraints)

    def test_missing_safe_axial_depth_is_reported_without_guessing(self):
        strategy = ProfileRecommendationStrategy("speed only", (
            self.recommendation("surface_speed", 100, "m/min"),))
        plan = plan_milling(
            self.context, (strategy,), stock_thickness=6, cut_through=0.2)
        self.assertIsNone(plan.depth_plan)
        self.assertIsNone(plan.safe_axial_depth)
        self.assertTrue(any("safe axial_depth" in item
                            for item in plan.missing_requirements))

    def test_fixed_value_over_machine_cap_fails_instead_of_being_changed(self):
        with self.assertRaisesRegex(ValueError, "fixed spindle_speed"):
            plan_milling(
                self.context, (),
                fixed_values=MillingConstraints("mm", spindle_speed=7000))

    def test_nonfixed_direct_operating_targets_are_capped_as_achieved_values(self):
        strategy = ProfileRecommendationStrategy("operating targets", (
            self.recommendation("surface_speed", 150, "m/min"),
            self.recommendation("spindle_speed", 8000, "rpm"),
            self.recommendation("chip_load", 0.04, "mm/tooth"),
            self.recommendation("feed_rate", 640, "mm/min"),
        ))
        plan = plan_milling(self.context, (strategy,))
        self.assertEqual(plan.solution.spindle_speed, 6000)
        self.assertEqual(plan.solution.feed_rate, 400)
        self.assertAlmostEqual(plan.solution.surface_speed,
                               math.pi * 6 * 6000 / 1000)
        self.assertAlmostEqual(plan.solution.chip_load, 400 / 12000)
        self.assertEqual([item.field for item in plan.active_constraints],
                         ["spindle_speed", "feed_rate"])


if __name__ == "__main__":
    unittest.main()
