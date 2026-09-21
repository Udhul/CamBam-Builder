import math
import unittest
from dataclasses import FrozenInstanceError

from cambam_builder import (
    ApplicableRange,
    MachineCapabilities,
    MaterialProfile,
    FixedRecommendationStrategy,
    OperatingConstraints,
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

    def test_rpm_only_cap_recalculates_nonfixed_feed_but_retains_fixed_feed(self):
        context = RecommendationContext(
            self.context.tool, self.context.material,
            MachineCapabilities("router", "mm", max_spindle_speed=6000,
                                max_feed_rate=1000), "slotting")
        recommendations = (
            self.recommendation("spindle_speed", 8000, "rpm"),
            self.recommendation("chip_load", 0.04, "mm/tooth"),
            self.recommendation("feed_rate", 640, "mm/min"),
            self.recommendation("axial_depth", 2, "mm"),
            self.recommendation("radial_engagement", 2, "mm"),
            self.recommendation("specific_cutting_force", 1800, "N/mm^2"),
        )
        plan = plan_milling(
            context, (ProfileRecommendationStrategy("targets", recommendations),))
        self.assertEqual(plan.solution.spindle_speed, 6000)
        self.assertEqual(plan.solution.feed_rate, 480)
        self.assertEqual(plan.solution.chip_load, 0.04)
        self.assertAlmostEqual(plan.solution.material_removal_rate, 1.92)
        self.assertAlmostEqual(plan.solution.cutting_power, 0.0576)
        self.assertAlmostEqual(
            plan.solution.torque, 0.0576 * 30000 / (math.pi * 6000))
        self.assertEqual([item.field for item in plan.active_constraints],
                         ["spindle_speed"])

        fixed_feed = self.recommendation(
            "feed_rate", 640, "mm/min", fixed=True)
        plan = plan_milling(context, (
            ProfileRecommendationStrategy("targets", recommendations[:-4]),
            FixedRecommendationStrategy("fixed feed", (fixed_feed,)),
        ))
        self.assertEqual(plan.solution.spindle_speed, 6000)
        self.assertEqual(plan.solution.feed_rate, 640)
        self.assertAlmostEqual(plan.solution.chip_load, 640 / 12000)

    def test_fixed_strategy_values_must_agree_with_explicit_and_tool_facts(self):
        fixed_rpm = FixedRecommendationStrategy("fixed rpm", (
            self.recommendation("spindle_speed", 5000, "rpm", fixed=True),))
        plan = plan_milling(
            self.context, (fixed_rpm,),
            fixed_values=MillingConstraints("mm", spindle_speed=5000))
        self.assertEqual(plan.solution.spindle_speed, 5000)
        with self.assertRaisesRegex(ValueError, "fixed recommendation.*conflicts"):
            plan_milling(
                self.context, (fixed_rpm,),
                fixed_values=MillingConstraints("mm", spindle_speed=4000))

        fixed_flutes = FixedRecommendationStrategy("fixed flutes", (
            self.recommendation("effective_flutes", 3, "count", fixed=True),))
        with self.assertRaisesRegex(ValueError, "effective_flutes.*conflicts"):
            plan_milling(self.context, (fixed_flutes,))

    def test_minimum_ranges_adjust_nonfixed_cut_and_entry_feeds(self):
        context = RecommendationContext(
            self.context.tool, self.context.material,
            MachineCapabilities("router", "mm", min_spindle_speed=5000,
                                max_spindle_speed=20000, min_feed_rate=300,
                                max_feed_rate=1000),
            "slotting", OperatingConstraints(
                "mm", min_spindle_speed=7000, max_spindle_speed=18000,
                min_feed_rate=350, max_feed_rate=900))
        strategy = ProfileRecommendationStrategy("low targets", (
            self.recommendation("spindle_speed", 6000, "rpm"),
            self.recommendation("chip_load", 0.02, "mm/tooth"),
            self.recommendation("feed_rate", 240, "mm/min"),
            self.recommendation("ramp_feed_rate", 200, "mm/min",
                                entry_mode="ramp", rule="tested ramp"),
        ))
        plan = plan_milling(context, (strategy,))
        self.assertEqual(plan.solution.spindle_speed, 7000)
        self.assertEqual(plan.solution.feed_rate, 350)
        self.assertAlmostEqual(plan.solution.chip_load, 350 / 14000)
        self.assertEqual(plan.ramp_feed_rate, 350)
        self.assertEqual(
            [(item.field, item.code) for item in plan.active_constraints],
            [("spindle_speed", "MINIMUM_BOUND_APPLIED"),
             ("feed_rate", "MINIMUM_BOUND_APPLIED"),
             ("ramp_feed_rate", "MINIMUM_BOUND_APPLIED")])
        self.assertEqual(plan.recommendations.get("feed_rate").value, 240)

        endpoint = plan_milling(context, (ProfileRecommendationStrategy(
            "endpoints", (
                self.recommendation("spindle_speed", 7000, "rpm"),
                self.recommendation("feed_rate", 350, "mm/min"),)),))
        self.assertFalse(endpoint.active_constraints)

    def test_fixed_and_imperial_operating_ranges(self):
        fixed = FixedRecommendationStrategy("fixed low rpm", (
            self.recommendation("spindle_speed", 6000, "rpm", fixed=True),))
        context = RecommendationContext(
            self.context.tool, self.context.material,
            MachineCapabilities("router", "mm", min_spindle_speed=7000),
            "slotting")
        with self.assertRaisesRegex(ValueError, "fixed spindle_speed.*below minimum"):
            plan_milling(context, (fixed,))

        imperial_tool = ToolProfile(
            "quarter-inch", "in", 0.25, 2, "flat end mill",
            entry_modes=("ramp",))
        imperial = RecommendationContext(
            imperial_tool, self.context.material,
            MachineCapabilities("mill", "in", min_feed_rate=12,
                                max_feed_rate=30),
            "slotting", OperatingConstraints("in", min_feed_rate=15,
                                              max_feed_rate=25))
        recommendation = Recommendation(
            "feed_rate", 10, "in/min", self.source,
            (ApplicableRange("cutter_diameter", "in", 0.1, 0.5),))
        plan = plan_milling(
            imperial, (ProfileRecommendationStrategy("imperial", (recommendation,)),))
        self.assertEqual(plan.solution.feed_rate, 15)
        self.assertEqual(plan.active_constraints[0].code,
                         "MINIMUM_BOUND_APPLIED")


if __name__ == "__main__":
    unittest.main()
