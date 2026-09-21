import math
import unittest
from dataclasses import FrozenInstanceError

from cambam_builder import (
    ApplicableRange,
    CallableRecommendationStrategy,
    DiameterRecommendationTable,
    FixedRecommendationStrategy,
    MachineCapabilities,
    MaterialProfile,
    MillingConstraints,
    ProfileRecommendationStrategy,
    Recommendation,
    RecommendationContext,
    RecommendationError,
    RecommendationProvenance,
    StrategyResult,
    ToolProfile,
    recommend_milling,
    solve_milling_constraints,
)


class RecommendationProfileTests(unittest.TestCase):
    def setUp(self):
        self.user_source = RecommendationProvenance(
            "user_override", "job setup", "fixture A", "2026-09-21")
        self.shop_source = RecommendationProvenance(
            "measured_shop_policy", "router test log", "run 42", "2026-09-20")
        self.context = RecommendationContext(
            ToolProfile("tool-6", "mm", 6, 2, "flat end mill",
                        substrate="carbide", entry_modes=("ramp", "helical")),
            MaterialProfile("al-6082-t6", "aluminium 6082", "T6"),
            MachineCapabilities("router", "mm", max_spindle_speed=24000,
                                max_feed_rate=3000, max_cutting_power=1.5),
            operation="slotting",
        )

    def recommendation(self, field, value, units, *, fixed=False,
                       entry_mode=None, rule=None):
        return Recommendation(
            field, value, units, self.user_source,
            (ApplicableRange("cutter_diameter", "mm", 3, 8),),
            fixed=fixed, entry_mode=entry_mode, rule=rule,
        )

    def test_profiles_are_immutable_and_machine_limits_are_explicit(self):
        with self.assertRaises(FrozenInstanceError):
            self.context.tool.cutter_diameter = 8
        self.assertEqual(self.context.tool.entry_modes, ("ramp", "helical"))
        self.assertEqual(self.context.machine.as_solver_limits().max_feed_rate, 3000)
        with self.assertRaises(RecommendationError):
            RecommendationContext(
                self.context.tool, self.context.material,
                MachineCapabilities("imperial", "in", max_feed_rate=100))

    def test_provenance_and_applicable_range_are_mandatory(self):
        with self.assertRaisesRegex(ValueError, "version_or_date"):
            RecommendationProvenance(
                "manufacturer_starting_point", "maker", "catalog", "")
        with self.assertRaisesRegex(ValueError, "at least one"):
            Recommendation("chip_load", 0.03, "mm/tooth", self.user_source, ())
        with self.assertRaisesRegex(ValueError, "boolean"):
            Recommendation(
                "chip_load", 0.03, "mm/tooth", self.user_source,
                (ApplicableRange("cutter_diameter", "mm", 3, 8),), fixed=1)
        with self.assertRaisesRegex(ValueError, "kind"):
            RecommendationProvenance("generic", "source", "ref", "date")

    def test_table_interpolates_only_inside_matching_profile(self):
        table = DiameterRecommendationTable(
            "chip_load", "mm", "mm/tooth", ((3, 0.02), (9, 0.05)),
            self.shop_source, "tool-6", "al-6082-t6", "slotting")
        result = recommend_milling(self.context, (table,))
        self.assertAlmostEqual(result.get("chip_load").value, 0.035)
        self.assertEqual(result.get("chip_load").applicable_ranges[0].minimum, 3)
        self.assertFalse(result.missing_requirements)

        outside = RecommendationContext(
            ToolProfile("tool-6", "mm", 10, 2, "flat end mill"),
            self.context.material, self.context.machine, "slotting")
        result = recommend_milling(outside, (table,))
        self.assertIsNone(result.get("chip_load"))
        self.assertIn("covers cutter_diameter", result.missing_requirements[0])

    def test_table_requires_exact_units_and_profile_identity(self):
        wrong_units = DiameterRecommendationTable(
            "surface_speed", "mm", "ft/min", ((3, 100), (9, 200)),
            self.shop_source, "tool-6", "al-6082-t6", "slotting")
        with self.assertRaisesRegex(RecommendationError, "m/min"):
            recommend_milling(self.context, (wrong_units,))

        other_material = DiameterRecommendationTable(
            "surface_speed", "mm", "m/min", ((3, 100), (9, 200)),
            self.shop_source, "tool-6", "steel", "slotting")
        result = recommend_milling(self.context, (other_material,))
        self.assertFalse(result.recommendations)
        self.assertIn("does not apply", result.missing_requirements[0])

    def test_fixed_user_rate_wins_independent_of_strategy_order(self):
        table = DiameterRecommendationTable(
            "chip_load", "mm", "mm/tooth", ((3, 0.02), (9, 0.05)),
            self.shop_source, "tool-6", "al-6082-t6", "slotting")
        fixed = FixedRecommendationStrategy(
            "operator values",
            (self.recommendation("chip_load", 0.025, "mm/tooth", fixed=True),),
        )
        for strategies in ((table, fixed), (fixed, table)):
            with self.subTest(strategies=strategies):
                result = recommend_milling(self.context, strategies)
                self.assertEqual(result.get("chip_load").value, 0.025)
                self.assertTrue(result.get("chip_load").fixed)

    def test_unresolved_nonfixed_conflict_is_rejected(self):
        first = ProfileRecommendationStrategy(
            "first", (Recommendation(
                "chip_load", 0.03, "mm/tooth", self.shop_source,
                (ApplicableRange("cutter_diameter", "mm", 3, 8),)),))
        second = ProfileRecommendationStrategy(
            "second", (Recommendation(
                "chip_load", 0.04, "mm/tooth", self.shop_source,
                (ApplicableRange("cutter_diameter", "mm", 3, 8),)),))
        with self.assertRaisesRegex(RecommendationError, "conflicting"):
            recommend_milling(self.context, (first, second))

    def test_custom_pure_callable_is_validated_and_composed(self):
        def calculate(context):
            value = context.tool.cutter_diameter / 2
            return StrategyResult((Recommendation(
                "axial_depth", value, "mm", self.shop_source,
                (ApplicableRange("cutter_diameter", "mm", 4, 8),)),))

        strategy = CallableRecommendationStrategy("half diameter", calculate)
        result = recommend_milling(self.context, (strategy,))
        self.assertEqual(result.get("axial_depth").value, 3)
        self.assertEqual(result.applied_strategies, ("half diameter",))

        bad = CallableRecommendationStrategy("bad", lambda context: ())
        with self.assertRaisesRegex(TypeError, "StrategyResult"):
            recommend_milling(self.context, (bad,))

    def test_entry_recommendation_requires_capability_and_own_rule(self):
        with self.assertRaisesRegex(ValueError, "own rule"):
            self.recommendation(
                "plunge_feed_rate", 100, "mm/min", entry_mode="plunge")

        ramp = self.recommendation(
            "ramp_feed_rate", 250, "mm/min", entry_mode="ramp",
            rule="measured 3 degree linear ramp limit")
        result = recommend_milling(
            self.context, (ProfileRecommendationStrategy("entry", (ramp,)),))
        self.assertEqual(result.get("ramp_feed_rate").value, 250)

        plunge = self.recommendation(
            "plunge_feed_rate", 100, "mm/min", entry_mode="plunge",
            rule="tool maker centre-cutting plunge row")
        with self.assertRaisesRegex(RecommendationError, "not plunge-capable"):
            recommend_milling(
                self.context, (ProfileRecommendationStrategy("entry", (plunge,)),))

    def test_out_of_range_custom_value_is_reported_not_applied(self):
        value = Recommendation(
            "surface_speed", 150, "m/min", self.shop_source,
            (ApplicableRange("max_spindle_speed", "rpm", 30000, 40000),))
        result = recommend_milling(
            self.context, (ProfileRecommendationStrategy("fast spindle", (value,)),))
        self.assertIsNone(result.get("surface_speed"))
        self.assertIn("outside applicable", result.missing_requirements[0])

    def test_recommendations_feed_the_formula_solver_without_hidden_conversion(self):
        speed = DiameterRecommendationTable(
            "surface_speed", "mm", "m/min", ((3, 120), (9, 180)),
            self.shop_source, "tool-6", "al-6082-t6", "slotting")
        chip = DiameterRecommendationTable(
            "chip_load", "mm", "mm/tooth", ((3, 0.02), (9, 0.05)),
            self.shop_source, "tool-6", "al-6082-t6", "slotting")
        profile = recommend_milling(self.context, (speed, chip))
        solution = solve_milling_constraints(MillingConstraints(
            "mm", cutter_diameter=self.context.tool.cutter_diameter,
            effective_flutes=self.context.tool.effective_flutes,
            surface_speed=profile.get("surface_speed").value,
            chip_load=profile.get("chip_load").value,
        ), self.context.machine.as_solver_limits())
        self.assertAlmostEqual(solution.surface_speed, 150)
        self.assertAlmostEqual(solution.spindle_speed, 150000 / (math.pi * 6))
        self.assertAlmostEqual(solution.feed_rate,
                               solution.spindle_speed * 2 * 0.035)

    def test_invalid_profiles_tables_and_output_units_fail_closed(self):
        with self.assertRaises(ValueError):
            ToolProfile("tool", "mm", 6, 2, "mill", entry_modes=("plunge", "plunge"))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            DiameterRecommendationTable(
                "chip_load", "mm", "mm/tooth", ((6, 0.03), (3, 0.02)),
                self.shop_source, "tool-6", "al-6082-t6")
        bad_output = self.recommendation("feed_rate", 100, "in/min")
        with self.assertRaisesRegex(RecommendationError, "mm/min"):
            recommend_milling(
                self.context,
                (ProfileRecommendationStrategy("wrong units", (bad_output,)),),
            )

        with self.assertRaisesRegex(ValueError, "fixed=True"):
            FixedRecommendationStrategy("not fixed", (bad_output,))


if __name__ == "__main__":
    unittest.main()
