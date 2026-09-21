import math
import sys
import unittest
from dataclasses import FrozenInstanceError

from cambam_builder import (
    MachineLimits,
    MachiningConstraintError,
    MillingConstraints,
    chip_load_from_feed,
    cutting_power,
    cutting_torque,
    feed_from_chip_load,
    material_removal_rate,
    rpm_from_surface_speed,
    solve_milling_constraints,
    surface_speed_from_rpm,
)


class FormulaKernelTests(unittest.TestCase):
    def test_kennametal_metric_worked_example(self):
        rpm = rpm_from_surface_speed(150, 20, units="mm")
        self.assertAlmostEqual(rpm, 2387.324146, places=6)
        self.assertAlmostEqual(feed_from_chip_load(0.1, rpm, 1, units="mm"),
                               238.732415, places=6)

    def test_kennametal_imperial_worked_example(self):
        rpm = rpm_from_surface_speed(500, 0.79, units="in")
        self.assertAlmostEqual(rpm, 6000 / (math.pi * 0.79), places=12)
        self.assertEqual(round(rpm), 2418)
        self.assertAlmostEqual(feed_from_chip_load(0.004, rpm, 1, units="in"),
                               9.670174, places=6)

    def test_formula_inverses(self):
        for units, speed, diameter, chip in (
                ("mm", 275.25, 6.35, 0.031),
                ("in", 902.887139, 0.25, 0.001220472440944882)):
            with self.subTest(units=units):
                rpm = rpm_from_surface_speed(speed, diameter, units=units)
                self.assertAlmostEqual(
                    surface_speed_from_rpm(rpm, diameter, units=units), speed)
                feed = feed_from_chip_load(chip, rpm, 3, units=units)
                self.assertAlmostEqual(
                    chip_load_from_feed(feed, rpm, 3, units=units), chip)

    def test_metric_imperial_equivalence(self):
        rpm_metric = rpm_from_surface_speed(100, 12.7, units="mm")
        rpm_imperial = rpm_from_surface_speed(
            100 * 3.280839895013123, 0.5, units="in")
        self.assertAlmostEqual(rpm_metric, rpm_imperial, places=9)

        feed_mm = feed_from_chip_load(0.05, rpm_metric, 2, units="mm")
        feed_in = feed_from_chip_load(0.05 / 25.4, rpm_imperial, 2, units="in")
        self.assertAlmostEqual(feed_mm / 25.4, feed_in, places=12)
        mrr_metric = material_removal_rate(feed_mm, 2.54, 1.27, units="mm")
        mrr_imperial = material_removal_rate(
            feed_in, 0.1, 0.05, units="in")
        self.assertAlmostEqual(mrr_metric / 16.387064, mrr_imperial, places=12)

    def test_mrr_power_and_torque_follow_published_equations(self):
        self.assertEqual(material_removal_rate(1000, 2, 8, units="mm"), 16)
        power = cutting_power(16, 1800, units="mm")
        self.assertAlmostEqual(power, 0.48)
        self.assertAlmostEqual(cutting_torque(power, 6000, units="mm"),
                               power * 30000 / (math.pi * 6000))

        self.assertEqual(material_removal_rate(20, 0.1, 0.25, units="in"), 0.5)
        hp = cutting_power(0.5, 200000, units="in")
        self.assertAlmostEqual(hp, 100000 / 396000)
        self.assertAlmostEqual(cutting_torque(hp, 5000, units="in"),
                               hp * 16501 / (math.pi * 5000))

    def test_invalid_atomic_inputs_are_rejected(self):
        calls = (
            lambda: rpm_from_surface_speed(100, 5, units="cm"),
            lambda: rpm_from_surface_speed(True, 5, units="mm"),
            lambda: surface_speed_from_rpm(float("nan"), 5, units="mm"),
            lambda: feed_from_chip_load(0.1, 1000, 1.5, units="mm"),
            lambda: chip_load_from_feed(100, 0, 2, units="mm"),
            lambda: material_removal_rate(100, -1, 2, units="mm"),
            lambda: cutting_power(float("inf"), 1000, units="mm"),
            lambda: cutting_torque(1e308, 1e-308, units="mm"),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


class ConstraintSolverTests(unittest.TestCase):
    def test_chained_solution_and_unit_metadata(self):
        result = solve_milling_constraints(MillingConstraints(
            units="mm", cutter_diameter=10, surface_speed=120,
            chip_load=0.04, effective_flutes=3,
            axial_depth=2, radial_engagement=4,
            specific_cutting_force=1800,
        ))
        expected_rpm = 120000 / (math.pi * 10)
        expected_feed = 0.04 * expected_rpm * 3
        self.assertAlmostEqual(result.spindle_speed, expected_rpm)
        self.assertAlmostEqual(result.feed_rate, expected_feed)
        self.assertAlmostEqual(result.material_removal_rate,
                               expected_feed * 2 * 4 / 1000)
        self.assertAlmostEqual(result.cutting_power,
                               result.material_removal_rate * 1800 / 60000)
        self.assertAlmostEqual(result.torque,
                               result.cutting_power * 30000 /
                               (math.pi * expected_rpm))
        self.assertFalse(result.missing_requirements)
        labels = dict(result.unit_labels)
        self.assertEqual(labels["surface_speed"], "m/min")
        self.assertEqual(labels["material_removal_rate"], "cm^3/min")
        self.assertEqual(labels["torque"], "N m")

    def test_each_equation_can_solve_an_identifiable_missing_value(self):
        for units in ("mm", "in"):
            factor = 1000 if units == "mm" else 12
            mrr_scale = 1000 if units == "mm" else 1
            power_divisor = 60000 if units == "mm" else 396000
            torque_factor = 30000 if units == "mm" else 16501
            speed_values = {
                "cutter_diameter": 10, "surface_speed": 100,
                "spindle_speed": factor * 100 / (math.pi * 10),
            }
            feed_values = {
                "chip_load": 0.05, "spindle_speed": 2000,
                "effective_flutes": 2, "feed_rate": 200,
            }
            mrr_values = {
                "feed_rate": 300, "axial_depth": 2, "radial_engagement": 3,
                "material_removal_rate": 1800 / mrr_scale,
            }
            power_values = {
                "material_removal_rate": 12, "specific_cutting_force": 1800,
                "cutting_power": 12 * 1800 / power_divisor,
            }
            torque_values = {
                "cutting_power": 0.5, "spindle_speed": 6000,
                "torque": 0.5 * torque_factor / (math.pi * 6000),
            }
            for group in (speed_values, feed_values, mrr_values,
                          power_values, torque_values):
                for missing, expected in group.items():
                    supplied = {name: value for name, value in group.items()
                                if name != missing}
                    with self.subTest(units=units, missing=missing):
                        result = solve_milling_constraints(
                            MillingConstraints(units, **supplied))
                        self.assertAlmostEqual(getattr(result, missing), expected)
                        self.assertIn(missing, result.solved_fields)

    def test_partial_input_reports_requirements_without_guessing(self):
        result = solve_milling_constraints(MillingConstraints(
            units="in", cutter_diameter=0.25, axial_depth=0.1))
        self.assertIsNone(result.spindle_speed)
        self.assertIsNone(result.feed_rate)
        self.assertIsNone(result.radial_engagement)
        self.assertTrue(any("surface speed" in item
                            for item in result.missing_requirements))
        self.assertTrue(any("MRR" in item for item in result.missing_requirements))

    def test_consistent_overconstraint_passes_and_conflict_fails(self):
        rpm = rpm_from_surface_speed(100, 10, units="mm")
        result = solve_milling_constraints(MillingConstraints(
            "mm", cutter_diameter=10, surface_speed=100, spindle_speed=rpm))
        self.assertEqual(dict(result.requested_values)["spindle_speed"], rpm)
        with self.assertRaisesRegex(MachiningConstraintError, "surface-speed"):
            solve_milling_constraints(MillingConstraints(
                "mm", cutter_diameter=10, surface_speed=100,
                spindle_speed=rpm + 1))

        conflict_cases = (
            MillingConstraints("mm", chip_load=0.05, spindle_speed=2000,
                               effective_flutes=2, feed_rate=201),
            MillingConstraints("mm", feed_rate=300, axial_depth=2,
                               radial_engagement=3, material_removal_rate=2),
            MillingConstraints("mm", material_removal_rate=12,
                               specific_cutting_force=1800, cutting_power=1),
            MillingConstraints("mm", cutting_power=0.5, spindle_speed=6000,
                               torque=2),
        )
        for request in conflict_cases:
            with self.subTest(request=request), self.assertRaises(
                    MachiningConstraintError):
                solve_milling_constraints(request)

    def test_relative_conflicts_near_zero_are_not_masked(self):
        with self.assertRaisesRegex(MachiningConstraintError, "feed"):
            solve_milling_constraints(MillingConstraints(
                "mm", spindle_speed=1, chip_load=1e-15,
                effective_flutes=1, feed_rate=2e-15))

    def test_axial_and_radial_engagement_are_independent(self):
        first = solve_milling_constraints(MillingConstraints(
            "mm", feed_rate=200, axial_depth=2, radial_engagement=5))
        second = solve_milling_constraints(MillingConstraints(
            "mm", feed_rate=200, axial_depth=5, radial_engagement=2))
        self.assertEqual(first.material_removal_rate, second.material_removal_rate)
        self.assertEqual(first.axial_depth, 2)
        self.assertEqual(first.radial_engagement, 5)
        self.assertNotIn("target_depth", MillingConstraints.__dataclass_fields__)

    def test_radial_engagement_cannot_exceed_known_diameter(self):
        with self.assertRaisesRegex(MachiningConstraintError, "radial_engagement"):
            solve_milling_constraints(MillingConstraints(
                "mm", cutter_diameter=6, radial_engagement=7))
        with self.assertRaisesRegex(MachiningConstraintError, "radial_engagement"):
            solve_milling_constraints(MillingConstraints(
                "mm", surface_speed=100, spindle_speed=10000,
                radial_engagement=4))

    def test_derived_rpm_and_feed_caps_propagate_to_achieved_values(self):
        request = MillingConstraints(
            "mm", cutter_diameter=10, surface_speed=200,
            chip_load=0.05, effective_flutes=2,
            axial_depth=2, radial_engagement=3,
            specific_cutting_force=1800,
        )
        result = solve_milling_constraints(
            request, MachineLimits(max_spindle_speed=5000, max_feed_rate=400))
        self.assertEqual(result.spindle_speed, 5000)
        self.assertEqual(result.feed_rate, 400)
        self.assertAlmostEqual(result.surface_speed, math.pi * 10 * 5000 / 1000)
        self.assertAlmostEqual(result.chip_load, 400 / (5000 * 2))
        self.assertAlmostEqual(result.material_removal_rate, 2.4)
        self.assertEqual([item.field for item in result.active_constraints],
                         ["spindle_speed", "feed_rate"])
        self.assertEqual(dict(result.requested_values)["surface_speed"], 200)
        self.assertEqual(dict(result.requested_values)["chip_load"], 0.05)

    def test_caps_at_or_above_result_are_inactive(self):
        request = MillingConstraints(
            "mm", spindle_speed=5000, chip_load=0.04, effective_flutes=2)
        result = solve_milling_constraints(
            request, MachineLimits(max_spindle_speed=5000, max_feed_rate=400))
        self.assertEqual(result.feed_rate, 400)
        self.assertFalse(result.active_constraints)

    def test_fixed_operational_limit_conflicts(self):
        for request, limits, message in (
            (MillingConstraints("mm", spindle_speed=6000),
             MachineLimits(max_spindle_speed=5000), "fixed spindle_speed"),
            (MillingConstraints("mm", feed_rate=500),
             MachineLimits(max_feed_rate=400), "fixed feed_rate"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                    MachiningConstraintError, message):
                solve_milling_constraints(request, limits)

    def test_minimum_operating_limits_adjust_derived_values_and_reject_fixed(self):
        request = MillingConstraints(
            "mm", cutter_diameter=10, surface_speed=100, chip_load=0.02,
            effective_flutes=2, axial_depth=2, radial_engagement=3,
            specific_cutting_force=1000)
        result = solve_milling_constraints(
            request, MachineLimits(min_spindle_speed=4000, min_feed_rate=200))
        self.assertEqual(result.spindle_speed, 4000)
        self.assertEqual(result.feed_rate, 200)
        self.assertAlmostEqual(result.chip_load, 0.025)
        self.assertAlmostEqual(result.material_removal_rate, 1.2)
        self.assertEqual([item.code for item in result.active_constraints],
                         ["MINIMUM_BOUND_APPLIED", "MINIMUM_BOUND_APPLIED"])
        with self.assertRaisesRegex(MachiningConstraintError, "below min"):
            solve_milling_constraints(
                MillingConstraints("mm", spindle_speed=3000),
                MachineLimits(min_spindle_speed=4000))
        with self.assertRaisesRegex(MachiningConstraintError, "cannot exceed"):
            solve_milling_constraints(
                MillingConstraints("mm"),
                MachineLimits(max_feed_rate=100, min_feed_rate=200))

    def test_exact_fixed_value_is_retained_and_results_are_frozen(self):
        exact = 123.45678901234567
        result = solve_milling_constraints(MillingConstraints(
            "mm", spindle_speed=4000, feed_rate=exact))
        self.assertEqual(result.feed_rate, exact)
        self.assertEqual(dict(result.requested_values)["feed_rate"], exact)
        with self.assertRaises(FrozenInstanceError):
            result.feed_rate = 1

        # Consistent within the documented relative tolerance, but still a caller
        # value that must not be normalized when no cap is active.
        near_surface = 100.00000005
        result = solve_milling_constraints(MillingConstraints(
            "mm", cutter_diameter=10, spindle_speed=100000 / (math.pi * 10),
            surface_speed=near_surface))
        self.assertEqual(result.surface_speed, near_surface)
        self.assertFalse(result.active_constraints)

    def test_extreme_numeric_ranges_raise_value_error(self):
        calls = (
            lambda: rpm_from_surface_speed(10 ** 10000, 1, units="mm"),
            lambda: feed_from_chip_load(1, 1, 10 ** 10000, units="mm"),
            lambda: cutting_power(sys.float_info.max, sys.float_info.max,
                                  units="mm"),
            lambda: solve_milling_constraints(MillingConstraints(
                "mm", chip_load=5e-324, spindle_speed=5e-324,
                effective_flutes=1)),
            lambda: solve_milling_constraints(MillingConstraints(
                "mm", material_removal_rate=sys.float_info.max,
                cutting_power=sys.float_info.max)),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_invalid_constraints_and_limits(self):
        cases = (
            lambda: solve_milling_constraints(MillingConstraints("MM")),
            lambda: solve_milling_constraints(MillingConstraints("mm", chip_load=True)),
            lambda: solve_milling_constraints(MillingConstraints("mm", effective_flutes=0)),
            lambda: solve_milling_constraints(MillingConstraints("mm", feed_rate=float("nan"))),
            lambda: solve_milling_constraints(MillingConstraints("mm"),
                                               MachineLimits(max_feed_rate=-1)),
        )
        for call in cases:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
