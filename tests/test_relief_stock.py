"""Independent envelope references and fault-sensitive relief stock checks."""

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from cambam_builder.cam_core import relief, relief_stock
from cambam_builder.cam_core.ordered_job import AxialLimits, JobMove


def plane_target():
    return relief.CubicRelief((0, 0, 1, 1), 1, (-0.5, 0, 0, 0))


def stage_at(target, y=0.5, *, name="finish", radius=0.2, pitch=0.1,
             tip=-0.49, x0=None, x1=None, limits=None):
    x0 = target.stock_xy[0] if x0 is None else x0
    x1 = target.stock_xy[2] if x1 is None else x1
    home = (target.stock_xy[0], target.stock_xy[1], 1)
    moves = []
    at = home

    def add(role, end, feed=100):
        nonlocal at
        if end != at:
            moves.append(JobMove(role, at, end, 0 if role == "rapid" else feed))
            at = end

    add("rapid", (x0, y, 1))
    add("entry", (x0, y, tip))
    add("cut", (x1, y, tip))
    add("retract", (x1, y, 1))
    add("rapid", home)
    operation = relief_stock.ReliefOperation(name, "T1", radius, 1,
                                             target, "parallel", pitch)
    stage = SimpleNamespace(id=name, tool_id="T1", operation=operation,
                            source_revision=target.fingerprint,
                            motions=tuple(moves), axial_limits=limits)
    return stage, tuple(moves)


class ReliefStockTests(unittest.TestCase):
    def test_continuous_sweep_matches_independent_stationary_reference(self):
        radius = 0.4
        # Infinite-line optimum has envelope sqrt(r²-d_perp²)*sqrt(1+slope²)
        # above the linearly interpolated centre Z. Its optimum is interior.
        slope, perpendicular = 0.3, 0.1
        a, b = (-2, 0, -0.2 - 2 * slope), (2, 0, -0.2 + 2 * slope)
        expected = 0.2 + math.sqrt(radius ** 2 - perpendicular ** 2) * math.sqrt(1 + slope ** 2)
        actual = relief_stock._sweep_depth((0, perpendicular), a, b, radius)
        self.assertAlmostEqual(actual, expected, places=12)
        # A clipped segment chooses its endpoint rather than an infinite line.
        self.assertAlmostEqual(relief_stock._sweep_depth(
            (0, 0), (0, 0, -0.2), (0.1, 0, 1), radius), 0.6)
        self.assertEqual(relief_stock._sweep_depth(
            (0, 2), a, b, radius), 0)
        self.assertAlmostEqual(relief_stock._sweep_depth(
            (0.1, 0), (0, 0, 1), (0, 0, -0.2), radius),
            0.2 + math.sqrt(radius ** 2 - 0.1 ** 2))

    def test_single_pass_volume_oracle_and_refining_cell_bounds(self):
        target = plane_target()
        expected_residual = 0.5 - (2 * 0.2 * 0.29 + math.pi * 0.2 ** 2 / 2)
        intervals = []
        for pitch in (0.1, 0.05):
            stage, moves = stage_at(target, pitch=pitch)
            report = relief_stock.replay_stages((stage,), (moves,))
            self.assertEqual(report["status"], "pass")
            row = report["prefixes"][0]
            low, high = row["residual_volume_mm3"]
            self.assertLessEqual(low, expected_residual)
            self.assertGreaterEqual(high, expected_residual)
            self.assertEqual(row["protected_overcut_upper_mm3"], 0)
            self.assertGreater(row["newly_removed_volume_lower_mm3"], 0)
            self.assertEqual(row["worst_cell_xy"][0], target.stock_xy[0])
            intervals.append((low, high))
        self.assertLess(intervals[1][1] - intervals[1][0],
                        intervals[0][1] - intervals[0][0])

    def test_cumulative_stock_union_and_retrace_noncredit(self):
        target = plane_target()
        first, a = stage_at(target, 0.25, name="first", pitch=0.05)
        second, b = stage_at(target, 0.75, name="second", pitch=0.05)
        retrace, c = stage_at(target, 0.75, name="retrace", pitch=0.05)
        prefixes = relief_stock.replay_stages((first, second, retrace),
                                             (a, b, c))["prefixes"]
        self.assertGreater(prefixes[1]["newly_removed_volume_lower_mm3"], 0.1)
        self.assertEqual(prefixes[2]["newly_removed_volume_lower_mm3"], 0)
        self.assertEqual(prefixes[1]["residual_volume_mm3"],
                         prefixes[2]["residual_volume_mm3"])
        self.assertLess(prefixes[1]["residual_volume_mm3"][1],
                        prefixes[0]["residual_volume_mm3"][1])
        only = relief_stock.replay_stages((second,), (b,))["prefixes"][0]
        self.assertGreater(only["residual_volume_mm3"][0],
                           prefixes[1]["residual_volume_mm3"][0])

    def test_high_endpoints_do_not_hide_interior_ridge_gouge(self):
        target = relief.CubicRelief((-1, 0, 1, 1), 1, (-0.5, 0, -0.2, 0))
        stage, moves = stage_at(target, radius=0.1, tip=-0.65)
        for point in (moves[1].end, moves[2].end):
            self.assertLess(relief.contact(target, 0.1, *point[:2]).upper_tip_z_mm,
                            point[2])
        with self.assertRaisesRegex(ValueError, "protected relief"):
            relief_stock.replay_stages((stage,), (moves,))

    def test_contact_proof_budget_refuses_uncertified_tangency(self):
        stage, moves = stage_at(plane_target(), tip=-0.5)
        with self.assertRaisesRegex(ValueError, "proof budget"):
            relief_stock.replay_stages((stage,), (moves,), max_proof_splits=0)

    def test_point_residual_preserves_design_and_exact_sweep(self):
        target = plane_target()
        cuts = [((0, 0.5, -0.29), (1, 0.5, -0.29), 0.2)]
        low, high = relief_stock.residual_at(target, cuts, 0.5, 0.5)
        self.assertLessEqual(low, 0.01)
        self.assertGreaterEqual(high, 0.01)
        self.assertLess(high - low, 1e-8)
        low, high = relief_stock.residual_at(target, cuts, 0.5, 1)
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)
        self.assertLess(high - low, 1e-8)
        with self.assertRaisesRegex(ValueError, "outside stock"):
            relief_stock.residual_at(target, cuts, -1, 0)

    def test_revision_grid_tool_and_motion_faults_reject(self):
        target = plane_target()
        stage, moves = stage_at(target)
        for mutation, message in (
                ({"source_revision": "stale"}, "revision"),
                ({"tool_id": "wrong"}, "identity")):
            broken = SimpleNamespace(**{**vars(stage), **mutation})
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                relief_stock.replay_stages((broken,), (moves,))
        other, other_moves = stage_at(target, name="other", pitch=0.05)
        with self.assertRaisesRegex(ValueError, "same design and residual grid"):
            relief_stock.replay_stages((stage, other), (moves, other_moves))
        altered = list(moves)
        altered[0] = replace(altered[0], end=(0, 0.5, -0.1))
        with self.assertRaisesRegex(ValueError, "rapid"):
            relief_stock.replay_stages((stage,), (tuple(altered),))
        altered = list(moves)
        altered[1] = replace(altered[1], role="cleared_descent")
        with self.assertRaisesRegex(ValueError, "unsupported relief feed"):
            relief_stock.replay_stages((stage,), (tuple(altered),))
        altered = list(moves)
        altered[2] = SimpleNamespace(**{**vars(altered[2]), "arc_g": 2})
        with self.assertRaisesRegex(ValueError, "straight motion"):
            relief_stock.replay_stages((stage,), (tuple(altered),))
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            relief_stock.replay_stages((stage,), ((moves[0], *moves[2:]),))

    def test_body_and_explicit_axial_depth_limits(self):
        target = plane_target()
        stage, moves = stage_at(target, limits=AxialLimits(0.6, 0.6))
        result = relief_stock.replay_stages((stage,), (moves,))
        self.assertEqual(result["prefixes"][0]["axial_limits"]["status"], "pass")
        for limits, message in ((AxialLimits(0.4, 0.6), "stepdown"),
                                (AxialLimits(0.6, 0.4), "entry")):
            broken = SimpleNamespace(**{**vars(stage), "axial_limits": limits})
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, message):
                relief_stock.replay_stages((broken,), (moves,))
        broken = SimpleNamespace(**{**vars(stage), "operation": replace(
            stage.operation, cutting_length_mm=0.49)})
        with self.assertRaisesRegex(ValueError, "shank"):
            relief_stock.replay_stages((broken,), (moves,))

    def test_rapid_withdrawal_requires_cleared_endpoint_and_credits_no_stock(self):
        stage, moves = stage_at(plane_target())
        feed_report = relief_stock.replay_stages((stage,), (moves,))["prefixes"][0]
        retract = moves[-2]
        middle = (*retract.start[:2], -0.1)
        withdrawn = (*moves[:-2],
                     replace(retract, role="rapid_retract", end=middle, feed=0),
                     replace(retract, role="rapid_retract", start=middle, feed=0),
                     moves[-1])
        rapid_report = relief_stock.replay_stages((stage,), (withdrawn,))["prefixes"][0]
        self.assertEqual(rapid_report["cuts"], feed_report["cuts"] - 1)
        for key in ("residual_volume_mm3", "maximum_residual_thickness_mm",
                    "newly_removed_volume_lower_mm3"):
            self.assertEqual(rapid_report[key], feed_report[key])
        for role, end in (("rapid", retract.end),
                          ("rapid_retract", (retract.end[0] + 0.01, *retract.end[1:])),
                          ("rapid_retract", (*retract.start[:2], retract.start[2] - 0.01))):
            bad = (*moves[:-2], replace(retract, role=role, end=end, feed=0))
            with self.subTest(role=role, end=end), self.assertRaisesRegex(ValueError, "rapid"):
                relief_stock.replay_stages((stage,), (bad,))
        with self.assertRaisesRegex(ValueError, "rapid"):
            relief_stock.replay_stages((stage,), ((
                replace(retract, role="rapid_retract", feed=0),),))

    def test_public_input_and_arithmetic_limits(self):
        target = plane_target()
        for fields in ({"radius_mm": True}, {"pitch_mm": 1e-8},
                       {"cutting_length_mm": 0.1}, {"strategy": "unproved"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                relief_stock.ReliefOperation(**{
                    "name": "finish", "tool_id": "T1", "radius_mm": 0.2,
                    "cutting_length_mm": 1, "target": target,
                    "strategy": "parallel", **fields})
        with self.assertRaises(ValueError):
            relief_stock.replay_stages((), ())
        with self.assertRaisesRegex(ValueError, "representable"):
            relief_stock._sweep_depth((0, 0), (0, 0, 0), (1e300, 0, 0), 0.2)


if __name__ == "__main__":
    unittest.main()
