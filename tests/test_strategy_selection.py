"""Selection uses only current, complete, ordered motion and stock evidence."""

from dataclasses import replace
import hashlib
import unittest

from cambam_builder.cam_extensions.strategy import (
    Alternative, REQUIRED_GATES, ResidualBounds, StageAudit, select_strategy,
)


def sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


SOURCE = sha("edited source")


def stage(name, predecessor, area, volume, *, kind="framework"):
    return StageAudit(name, kind, SOURCE, predecessor, sha(name + " motion"),
                      sha(name + " emitted"), ResidualBounds(area, area + 0.1,
                                                                volume, volume + 0.2),
                      REQUIRED_GATES)


class StrategySelectionTests(unittest.TestCase):
    def select(self, alternatives, *, area=2, volume=20, tie=None, manual=None):
        return select_strategy(SOURCE, tuple(alternatives), max_area_mm2=area,
                               max_volume_mm3=volume,
                               tie_order=tuple(tie or [item.name for item in alternatives]),
                               manual_choice=manual)

    def test_feasible_ranking_uses_area_volume_then_declared_tie_order(self):
        native = Alternative("native", (stage("n", SOURCE, 1, 9, kind="native"),))
        custom = Alternative("custom", (stage("c", SOURCE, 1, 8, kind="custom_region"),))
        explicit = Alternative("explicit", (stage("e", SOURCE, 1, 8),))
        result = self.select((native, custom, explicit), tie=("explicit", "custom", "native"))
        self.assertEqual((result.status, result.chosen), ("selected", "explicit"))
        self.assertEqual([item.status for item in result.assessments],
                         ["feasible", "feasible", "feasible"])
        manual = self.select((native, custom, explicit), manual="native")
        self.assertEqual((manual.status, manual.chosen), ("selected", "native"))

    def test_mixed_ordered_stages_require_exact_predecessor(self):
        rough = stage("native rough", SOURCE, 10, 80, kind="native")
        cleanup = stage("framework cleanup", rough.chain_fingerprint, 1, 8)
        route = Alternative("mixed", (rough, cleanup))
        selected = self.select((route,))
        self.assertEqual(selected.chosen, "mixed")
        self.assertEqual(selected.assessments[0].stage_residuals,
                         ((rough.name, rough.residual), (cleanup.name, cleanup.residual)))
        broken = Alternative("broken", (rough, replace(cleanup,
                                                       predecessor_fingerprint=sha("other"))))
        result = self.select((broken,))
        self.assertEqual((result.status, result.chosen), ("infeasible", None))
        self.assertIn("missing ordered predecessor evidence",
                      result.assessments[0].reasons[0])

    def test_stale_unsafe_and_incomplete_routes_cannot_win_or_be_manually_chosen(self):
        good = Alternative("good", (stage("good", SOURCE, 1, 8),))
        bad_stage = replace(stage("bad", SOURCE, 0, 0), freshness="stale",
                            complete=False, passed_gates=frozenset({"motion"}))
        bad = Alternative("bad", (bad_stage,))
        selected = self.select((bad, good))
        self.assertEqual((selected.status, selected.chosen), ("selected", "good"))
        self.assertEqual(selected.assessments[0].status, "unsafe")
        rejected = self.select((bad, good), manual="bad")
        self.assertEqual((rejected.status, rejected.chosen), ("infeasible", None))
        self.assertEqual(len(rejected.assessments[0].reasons), 3)

    def test_safe_partial_and_no_audited_route_report_distinct_results(self):
        a = Alternative("area miss", (stage("area", SOURCE, 3, 7),))
        v = Alternative("volume miss", (stage("volume", SOURCE, 1, 30),))
        partial = self.select((a, v))
        self.assertEqual((partial.status, partial.chosen), ("partial", "volume miss"))
        self.assertIn("remaining volume exceeds budget", partial.assessments[1].reasons)
        missing = self.select((Alternative("missing", ()),))
        self.assertEqual((missing.status, missing.chosen), ("infeasible", None))
        self.assertEqual(missing.assessments[0].reasons, ("no audited stages",))

    def test_rejects_invalid_bounds_tie_order_and_source(self):
        with self.assertRaisesRegex(ValueError, "ordered"):
            ResidualBounds(2, 1, 0, 0)
        candidate = Alternative("one", (stage("one", SOURCE, 1, 8),))
        with self.assertRaisesRegex(ValueError, "tie order"):
            self.select((candidate,), tie=("other",))
        stale = Alternative("stale", (replace(stage("s", SOURCE, 1, 8),
                                              source_fingerprint=sha("old source")),))
        result = self.select((stale,))
        self.assertEqual(result.status, "infeasible")
        self.assertIn("stale source fingerprint", result.assessments[0].reasons[0])


if __name__ == "__main__":
    unittest.main()
