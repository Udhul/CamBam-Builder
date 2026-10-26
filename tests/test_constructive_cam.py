"""Independent source contact, decoded machining and stale-design witnesses."""

from dataclasses import replace
import math
import unittest

from cambam_builder.cam_core.constructive import ConstructiveSource, SweptRidge
from cambam_builder.cam_core import relief
from cambam_builder.cam_core.occupancy import Box, ToolBand, ToolBody
from cambam_builder.cam_core.ordered_job import AxialLimits
from cambam_builder.cam_extensions import relief_finishing as finishing
from cambam_builder.integrations import ordered_output


def source():
    return ConstructiveSource("cam-test", 1.2, 0.8, 0.6, -0.24,
        (SweptRidge("curved", (0, 1.2), (0.4, 0.4), (0.22, 0.28),
                    (0.08, 0.10), wave_amplitude_mm=0.07),))


def proposal(target=None, strategy="spine", **options):
    settings = dict(tool=ToolBody("T1", (ToolBand("cutter", 0, 1, 0.15),
        ToolBand("shank", 1, 2, 0.15), ToolBand("holder", 2, 3, 0.3))),
        stepover_mm=0.2, sample_step_mm=0.12, safe_z_mm=1,
        feed_mm_min=240, plunge_mm_min=120, rpm=10000,
        axial_limits=AxialLimits(0.5, 0.5), pitch_mm=0.05,
        contact_tolerance_mm=0.003, allowance_mm=0.004)
    settings.update(options)
    return finishing.generate(source() if target is None else target, strategy, **settings)


class ConstructiveCAMTests(unittest.TestCase):
    def test_contact_brackets_independent_dense_ball_envelope(self):
        target = source()
        radius = 0.15
        # Deliberately independent sampling of the lower sphere envelope.
        # The dense witness is a lower estimate; its resolution adds no proof.
        for x, y in ((0, 0.4), (0.3, 0.35), (0.6, 0.4), (1.2, 0.8)):
            result = relief.contact(target, radius, x, y, tolerance_mm=0.001)
            dense = max(target.height_at(x + radius * i / 50,
                                         y + radius * j / 50) - radius +
                        math.sqrt(max(0, radius**2 * (1 - (i*i+j*j)/2500)))
                        for i in range(-50, 51) for j in range(-50, 51)
                        if i*i + j*j <= 2500)
            self.assertLessEqual(dense, result.upper_tip_z_mm + 1e-12)
            self.assertLess(result.upper_tip_z_mm - dense, 0.0012)

    def test_feature_families_depth_and_both_decoded_dialects(self):
        jobs = {family: proposal(strategy=family)
                for family in ("parallel", "spine", "section")}
        for family, job in jobs.items():
            finishing.check_source(job, source())
            depths = {move.end[2] for move in job.stages[0].motions if move.role == "cut"}
            self.assertGreater(len(depths), 3)
            for dialect in ("uccnc", "grbl"):
                with self.subTest(family=family, dialect=dialect):
                    _, report = ordered_output.emit(job, dialect)
                    self.assertEqual(report["stock_access_residual"]["status"], "pass")
                    self.assertEqual(report["tool_fixture_occupancy"]["status"], "pass")
                    self.assertLess(report["stock_access_residual"]["prefixes"][-1]
                                    ["maximum_residual_thickness_mm"][1], 0.18)
        self.assertNotEqual(jobs["spine"].fingerprint, jobs["section"].fingerprint)

    def test_cap_retains_original_design_and_leaves_measurable_stock(self):
        target = source()
        job = proposal(target, depth_cap_mm=0.1)
        self.assertEqual(job.stages[0].operation.target, target)
        self.assertGreaterEqual(min(m.end[2] for m in job.stages[0].motions), -0.1)
        _, report = ordered_output.emit(job, "uccnc")
        self.assertGreater(report["stock_access_residual"]["prefixes"][-1]
                           ["maximum_residual_thickness_mm"][0], 0.08)
        non_grid_cap = proposal(target, depth_cap_mm=0.10005)
        self.assertGreaterEqual(min(m.end[2] for m in non_grid_cap.stages[0].motions), -0.10005)

    def test_narrow_join_has_independently_unreachable_detail(self):
        target = ConstructiveSource("narrow-valley", 1, 0.8, 0.6, -0.3,
            tuple(SweptRidge(name, (0, 1), (center, center), (0.2, 0.2),
                (0.15, 0.15), profile="tent", cap_ends=False)
                for name, center in (("left", 0.25), ("right", 0.55))))
        query = relief.contact(target, 0.25, 0.5, 0.4, tolerance_mm=0.001)
        # Each ridge peak independently forces tip Z >= -.15-.25+sqrt(.25²-.15²).
        necessary_tip = -0.15 - 0.25 + math.sqrt(0.25**2 - 0.15**2)
        valley_z = -0.3 + 0.15 * (1 - 0.15 / 0.2)
        self.assertAlmostEqual(target.height_at(0.5, 0.4), valley_z)
        self.assertGreaterEqual(query.upper_tip_z_mm, necessary_tip)
        self.assertGreater(query.lower_tip_z_mm - valley_z, 0.06)

    def test_parameter_and_modifier_edits_require_regeneration(self):
        target = source()
        job = proposal(target)
        for changed in (replace(target, repeat_count=2),
                        replace(target, scale_xyz=(1.1, 1, 1)),
                        replace(target, ridges=(replace(target.ridges[0],
                            widths_mm=(0.2, 0.3)),))):
            with self.assertRaisesRegex(ValueError, "stale"):
                finishing.check_source(job, changed)
            fresh = proposal(changed)
            finishing.check_source(fresh, changed)

    def test_unsafe_depth_links_axial_and_fixture_choices_reject(self):
        job = proposal()
        # A continuous synthetically lowered path cannot pass the stock oracle.
        def lowered(point):
            return (point[0], point[1], point[2] - 0.3) if point[2] < 0 else point
        stage = job.stages[0]
        motions = tuple(replace(m, start=lowered(m.start), end=lowered(m.end))
                        for m in stage.motions)
        with self.assertRaisesRegex(ValueError, "protected relief|depth"):
            ordered_output.emit(replace(job, stages=(replace(stage, motions=motions),)), "uccnc")
        motions = tuple(replace(m, role="rapid", feed=0) if m.role == "cut" else m
                        for m in stage.motions)
        with self.assertRaisesRegex(ValueError, "rapid|stock"):
            ordered_output.emit(replace(job, stages=(replace(stage, motions=motions),)), "uccnc")
        with self.assertRaisesRegex(ValueError, "depth|stepdown"):
            ordered_output.emit(proposal(axial_limits=AxialLimits(0.05, 0.05)), "uccnc")
        with self.assertRaisesRegex(ValueError, "fixture|collision"):
            ordered_output.emit(proposal(fixtures=(Box("clamp", (0.4, 0.2, 0, 0.6, 0.6, 2)),)), "uccnc")

    def test_evolving_stock_replays_both_stages_against_same_source(self):
        combined = finishing.sequence(proposal(depth_cap_mm=0.05), proposal(strategy="section"))
        _, report = ordered_output.emit(combined, "uccnc")
        first, last = report["stock_access_residual"]["prefixes"]
        self.assertLess(last["residual_volume_mm3"][1], first["residual_volume_mm3"][1])
        self.assertGreater(last["newly_removed_volume_lower_mm3"], 0)


if __name__ == "__main__":
    unittest.main()
