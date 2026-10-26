"""Reusable SF01 relief consumer and measured strategy comparison.

Run from the repository root with its declared Python, or copy this application
outside the checkout and run against an installed package. Outputs go only to a
new supplied directory (default: unique ignored output/ directory). The sampled
ornament defines an exact bilinear design; it is not certified against its sine
recipe between samples. No CamBam, planar extra or machine is required.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import uuid

import cambam_builder
from cambam_builder.cam_core.occupancy import ToolBand, ToolBody
from cambam_builder.cam_core.ordered_job import AxialLimits
from cambam_builder.cam_core.relief import CubicRelief, HeightField
from cambam_builder.cam_core.relief_stock import residual_at
from cambam_builder.cam_extensions.relief_finishing import generate, motion_cost, sequence
from cambam_builder.integrations.ordered_dialects import decode
from cambam_builder.integrations.ordered_output import write_bundle


def designs() -> dict[str, CubicRelief | HeightField]:
    """Two independent design representations sharing one verifier/policy."""
    xs = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    ys = (0.0, 0.5, 1.0, 1.5, 2.0)
    heights = tuple(tuple(-0.55 + 0.16 * math.cos(math.pi * x) +
                          0.07 * math.sin(math.pi * y / 2) * (x / 1.5)
                          for x in xs) for y in ys)
    return {"valley-ridge": CubicRelief((-2, 0, 2, 3), 1.2, (-0.55, -0.3, 0, 0.1)),
            "sampled-ornament": HeightField(xs, ys, heights, 1.2)}


def proposal(target, strategy: str, *, stepover_mm: float,
             sample_step_mm: float = 0.04, allowance_mm: float = 0.001):
    tool = ToolBody("T1", (ToolBand("cutter", 0, 2, 0.25),
                           ToolBand("shank", 2, 3, 0.25),
                           ToolBand("holder", 3, 4, 0.5)))
    return generate(target, strategy, tool=tool, stepover_mm=stepover_mm,
                    sample_step_mm=sample_step_mm, safe_z_mm=2,
                    feed_mm_min=240, plunge_mm_min=120, rpm=10000,
                    axial_limits=AxialLimits(1, 1), pitch_mm=0.1,
                    waterline_step_mm=0.07, allowance_mm=allowance_mm)


def _measure(directory: Path, job, detail_xy: tuple) -> dict:
    report = write_bundle(directory, job, "uccnc")
    files = tuple((directory / f"stage-{i + 1}.nc").read_bytes()
                  for i in range(len(job.stages)))
    decoded = decode(files, "uccnc", initial_work_tip=job.initial_tip)
    actual_stages, cuts = [], []
    for stage, read in zip(job.stages, decoded.stages):
        motions = tuple(replace(expected, start=observed.start, end=observed.end,
                                feed=observed.feed)
                        for expected, observed in zip(stage.motions, read.moves))
        actual_stages.append(replace(stage, motions=motions))
        radius = stage.relief_operation.radius_mm
        cuts.extend(((*move.start[:2], move.start[2] + radius),
                     (*move.end[:2], move.end[2] + radius), radius)
                    for move in motions if move.role in ("entry", "cut", "retract"))
    target = job.stages[0].relief_operation.target
    return {"design_fingerprint": target.fingerprint,
            "job_fingerprint": job.fingerprint,
            "program_sha256": [hashlib.sha256(data).hexdigest() for data in files],
            "stock": report["stock_access_residual"],
            "body_status": report["tool_fixture_occupancy"]["status"],
            "motion": motion_cost(replace(job, stages=tuple(actual_stages)), rapid_mm_min=1200),
            "detail_residual_mm": [{"xy": point, "interval": residual_at(target, cuts, *point)}
                                   for point in detail_xy]}


def run(directory: Path) -> dict:
    """Write and verify six comparisons plus one cumulative finish/rest job."""
    directory.mkdir(parents=True, exist_ok=False)
    summary = {"version": cambam_builder.__version__,
               "installed_module": cambam_builder.__file__,
               "finish_limit_mm": 0.20,
               "scope": "offline decoded geometry; no machine, dynamics or cutting-load acceptance",
               "cases": {}}
    for name, target in designs().items():
        points = ((-1.0, 1.5), (1.0, 1.5)) if name == "valley-ridge" else ((-1.0, 1.0), (0.0, 1.0), (1.0, 1.0))
        cases = {}
        for strategy, step in (("guided", 0.3), ("parallel", 0.16), ("waterline", 0.3)):
            job = proposal(target, strategy, stepover_mm=step)
            row = _measure(directory / f"{name}-{strategy}", job, points)
            final = row["stock"]["prefixes"][-1]
            row["meets_finish_limit"] = final["maximum_residual_thickness_mm"][1] <= summary["finish_limit_mm"]
            if strategy != "waterline" and not row["meets_finish_limit"]:
                raise AssertionError(f"{name}/{strategy} exceeds finish limit")
            cases[strategy] = row
        summary["cases"][name] = cases
        print(f"Verified {name}: guided, parallel and waterline", flush=True)
    cubic = summary["cases"]["valley-ridge"]
    if not cubic["guided"]["motion"]["feed_distance_mm"] < cubic["parallel"]["motion"]["feed_distance_mm"]:
        raise AssertionError("guided moulding did not reduce motion at the declared finish limit")
    if any(point["interval"][1] > 0.005 for point in cubic["guided"]["detail_residual_mm"]):
        raise AssertionError("guided valley/ridge detail exceeds 0.005 mm")
    target = designs()["valley-ridge"]
    coarse = proposal(target, "parallel", stepover_mm=0.6, sample_step_mm=0.12,
                      allowance_mm=0.20)
    finish = proposal(target, "guided", stepover_mm=0.3)
    evolution = _measure(directory / "evolving-stock", sequence(coarse, finish), ((-1.0, 1.5), (1.0, 1.5)))
    prefixes = evolution["stock"]["prefixes"]
    if prefixes[-1]["newly_removed_volume_lower_mm3"] <= 0.1:
        raise AssertionError("later finish lacks useful-removal evidence")
    if prefixes[-1]["residual_volume_mm3"][1] >= prefixes[0]["residual_volume_mm3"][1]:
        raise AssertionError("cumulative finish failed to improve residual bound")
    summary["evolving_stock"] = evolution
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Verified cumulative finish; report: {directory / 'comparison.json'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output") / f"sf01-relief-{uuid.uuid4().hex[:8]}")
    run(parser.parse_args().output)
