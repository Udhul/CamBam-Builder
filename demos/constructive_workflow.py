"""CG01 compact constructive geometry, native handoff and audited CAM consumer.

Run from the repository root using its declared interpreter, or copy this file
outside the checkout and run against the installed library. Every emitted NC
file is the exact byte sequence independently decoded and replayed by emit().
The finite comparison covers these candidates only; no universal route optimum,
physical cutting, controller-runtime or arbitrary reverse-engineering claim.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import cambam_builder
from cambam_builder.cam_core.constructive import ConstructiveSource, SweptRidge, reconstruct_spine
from cambam_builder.cam_core.occupancy import ToolBand, ToolBody
from cambam_builder.cam_core.ordered_job import AxialLimits
from cambam_builder.cam_core.relief import contact
from cambam_builder.cam_core.relief_stock import residual_at
from cambam_builder.cam_extensions.relief_finishing import check_source, generate, motion_cost, sequence
from cambam_builder.integrations.cambam.native_constructive import create_document, read_document, regenerate
from cambam_builder.integrations.ordered_dialects import decode
from cambam_builder.integrations.ordered_output import emit
from cambam_builder.native.reader import read_cambam_bytes


FINISH_LIMIT_MM = 0.22
NATIVE_FIDELITY_LIMIT_MM = 0.05


def designs() -> dict[str, ConstructiveSource]:
    """Crossed repeated ornament and a broad changing-section sweep."""
    return {
        "frieze": ConstructiveSource("decorative-frieze", 3, 1.5, 0.6, -0.3, (
            SweptRidge("ascending-petal", (0, 3), (0.35, 1.15), (0.14, 0.22), (0.12, 0.18),
                       wave_amplitude_mm=0.08),
            SweptRidge("descending-sharp-petal", (0, 3), (1.15, 0.35), (0.22, 0.14), (0.18, 0.12),
                       wave_amplitude_mm=-0.08, profile="tent"),
        ), repeat_count=2, provenance=("authored crossing petals; shared max-union branches",)),
        "moulding": ConstructiveSource("changing-section-moulding", 3, 1.5, 0.6, -0.3, (
            SweptRidge("varying-crown", (0, 3), (0.65, 0.85), (0.30, 0.56), (0.10, 0.18),
                       wave_amplitude_mm=0.05, cap_ends=False),
        ), provenance=("authored uncapped sweep; width, section height and center change",)),
    }


def proposal(source: ConstructiveSource, strategy: str, *, stepover_mm: float = 0.15,
             depth_cap_mm: float | None = None):
    tool = ToolBody("T1", (ToolBand("cutter", 0, 1, 0.12),
                           ToolBand("shank", 1, 2, 0.12),
                           ToolBand("holder", 2, 3, 0.30)))
    return generate(source, strategy, tool=tool, stepover_mm=stepover_mm,
                    sample_step_mm=0.04, safe_z_mm=1, feed_mm_min=240,
                    plunge_mm_min=120, rpm=10000, axial_limits=AxialLimits(0.6, 0.6),
                    pitch_mm=0.05, contact_tolerance_mm=0.002,
                    allowance_mm=0.003, depth_cap_mm=depth_cap_mm)


def _measure(directory: Path, job, detail_xy: dict[str, tuple[float, float]]) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # emit independently decodes and checks the complete motion, original source,
    # cumulative stock, axial limits and non-cutting bodies before bytes escape.
    files, evidence = emit(job, "uccnc")
    for index, data in enumerate(files, 1):
        (directory / f"stage-{index}.nc").write_bytes(data)
    decoded = decode(files, "uccnc", initial_work_tip=job.initial_tip)
    stages, cuts, cut_lengths, air_feed = [], [], 0.0, 0.0
    turning_angles = []
    for stage, actual in zip(job.stages, decoded.stages):
        moves = tuple(replace(expected, start=observed.start, end=observed.end, feed=observed.feed)
                      for expected, observed in zip(stage.motions, actual.moves))
        stages.append(replace(stage, motions=moves))
        radius = stage.relief_operation.radius_mm
        previous_direction = None
        for move in moves:
            distance = math.dist(move.start, move.end)
            if move.role == "cut":
                cut_lengths += distance
                direction = tuple(b-a for a, b in zip(move.start, move.end))
                if previous_direction is not None:
                    denominator = math.hypot(*direction) * math.hypot(*previous_direction)
                    cosine = sum(a*b for a, b in zip(direction, previous_direction)) / denominator
                    turning_angles.append(math.degrees(math.acos(max(-1, min(1, cosine)))))
                previous_direction = direction
            else:
                previous_direction = None
                if move.role not in ("rapid", "rapid_retract"):
                    air_feed += distance
            if move.role in ("entry", "cut", "retract"):
                cuts.append(((*move.start[:2], move.start[2]+radius),
                             (*move.end[:2], move.end[2]+radius), radius))
    source = job.stages[0].relief_operation.target
    stock = evidence["stock_access_residual"]
    motion = motion_cost(replace(job, stages=tuple(stages)), rapid_mm_min=1200)
    cut_depths = [-p[2] for stage in stages for move in stage.motions if move.role == "cut"
                  for p in (move.start, move.end)]
    row = {
        "source_fingerprint": source.fingerprint, "job_fingerprint": job.fingerprint,
        "program_sha256": [hashlib.sha256(data).hexdigest() for data in files],
        "stock": stock, "body_status": evidence["tool_fixture_occupancy"]["status"],
        "motion": {**motion, "cut_distance_mm": cut_lengths,
                   "entry_retract_feed_mm": air_feed,
                   "mean_cut_turn_degrees": sum(turning_angles)/len(turning_angles) if turning_angles else 0,
                   "maximum_cut_turn_degrees": max(turning_angles, default=0),
                   "tool_changes": max(0, len({stage.tool_id for stage in stages})-1)},
        "geometry_dependent_depth_mm": [min(cut_depths), max(cut_depths)],
        "located_residual_mm": [{"feature": name, "xy": point,
                                 "interval": residual_at(source, cuts, *point)}
                                for name, point in detail_xy.items()],
        "meets_finish_limit": stock["prefixes"][-1]["maximum_residual_thickness_mm"][1] <= FINISH_LIMIT_MM,
        "audit_seconds": time.monotonic()-started,
    }
    (directory / "evidence.json").write_text(json.dumps({"summary": row, "audit": evidence}, indent=2)+"\n", encoding="utf-8")
    return row


def _mesh_fidelity(source: ConstructiveSource, document) -> dict:
    """Independent barycentric/edge checks on parsed native triangle bytes."""
    project = read_cambam_bytes(document.xml_bytes, source_name="constructive consumer native bytes")
    surface = next(p for p in project.list_primitives() if p.user_identifier == document.surface_identifier)
    geometry = surface.get_absolute_coordinates_xyz()
    vertices, faces = geometry["vertices"], geometry["faces"]
    incidence = Counter()
    maximum, witness = 0.0, None
    for face in faces:
        a, b, c = (vertices[index] for index in face)
        signed_twice_area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if signed_twice_area <= 0:
            raise AssertionError("native triangle orientation or area failed")
        for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            incidence[tuple(sorted((i, j)))] += 1
        for weights in ((1/3, 1/3, 1/3), (0.1, 0.3, 0.6), (0.6, 0.3, 0.1), (0.2, 0.6, 0.2)):
            point = tuple(sum(w*p[axis] for w, p in zip(weights, (a, b, c))) for axis in range(3))
            error = abs(point[2]-source.height_at(*point[:2]))
            if error > maximum:
                maximum, witness = error, point
    boundary = [edge for edge, count in incidence.items() if count == 1]
    degrees = Counter(index for edge in boundary for index in edge)
    if any(count not in (1, 2) for count in incidence.values()) or any(v != 2 for v in degrees.values()):
        raise AssertionError("native mesh has nonmanifold or branching boundary")
    if len(vertices)-len(incidence)+len(faces) != 1:
        raise AssertionError("native rectangular surface lost disk topology")
    if maximum > document.source_deviation_bound_mm:
        raise AssertionError("native mesh exceeds source approximation certificate")
    return {"vertices": len(vertices), "triangles": len(faces), "boundary_edges": len(boundary),
            "euler_characteristic": 1, "maximum_observed_vertical_error_mm": maximum,
            "error_witness_xyz": witness, "certified_vertical_error_mm": document.source_deviation_bound_mm,
            "scope": "positive oriented disk topology and dense barycentric source deviation; construction controls retained separately"}


def _native_fidelity(directory: Path, source: ConstructiveSource) -> dict:
    """A useful certified native projection, independent of CAM resolution."""
    nx, ny = 129, 97
    document = create_document(source, x_samples=nx, y_samples=ny)
    if document.source_deviation_bound_mm > NATIVE_FIDELITY_LIMIT_MM:
        raise AssertionError("native projection exceeds the fixed fidelity ceiling")
    with (directory / "surface-native-fidelity.cb").open("xb") as stream:
        stream.write(document.xml_bytes)
    loaded = read_document(document.xml_bytes)
    if loaded.source != source:
        raise AssertionError("high-fidelity native handoff lost constructive source")
    return {"samples": [nx, ny], "file": "surface-native-fidelity.cb",
            "native_fidelity_ceiling_mm": NATIVE_FIDELITY_LIMIT_MM,
            **_mesh_fidelity(source, document)}


def _source_and_native(directory: Path, source: ConstructiveSource) -> dict:
    (directory / "source.json").write_text(source.canonical_json()+"\n", encoding="utf-8")
    reloaded = ConstructiveSource.from_json((directory / "source.json").read_text(encoding="utf-8"))
    if reloaded != source:
        raise AssertionError("editable source JSON roundtrip changed construction")
    rows = []
    documents = []
    for nx, ny in ((25, 17), (49, 33)):
        document = create_document(source, x_samples=nx, y_samples=ny)
        (directory / f"surface-{nx}x{ny}.cb").write_bytes(document.xml_bytes)
        loaded = read_document(document.xml_bytes)
        if loaded.source != source:
            raise AssertionError("native handoff lost editable construction")
        rows.append({"samples": [nx, ny], **_mesh_fidelity(source, document)})
        documents.append(document)
    rows.append(_native_fidelity(directory, source))
    if rows[1]["certified_vertical_error_mm"] >= rows[0]["certified_vertical_error_mm"]:
        raise AssertionError("finer native projection did not improve certified fidelity")
    edited_ridge = replace(source.ridges[0], widths_mm=tuple(w*1.1 for w in source.ridges[0].widths_mm),
                           heights_mm=tuple(h*0.9 for h in source.ridges[0].heights_mm))
    edited = replace(source, ridges=(edited_ridge, *source.ridges[1:]),
                     repeat_count=source.repeat_count+1 if source.ridges[0].cap_ends else 1,
                     scale_xyz=(1.15, 0.9, 1.05))
    try:
        documents[0].check_document(current_source=edited)
    except ValueError:
        stale_rejected = True
    else:
        raise AssertionError("edited source reused stale native Surface")
    fresh = regenerate(edited, x_samples=49, y_samples=33,
                       surface_identifier=documents[0].surface_identifier)
    fresh.check_document(current_source=edited)
    if fresh.source.source_id != source.source_id:
        raise AssertionError("regeneration changed source identity")
    (directory / "edited-source.json").write_text(edited.canonical_json()+"\n", encoding="utf-8")
    (directory / "edited-regenerated.cb").write_bytes(fresh.xml_bytes)
    # Explicit repeat seam observations are independent of tessellation density.
    joins = []
    for repeat in range(1, source.repeat_count):
        x = source.origin_xy[0]+repeat*source.template_length_mm*source.scale_xyz[0]
        for y in (0.25, 0.75, 1.25):
            eps = 1e-7
            delta = abs(source.height_at(x-eps, y)-source.height_at(x+eps, y))
            if delta > 1e-6:
                raise AssertionError("authored repeat join is discontinuous")
            joins.append({"xy": [x, y], "one_sided_height_difference_mm": delta})
    return {"source_id": source.source_id, "source_fingerprint": source.fingerprint,
            "source_json_bytes": len(source.canonical_json().encode()),
            "sweep_controls": len(source.ridges), "repeat_count": source.repeat_count,
            "native_resolutions": rows, "repeat_joins": joins,
            "source_edit_stale_rejected": stale_rejected,
            "edited_fingerprint": edited.fingerprint, "regenerated_fresh": True,
            "edited_parameters": {"width_multiplier": 1.1, "height_multiplier": 0.9,
                                   "repeat_count": edited.repeat_count, "scale_xyz": edited.scale_xyz}}


def _reconstruction(directory: Path) -> dict:
    truth = lambda x: 0.7+0.08*x+0.24*math.sin(2*math.pi*x/4+0.3)
    observations = tuple((4*i/24, round(truth(4*i/24), 2)+(0.002 if i % 2 else -0.002))
                         for i in range(25))
    fit = reconstruct_spine(observations, x_interval_mm=(0, 4), residual_limit_mm=0.012)
    if fit.ambiguous or fit.selected_index is None:
        raise AssertionError("known degraded correspondence lacked identifiable candidate")
    chosen = fit.candidates[fit.selected_index]
    expected_centers = (0.7, 1.02)
    linear_error = max(abs(a-b) for a, b in zip(chosen.centers_mm, expected_centers))
    sine_error = math.hypot(chosen.wave_amplitude_mm*math.cos(chosen.wave_phase_rad)-0.24*math.cos(0.3),
                            chosen.wave_amplitude_mm*math.sin(chosen.wave_phase_rad)-0.24*math.sin(0.3))
    bound = linear_error+sine_error
    if chosen.cycles != 1 or bound > 0.01:
        raise AssertionError("reconstruction exceeded synthetic truth bound")
    ambiguous = reconstruct_spine(tuple((i/8, 1.0) for i in range(9)), x_interval_mm=(0, 1))
    if not ambiguous.ambiguous or ambiguous.recovered_author_intent:
        raise AssertionError("ambiguous observations asserted recovered author intent")
    data = {"known_truth": {"centers_mm": expected_centers, "amplitude_mm": 0.24, "cycles": 1, "phase_rad": 0.3},
            "input_degradation_bound_mm": 0.007,
            "uniform_known_truth_spine_error_bound_mm": bound,
            "correspondence_fit": asdict(fit), "ambiguous_fit": asdict(ambiguous),
            "recovered_author_intent": False,
            "scope": "finite declared linear-plus-sinusoid models with supplied X correspondence; observed residuals plus synthetic coefficient oracle"}
    (directory / "reconstruction.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    return data


def _unreachable_detail(directory: Path) -> dict:
    """Independent sharp-valley/ball geometry witness, not a routing verdict."""
    source = ConstructiveSource("unreachable-sharp-valley", 2, 1.2, 0.6, -0.3, (
        SweptRidge("left-flank", (0, 2), (0.4, 0.4), (0.4, 0.4), (0.12, 0.12),
                   profile="tent", cap_ends=False),
        SweptRidge("right-flank", (0, 2), (0.8, 0.8), (0.4, 0.4), (0.12, 0.12),
                   profile="tent", cap_ends=False),
    ))
    radius, slope = 0.12, 0.3
    valley_z = -0.24
    excess = radius*(math.sqrt(1+slope*slope)-1)
    exact = valley_z+excess
    query = contact(source, radius, 1, 0.6, tolerance_mm=0.0001)
    if not query.lower_tip_z_mm <= exact <= query.upper_tip_z_mm:
        raise AssertionError("independent V-valley ball-contact formula escaped contact interval")
    if query.lower_tip_z_mm-valley_z <= 0:
        raise AssertionError("sharp concave detail lost positive tool-limited contact residual")
    (directory / "unreachable-valley-source.json").write_text(source.canonical_json()+"\n", encoding="utf-8")
    return {"source_fingerprint": source.fingerprint, "valley_xy": [1, 0.6],
            "design_valley_z_mm": valley_z, "local_flank_slope": slope, "ball_radius_mm": radius,
            "centered_contact_exact_z_mm": exact,
            "contact_interval_z_mm": [query.lower_tip_z_mm, query.upper_tip_z_mm],
            "centered_contact_excess_mm": excess,
            "exact_vertex_access": "unreachable by a smooth ball without gouging a neighboring flank",
            "proof": "At the concave vertex z=z0+m*abs(y-y0), a differentiable ball envelope cannot simultaneously have right derivative >=m and left derivative <=-m. The centered-contact formula is exact because its witnesses lie within both linear flanks.",
            "scope": "fixed-axis ball and this declared sharp valley; design is retained, not rounded to the tool"}


def _known_best(directory: Path) -> dict:
    source = ConstructiveSource("enumerable-plane", 2, 0.5, 0.6, -0.2, ())
    candidates = {}
    for family in ("parallel", "section"):
        candidates[family] = _measure(directory / family, proposal(source, family, stepover_mm=0.25),
                                      {"flat_center": (1, 0.25)})
    expected_counts = {"parallel": math.ceil(0.5/0.25)+1, "section": math.ceil(2/0.25)+1}
    analytic_minutes = {}
    for family, count in expected_counts.items():
        along, across = (2, 0.5) if family == "parallel" else (0.5, 2)
        # Odd serpentine row counts finish at the far opposite stock corner.
        feed_cut = count*along
        vertical = 2*count*(1-(-0.197))
        rapid = across+math.hypot(2, 0.5)
        analytic_minutes[family] = feed_cut/240+vertical/120+rapid/1200
        if not math.isclose(candidates[family]["motion"]["estimated_minutes"], analytic_minutes[family], abs_tol=1e-8):
            raise AssertionError("enumerable candidate differs from independent plane motion oracle")
    best = min(analytic_minutes, key=analytic_minutes.get)
    measured = min(candidates, key=lambda key: candidates[key]["motion"]["estimated_minutes"])
    if best != "parallel" or measured != best:
        raise AssertionError("finite routing choice missed independently known best")
    return {"domain": ["parallel along length", "section across width"],
            "objective": "constant-feed minutes including safe links/entries", "completeness": "all two declared choices",
            "independent_entry_counts": expected_counts, "independent_minutes": analytic_minutes,
            "known_best": best, "measured_best": measured, "candidates": candidates}


def _rejections(job) -> dict:
    results = {}
    stage = job.stages[0]
    unsafe_depth = replace(job, stages=(replace(stage, axial_limits=AxialLimits(0.01, 0.01)),))
    index = next(i for i, move in enumerate(stage.motions) if move.role == "cut" and min(move.start[2], move.end[2]) < 0)
    moves = list(stage.motions)
    moves[index] = replace(moves[index], role="rapid", feed=0)
    unsafe_link = replace(job, stages=(replace(stage, motions=tuple(moves)),))
    edited = replace(stage.relief_operation.target, base_height_mm=-0.31)
    for label, candidate in (("excess_virgin_depth", unsafe_depth), ("below_stock_rapid", unsafe_link)):
        try:
            emit(candidate, "uccnc")
        except ValueError as exc:
            results[label] = str(exc)
        else:
            raise AssertionError(f"unsafe candidate accepted: {label}")
    try:
        check_source(job, edited)
    except ValueError as exc:
        results["source_edit_invalidates_paths"] = str(exc)
    else:
        raise AssertionError("edited source reused stale paths")
    return results


def run(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"library_version": cambam_builder.__version__, "installed_module": cambam_builder.__file__,
               "finish_ceiling_mm": FINISH_LIMIT_MM,
               "native_fidelity_ceiling_mm": NATIVE_FIDELITY_LIMIT_MM,
               "comparison_controls": {"stepover_mm": 0.15, "sample_step_mm": 0.04, "stock_pitch_mm": 0.05,
                                       "ball_radius_mm": 0.12, "contact_tolerance_mm": 0.002,
                                       "allowance_mm": 0.003, "feed_mm_min": 240, "plunge_mm_min": 120,
                                       "safe_z_mm": 1, "virgin_depth_limits_mm": 0.6},
               "scope": "original analytic design, exact emitted bytes, bounded decoded continuous motion/stock/body evidence; no physical cutting or universal optimum", "cases": {}}
    saved_jobs = {}
    for name, source in designs().items():
        case_directory = directory / name
        case_directory.mkdir()
        source_record = _source_and_native(case_directory, source)
        points = ({"broad_background": (0.1, 0.1), "narrow_between_petals": (0.5, 0.75),
                   "branch_crossing": (1.5, 0.75), "repeat_seam": (3, 0.75),
                   "sharp_descending_ridge": (0.75, source.ridges[1].center_at(0.25))}
                  if name == "frieze" else {"narrow_section": (0.15, 0.65), "broad_crown": (2.85, 0.85),
                                            "varying_curvature": (1.5, 1.0), "background": (1.5, 0.05)})
        rows = {}
        for family in ("parallel", "spine", "section"):
            job = proposal(source, family)
            saved_jobs[name, family] = job
            check_source(job, source)
            row = _measure(case_directory / family, job, points)
            rows[family] = row
            print(f"Verified {name}/{family}: {row['audit_seconds']:.1f}s audit; residual upper {row['stock']['prefixes'][-1]['maximum_residual_thickness_mm'][1]:.5f}", flush=True)
        feasible = [family for family, row in rows.items() if row["meets_finish_limit"]]
        if not feasible:
            raise AssertionError(f"no finish-feasible candidate for {name}")
        best = min(feasible, key=lambda family: rows[family]["motion"]["estimated_minutes"])
        baseline_time = rows["parallel"]["motion"]["estimated_minutes"]
        for row in rows.values():
            row["minutes_delta_from_parallel"] = row["motion"]["estimated_minutes"]-baseline_time
            row["residual_upper_delta_from_parallel_mm"] = (row["stock"]["prefixes"][-1]["maximum_residual_thickness_mm"][1]-
                                                          rows["parallel"]["stock"]["prefixes"][-1]["maximum_residual_thickness_mm"][1])
        summary["cases"][name] = {"source": source_record, "strategies": rows,
                                   "selection": {"domain": ["parallel", "spine+parallel fill", "section+transverse fill"],
                                                 "objective": "minimum estimated minutes subject to fixed global finish ceiling",
                                                 "completeness": "all three declared candidates only", "feasible": feasible, "best": best}}
    frieze_source = designs()["frieze"]
    coarse = proposal(frieze_source, "parallel", depth_cap_mm=0.05)
    finish = saved_jobs["frieze", "spine"]
    evolution = _measure(directory / "evolving-stock", sequence(coarse, finish),
                         {"branch": (1.5, 0.75), "background": (0.1, 0.1)})
    prefixes = evolution["stock"]["prefixes"]
    if prefixes[-1]["residual_volume_mm3"][1] >= prefixes[0]["residual_volume_mm3"][1] or prefixes[-1]["newly_removed_volume_lower_mm3"] <= 0:
        raise AssertionError("later finish did not improve independently replayed cumulative stock")
    summary["evolving_stock"] = evolution
    summary["rejected_candidates"] = _rejections(saved_jobs["moulding", "parallel"])
    summary["bounded_reconstruction"] = _reconstruction(directory)
    summary["unreachable_detail"] = _unreachable_detail(directory)
    summary["enumerable_known_best"] = _known_best(directory / "known-best")
    summary["elapsed_seconds"] = time.monotonic()-started
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(f"CG01 consumer complete: {directory / 'comparison.json'}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output") / f"cg01-constructive-{uuid.uuid4().hex[:8]}")
    run(parser.parse_args().output)
