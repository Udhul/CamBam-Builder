"""Edited curved Region comparison with native and direct motion evidence.

The synthetic T1 option is an explicit proof input. Selection requires actual
Default posts for both native candidates; neither planned paths nor an MOP
setting supplies stock authority.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ..cam_core import replay, v_region
from ..cam_extensions.strategy import (Alternative, REQUIRED_GATES,
                                       ResidualBounds, StageAudit,
                                       select_strategy)
from ..native.reader import read_cambam_bytes
from .cambam import native_series, native_v_region as m3
from .cambam.rc01_post import read_default_post


ROUNDED = v_region.VProfile("rounded", 60, 0.5, 4, 3)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _semantic(data):
    project, _, _, _ = m3._source(data, "annulus", 2)
    value, has_mops = native_series._source_semantic(project, data)
    if has_mops:
        raise ValueError("M4 edited source must be MOP-free")
    return value


def render_direct(plan, prior, start):
    """Strict absolute-mm reference program for one complete T1/T3 chain."""
    moves = m3._motion_sequence(plan, start)
    return ("( M4 curved rounded reference; initial tip position declared externally )\n"
            "G21 G90 G61 G40 G17\n"
            f"T1 M6\nM3 S{m3.RPM}\n"
            + m3._combined_script(prior, moves) + "\nM5\nM30\n")


def render_endmill_direct(prior):
    """Complete T1-only reference output for the partial comparison route."""
    script = m3._script(tuple(item for item in prior.items[2:-1]
                              if type(item) is replay.Motion))
    return ("( M4 curved endmill-only reference; initial tip declared externally )\n"
            "G21 G90 G61 G40 G17\n"
            f"T1 M6\nM3 S{m3.RPM}\n" + script + "\nM5\nM30\n")


def _audit_endmill_text(text, plan, prior, start):
    if text != render_endmill_direct(prior):
        raise ValueError("M4 endmill program differs from source-bound rendering")
    actual, warnings = read_default_post(text, initial_position=start)
    if warnings or len(actual) != len(prior.items):
        raise ValueError("M4 endmill direct stream incomplete or unsupported")
    observed = []
    for index, (want, got) in enumerate(zip(prior.items, actual)):
        if type(want) is replay.Event:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or
                    tuple(got["position"]) != want.position or
                    (want.kind == "spindle_start" and got["rpm"] != m3.RPM)):
                raise ValueError(f"M4 endmill event {index} differs")
            observed.append(replace(want, position=tuple(got["position"])))
        else:
            if (got["type"] != "move" or got["tool"] != want.tool or
                    got["g"] != (0 if want.role == "rapid" else 1) or
                    got["feed"] != want.feed or
                    tuple(got["start"]) != want.start or
                    tuple(got["end"]) != want.end):
                raise ValueError(f"M4 endmill move {index} differs")
            observed.append(replace(want, start=tuple(got["start"]),
                                    end=tuple(got["end"]), feed=got["feed"]))
    posted = replace(prior, items=tuple(observed))
    rest = v_region.with_prior(plan, posted)
    area = v_region.section_report(rest, 1, final=False)
    volume = v_region.volume_bounds(rest, final=False)
    return {"status": "bounded_m4_endmill_direct_pass",
            "item_count": len(actual), "section_1_mm2": area,
            "volume_mm3": volume,
            "motion_fingerprint": posted.motion_fingerprint}


def _audit_direct_text(text, plan, prior, start):
    if text != render_direct(plan, prior, start):
        raise ValueError("M4 direct program differs from source-bound rendering")
    actual, warnings = read_default_post(text, initial_position=start)
    if warnings:
        raise ValueError(f"M4 direct parser warning: {warnings[0]}")
    expected = list(prior.items) + [
        replay.Event("tool_change", "T3", start),
        replay.Event("spindle_start", "T3", start),
        *m3._motion_sequence(plan, start),
        replay.Event("spindle_stop", "T3", start)]
    if len(actual) != len(expected):
        raise ValueError("M4 direct program has missing or extra items")
    observed_prior = []
    for index, (want, got) in enumerate(zip(expected, actual)):
        if type(want) is replay.Event:
            if (got["type"] != "event" or got["kind"] != want.kind or
                    got["tool"] != want.tool or
                    tuple(got["position"]) != want.position or
                    (want.kind == "spindle_start" and got["rpm"] != m3.RPM)):
                raise ValueError(f"M4 direct event {index} differs")
            if index < len(prior.items):
                observed_prior.append(replace(want,
                                              position=tuple(got["position"])))
            continue
        is_prior = type(want) is replay.Motion
        tool = want.tool if is_prior else "T3"
        feed = (want.feed if is_prior else 0 if want.role == "rapid" else
                m3.ENTRY_FEED if want.role == "entry" else m3.CUT_FEED)
        if (got["type"] != "move" or got["tool"] != tool or
                got["g"] != (0 if want.role == "rapid" else 1) or
                got["feed"] != feed or tuple(got["start"]) != want.start or
                tuple(got["end"]) != want.end):
            raise ValueError(f"M4 direct motion {index} differs")
        if is_prior:
            observed_prior.append(replace(want, start=tuple(got["start"]),
                                          end=tuple(got["end"]), feed=got["feed"]))
    posted_prior = replace(prior, items=tuple(observed_prior))
    rest = v_region.with_prior(plan, posted_prior)
    prior_area = v_region.section_report(rest, 1, final=False)
    final_area = v_region.section_report(rest, 1)
    prior_volume = v_region.volume_bounds(rest, final=False)
    final_volume = v_region.volume_bounds(rest)
    if final_area[2] >= 1e-7 or final_area[1] > 2 or final_volume[1] > 80:
        raise ValueError("M4 direct stock, area or volume gate failed")
    return {"status": "bounded_m4_direct_pass", "item_count": len(actual),
            "prior_cuts": len(rest.prior_stock.cuts),
            "prior_section_1_mm2": prior_area,
            "section_1_mm2": final_area,
            "prior_volume_mm3": prior_volume,
            "volume_mm3": final_volume,
            "prior_motion_fingerprint": posted_prior.motion_fingerprint}


def build_comparison(directory, *, source_path, case="annulus",
                     prior_path=None, synthetic_prior=False):
    """Prepare two complete routes from one reopened native source."""
    if case != "annulus":
        raise ValueError("M4 bounded comparison currently requires an annulus")
    if (prior_path is None) == (synthetic_prior is False):
        raise ValueError("provide one supplied prior or explicit synthetic proof input")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("M4 output directory must be new")
    source_data = Path(source_path).read_bytes()
    source_semantic = _semantic(source_data)
    directory.mkdir(parents=True)
    (directory / "source.cb").write_bytes(source_data)
    plan, start = m3._plan(source_data, case, ROUNDED, 2, 1)
    supplied = (None if synthetic_prior else
                json.loads(Path(prior_path).read_text(encoding="utf-8")))
    prior = m3._prior_trace(source_data, plan, start, supplied)
    prior_data = (json.dumps(m3._supplied_prior(source_data, plan, prior, start),
                             indent=2) + "\n").encode("utf-8")
    (directory / "prior.json").write_bytes(prior_data)
    endmill_text = render_endmill_direct(prior)
    endmill_result = _audit_endmill_text(endmill_text, plan, prior, start)
    endmill_file = directory / "endmill-direct.nc"
    endmill_file.write_bytes(endmill_text.encode("ascii"))
    jobs = {}
    for fill in ("raster", "offset"):
        folder = directory / fill
        native = m3.build_workflow(folder, case=case, tool=ROUNDED,
            source_path=directory / "source.cb",
            prior_path=directory / "prior.json", fill_pattern=fill)
        path_plan, path_start = m3._plan(source_data, case, ROUNDED, 2, 1,
                                         fill)
        path_prior = m3._prior_trace(source_data, path_plan, path_start,
            json.loads(prior_data))
        text = render_direct(path_plan, path_prior, path_start)
        direct_report = _audit_direct_text(text, path_plan, path_prior,
                                           path_start)
        direct_file = folder / "direct-reference.nc"
        direct_file.write_bytes(text.encode("ascii"))
        jobs[fill] = {"native_manifest_sha256": _sha(
            (folder / "expected-motion.json").read_bytes()),
            "direct_sha256": _sha(direct_file.read_bytes()),
            "plan_fingerprint": path_plan.fingerprint,
            "direct_result": direct_report,
            "native_post_status": native["post_status"]}
    manifest = {"format": "m4-edited-curved-v1", "case": case,
                "source_sha256": _sha(source_data),
                "source_semantic_sha256": source_semantic,
                "prior_sha256": _sha(prior_data),
                "prior_motion_fingerprint": prior.motion_fingerprint,
                "endmill_direct_sha256": _sha(endmill_file.read_bytes()),
                "endmill_direct_result": endmill_result,
                "synthetic_prior": synthetic_prior,
                "initial_xyz_mm": start, "jobs": jobs,
                "selection_status": "pending_actual_CamBam_posts"}
    (directory / "comparison.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load(manifest_path, current_source_path=None):
    root = Path(manifest_path).parent
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("format") != "m4-edited-curved-v1":
        raise ValueError("not an M4 edited curved comparison")
    source = (root / "source.cb").read_bytes()
    prior_data = (root / "prior.json").read_bytes()
    if (_sha(source) != manifest["source_sha256"] or
            _sha(prior_data) != manifest["prior_sha256"] or
            _semantic(source) != manifest["source_semantic_sha256"]):
        raise ValueError("M4 source or prior evidence changed")
    if current_source_path is not None:
        current = Path(current_source_path).read_bytes()
        if _semantic(current) != manifest["source_semantic_sha256"]:
            raise ValueError("M4 source geometry or stock edit invalidated evidence")
    return root, manifest, source, json.loads(prior_data)


def audit_direct(manifest_path, fill, *, current_source_path=None):
    root, manifest, source, supplied = _load(manifest_path,
                                             current_source_path)
    if fill not in ("raster", "offset"):
        raise ValueError("unknown M4 fill")
    folder = root / fill
    native_file = folder / "expected-motion.json"
    direct_file = folder / "direct-reference.nc"
    if (_sha(native_file.read_bytes()) != manifest["jobs"][fill][
            "native_manifest_sha256"] or
            _sha(direct_file.read_bytes()) != manifest["jobs"][fill][
            "direct_sha256"] or
            (folder / "source.cb").read_bytes() != source or
            json.loads((folder / "prior.json").read_text(encoding="utf-8"))
            != supplied):
        raise ValueError("M4 direct or native evidence changed")
    plan, start = m3._plan(source, manifest["case"], ROUNDED, 2, 1, fill)
    prior = m3._prior_trace(source, plan, start, supplied)
    if (plan.fingerprint != manifest["jobs"][fill]["plan_fingerprint"] or
            prior.motion_fingerprint != manifest["prior_motion_fingerprint"]):
        raise ValueError("M4 source-bound plan changed")
    report = _audit_direct_text(direct_file.read_text(encoding="ascii"),
                                plan, prior, start)
    report = json.loads(json.dumps(report))
    if report != manifest["jobs"][fill]["direct_result"]:
        raise ValueError("M4 direct stock result changed")
    return {"direct_sha256": manifest["jobs"][fill]["direct_sha256"],
            **report}


def audit_endmill_direct(manifest_path, *, current_source_path=None):
    root, manifest, source, supplied = _load(manifest_path,
                                             current_source_path)
    path = root / "endmill-direct.nc"
    if _sha(path.read_bytes()) != manifest["endmill_direct_sha256"]:
        raise ValueError("M4 endmill direct file changed")
    plan, start = m3._plan(source, manifest["case"], ROUNDED, 2, 1)
    prior = m3._prior_trace(source, plan, start, supplied)
    report = json.loads(json.dumps(_audit_endmill_text(
        path.read_text(encoding="ascii"), plan, prior, start)))
    if report != manifest["endmill_direct_result"]:
        raise ValueError("M4 endmill direct stock result changed")
    return {"direct_sha256": manifest["endmill_direct_sha256"], **report}


def _bounds(report, *, final):
    area = report["section_1_mm2" if final else "prior_section_1_mm2"]
    volume = report["volume_mm3" if final else "prior_volume_mm3"]
    return ResidualBounds(area[0], area[1], volume[0], volume[1])


def audit_comparison(manifest_path, *, current_source_path=None,
                     raster_post=None, offset_post=None,
                     raster_preview=None, offset_preview=None,
                     manual_choice=None):
    """Select only routes whose complete native post and direct file both pass."""
    root, manifest, _, _ = _load(manifest_path, current_source_path)
    reports = {}
    routes = {}
    source_key = manifest["source_semantic_sha256"]
    endmill_direct = audit_endmill_direct(manifest_path,
                                         current_source_path=current_source_path)
    for fill, post, preview in (("raster", raster_post, raster_preview),
                                ("offset", offset_post, offset_preview)):
        try:
            direct = audit_direct(manifest_path, fill,
                                  current_source_path=current_source_path)
        except (ValueError, OSError) as exc:
            direct = {"status": "unverified", "reason": str(exc)}
        try:
            native = (m3.audit_post(root / fill / "expected-motion.json", post)
                      if post is not None else {"status": "missing_actual_post"})
        except (ValueError, OSError) as exc:
            native = {"status": "unverified", "reason": str(exc)}
        try:
            visible = (m3.audit_preview(root / fill / "expected-motion.json",
                                        preview) if preview is not None else
                       {"status": "missing_actual_preview"})
        except (ValueError, OSError) as exc:
            visible = {"status": "unverified", "reason": str(exc)}
        reports[fill] = {"direct": direct, "native": native,
                         "preview": visible}
        if (direct["status"] != "bounded_m4_direct_pass" or
                native["status"] != "bounded_m3_v_post_pass" or
                visible["status"] != "m3_v_preview_centerlines_match"):
            routes[fill] = Alternative("rounded_" + fill, ())
            continue
        if (tuple(native["prior_section_1_mm2"]) !=
                tuple(direct["prior_section_1_mm2"]) or
                tuple(native["section_1_mm2"]) !=
                tuple(direct["section_1_mm2"])):
            raise ValueError("M4 native and direct stock results differ")
        emitted = _sha((native["post_sha256"] + direct["direct_sha256"])
                       .encode("ascii"))
        prior = StageAudit("T1 endmill prior", "framework", source_key,
            source_key, manifest["prior_motion_fingerprint"],
            _sha((emitted + direct["prior_motion_fingerprint"])
                 .encode("ascii")), _bounds(direct, final=False),
            REQUIRED_GATES)
        finish = StageAudit("T3 rounded " + fill, "framework", source_key,
            prior.chain_fingerprint, manifest["jobs"][fill]["plan_fingerprint"],
            _sha((emitted + manifest["jobs"][fill]["plan_fingerprint"])
                 .encode("ascii")),
            _bounds(direct, final=True), REQUIRED_GATES)
        routes[fill] = Alternative("rounded_" + fill, (prior, finish))
    prior_route = StageAudit("T1 endmill only", "framework", source_key,
        source_key, manifest["prior_motion_fingerprint"],
        _sha((endmill_direct["direct_sha256"] +
              endmill_direct["motion_fingerprint"]).encode("ascii")),
        ResidualBounds(endmill_direct["section_1_mm2"][0],
            endmill_direct["section_1_mm2"][1],
            endmill_direct["volume_mm3"][0],
            endmill_direct["volume_mm3"][1]), REQUIRED_GATES)
    endmill = Alternative("endmill_only", (prior_route,))
    alternatives = (endmill, routes["raster"], routes["offset"])
    selection = select_strategy(source_key, alternatives, max_area_mm2=2,
        max_volume_mm3=80, tie_order=("rounded_offset", "rounded_raster",
                                      "endmill_only"), manual_choice=manual_choice)
    return {"status": selection.status, "chosen": selection.chosen,
            "source_semantic_sha256": source_key,
            "assessments": [{"name": a.name, "status": a.status,
                "reasons": a.reasons, "stage_residuals": [
                    {"name": name, "area_mm2": (bounds.area_lower_mm2,
                     bounds.area_upper_mm2), "volume_mm3": (
                     bounds.volume_lower_mm3, bounds.volume_upper_mm3)}
                    for name, bounds in a.stage_residuals]}
                for a in selection.assessments],
            "endmill_direct": endmill_direct, "reports": reports}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M4 edited curved comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("directory")
    build.add_argument("--source", required=True)
    build.add_argument("--prior")
    build.add_argument("--synthetic-prior", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("manifest")
    audit.add_argument("--source")
    audit.add_argument("--raster-post")
    audit.add_argument("--offset-post")
    audit.add_argument("--raster-preview")
    audit.add_argument("--offset-preview")
    audit.add_argument("--manual-choice", choices=("endmill_only",
        "rounded_raster", "rounded_offset"))
    args = parser.parse_args()
    if args.command == "build":
        result = build_comparison(args.directory, source_path=args.source,
            prior_path=args.prior, synthetic_prior=args.synthetic_prior)
    else:
        result = audit_comparison(args.manifest,
            current_source_path=args.source, raster_post=args.raster_post,
            offset_post=args.offset_post, raster_preview=args.raster_preview,
            offset_preview=args.offset_preview,
            manual_choice=args.manual_choice)
    print(json.dumps(result, indent=2))
