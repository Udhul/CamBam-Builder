"""Controller output and hash-bound offline evidence for supported ordered jobs."""

import hashlib
import json
import math
from pathlib import Path

from ..cam_core import ordered_job
from . import ordered_dialects


FORMAT = "ordered-output-v1"
PROFILES = {"uccnc": "ordered-uccnc-g49-v1",
            "grbl": "ordered-grbl-v1.1-g43.1-v1"}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _profile(dialect):
    if dialect not in PROFILES:
        raise ValueError("unsupported ordered output dialect")
    return PROFILES[dialect]


def _effect(effect, stage):
    """Decode a caller-supplied host effect independently of the NC writer."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate external effect field")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("nonfinite external effect value")

    try:
        data = json.loads(effect.decode("utf-8"), object_pairs_hook=unique,
                          parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid external effect bytes") from exc
    if (type(data) is not dict or set(data) != {"version", "stage_id", "tool_id",
                                          "model", "travel_program_tip_xyz_mm"} or
            data["version"] != "ordered-effect-v1" or
            data["stage_id"] != stage.id or data["tool_id"] != stage.tool_id or
            data["model"] != stage.transition.effect_model):
        raise ValueError("unknown or mismatched external effect")
    raw = data["travel_program_tip_xyz_mm"]
    if (type(raw) is not list or len(raw) < 2 or
            any(type(point) is not list or len(point) != 3 or
                any(type(value) not in (int, float) or
                    not math.isfinite(value) for value in point)
                for point in raw)):
        raise ValueError("invalid external travel")
    points = tuple(tuple(point) for point in raw)
    if points != stage.transition.travel:
        raise ValueError("external travel differs from declared model")
    return points


def audit_files(job, dialect, files, *, effects=None,
                expected_hashes=None, fixture_top_z_mm=0,
                source_binding=None):
    """Decode every file; compare intended motion and replay decoded stock."""
    _profile(dialect)
    if type(job) is not ordered_job.Job or type(files) is not tuple or not files:
        raise ValueError("ordered job and complete file tuple required")
    if (type(fixture_top_z_mm) not in (int, float) or
            not math.isfinite(fixture_top_z_mm)):
        raise ValueError("finite fixture top required")
    if job.source_kind == "native":
        if source_binding is None:
            raise ValueError("native source freshness binding required")
        source_binding.check(job)
    elif source_binding is not None:
        raise ValueError("direct job has unexpected native source binding")
    hashes = tuple(_sha(data) for data in files)
    if expected_hashes is not None and hashes != tuple(expected_hashes):
        raise ValueError("ordered output bytes changed")
    initial = tuple(a + b for a, b in zip(job.initial_tip,
                                          job.translation_xyz_mm))
    decoded = ordered_dialects.decode(files, dialect, initial_work_tip=initial)
    for stage, read in zip(job.stages, decoded.stages):
        for move in read.transition_moves:
            physical = tuple(
                (point[0] - job.translation_xyz_mm[0],
                 point[1] - job.translation_xyz_mm[1],
                 point[2] - job.translation_xyz_mm[2] + stage.offset_mm -
                 stage.tool_length_mm)
                for point in (move.start, move.end))
            ordered_job.verify_safe_travel(physical,
                fixture_top_z_mm=fixture_top_z_mm)
    report = ordered_job.audit(job, decoded, dialect=dialect,
                               expected_fingerprint=job.fingerprint)
    effects = {} if effects is None else effects
    if type(effects) is not dict:
        raise ValueError("external effects must be keyed by stage ID")
    expected = {stage.id for stage in job.stages[1:]
                if stage.transition.actor == "synthetic_host"}
    if set(effects) != expected:
        raise ValueError("missing or unmodeled external effect")
    effect_hashes = {}
    for index, stage in enumerate(job.stages[1:], 1):
        if stage.id not in expected:
            continue
        points = _effect(effects[stage.id], stage)
        if (points[0] != job.stages[index - 1].motions[-1].end or
                points[-1] != stage.transition.resume_tip):
            raise ValueError("external travel does not join ordered stages")
        ordered_job.verify_safe_travel(points,
                                       fixture_top_z_mm=fixture_top_z_mm)
        effect_hashes[stage.id] = _sha(effects[stage.id])
    report["transition_evidence"]["external_effect_hashes"] = effect_hashes
    report["motion_equivalence"]["program_sha256"] = hashes
    report["profile"] = _profile(dialect)
    return report


def emit(job, dialect, *, effects=None, fixture_top_z_mm=0,
         source_binding=None):
    """Return program bytes and independently checked offline evidence."""
    files = ordered_dialects.render(job, dialect)
    report = audit_files(job, dialect, files, effects=effects,
                         fixture_top_z_mm=fixture_top_z_mm,
                         source_binding=source_binding)
    return files, report


def write_bundle(directory, job, dialect, *, effects=None, fixture_top_z_mm=0,
                 source_binding=None):
    """Write a new versioned bundle; audit it again from its final bytes."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError("ordered output directory must be new")
    files, _ = emit(job, dialect, effects=effects,
                    fixture_top_z_mm=fixture_top_z_mm,
                    source_binding=source_binding)
    names = tuple(f"stage-{i + 1}.nc" for i in range(len(files)))
    directory.mkdir(parents=True)
    for name, data in zip(names, files):
        (directory / name).write_bytes(data)
    effects = {} if effects is None else effects
    effect_names = {}
    for index, stage in enumerate(job.stages):
        if stage.id not in effects:
            continue
        stage_id, data = stage.id, effects[stage.id]
        name = f"effect-{index + 1}.json"
        (directory / name).write_bytes(data)
        effect_names[stage_id] = {"file": name, "sha256": _sha(data)}
    manifest = {
        "format": FORMAT, "profile": _profile(dialect), "dialect": dialect,
        "job_fingerprint": job.fingerprint,
        "source_fingerprint": job.source_fingerprint,
        "prefix_fingerprints": job.prefixes,
        "verifier": ordered_job.VERSION,
        "numerical": {"coordinate_decimals": 4,
                      "match_tolerance_mm": ordered_job.MATCH_TOLERANCE_MM,
                      "fixture_top_z_mm": fixture_top_z_mm},
        "programs": [{"stage_ids": [stage.id for stage in job.stages]
                      if dialect == "grbl" else [job.stages[i].id],
                      "file": name, "sha256": _sha(data)}
                     for i, (name, data) in enumerate(zip(names, files))],
        "effects": effect_names,
    }
    manifest["evidence"] = audit_files(
        job, dialect, files, effects=effects,
        fixture_top_z_mm=fixture_top_z_mm,
        source_binding=source_binding)
    path = directory / "handoff.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit_bundle(path, job, source_binding=source_binding)


def audit_bundle(manifest_path, job, *, source_binding=None):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dialect = manifest.get("dialect")
    if (manifest.get("format") != FORMAT or
            manifest.get("profile") != _profile(dialect) or
            manifest.get("job_fingerprint") != job.fingerprint or
            manifest.get("source_fingerprint") != job.source_fingerprint or
            tuple(manifest.get("prefix_fingerprints", ())) != job.prefixes or
            manifest.get("verifier") != ordered_job.VERSION):
        raise ValueError("stale ordered output setup or source")
    numerical = manifest.get("numerical", {})
    if (numerical.get("coordinate_decimals") != 4 or
            numerical.get("match_tolerance_mm") !=
            ordered_job.MATCH_TOLERANCE_MM or
            type(numerical.get("fixture_top_z_mm")) not in (int, float)):
        raise ValueError("ordered output numerical policy changed")
    programs = manifest.get("programs", ())
    names = tuple(f"stage-{i + 1}.nc" for i in range(
        1 if dialect == "grbl" else len(job.stages)))
    if (len(programs) != len(names) or
            tuple(row.get("file") for row in programs) != names or
            any(row.get("stage_ids") != ([s.id for s in job.stages]
                if dialect == "grbl" else [job.stages[i].id])
                for i, row in enumerate(programs))):
        raise ValueError("ordered output file or stage order changed")
    files = tuple((path.parent / name).read_bytes() for name in names)
    effects = {}
    expected = {stage.id for stage in job.stages[1:]
                if stage.transition.actor == "synthetic_host"}
    if set(manifest.get("effects", {})) != expected:
        raise ValueError("ordered effect list changed")
    for index, stage in enumerate(job.stages):
        stage_id = stage.id
        if stage_id not in expected:
            continue
        row = manifest["effects"][stage_id]
        if row.get("file") != f"effect-{index + 1}.json":
            raise ValueError("ordered effect file changed")
        data = (path.parent / row["file"]).read_bytes()
        if _sha(data) != row.get("sha256"):
            raise ValueError("ordered effect bytes changed")
        effects[stage_id] = data
    report = audit_files(job, dialect, files, effects=effects,
                         expected_hashes=tuple(row.get("sha256")
                                               for row in programs),
                         fixture_top_z_mm=numerical["fixture_top_z_mm"],
                         source_binding=source_binding)
    if json.loads(json.dumps(report)) != manifest.get("evidence"):
        raise ValueError("ordered output evidence changed")
    return {"status": "ordered_output_pass", "manifest_sha256":
            _sha(path.read_bytes()), **report}
