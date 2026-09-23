"""Select RC01 stock evidence from a pinned native post or a core program.

Native evidence is a bounded geometric observation of posted T1 motion. It does
not turn the RC01 Pocket/Default output into an accepted machining program.
"""

import hashlib
import json
import math
from pathlib import Path

from ...cam_core.rc01 import Job, Program, verify
from ...cambam_reader import read_cambam_bytes
from .rc01_adapter import normalize
from .rc01_native_post import _area_by_depth, _motion_findings
from .rc01_post import read_default_post


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _motion_fingerprint(items):
    stable = [{key: value for key, value in item.items() if key != "line"}
              for item in items]
    data = json.dumps(stable, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def _native_paths(evidence_path):
    path = Path(evidence_path)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("format") != "rc01-native-rough-stock-v1":
        raise ValueError("unsupported native stock evidence format")
    directory = path.parent
    paths = {key: directory / evidence[name] for key, name in (
        ("manifest", "manifest_file"), ("setup", "setup_file"),
        ("post", "post_file"))}
    if any(paths[key].name != evidence[name] for key, name in (
            ("manifest", "manifest_file"), ("setup", "setup_file"),
            ("post", "post_file"))):
        raise ValueError("native stock files must be colocated")
    for key in paths:
        if _sha256(paths[key]) != evidence[f"{key}_sha256"]:
            raise ValueError(f"stale native {key} fingerprint")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("format") != "rc01-native-v1":
        raise ValueError("unsupported RC01 native manifest")
    paths["source"] = directory / manifest["source_file"]
    paths["candidate"] = directory / manifest["variants"]["rough"]["file"]
    if (paths["source"].name != manifest["source_file"] or
            paths["candidate"].name != manifest["variants"]["rough"]["file"]):
        raise ValueError("native source and candidate must be colocated")
    if _sha256(paths["source"]) != manifest["source_sha256"]:
        raise ValueError("stale native source fingerprint")
    if _sha256(paths["candidate"]) != manifest["variants"]["rough"]["sha256"]:
        raise ValueError("stale native candidate fingerprint")
    paths["evidence"] = path
    return evidence, manifest, paths


def check_native_freshness(evidence_path, result):
    """Reject an earlier result if any input or posted bytes have changed."""
    if result.get("authority") != "native_posted":
        raise ValueError("result is not native posted-motion evidence")
    _, _, paths = _native_paths(evidence_path)
    actual = {key: _sha256(path) for key, path in paths.items()}
    if actual != result.get("input_sha256"):
        raise ValueError("native stock result belongs to different evidence bytes")
    return True


def analyze_rc01_stock(authority, *, evidence_path=None, program=None):
    """Replay one explicitly selected RC01 motion authority.

    The native branch accepts only the pinned T1 Region/Pocket post. The core
    branch verifies a supplied complete RC01 Program independently. Neither
    branch falls back to the other when evidence is missing or stale.
    """
    job = Job()
    if authority == "framework_generated":
        if type(program) is not Program or evidence_path is not None:
            raise ValueError("framework authority requires only an RC01 Program")
        certificate = verify(program, job)
        return {
            "authority": authority, "status": certificate.status,
            "job_fingerprint": certificate.fingerprint,
            "motion_fingerprint": certificate.motion_fingerprint,
            "rough_rest_by_depth_mm2": [list(row) for row in
                                         certificate.rough_rest_by_depth],
            "final_rest_by_depth_mm2": [list(row) for row in
                                         certificate.final_rest_by_depth],
            "stock_dependent_use": "bounded_core_certificate",
        }
    if authority != "native_posted":
        raise ValueError("unknown RC01 stock authority")
    if evidence_path is None or program is not None:
        raise ValueError("native authority requires only posted evidence")
    evidence, manifest, paths = _native_paths(evidence_path)
    setup = json.loads(paths["setup"].read_text(encoding="utf-8"))
    for key in ("source", "candidate"):
        document = read_cambam_bytes(paths[key].read_bytes(), source_name=str(paths[key]))
        normalized = normalize(document, setup, allow_attachments=key == "candidate")
        if normalized != job or normalized.fingerprint != manifest["job_fingerprint"]:
            raise ValueError(f"native {key} no longer describes the RC01 job")
        if key == "candidate":
            enabled = [mop.name for mop in document.list_mops() if mop.enabled]
            if enabled != manifest["variants"]["rough"]["enabled_mops"]:
                raise ValueError("native roughing MOP selection changed")
    post = paths["post"].read_text(encoding="utf-8-sig")
    header = post.splitlines()[:12]
    if (not any(line.startswith(f"( {paths['candidate'].stem} ") for line in header)
            or "( Post processor: Default )" not in header):
        raise ValueError("native post header does not match candidate/Default")
    name = manifest["variants"]["rough"]["enabled_mops"][0]
    if sum(line.strip() == f"( {name} )" for line in post.splitlines()) != 1:
        raise ValueError("native post lacks the selected roughing section")
    items, warnings = read_default_post(post, allow_arcs=True)
    moves = [item for item in items if item["type"] == "move"]
    if not moves or any(item["tool"] != "T1" for item in items):
        raise ValueError("native roughing post has missing or unexpected tools")
    rest = _area_by_depth(moves, job, 3)
    findings = _motion_findings(items, warnings, job)
    ideal_corner_rest = (4 - math.pi) * 9
    rest_budget_met = all(
        row["rest_area_mm2"][1] <= ideal_corner_rest + 0.5 and
        row["residual_outside_ideal_or_boundary_0_05mm_mm2"] <= 1e-7
        for row in rest)
    result = {
        "authority": authority,
        "status": "bounded_posted_stock_observation",
        "job_fingerprint": job.fingerprint,
        "motion_fingerprint": _motion_fingerprint(items),
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "rough_rest_by_depth": rest,
        "rough_rest_budget_met": rest_budget_met,
        "final_rest_by_depth": None,
        "motion_role_findings": findings,
        "stock_dependent_use": "blocked_by_motion_or_rest" if findings or
                               not rest_budget_met else "bounded_geometric_observation",
        "numeric_limit": "GEOS floating topology is not an interval proof",
    }
    if evidence.get("motion_sha256") != result["motion_fingerprint"]:
        raise ValueError("native posted-motion interpretation changed")
    return result
