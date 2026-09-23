"""Select RC01 stock evidence from pinned native posts or a core program.

Native evidence is a bounded geometric observation of posted T1/T2 motion. It
does not turn the RC01 Pocket/Default output into an accepted machining program.
"""

import hashlib
import json
import math
from pathlib import Path

from ...cam_core.rc01 import Job, Program, verify
from ...cambam_reader import read_cambam_bytes
from .rc01_adapter import normalize
from .rc01_native_post import _area_by_depth, _motion_findings, audit_native_posts
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
    paired = evidence.get("format") == "rc01-native-paired-stock-v1"
    if not paired and evidence.get("format") != "rc01-native-rough-stock-v1":
        raise ValueError("unsupported native stock evidence format")
    directory = path.parent
    names = [
        ("manifest", "manifest_file"), ("setup", "setup_file"),
        ("rough_post", "rough_post_file") if paired else ("post", "post_file"),
    ]
    if paired:
        names.append(("combined_post", "combined_post_file"))
    paths = {key: directory / evidence[name] for key, name in names}
    if any(paths[key].name != evidence[name] for key, name in names):
        raise ValueError("native stock files must be colocated")
    for key in paths:
        if _sha256(paths[key]) != evidence[f"{key}_sha256"]:
            raise ValueError(f"stale native {key} fingerprint")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("format") != "rc01-native-v1":
        raise ValueError("unsupported RC01 native manifest")
    paths["source"] = directory / manifest["source_file"]
    paths["candidate"] = directory / manifest["variants"]["rough"]["file"]
    if paired:
        paths["combined_candidate"] = directory / manifest["variants"]["combined"]["file"]
    if (paths["source"].name != manifest["source_file"] or
            paths["candidate"].name != manifest["variants"]["rough"]["file"] or
            (paired and paths["combined_candidate"].name !=
             manifest["variants"]["combined"]["file"])):
        raise ValueError("native source and candidate must be colocated")
    if _sha256(paths["source"]) != manifest["source_sha256"]:
        raise ValueError("stale native source fingerprint")
    if _sha256(paths["candidate"]) != manifest["variants"]["rough"]["sha256"]:
        raise ValueError("stale native candidate fingerprint")
    if paired and _sha256(paths["combined_candidate"]) != manifest["variants"]["combined"]["sha256"]:
        raise ValueError("stale native combined candidate fingerprint")
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

    The native branch accepts pinned T1 or T1/T2 Region/Pocket posts. The core
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
    paired = evidence["format"] == "rc01-native-paired-stock-v1"
    for key in (("source", "candidate", "combined_candidate") if paired else
                ("source", "candidate")):
        document = read_cambam_bytes(paths[key].read_bytes(), source_name=str(paths[key]))
        normalized = normalize(document, setup, allow_attachments=key != "source")
        if normalized != job or normalized.fingerprint != manifest["job_fingerprint"]:
            raise ValueError(f"native {key} no longer describes the RC01 job")
        if key != "source":
            variant = "combined" if key == "combined_candidate" else "rough"
            enabled = [mop.name for mop in document.list_mops() if mop.enabled]
            if enabled != manifest["variants"][variant]["enabled_mops"]:
                raise ValueError(f"native {variant} MOP selection changed")
    if paired:
        return _analyze_native_pair(evidence, manifest, paths, job)
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


def _analyze_native_pair(evidence, manifest, paths, job):
    """Keep geometric coverage separate from the Pocket/Default motion verdict."""
    if paths["setup"].name != "setup.json":
        raise ValueError("paired native audit requires colocated setup.json")
    parsed = {}
    for variant, candidate_key, post_key in (
            ("rough", "candidate", "rough_post"),
            ("combined", "combined_candidate", "combined_post")):
        post = paths[post_key].read_text(encoding="utf-8-sig")
        lines = post.splitlines()
        if (not any(line.startswith(f"( {paths[candidate_key].stem} ")
                    for line in lines[:12]) or
                "( Post processor: Default )" not in lines[:12]):
            raise ValueError(f"native {variant} post header does not match candidate/Default")
        names = manifest["variants"][variant]["enabled_mops"]
        if any(sum(line.strip() == f"( {name} )" for line in lines) != 1
               for name in names):
            raise ValueError(f"native {variant} post lacks a selected MOP section")
        items, _ = read_default_post(post, allow_arcs=True)
        parsed[variant] = items
    rough_items, combined_items = parsed["rough"], parsed["combined"]
    # Compare all parsed T1 events and moves through the final T1 move. The
    # rough-only footer may differ after that point, but T2 must inherit the
    # same actual stock state, including tool, spindle and feed history.
    def t1_prefix(items):
        t1_moves = [index for index, item in enumerate(items)
                    if item["type"] == "move" and item["tool"] == "T1"]
        if not t1_moves:
            raise ValueError("native post has no T1 roughing motion")
        end = t1_moves[-1]
        return [{key: value for key, value in item.items() if key != "line"}
                for item in items[:end + 1]]

    if t1_prefix(rough_items) != t1_prefix(combined_items):
        raise ValueError("native combined T1 prefix differs from rough-only post")
    if not any(item["type"] == "move" and item["tool"] == "T2"
               for item in combined_items):
        raise ValueError("native combined post has no T2 motion")
    for variant, items in parsed.items():
        if evidence[f"{variant}_motion_sha256"] != _motion_fingerprint(items):
            raise ValueError(f"native {variant} posted-motion interpretation changed")
    audit = audit_native_posts(paths["manifest"], paths["rough_post"],
                               paths["combined_post"])
    columns = audit["t2_vertical_columns_from_t1"]
    required = audit["required_corner_columns_from_t1"]
    access_witnessed = bool(columns) and all(
        row["exact_single_t1_cut_witness_line"] is not None
        for row in columns + required)
    coverage_met = (audit["rough_rest_budget_met"] and
                    audit["final_rest_budget_met"] and access_witnessed)
    result = {
        "authority": "native_posted",
        "status": "bounded_paired_posted_stock_observation",
        "job_fingerprint": job.fingerprint,
        "motion_fingerprint": {variant: _motion_fingerprint(items)
                               for variant, items in parsed.items()},
        "input_sha256": {key: _sha256(path) for key, path in paths.items()},
        "rough_prefix_identical": audit["rough_prefix_identical"],
        "rough_rest_by_depth": audit["rough_rest_by_depth"],
        "final_rest_by_depth": audit["final_rest_by_depth"],
        "rough_rest_budget_met": audit["rough_rest_budget_met"],
        "final_rest_budget_met": audit["final_rest_budget_met"],
        "t2_vertical_columns_from_t1": columns,
        "required_corner_columns_from_t1": required,
        "t2_vertical_access_witnessed": access_witnessed,
        "coverage_budget_met": coverage_met,
        "motion_role_issue_counts": audit["issue_counts"],
        "motion_role_issue_kinds": audit["issue_kinds"],
        "stock_dependent_use": "blocked_by_motion_or_rest"
        if audit["status"] != "bounded_checks_passed_access_unverified" or
           not coverage_met else "bounded_geometric_observation_only",
        "numeric_limit": audit["numeric_limit"],
    }
    return result
