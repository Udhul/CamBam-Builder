"""Nominal generated RC01 through the bounded split-file UCCNC profile."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path

from ..cam_core import ordered_job, rc01
from . import ordered_dialects, ordered_output


FORMAT = "rc01-uccnc-v1"
EVIDENCE_NAME = "rc01-evidence.json"
HANDOFF_NAME = "handoff.json"
DEFAULT_TRANSITION_TOKEN = "offline-synthetic-operator-state"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _source(transition_token):
    if type(transition_token) is not str or not transition_token:
        raise ValueError("declared RC01 tool handoff state required")
    source = rc01.Job()
    program = rc01.generate(source)
    operations = rc01.replay_trace(program).operations
    start = tuple(map(float, rc01.SETUP))
    stages = []
    for index, operation in enumerate(operations):
        moves = tuple(ordered_job.JobMove(
            move.role, tuple(map(float, move.start)),
            tuple(map(float, move.end)), move.feed)
            for move in program.items
            if isinstance(move, rc01.Move) and move.operation == operation.name)
        transition = None if index == 0 else ordered_job.Transition(
            "operator", "split", operation.tool.name, start, transition_token)
        stages.append(ordered_job.Stage(
            operation.name, operation.tool.name, moves,
            source.tools[index].rpm, operation=operation,
            source_revision=program.motion_fingerprint,
            transition=transition))
    job = ordered_job.Job(source.fingerprint, tuple(stages), start,
                          program_frame=source.frame)
    return source, program, job


def _decoded_rc01(source, program, job, files):
    decoded = ordered_dialects.decode(files, "uccnc",
                                      initial_work_tip=job.initial_tip)
    if len(decoded.stages) != len(job.stages):
        raise ValueError("RC01 decoded stage count differs")
    moves = iter(move for stage in decoded.stages for move in stage.moves)
    items = []
    for item in program.items:
        if not isinstance(item, rc01.Move):
            items.append(item)
            continue
        try:
            observed = next(moves)
        except StopIteration as exc:
            raise ValueError("RC01 decoded motion missing") from exc
        items.append(rc01.Move(
            item.role, item.tool,
            tuple(Fraction(str(value)) for value in observed.start),
            tuple(Fraction(str(value)) for value in observed.end),
            Fraction(str(observed.feed)), item.operation))
    if next(moves, None) is not None:
        raise ValueError("RC01 decoded motion extra")
    return rc01.Program(source.fingerprint, tuple(items), source.frame)


def _result(certificate, report):
    return {
        "status": "rc01_uccnc_offline_pass",
        "profile": report["profile"],
        "transition_evidence": report["transition_evidence"],
        "program_sha256": report["motion_equivalence"]["program_sha256"],
        "decoded_motion_fingerprint": certificate.motion_fingerprint,
        "move_count": certificate.moves,
        "rough_rest_by_depth_mm2": certificate.rough_rest_by_depth,
        "final_rest_by_depth_mm2": certificate.final_rest_by_depth,
        "rough_rest_volume_mm3": certificate.rough_rest_volume,
        "final_rest_volume_mm3": certificate.final_rest_volume,
        "area_coordinate_enclosure_mm": certificate.area_coordinate_enclosure_mm,
        "location_polygon_sagitta_mm": certificate.location_polygon_sagitta_mm,
        "location_numeric_enclosure_mm": certificate.location_numeric_enclosure_mm,
        "completion": certificate.status,
        "runtime_parity": report["runtime_parity"],
        "physical_setup": report["physical_setup"],
    }


def build_controller_bundle(directory, *,
                            transition_token=DEFAULT_TRANSITION_TOKEN):
    """Write nominal T1/T2 UCCNC files and their decoded RC01 certificate."""
    source, program, job = _source(transition_token)
    directory = Path(directory)
    ordered_output.write_bundle(directory, job, "uccnc",
                                coordinate_decimals=6)
    handoff = directory / HANDOFF_NAME
    report = ordered_output.audit_bundle(handoff, job)
    files = tuple((directory / f"stage-{i + 1}.nc").read_bytes()
                  for i in range(len(job.stages)))
    certificate = rc01.verify(_decoded_rc01(source, program, job, files),
                              source)
    result = _result(certificate, report)
    evidence = {
        "format": FORMAT,
        "source_fingerprint": source.fingerprint,
        "source_motion_fingerprint": program.motion_fingerprint,
        "ordered_job_fingerprint": job.fingerprint,
        "transition_token": transition_token,
        "handoff_sha256": _sha(handoff.read_bytes()),
        "result": result,
    }
    (directory / EVIDENCE_NAME).write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return audit_controller_bundle(directory / EVIDENCE_NAME,
                                   transition_token=transition_token)


def audit_controller_bundle(evidence_path, *,
                            transition_token=DEFAULT_TRANSITION_TOKEN):
    """Re-audit final bytes and RC01's all-height stock/process obligations."""
    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    source, program, job = _source(transition_token)
    handoff = evidence_path.parent / HANDOFF_NAME
    if (evidence.get("format") != FORMAT or
            evidence.get("source_fingerprint") != source.fingerprint or
            evidence.get("source_motion_fingerprint") !=
            program.motion_fingerprint or
            evidence.get("ordered_job_fingerprint") != job.fingerprint or
            evidence.get("transition_token") != transition_token or
            evidence.get("handoff_sha256") != _sha(handoff.read_bytes())):
        raise ValueError("stale RC01 source or controller handoff")
    report = ordered_output.audit_bundle(handoff, job)
    if report["stock_access_residual"]["status"] != "pass":
        raise ValueError("RC01 ordered stock replay did not pass")
    files = tuple((evidence_path.parent / f"stage-{i + 1}.nc").read_bytes()
                  for i in range(len(job.stages)))
    certificate = rc01.verify(_decoded_rc01(source, program, job, files),
                              source)
    result = _result(certificate, report)
    if json.loads(json.dumps(result)) != evidence.get("result"):
        raise ValueError("RC01 controller evidence differs")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or audit nominal RC01 UCCNC")
    parser.add_argument("mode", choices=("build", "audit"))
    parser.add_argument("path", help="new bundle directory or rc01-evidence.json")
    parser.add_argument("--transition-token", default=DEFAULT_TRANSITION_TOKEN)
    args = parser.parse_args()
    action = (build_controller_bundle if args.mode == "build" else
              audit_controller_bundle)
    print(json.dumps(action(args.path, transition_token=args.transition_token),
                     indent=2))
