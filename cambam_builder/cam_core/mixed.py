"""One synthetic RC01 plus pointed-cone sequence proving shared replay reuse."""

from dataclasses import dataclass
import hashlib

from . import rc01, replay, vcarve


SLOT_OFFSET = (50, 0)
FRAME = "rc01-cone-drawing"


def mixed_trace(program, slot_plan, *, job=rc01.Job()):
    """Place the slot beside RC01 stock and join both operations in one frame."""
    rc = rc01.replay_trace(program, job, frame=FRAME)
    cone = vcarve.replay_trace(slot_plan, frame=FRAME, offset=SLOT_OFFSET)
    source = hashlib.sha256(repr((job.fingerprint, slot_plan.slot,
                                  slot_plan.tool, SLOT_OFFSET,
                                  FRAME)).encode("utf-8")).hexdigest()
    start = rc.items[-1].position
    cone_start = cone.initial_position
    items = rc.items + (
        replay.Event("tool_change", "cone", start),
        replay.Event("spindle_start", "cone", start),
        replay.Motion("rapid", "cone", "slot", start, cone_start),
    ) + cone.items[2:]
    return replay.Trace(source, FRAME, rc.initial_position,
                        rc.operations + cone.operations, items)


@dataclass(frozen=True)
class MixedResult:
    trace: replay.Trace
    stock: replay.ReplayResult
    rc_certificate: rc01.Certificate
    slot_result: vcarve.SlotResult


def verify_mixed(program=None, slot_plan=None, *, job=rc01.Job()):
    """Keep each independent oracle while replaying their single ordered trace."""
    if program is None:
        program = rc01.generate(job)
    if slot_plan is None:
        slot_plan = vcarve.generate_slot()
    rc_certificate = rc01.verify(program, job)
    slot_result = vcarve.verify_slot(slot_plan)
    trace = mixed_trace(program, slot_plan, job=job)
    stock = replay.replay(trace, expected_source=trace.source_fingerprint)
    return MixedResult(trace, stock, rc_certificate, slot_result)
