"""Adapt a strictly normalized native linear series to an ordered job.

The native source and Default post remain separate immutable inputs. This
adapter accepts only already parsed, replayable per-MOP safe-return stages.
"""

from dataclasses import dataclass

from ...cam_core import ordered_job, replay
from .native_series import NativeSeries, PostedEvent, PostedMove


@dataclass(frozen=True)
class NativeBinding:
    series: NativeSeries
    source_path: object
    candidate_path: object
    post_path: object
    setup: object = None

    def check(self, job):
        if (type(job) is not ordered_job.Job or job.source_kind != "native" or
                type(self.series) is not NativeSeries or
                job.source_fingerprint != self.series.evidence_fingerprint):
            raise ValueError("native ordered source binding differs")
        self.series.check_freshness(self.source_path, self.candidate_path,
                                    self.post_path, setup=self.setup)
        if len(job.stages) != len(self.series.stages):
            raise ValueError("native ordered stage count differs from post")
        for stage, native in zip(job.stages, self.series.stages):
            posted = tuple(item for item in self.series.items
                           if type(item) is PostedMove and
                           item.operation == native.name)
            if (stage.id != native.name or stage.tool_id != native.tool or
                    stage.source_revision != self.series.evidence_fingerprint or
                    stage.operation is None or
                    stage.operation.tool.kind != "cylinder" or
                    stage.operation.tool.radius * 2 != native.diameter_mm or
                    len(stage.motions) != len(posted)):
                raise ValueError("native ordered stage identity differs from post")
            for move, source in zip(stage.motions, posted):
                if (move.start != source.start or move.end != source.end or
                        (0 if move.role == "rapid" else 1) != source.g or
                        move.feed != (0 if source.g == 0 else source.feed)):
                    raise ValueError("native ordered motion differs from post")
        if job.stock_present:
            from .native_series_audit import _bind_source_target
            targets = tuple(stage.operation.target for stage in job.stages)
            if any(target != targets[0] for target in targets[1:]):
                raise ValueError("native ordered stock targets differ")
            _bind_source_target(self.source_path, self.candidate_path,
                                self.series, targets[0])
        return True


def from_native_series(series, *, targets, cutting_lengths_mm, entry_modes,
                       boundary="split", stock_present=True):
    if type(series) is not NativeSeries or boundary not in ("split", "pause"):
        raise ValueError("normalized native series and transition boundary required")
    trace = series.to_trace(targets, cutting_lengths_mm, entry_modes)
    replay.replay(trace, expected_source=series.evidence_fingerprint)
    operations = {op.name: op for op in trace.operations}
    stages = []
    active, running, rpm, at, moves = None, False, None, trace.initial_position, []
    operation = None
    # Native event and replay motion streams have matching order; use the
    # posted G mode to exclude roles this bounded controller subset cannot
    # faithfully lower (for example a low G0 cleared descent).
    for source, item in zip(series.items, trace.items):
        if type(source) is PostedEvent:
            if source.kind == "tool_change":
                if running:
                    raise ValueError("native tool change inside stage")
                active = source.tool
            elif source.kind == "spindle_start":
                if running or active != source.tool or moves:
                    raise ValueError("native stage spindle state differs")
                running, rpm = True, source.rpm
            else:
                if not running or not moves or operation is None:
                    raise ValueError("native stage lacks complete safe return")
                transition = (None if not stages else
                              ordered_job.Transition(
                                  "operator", boundary, active, moves[0].start,
                                  f"native-{operation}-installation"))
                stages.append(ordered_job.Stage(
                    operation, active, tuple(moves), rpm,
                    operation=operations[operation],
                    source_revision=series.evidence_fingerprint,
                    transition=transition))
                running, moves, operation = False, [], None
            continue
        if type(source) is not PostedMove or not running:
            raise ValueError("native motion outside spindle stage")
        if operation is None:
            operation = item.operation
        elif item.operation != operation:
            raise ValueError("native spindle stage contains multiple MOPs")
        if source.g != (0 if item.role == "rapid" else 1):
            raise ValueError("native G mode requires an unsupported motion role")
        moves.append(ordered_job.JobMove(item.role, item.start, item.end,
                                         0 if source.g == 0 else item.feed))
        at = item.end
    if running or not stages or len(stages) != len(series.stages):
        raise ValueError("native series has incomplete stages")
    return ordered_job.Job(series.evidence_fingerprint, tuple(stages),
                           trace.initial_position, stock_present=stock_present,
                           program_frame=trace.frame, source_kind="native")
