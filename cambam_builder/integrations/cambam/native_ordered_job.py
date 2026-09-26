"""Adapt a strictly normalized native linear series to an ordered job.

The native source and Default post remain separate immutable inputs. This
adapter accepts only already parsed, replayable per-MOP safe-return stages.
"""

from dataclasses import dataclass

from ...cam_core import ordered_job, replay, v_region
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
        native_count = len(self.series.stages)
        hybrid = (len(job.stages) == native_count + 1 and native_count == 1 and
                  job.stages[-1].v_plan is not None and job.stock_present)
        if len(job.stages) != native_count and not hybrid:
            raise ValueError("native ordered stage count differs from post")
        spindle = tuple(item.rpm for item in self.series.items
                        if type(item) is PostedEvent and
                        item.kind == "spindle_start")
        if len(spindle) != native_count:
            raise ValueError("native ordered spindle stage count differs")
        for index, (stage, native) in enumerate(zip(
                job.stages[:native_count], self.series.stages)):
            posted = tuple(item for item in self.series.items
                           if type(item) is PostedMove and
                           item.operation == native.name)
            if (stage.id != native.name or stage.tool_id != native.tool or
                    stage.source_revision != self.series.evidence_fingerprint or
                    stage.operation is None or
                    stage.operation.tool.kind != "cylinder" or
                    stage.operation.tool.radius * 2 != native.diameter_mm or
                    stage.rpm != spindle[index] or
                    len(stage.motions) != len(posted)):
                raise ValueError("native ordered stage identity differs from post")
            for move, source in zip(stage.motions, posted):
                if (move.start != source.start or move.end != source.end or
                        (0 if move.role in ("rapid", "rapid_retract") else 1)
                        != source.g or
                        move.feed != (0 if source.g == 0 else source.feed)):
                    raise ValueError("native ordered motion differs from post")
        if job.stock_present:
            from .native_series_audit import _bind_source_target
            targets = tuple(stage.operation.target for stage in job.stages[:native_count])
            if any(target != targets[0] for target in targets[1:]):
                raise ValueError("native ordered stock targets differ")
            _bind_source_target(self.source_path, self.candidate_path,
                                self.series, targets[0])
            if hybrid:
                v_stage = job.stages[-1]
                shape = v_stage.v_plan.target.safe
                from shapely.geometry import Polygon
                if (v_stage.source_revision != v_stage.v_plan.fingerprint or
                        v_stage.v_plan.target.source_id !=
                        self.series.evidence_fingerprint or
                        not targets[0].region_shell or
                        targets[0].depth != v_stage.v_plan.target.cap_depth or
                        not Polygon(targets[0].region_shell,
                                    targets[0].region_holes).equals(shape)):
                    raise ValueError("generated V stage differs from native source")
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
        role = "rapid_retract" if source.g == 0 and item.role == "retract" else item.role
        if source.g != (0 if role in ("rapid", "rapid_retract") else 1):
            raise ValueError("native G mode requires an unsupported motion role")
        moves.append(ordered_job.JobMove(role, item.start, item.end,
                                         0 if source.g == 0 else item.feed))
        at = item.end
    if running or not stages or len(stages) != len(series.stages):
        raise ValueError("native series has incomplete stages")
    return ordered_job.Job(series.evidence_fingerprint, tuple(stages),
                           trace.initial_position, stock_present=stock_present,
                           program_frame=trace.frame, source_kind="native")


def from_native_v(series, plan, *, target, cutting_length_mm,
                  entry_mode="virgin", tool_id="T3", rpm=12000,
                  entry_feed=60, cut_feed=300, boundary="split"):
    """Join one posted native cylinder and a generated V finish on one source.

    The native MOP's complete posted motion establishes predecessor stock. The
    caller still supplies the independently derived source target and V plan;
    output requires a NativeBinding that rechecks the current source/post.
    """
    if type(series) is not NativeSeries or len(series.stages) != 1 or (
            type(plan) is not v_region.VPlan or type(target) is not replay.Target or
            plan.target.source_id != series.evidence_fingerprint or
            series.stages[0].target_ids != (target.name,) or
            tool_id == series.stages[0].tool):
        raise ValueError("one source-bound native predecessor and V plan required")
    native = from_native_series(
        series, targets={series.stages[0].name: target},
        cutting_lengths_mm={series.stages[0].tool: cutting_length_mm},
        entry_modes={series.stages[0].name: entry_mode}, boundary=boundary)
    first = native.stages[0]
    trace = series.to_trace({first.id: target},
                            {first.tool_id: cutting_length_mm},
                            {first.id: entry_mode})
    v_region.with_prior(plan, trace)
    start = first.motions[-1].end
    moves = v_region.complete_motion(plan, start)
    second = ordered_job.Stage(
        "v-finish", tool_id,
        tuple(ordered_job.JobMove(
            move.role, move.start, move.end,
            0 if move.role == "rapid" else
            entry_feed if move.role == "entry" else cut_feed)
            for move in moves),
        rpm, v_plan=plan, source_revision=plan.fingerprint,
        transition=ordered_job.Transition(
            "operator", boundary, tool_id, start,
            "operator-confirmed-installation"))
    return ordered_job.Job(series.evidence_fingerprint, (first, second),
                           native.initial_tip, program_frame=native.program_frame,
                           source_kind="native")
