"""Small, controller-neutral ordered machining job and decoded-motion audit.

The supported stock evaluators are cylindrical endmill replay and one terminal
rounded-V finish after a single cylindrical predecessor. Other stage mixes
remain explicit capability limits; source/posted-motion comparison still works.
"""

from dataclasses import dataclass, replace
import hashlib
import math

from . import replay, v_region


VERSION = "ordered-job-v1"
MATCH_TOLERANCE_MM = 0.000051


def _xyz(value):
    if (type(value) is not tuple or len(value) != 3 or
            any(type(n) not in (int, float) or not math.isfinite(n)
                for n in value)):
        raise ValueError("finite XYZ tuple required")
    return value


def _near(a, b):
    return all(abs(x - y) <= MATCH_TOLERANCE_MM for x, y in zip(a, b))


def _hash(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def verify_safe_travel(points, *, fixture_top_z_mm=0):
    """Check continuous straight travel over the supported flat stock/fixture."""
    if (type(points) is not tuple or len(points) < 2 or
            type(fixture_top_z_mm) not in (int, float) or
            not math.isfinite(fixture_top_z_mm)):
        raise ValueError("finite flat-fixture travel required")
    for point in points:
        _xyz(point)
        if point[2] <= max(0, fixture_top_z_mm):
            raise ValueError("travel contacts stock or fixture")
    if any(a == b for a, b in zip(points, points[1:])):
        raise ValueError("zero-length external travel")
    return True


@dataclass(frozen=True)
class JobMove:
    role: str
    start: tuple
    end: tuple
    feed: float = 0.0

    def __post_init__(self):
        if self.role not in ("rapid", "rapid_retract", "approach", "entry",
                            "cleared_descent", "cut", "retract"):
            raise ValueError("unsupported ordered motion role")
        _xyz(self.start)
        _xyz(self.end)
        if (self.start == self.end or type(self.feed) not in (int, float) or
                not math.isfinite(self.feed) or
                (self.feed != 0 if self.role in ("rapid", "rapid_retract")
                 else self.feed <= 0)):
            raise ValueError("invalid ordered motion/feed")


@dataclass(frozen=True)
class Transition:
    actor: str                         # operator or synthetic_host
    boundary: str                      # split or pause
    installed_tool: str
    resume_tip: tuple                  # program-frame tip after external action
    completion_token: str              # caller assertion identifier, not telemetry
    travel: tuple = ()                 # modeled physical-tip XYZ polyline
    effect_model: str = "none"

    def __post_init__(self):
        if (self.actor not in ("operator", "synthetic_host") or
                self.boundary not in ("split", "pause") or
                not self.installed_tool or not self.completion_token or
                not self.effect_model or type(self.travel) is not tuple):
            raise ValueError("invalid ordered transition")
        _xyz(self.resume_tip)
        for point in self.travel:
            _xyz(point)
        if self.actor == "synthetic_host" and (
                len(self.travel) < 2 or self.effect_model == "none"):
            raise ValueError("automatic transition needs modeled external travel")


@dataclass(frozen=True)
class Stage:
    id: str
    tool_id: str
    motions: tuple
    rpm: float
    offset_mm: float = 0.0
    operation: object = None          # replay.Operation for cylindrical stock
    v_plan: object = None             # v_region.VPlan for terminal V finish
    source_revision: str = ""
    transition: object = None
    tool_length_mm: float = 0.0

    def __post_init__(self):
        if (type(self.id) is not str or not self.id or
                type(self.tool_id) is not str or not self.tool_id or
                type(self.source_revision) is not str or
                not self.source_revision or
                type(self.motions) is not tuple or
                not self.motions or any(type(m) is not JobMove for m in self.motions)
                or type(self.rpm) not in (int, float) or not math.isfinite(self.rpm)
                or self.rpm <= 0 or any(type(n) not in (int, float) or
                not math.isfinite(n) for n in (self.offset_mm, self.tool_length_mm))
                or (self.operation is not None and self.v_plan is not None)
                or (self.transition is not None and
                    type(self.transition) is not Transition)):
            raise ValueError("invalid ordered stage")
        if (self.operation is not None and
                (type(self.operation) is not replay.Operation or
                 self.operation.name != self.id or
                 self.operation.tool.name != self.tool_id)):
            raise ValueError("stage endmill operation identity differs")
        if self.v_plan is not None and type(self.v_plan) is not v_region.VPlan:
            raise ValueError("stage V plan required")
        at = self.motions[0].start
        for motion in self.motions:
            if motion.start != at:
                raise ValueError("stage motion discontinuity")
            at = motion.end
        if self.motions[0].start[2] <= 0 or at[2] <= 0:
            raise ValueError("stage needs safe start and end tips")


@dataclass(frozen=True)
class Job:
    source_fingerprint: str
    stages: tuple
    initial_tip: tuple
    translation_xyz_mm: tuple = (0, 0, 0)
    program_frame: str = "program"
    work_frame: str = "G54"
    stock_present: bool = True
    units: str = "mm"
    source_kind: str = "direct"

    def __post_init__(self):
        if (not self.source_fingerprint or type(self.stages) is not tuple or
                not self.stages or any(type(s) is not Stage for s in self.stages)
                or len({s.id for s in self.stages}) != len(self.stages) or
                not self.program_frame or self.work_frame != "G54" or
                type(self.stock_present) is not bool or self.units != "mm" or
                self.source_kind not in ("direct", "native")):
            raise ValueError("invalid ordered job")
        _xyz(self.initial_tip)
        _xyz(self.translation_xyz_mm)
        at = self.initial_tip
        for index, stage in enumerate(self.stages):
            if stage.motions[0].start != at:
                raise ValueError("ordered stage initial tip differs")
            if index and (stage.transition is None or
                          stage.transition.installed_tool != stage.tool_id or
                          stage.transition.resume_tip != at):
                raise ValueError("missing or mismatched ordered transition")
            if index == 0 and stage.transition is not None:
                raise ValueError("first stage has unexpected transition")
            at = stage.motions[-1].end

    @property
    def fingerprint(self):
        return _hash((VERSION, self.source_fingerprint, self.stages,
                      self.initial_tip, self.translation_xyz_mm,
                      self.program_frame, self.work_frame, self.stock_present,
                      self.units, self.source_kind, MATCH_TOLERANCE_MM))

    @property
    def prefixes(self):
        return tuple(_hash((VERSION, self.source_fingerprint,
                            self.stages[:n], self.initial_tip,
                            self.translation_xyz_mm, self.stock_present,
                            self.source_kind))
                     for n in range(1, len(self.stages) + 1))


def from_prior_v(plan, prior, start, *, tool_id="T3", rpm=12000,
                 entry_feed=60, cut_feed=300, translation_xyz_mm=(0, 0, 0),
                 boundary="split"):
    """Adapt existing verified endmill/V values without choosing a strategy."""
    if (type(prior) is not replay.Trace or type(plan) is not v_region.VPlan or
            len(prior.operations) != 1 or prior.initial_position != start or
            prior.source_fingerprint != plan.target.source_id):
        raise ValueError("incompatible endmill/V source")
    v_region.with_prior(plan, prior)
    raw = tuple(item for item in prior.items if type(item) is replay.Motion)
    if not raw:
        raise ValueError("missing prior endmill motion")
    first = Stage(prior.operations[0].name, prior.operations[0].tool.name,
                  tuple(JobMove(m.role, m.start, m.end,
                                0 if m.role == "rapid" else m.feed)
                        for m in raw), rpm, operation=prior.operations[0],
                  source_revision=prior.source_fingerprint)
    motion = v_region.complete_motion(plan, start)
    second = Stage("v-finish", tool_id,
                   tuple(JobMove(m.role, m.start, m.end,
                                 0 if m.role == "rapid" else
                                 entry_feed if m.role == "entry" else cut_feed)
                         for m in motion), rpm, v_plan=plan,
                   source_revision=plan.fingerprint,
                   transition=Transition("operator", boundary, tool_id, start,
                                         "operator-confirmed-installation"))
    return Job(plan.target.source_id, (first, second), start,
               translation_xyz_mm, program_frame=prior.frame)


def _observed_stage(stage, decoded, translation):
    if (decoded.tool_id != stage.tool_id or decoded.rpm != stage.rpm or
            decoded.offset_mm != stage.offset_mm or
            len(decoded.moves) != len(stage.motions)):
        raise ValueError("decoded stage tool, spindle, offset or length differs")
    actual = []
    for index, (got, want) in enumerate(zip(decoded.moves, stage.motions)):
        start = tuple(round(a - b, 7) for a, b in zip(got.start, translation))
        end = tuple(round(a - b, 7) for a, b in zip(got.end, translation))
        if (got.g != (0 if want.role in ("rapid", "rapid_retract") else 1) or
                abs(got.feed - want.feed) > 1e-9 or
                not _near(start, want.start) or not _near(end, want.end)):
            raise ValueError(f"decoded stage {stage.id} motion {index} differs")
        # The decoded feed is checked above. Retain the resolved numeric type
        # so a legacy trace fingerprint does not change for 60 versus 60.0.
        actual.append(JobMove(want.role, start, end, want.feed))
    if not _near(actual[-1].end, stage.motions[-1].end):
        raise ValueError("decoded stage lacks safe return")
    return tuple(actual)


def _replay_endmills(job, stages, motions):
    operations, items, active, at = [], [], None, job.initial_tip
    results = []
    for stage, observed in zip(stages, motions):
        operations.append(stage.operation)
        if active != stage.tool_id:
            items.append(replay.Event("tool_change", stage.tool_id, at))
            active = stage.tool_id
        items.append(replay.Event("spindle_start", stage.tool_id, at))
        for motion in observed:
            role = "retract" if motion.role == "rapid_retract" else motion.role
            items.append(replay.Motion(role, stage.tool_id, stage.id,
                                       motion.start, motion.end, motion.feed))
            at = motion.end
        items.append(replay.Event("spindle_stop", stage.tool_id, at))
        trace = replay.Trace(job.source_fingerprint, job.program_frame,
                             job.initial_tip, tuple(operations), tuple(items))
        results.append((trace, replay.replay(
            trace, expected_source=job.source_fingerprint)))
    return results


def _decoded_v_plan(plan, motions):
    """Rebuild V paths from decoded coordinates before stock calculations."""
    intended = v_region.complete_motion(plan, motions[0].start)
    if len(motions) != len(intended):
        raise ValueError("decoded V motion count differs")
    prefix = int(intended[0].role == "rapid")
    middle = motions[prefix:prefix + len(plan.motions)]
    paths, points = [], None
    for move in middle:
        if move.role == "rapid":
            if points is not None:
                raise ValueError("V rapid inside path")
        elif move.role == "entry":
            if points is not None:
                raise ValueError("V entry before retract")
            points = [(move.end[0], move.end[1], -move.end[2])]
        elif move.role == "cut":
            if points is None:
                raise ValueError("V cut without entry")
            points.append((move.end[0], move.end[1], -move.end[2]))
        elif move.role == "retract":
            if points is None or len(points) < 2 or len(paths) >= len(plan.paths):
                raise ValueError("V retract without matching path")
            paths.append(v_region.VPath(plan.paths[len(paths)].role,
                                        tuple(points)))
            points = None
    if points is not None or len(paths) != len(plan.paths):
        raise ValueError("decoded V path count differs")
    decoded = replace(plan, paths=tuple(paths),
                      motions=tuple(v_region.VMotion(m.role, m.start, m.end)
                                    for m in middle))
    v_region.verify(decoded)
    return decoded


def audit(job, decoded, *, dialect, expected_fingerprint=None):
    """Compare independent decoded stages, then replay their actual coordinates."""
    if type(job) is not Job or (expected_fingerprint is not None and
                                expected_fingerprint != job.fingerprint):
        raise ValueError("stale ordered job evidence")
    if dialect not in ("uccnc", "grbl"):
        raise ValueError("unsupported decoded dialect")
    if len(decoded.stages) != len(job.stages):
        raise ValueError("missing or extra decoded stage")
    actual = []
    work_tip = tuple(a + b for a, b in zip(job.initial_tip,
                                            job.translation_xyz_mm))
    old_offset = 0.0
    for index, (stage, read) in enumerate(zip(job.stages, decoded.stages)):
        if (read.pause_after != (dialect == "grbl" and
                                 index < len(job.stages) - 1) or
                read.program_ended != (dialect == "uccnc" or
                                       index == len(job.stages) - 1)):
            raise ValueError("decoded boundary or end differs")
        if index and stage.transition.boundary != (
                "split" if dialect == "uccnc" else "pause"):
            raise ValueError("transition policy differs from decoded boundary")
        transition_moves = read.transition_moves
        delta = stage.offset_mm - old_offset if dialect == "grbl" else 0.0
        expected_start = tuple(a + b for a, b in zip(
            stage.motions[0].start, job.translation_xyz_mm))
        if abs(delta) > 1e-9 or (dialect == "grbl" and
                                  not _near(work_tip, expected_start)):
            initial = (work_tip[0], work_tip[1], work_tip[2] - delta)
            if (len(transition_moves) != 1 or transition_moves[0].g != 0 or
                    transition_moves[0].feed != 0 or
                    not _near(transition_moves[0].start, initial) or
                    not _near(transition_moves[0].end, expected_start)):
                raise ValueError("missing or changed offset compensation travel")
            for point in (initial, expected_start):
                tip = tuple(a - b for a, b in zip(point,
                                                  job.translation_xyz_mm))
                if job.stock_present and tip[2] <= 0:
                    raise ValueError("offset compensation travels into stock")
        elif transition_moves:
            raise ValueError("unmodeled transition travel")
        actual.append(_observed_stage(stage, read, job.translation_xyz_mm))
        work_tip = tuple(a + b for a, b in zip(stage.motions[-1].end,
                                                job.translation_xyz_mm))
        old_offset = stage.offset_mm
    # A declared length correction must agree with the installed tool length.
    # The offset word alone is not evidence that the tool was installed.
    for stage in job.stages:
        if abs(stage.offset_mm - stage.tool_length_mm) > 1e-9:
            raise ValueError("effective tool-tip offset differs from setup")
    assumptions = [s.transition.completion_token for s in job.stages[1:]]
    report = {
        "job_fingerprint": job.fingerprint,
        "prefix_fingerprints": job.prefixes,
        "input_resolution": {"status": "pass", "scope": "resolved ordered job",
                             "source_fingerprint": job.source_fingerprint},
        "document_fidelity": ({"status": "pass", "scope":
                               "supplied normalized native source/candidate/post"}
                              if job.source_kind == "native" else
                              {"status": "not_evaluated",
                               "reason": "direct Python job has no native document"}),
        "motion_equivalence": {"status": "pass", "scope": "every decoded stage",
                               "moves": tuple(len(m) for m in actual),
                               "tolerance_mm": MATCH_TOLERANCE_MM},
        "transition_evidence": {"status": "pass_with_assumptions",
                                "scope": "ordered stage boundaries and effective tip",
                                "policies": tuple((s.id, s.transition.actor,
                                                   s.transition.boundary,
                                                   s.transition.effect_model)
                                                  for s in job.stages[1:]),
                                "effective_start_tip_program_xyz_mm": tuple(
                                    (s.motions[0].start[0],
                                     s.motions[0].start[1],
                                     s.motions[0].start[2] + s.offset_mm -
                                     s.tool_length_mm) for s in job.stages),
                                "completion_assumed_not_observed": tuple(assumptions)},
        "runtime_parity": {"status": "not_evaluated",
                           "reason": "no controller runtime trace"},
        "physical_setup": {"status": "not_evaluated",
                           "reason": "installation and measurement are caller assertions"},
    }
    if not job.stock_present:
        report["stock_access_residual"] = {"status": "not_evaluated",
                                           "reason": "no supplied initial stock"}
        return report
    endmill_count = next((i for i, s in enumerate(job.stages)
                          if s.v_plan is not None), len(job.stages))
    if any(s.operation is None for s in job.stages[:endmill_count]) or any(
            s.v_plan is not None for s in job.stages[endmill_count + 1:]):
        report["stock_access_residual"] = {"status": "unsupported",
                                           "reason": "unsupported stock evaluator sequence"}
        return report
    prefixes = _replay_endmills(job, job.stages[:endmill_count],
                               actual[:endmill_count]) if endmill_count else []
    if endmill_count == len(job.stages):
        report["stock_access_residual"] = {
            "status": "pass", "scope": "decoded ordered cylindrical sweeps",
            "cuts_by_prefix": tuple(len(stock.cuts) for _, stock in prefixes),
            "trace_fingerprint": prefixes[-1][0].motion_fingerprint}
        return report
    if endmill_count != 1 or endmill_count != len(job.stages) - 1:
        report["stock_access_residual"] = {"status": "unsupported",
                                           "reason": "V stock requires one endmill predecessor"}
        return report
    plan = _decoded_v_plan(job.stages[-1].v_plan, actual[-1])
    rest = v_region.with_prior(plan, prefixes[-1][0])
    report["stock_access_residual"] = {
        "status": "pass", "scope": "decoded endmill then decoded rounded V",
        "cuts_by_prefix": (len(rest.prior_stock.cuts),),
        "prior_section_1_mm2": v_region.section_report(rest, 1, final=False),
        "section_1_mm2": v_region.section_report(rest, 1),
        "prior_volume_mm3": v_region.volume_bounds(rest, final=False),
        "volume_mm3": v_region.volume_bounds(rest),
        "decoded_prior_fingerprint": prefixes[-1][0].motion_fingerprint,
        "decoded_v_plan_fingerprint": plan.fingerprint}
    return report
