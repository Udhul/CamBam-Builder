"""Finite planar search; path, decoded-stock and safety owners remain independent.

Time uses constant declared feeds/setup costs, without acceleration or load.
"""
from dataclasses import dataclass, replace
from itertools import permutations
import math

from shapely.errors import GEOSException

from cambam_builder.cam_core import ordered_job as oj, planar_rest, v_region as v
from cambam_builder.cam_core.occupancy import OccupancySetup
from cambam_builder.integrations import ordered_dialects
from cambam_builder.integrations.ordered_output import emit


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            value < 0 or (positive and value == 0)):
        raise ValueError(f"finite {'positive' if positive else 'nonnegative'} {name} required")


def _count(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{'positive' if positive else 'nonnegative'} integer {name} required")


@dataclass(frozen=True)
class Tool:
    tool_id: str
    profile: v.VProfile
    axial_limits: oj.AxialLimits
    rpm: float
    cut_feed_mm_min: float
    entry_feed_mm_min: float
    retract_feed_mm_min: float

    def __post_init__(self):
        if (type(self.tool_id) is not str or not self.tool_id or
                type(self.profile) is not v.VProfile or
                type(self.axial_limits) is not oj.AxialLimits):
            raise ValueError('named V tool and axial/entry limits required')
        ordered_dialects._tool(self.tool_id)
        for name in ('rpm', 'cut_feed_mm_min', 'entry_feed_mm_min', 'retract_feed_mm_min'):
            _number(getattr(self, name), name, positive=True)


@dataclass(frozen=True)
class Family:
    family_id: str
    kind: str                      # raster, offset, feature
    stepover_mm: float
    xy_step_mm: float
    margin_mm: float = .02
    max_cusp_mm: float = .25        # feature pitch control, not finish acceptance
    max_paths: int = 1000
    max_sites: int = 1500

    def __post_init__(self):
        if (type(self.family_id) is not str or not self.family_id or
                self.kind not in ('raster', 'offset', 'feature')):
            raise ValueError('named supported path family required')
        for name in ('stepover_mm', 'xy_step_mm', 'margin_mm', 'max_cusp_mm'):
            _number(getattr(self, name), name, positive=True)
        if self.margin_mm <= 1e-5:
            raise ValueError('path margin must exceed geometry tolerance')
        _count(self.max_paths, 'path budget', positive=True)
        _count(self.max_sites, 'site budget', positive=True)


@dataclass(frozen=True)
class Constraints:
    max_area_mm2: float             # section depth is min(1 mm, target cap)
    max_volume_mm3: float
    max_time_s: object = None
    max_tool_changes: object = None # initial installation is not a change
    max_setup_boundaries: object = None
    max_floor_cusp_mm: object = None

    def __post_init__(self):
        if self.max_area_mm2 is None or self.max_volume_mm3 is None:
            raise ValueError('explicit area and volume finish budgets required')
        for name in ('max_area_mm2', 'max_volume_mm3', 'max_time_s'):
            if getattr(self, name) is not None:
                _number(getattr(self, name), name)
        for name in ('max_tool_changes', 'max_setup_boundaries'):
            if getattr(self, name) is not None:
                _count(getattr(self, name), name)
        if self.max_floor_cusp_mm is not None:
            _number(self.max_floor_cusp_mm, 'floor cusp', positive=True)


@dataclass(frozen=True)
class CostModel:
    rapid_mm_min: float
    initial_setup_s: float
    boundary_s: float               # every split/pause, including depth passes
    tool_change_s: float            # additional time when tool ID changes

    def __post_init__(self):
        _number(self.rapid_mm_min, 'rapid rate', positive=True)
        for name in ('initial_setup_s', 'boundary_s', 'tool_change_s'):
            _number(getattr(self, name), name)


@dataclass(frozen=True)
class Cost:
    estimated_time_s: float
    below_stock_travel_length_mm: float
    above_stock_travel_length_mm: float     # above-stock portions only
    tool_changes: int
    setup_boundaries: int
    confidence: str = 'constant-feed estimate; declared setup times; no runtime observation'


@dataclass(frozen=True)
class Bundle:
    job: oj.Job
    files: tuple
    dialect: str
    area_mm2: tuple
    volume_mm3: tuple
    section_depth_mm: float
    unproved_floor_area_mm2: object
    cost: Cost
    program_sha256: tuple


@dataclass(frozen=True)
class Assessment:
    actions: tuple                  # requested (tool ID, family ID) order
    executed_actions: tuple
    omitted_actions: tuple
    status: str                    # feasible, constrained, empty, rejected
    reasons: tuple
    bundle: object = None


@dataclass(frozen=True)
class SearchResult:
    status: str                    # selected, infeasible within declared orders, unresolved
    chosen: object                 # Assessment or None
    assessments: tuple
    frontier_actions: tuple
    dominated_actions: tuple
    total_orders: int
    evaluated_orders: int
    enumeration_complete: bool
    unresolved_orders: int
    objective: str
    quality: str
    design_fingerprint: str


def _composition(job, files, dialect):
    # Reuse the stock owner's reconstruction; cached reports never guide cuts.
    decoded = ordered_dialects.decode(files, dialect, initial_work_tip=job.initial_tip)
    plans = tuple(oj._decoded_v_plan(s.v_plan, oj._observed_stage(s, d, job.translation_xyz_mm))
                  for s, d in zip(job.stages, decoded.stages))
    return v.VComposition(plans[0].target, plans), decoded


def _cost(job, decoded, model):
    changes = sum(a.tool_id != b.tool_id for a, b in zip(job.stages, job.stages[1:]))
    boundaries = len(job.stages)-1
    seconds = model.initial_setup_s + boundaries*model.boundary_s + changes*model.tool_change_s
    cutting = air = 0.0
    for stage, observed in zip(job.stages, decoded.stages):
        for intended, move in zip(stage.motions, observed.moves):
            length = math.dist(move.start, move.end)
            rate = model.rapid_mm_min if intended.role == 'rapid' else move.feed
            seconds += 60*length/rate
            za, zb = move.start[2], move.end[2]
            above = (1.0 if min(za, zb) >= 0 else 0.0 if max(za, zb) <= 0
                     else max(za, zb)/abs(zb-za))
            air += length*above
            cutting += length*(1-above)
    for value in (seconds, cutting, air):
        _number(value, 'computed cost')
    return Cost(seconds, cutting, air, changes, boundaries)


def _plan(prior, tool, family, safe_z):
    controls = dict(stepover_mm=family.stepover_mm, xy_step_mm=family.xy_step_mm,
                    margin_mm=family.margin_mm, safe_z=safe_z, max_paths=family.max_paths)
    if family.kind == 'feature':
        return planar_rest.generate(prior, tool.profile, max_cusp_mm=family.max_cusp_mm,
                                    max_sites=family.max_sites, **controls).plan
    return v.plan(prior.target, tool.profile, fill_pattern=family.kind, **controls)


def _evaluate(actions, target, tools, families, constraints, model, setup,
              initial_tip, safe_z, dialect):
    stages, executed, omitted = [], [], []
    seed = v.VPlan(target, next(iter(tools.values())).profile, (), (),
                   safe_z, .02, 1, 'infeasible', 'virgin stock; no removal')
    prior = v.VComposition(target, (seed,))
    boundary = 'pause' if dialect == 'grbl' else 'split'
    try:
        for action in actions:
            tool, family = tools[action[0]], families[action[1]]
            plan = _plan(prior, tool, family, safe_z)
            v.verify(plan)
            # Retain unproved benefit. Omit only wholly proved cutting-profile
            # air against previously decoded stock, never caller assertions.
            if not plan.paths or (prior.stages and all(
                    planar_rest.cutting_sweep_clear(prior, tool.profile, a, b)
                    for path in plan.paths for a, b in zip(path.points, path.points[1:]))):
                omitted.append(action)
                continue
            for passed in v.depth_passes(plan, tool.axial_limits.max_stepdown_mm):
                moves = tuple(oj.JobMove(m.role, m.start, m.end,
                    0 if m.role == 'rapid' else tool.entry_feed_mm_min if m.role == 'entry'
                    else tool.retract_feed_mm_min if m.role == 'retract' else tool.cut_feed_mm_min)
                    for m in v.complete_motion(passed, initial_tip))
                transition = None if not stages else oj.Transition('operator', boundary,
                    tool.tool_id, initial_tip, 'declared-offline-resume')
                stages.append(oj.Stage(f'search-{len(stages)+1}', tool.tool_id, moves,
                    tool.rpm, v_plan=passed, source_revision=passed.fingerprint,
                    transition=transition, axial_limits=tool.axial_limits))
            used = {stage.tool_id for stage in stages}
            bounded = replace(setup, tools=tuple(b for b in setup.tools if b.tool_id in used))
            job = oj.Job(target.source_id, tuple(stages), initial_tip,
                         program_frame=setup.frame, occupancy_setup=bounded)
            files, report = emit(job, dialect, coordinate_decimals=6 if dialect == 'uccnc' else 4)
            prior, decoded = _composition(job, files, dialect)
            executed.append(action)
        if not stages:
            return Assessment(actions, (), tuple(omitted), 'empty', ('no executable cutting paths',))
        for gate in ('motion_equivalence', 'stock_access_residual',
                     'tool_fixture_occupancy', 'axial_process_limits'):
            if report[gate]['status'] != 'pass':
                raise ValueError(f'missing passing {gate}')
        stock = report['stock_access_residual']
        cost = _cost(job, decoded, model)
        floor_gap = None
        if constraints.max_floor_cusp_mm is not None:
            depth = max(0, target.cap_depth-constraints.max_floor_cusp_mm)
            floor_gap = target.section(target.cap_depth, outer=True).difference(
                v.section_evidence(prior, depth).known_free_inner).area
        bundle = Bundle(job, files, dialect, tuple(stock['section_1_mm2']),
            tuple(stock['volume_mm3']), stock['section_depth_mm'], floor_gap, cost,
            tuple(report['motion_equivalence']['program_sha256']))
        reasons = []
        for name, actual, limit in (
                ('residual area', bundle.area_mm2[1], constraints.max_area_mm2),
                ('residual volume', bundle.volume_mm3[1], constraints.max_volume_mm3),
                ('estimated time', cost.estimated_time_s, constraints.max_time_s),
                ('tool changes', cost.tool_changes, constraints.max_tool_changes),
                ('setup boundaries', cost.setup_boundaries, constraints.max_setup_boundaries)):
            if limit is not None and actual > limit:
                reasons.append(f'{name} exceeds declared limit')
        if floor_gap is not None and floor_gap > 0:
            reasons.append('floor cusp unproved over located floor area')
        return Assessment(actions, tuple(executed), tuple(omitted),
                          'constrained' if reasons else 'feasible', tuple(reasons), bundle)
    except (ValueError, OverflowError, GEOSException) as exc:
        return Assessment(actions, tuple(executed), tuple(omitted), 'rejected', (str(exc),))


def _metrics(assessment):
    b = assessment.bundle
    return (b.area_mm2[1], b.volume_mm3[1], b.cost.estimated_time_s,
            b.cost.tool_changes, b.cost.setup_boundaries)


def _dominates(a, other):
    # Distinct interval-valued stock results cannot establish dominance while
    # their enclosures overlap, even when one upper-bound score is smaller.
    x, y = a.bundle, other.bundle
    comparisons = ((x.area_mm2[1], y.area_mm2[0]),
                   (x.volume_mm3[1], y.volume_mm3[0]),
                   (x.cost.estimated_time_s, y.cost.estimated_time_s),
                   (x.cost.tool_changes, y.cost.tool_changes),
                   (x.cost.setup_boundaries, y.cost.setup_boundaries))
    return all(a <= b for a, b in comparisons) and any(a < b for a, b in comparisons)


def _rank(assessment, objective):
    area, volume, time, changes, boundaries = _metrics(assessment)
    return {'residual': (area, volume, time, changes, boundaries),
            'time': (time, area, volume, changes, boundaries),
            'tool_changes': (changes, time, area, volume, boundaries)}[objective] + (assessment.actions,)


def search(target, tools, families, *, constraints, cost_model, setup,
           initial_tip, safe_z, max_operations, max_evaluations,
           objective='residual', dialect='uccnc', manual_order=None):
    """Enumerate distinct tool/family actions, single operations before longer orders.

    Manual order is evaluated first, consuming the same budget. It may choose a
    dominated feasible bundle but cannot bypass safety or constraints. Complete
    enumeration optimizes reported bounds/estimated costs in these finite
    families, never all possible paths or real machining time.
    """
    if (type(target) is not v.VTarget or target.design_angle_degrees is None or
            type(constraints) is not Constraints or type(cost_model) is not CostModel or
            type(setup) is not OccupancySetup):
        raise ValueError('frozen design, constraints, cost model and whole-tool setup required')
    for values, cls, name in ((tools, Tool, 'tool_id'), (families, Family, 'family_id')):
        if (type(values) is not tuple or not values or any(type(x) is not cls for x in values)
                or len({getattr(x, name) for x in values}) != len(values)):
            raise ValueError('distinct nonempty tool inventory and family tuples required')
    if {b.tool_id for b in setup.tools} != {t.tool_id for t in tools}:
        raise ValueError('setup must declare exactly the inventory tool bodies')
    if target.frame not in (None, setup.frame):
        raise ValueError('design and setup frames differ')
    _number(safe_z, 'safe height', positive=True)
    oj._xyz(initial_tip)
    if initial_tip[2] != safe_z:
        raise ValueError('initial tip must use declared safe height')
    _count(max_operations, 'operation bound', positive=True)
    _count(max_evaluations, 'evaluation budget', positive=True)
    if objective not in ('residual', 'time', 'tool_changes') or dialect not in ('uccnc', 'grbl'):
        raise ValueError('unsupported objective or dialect')
    actions = tuple((t.tool_id, f.family_id) for t in tools for f in families)
    if max_operations > len(actions):
        raise ValueError('operation bound exceeds distinct action inventory')
    if manual_order is not None and (type(manual_order) is not tuple or not manual_order or
            len(manual_order) > max_operations or any(a not in actions for a in manual_order)
            or len(set(manual_order)) != len(manual_order)):
        raise ValueError('manual order must contain distinct declared actions within the bound')
    total = sum(math.perm(len(actions), n) for n in range(1, max_operations+1))

    def orders():
        if manual_order is not None:
            yield manual_order
        for n in range(1, max_operations+1):
            for order in permutations(actions, n):
                if order != manual_order:
                    yield order

    assessments = []
    for order in orders():
        assessments.append(_evaluate(order, target, {t.tool_id: t for t in tools},
            {f.family_id: f for f in families}, constraints, cost_model, setup,
            initial_tip, safe_z, dialect))
        if len(assessments) >= max_evaluations:
            break
    feasible = [a for a in assessments if a.status == 'feasible']
    frontier, dominated = [], []
    for a in feasible:
        worse = any(_dominates(b, a) for b in feasible)
        (dominated if worse else frontier).append(a.actions)
    chosen = None
    if manual_order is not None:
        if assessments[0].status == 'feasible':
            chosen = assessments[0]
    elif feasible:
        chosen = min(feasible, key=lambda a: _rank(a, objective))
    complete = len(assessments) == total
    unresolved = sum(a.status == 'rejected' for a in assessments)
    quality = ('complete declared enumeration' if complete and not unresolved
               else 'best verified subset; unresolved or unevaluated orders remain')
    status = ('selected' if chosen else 'infeasible' if
              ((complete and not unresolved) or (manual_order is not None and
               assessments[0].status in ('constrained', 'empty'))) else 'unresolved')
    return SearchResult(status, chosen, tuple(assessments),
        tuple(frontier), tuple(dominated), total, len(assessments), complete, unresolved,
        objective, quality, target.fingerprint)
