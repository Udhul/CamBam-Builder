"""Finite tapered-inlay consumer of BO01 policy and independent IN01 audits."""
from dataclasses import dataclass, replace
import math

from shapely.errors import GEOSException

from . import bundle_search as b
from ..cam_core import composite_inlay as ci, tapered_inlay as ti, replay
from ..cam_core.occupancy import OccupancySetup
from ..integrations import inlay_output as io, ordered_dialects, ordered_output


@dataclass(frozen=True)
class PartRequest:
    """One independently set up component, in Design.targets(side) order."""
    tools: tuple
    families: tuple
    constraints: b.Constraints
    setup: OccupancySetup
    initial_tip: tuple
    safe_z: float
    max_operations: int


@dataclass(frozen=True)
class Facing:
    """Caller declarations; setup coordinates use the assembled backing top.

    The setup frame must be 'assembled-backing-top'; its boxes and tool body
    are rebound to the candidate's Finish frame without changing coordinates.
    """
    tool: b.Tool
    setup: OccupancySetup
    stepover_mm: float
    safe_z: float
    max_paths: int
    removal_mm: float
    plane_tolerance_mm: float
    motif_tolerance_mm: float
    minimum_plug_core_mm: float
    minimum_receiver_floor_mm: float
    edge_access_mm: float
    assembly_token: str
    cure_token: str
    renewed_setup_token: str
    assembly_time_s: float
    cure_time_s: float

    def __post_init__(self):
        if (type(self.tool) is not b.Tool or type(self.tool.profile) is not replay.ToolProfile or
                type(self.setup) is not OccupancySetup or self.setup.frame != 'assembled-backing-top' or
                tuple(t.tool_id for t in self.setup.tools) != (self.tool.tool_id,)):
            raise ValueError('cylindrical facing tool and assembled-backing-top setup required')
        for name in ('stepover_mm', 'safe_z', 'minimum_plug_core_mm',
                     'minimum_receiver_floor_mm', 'edge_access_mm'):
            b._number(getattr(self, name), name, positive=True)
        for name in ('removal_mm', 'plane_tolerance_mm', 'motif_tolerance_mm',
                     'assembly_time_s', 'cure_time_s'):
            b._number(getattr(self, name), name)
        b._count(self.max_paths, 'facing path budget', positive=True)
        for name in ('assembly_token', 'cure_token', 'renewed_setup_token'):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError('declared assembly/cure/renewed setup required')


@dataclass(frozen=True)
class PairBundle:
    parts: tuple                    # receiver components, then plug components
    facing: io.ComponentOutput
    finish: ci.Finish
    assembly_report: dict
    facing_report: dict
    area_mm2: tuple                 # sum of independent part section residual bounds
    volume_mm3: tuple
    cost: b.Cost


@dataclass(frozen=True)
class PairAssessment(b.Assessment):
    part_assessments: tuple = ()
    assembly_report: object = None
    facing_report: object = None


def _output(bundle):
    return io.ComponentOutput(bundle.job, bundle.files, bundle.job.fingerprint,
                              bundle.program_sha256)


def _face(assembly, policy, dialect):
    finish = ci.Finish(assembly, policy.removal_mm, policy.plane_tolerance_mm,
        policy.motif_tolerance_mm, policy.minimum_plug_core_mm,
        policy.minimum_receiver_floor_mm, policy.edge_access_mm,
        policy.assembly_token, policy.cure_token, policy.renewed_setup_token)
    rows = max(1, math.ceil((assembly.footprint.bounds[3]-assembly.footprint.bounds[1]) /
                            policy.stepover_mm)) + 1
    passes = max(1, math.ceil((assembly.top_mm+policy.removal_mm) /
                              policy.tool.axial_limits.max_stepdown_mm))
    if rows*passes > policy.max_paths:
        raise ValueError('facing path budget exceeded')
    trace = ci.generate(finish, policy.tool.profile, stepover_mm=policy.stepover_mm,
        stepdown_mm=policy.tool.axial_limits.max_stepdown_mm, clearance_mm=policy.safe_z)
    job = io.facing_job(finish, trace, feed_mm_min=policy.tool.cut_feed_mm_min, rpm=policy.tool.rpm)
    # Keep entry and retract feeds explicit, as in the planar consumer.
    stage = job.stages[0]
    moves = tuple(replace(m, feed=(0 if m.role == 'rapid' else
        policy.tool.entry_feed_mm_min if m.role == 'entry' else
        policy.tool.retract_feed_mm_min if m.role == 'retract' else policy.tool.cut_feed_mm_min))
        for m in stage.motions)
    job = replace(job, stages=(replace(stage, motions=moves, axial_limits=policy.tool.axial_limits),),
                  occupancy_setup=replace(policy.setup, frame=finish.frame))
    files, report = ordered_output.emit(job, dialect,
        coordinate_decimals=6 if dialect == 'uccnc' else 4)
    output = io.ComponentOutput(job, files, job.fingerprint,
                               tuple(report['motion_equivalence']['program_sha256']))
    result = io.audit_facing(finish, output, dialect)
    for gate in ('motion_equivalence', 'stock_access_residual',
                 'tool_fixture_occupancy', 'axial_process_limits'):
        if result['output'][gate]['status'] != 'pass':
            raise ValueError(f'missing passing facing {gate}')
    decoded = ordered_dialects.decode(files, dialect, initial_work_tip=job.initial_tip)
    return finish, output, result, decoded


def _compound_orders(spaces, bounds, prefix=()):
    # Lazy recursion avoids materializing permutation pools or Cartesian products.
    if len(prefix) == len(spaces):
        yield prefix
    else:
        i = len(prefix)
        for order in b._orders(spaces[i], bounds[i]):
            yield from _compound_orders(spaces, bounds, prefix+(order,))


def search_pair(design, receiver, plug, *, facing, constraints, cost_model,
                max_evaluations, objective='time', dialect='uccnc',
                registration_xy=(0.0, 0.0), slabs=16, manual_orders=None):
    """Search the Cartesian space of independent component orders, then face.

    Every part order must satisfy its limits. Only decoded IN01 insertion and
    final-facing passes may win. Evaluation budget counts complete compound
    orders; repeated component orders share their independently audited result.
    Manual orders list receiver components followed by plug components.
    """
    if (type(design) is not ti.Design or type(facing) is not Facing or
            type(constraints) is not b.Constraints or type(cost_model) is not b.CostModel):
        raise ValueError('tapered design, facing declarations, constraints and costs required')
    b._count(max_evaluations, 'pair evaluation budget', positive=True)
    b._count(slabs, 'assembly slabs', positive=True)
    if constraints.max_floor_cusp_mm is not None:
        raise ValueError('compound floor cusp is unsupported; set per-component limits')
    if (type(registration_xy) is not tuple or len(registration_xy) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in registration_xy)):
        raise ValueError('finite assembly XY registration required')
    targets, requests, spaces, labels = [], [], [], []
    for side, group in (('receiver', receiver), ('plug', plug)):
        expected = design.targets(side)
        if (type(group) is not tuple or len(group) != len(expected) or
                any(type(r) is not PartRequest for r in group)):
            raise ValueError('one PartRequest for every independent design component required')
        for i, (target, request) in enumerate(zip(expected, group)):
            spaces.append(b._validate(target, request.tools, request.families,
                request.constraints, cost_model, request.setup, request.initial_tip,
                request.safe_z, request.max_operations, max_evaluations,
                objective, dialect, None))
            targets.append(target)
            requests.append(request)
            labels.append(f'{side} component {i+1}')
    # A tool ID denotes one physical profile across the compound workflow.
    profiles = {}
    for tool in tuple(t for r in requests for t in r.tools)+(facing.tool,):
        if tool.tool_id in profiles and profiles[tool.tool_id] != tool.profile:
            raise ValueError('compound tool ID has conflicting physical profiles')
        profiles[tool.tool_id] = tool.profile
    bounds = tuple(r.max_operations for r in requests)
    if manual_orders is not None:
        if type(manual_orders) is not tuple or len(manual_orders) != len(requests):
            raise ValueError('manual orders must cover all independent components')
        for target, request, order in zip(targets, requests, manual_orders):
            b._validate(target, request.tools, request.families, request.constraints, cost_model,
                request.setup, request.initial_tip, request.safe_z, request.max_operations,
                max_evaluations, objective, dialect, order)
            if order is None:
                raise ValueError('each manual component order must be explicit')
    total = math.prod(sum(math.perm(len(s), n) for n in range(1, bound+1))
                      for s, bound in zip(spaces, bounds))
    cache, assessments = {}, []

    def evaluate(orders):
        parts, assembly_report, facing_report = [], None, None
        try:
            for i, (order, target, request) in enumerate(zip(orders, targets, requests)):
                key = (i, order)
                if key not in cache:
                    cache[key] = b._evaluate(order, target,
                        {t.tool_id: t for t in request.tools},
                        {f.family_id: f for f in request.families}, request.constraints,
                        cost_model, request.setup, request.initial_tip, request.safe_z, dialect)
                parts.append(cache[key])
            executed = tuple(p.executed_actions for p in parts)
            omitted = tuple(p.omitted_actions for p in parts)
            failed = [(label, p) for label, p in zip(labels, parts) if p.status != 'feasible']
            if failed:
                status = 'rejected' if any(p.status == 'rejected' for _, p in failed) else 'constrained'
                reasons = tuple(f'{label}: {reason}' for label, p in failed for reason in p.reasons)
                return PairAssessment(orders, executed, omitted, status, reasons,
                                      part_assessments=tuple(parts))
            bundles = tuple(p.bundle for p in parts)
            outputs = tuple(_output(bundle) for bundle in bundles)
            split = len(receiver)
            assembly_report = io.audit_tapered_pair(design, outputs[:split], outputs[split:],
                dialect, registration_xy=registration_xy, slabs=slabs)
            if assembly_report['assembly']['status'] != 'pass':
                return PairAssessment(orders, executed, omitted, 'constrained',
                    ('IN01 insertion/fit: '+assembly_report['assembly']['status'],),
                    part_assessments=tuple(parts), assembly_report=assembly_report)
            assembly, _ = io.assemble_outputs(design, outputs[:split], outputs[split:], dialect,
                registration_xy=registration_xy, slabs=slabs)
            finish, face_output, facing_report, decoded = _face(assembly, facing, dialect)
            if facing_report['finishing']['status'] != 'pass':
                failures = tuple(name for name in ('plane_ok', 'motif_ok', 'thickness_ok')
                                 if not facing_report['finishing'][name])
                return PairAssessment(orders, executed, omitted, 'constrained',
                    ('IN01 final facing: '+', '.join(failures),), part_assessments=tuple(parts),
                    assembly_report=assembly_report, facing_report=facing_report)
            costs = tuple(x.cost for x in bundles)+(b._cost(face_output.job, decoded, cost_model),)
            jobs = tuple(x.job for x in bundles)+(face_output.job,)
            transfers = len(jobs)-1
            changes = sum(a.stages[-1].tool_id != z.stages[0].tool_id for a, z in zip(jobs, jobs[1:]))
            cost = b.Cost(sum(c.estimated_time_s for c in costs)+transfers*cost_model.boundary_s+
                changes*cost_model.tool_change_s+facing.assembly_time_s+facing.cure_time_s,
                sum(c.below_stock_travel_length_mm for c in costs),
                sum(c.above_stock_travel_length_mm for c in costs),
                sum(c.tool_changes for c in costs)+changes,
                sum(c.setup_boundaries for c in costs)+transfers,
                'constant-feed estimate; declared setup/assembly/cure times; no runtime observation')
            for value in (cost.estimated_time_s, cost.below_stock_travel_length_mm,
                          cost.above_stock_travel_length_mm):
                b._number(value, 'compound cost')
            area = tuple(sum(x.area_mm2[i] for x in bundles) for i in (0, 1))
            volume = tuple(sum(x.volume_mm3[i] for x in bundles) for i in (0, 1))
            bundle = PairBundle(bundles, face_output, finish, assembly_report, facing_report,
                                area, volume, cost)
            reasons = b._limit_reasons(area, volume, cost, constraints)
            return PairAssessment(orders, executed, omitted, 'constrained' if reasons else 'feasible',
                tuple(reasons), bundle, tuple(parts), assembly_report, facing_report)
        except (ValueError, OverflowError, GEOSException) as exc:
            return PairAssessment(orders, tuple(p.executed_actions for p in parts),
                tuple(p.omitted_actions for p in parts), 'rejected', (str(exc),),
                part_assessments=tuple(parts), assembly_report=assembly_report,
                facing_report=facing_report)

    def orders():
        if manual_orders is not None:
            yield manual_orders
        for candidate in _compound_orders(tuple(spaces), bounds):
            if candidate != manual_orders:
                yield candidate

    for order in orders():
        assessments.append(evaluate(order))
        if len(assessments) >= max_evaluations:
            break
    return b._result(assessments, total, objective, design.fingerprint, manual_orders)
