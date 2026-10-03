"""Complete-byte assembly audits for independent straight-wall and tapered jobs."""
from dataclasses import asdict, dataclass

from . import ordered_dialects, ordered_output
from ..cam_core import ordered_job, ornamental_inlay, replay


def audit_pair(design, receiver_job, plug_job, dialect, receiver_files, plug_files,
               *, expected_receiver_job, expected_plug_job,
               expected_receiver_hashes, expected_plug_hashes,
               registration_xy=(0.0, 0.0)):
    """Bind both complete outputs, then assemble their decoded remaining stock.

    Detached cylindrical jobs only. Native freshness and externally simulated
    transitions need their own explicit input bindings and are not inferred.
    """
    traces, reports = [], []
    for side, job, files, fingerprint, hashes in (
            ("receiver", receiver_job, receiver_files, expected_receiver_job,
             expected_receiver_hashes),
            ("plug", plug_job, plug_files, expected_plug_job, expected_plug_hashes)):
        if (type(job) is not ordered_job.Job or job.fingerprint != fingerprint or
                job.source_kind != "direct" or not job.stock_present or
                job.source_fingerprint != design.source(side) or
                job.program_frame != design.frame(side) or
                hashes is None or any(type(s.operation) is not replay.Operation
                                      for s in job.stages)):
            raise ValueError("stale or unsupported paired job/source binding")
        report = ordered_output.audit_files(job, dialect, files,
                                             expected_hashes=hashes)
        decoded = ordered_dialects.decode(files, dialect,
            initial_work_tip=tuple(a+b for a, b in zip(job.initial_tip,
                                                      job.translation_xyz_mm)))
        motions = tuple(ordered_job._observed_stage(s, d, job.translation_xyz_mm)
                        for s, d in zip(job.stages, decoded.stages))
        trace, _ = ordered_job._replay_endmills(job, job.stages, motions)[-1]
        traces.append(trace)
        reports.append(report)
    result = ornamental_inlay.verify_pair(design, *traces,
        expected_receiver_motion=traces[0].motion_fingerprint,
        expected_plug_motion=traces[1].motion_fingerprint,
        registration_xy=registration_xy)
    return {"assembly": asdict(result), "parts": tuple(reports),
            "scope": "decoded straight-wall stock and rigid insertion; no physical fit"}


@dataclass(frozen=True)
class ComponentOutput:
    """Complete ordered output for one tapered removal component."""
    job: ordered_job.Job
    files: tuple
    expected_job: str
    expected_hashes: tuple


def assemble_outputs(design, receiver_outputs, plug_outputs, dialect, *,
                     registration_xy=(0.0, 0.0), slabs=16):
    """Build facing input from complete decoded independent component outputs.

    Straight-wall parts use one ComponentOutput each; tapered parts may contain
    several complete components. Existing paired audits own byte/setup gates.
    """
    from ..cam_core import composite_inlay, tapered_inlay, v_region

    groups = (receiver_outputs, plug_outputs)
    if any(type(g) is not tuple or not g or
           any(type(o) is not ComponentOutput for o in g) for g in groups):
        raise ValueError("nonempty ComponentOutput tuples required")
    if type(design) is ornamental_inlay.Design:
        if any(len(g) != 1 for g in groups):
            raise ValueError("one complete ordered output per straight-wall part required")
        r, p = receiver_outputs[0], plug_outputs[0]
        report = audit_pair(design, r.job, p.job, dialect, r.files, p.files,
            expected_receiver_job=r.expected_job, expected_plug_job=p.expected_job,
            expected_receiver_hashes=r.expected_hashes, expected_plug_hashes=p.expected_hashes,
            registration_xy=registration_xy)
    else:
        report = audit_tapered_pair(design, *groups, dialect,
            registration_xy=registration_xy, slabs=slabs)
    stocks = []
    for group in groups:
        components = []
        for output in group:
            job = output.job
            decoded = ordered_dialects.decode(output.files, dialect,
                initial_work_tip=tuple(a+b for a, b in zip(job.initial_tip, job.translation_xyz_mm)))
            motions = tuple(ordered_job._observed_stage(s, d, job.translation_xyz_mm)
                            for s, d in zip(job.stages, decoded.stages))
            if type(design) is ornamental_inlay.Design:
                stock = ordered_job._replay_endmills(job, job.stages, motions)[-1][0]
            else:
                sweeps = tuple(ordered_job._decoded_v_plan(s.v_plan, m)
                    if s.v_plan is not None else
                    ordered_job._replay_endmills(job, (s,), (m,), initial_tip=m[0].start)[0][0]
                    for s, m in zip(job.stages, motions))
                target = next(s.v_plan.target for s in job.stages if s.v_plan is not None)
                components.append(v_region.VComposition(target, sweeps))
        if type(design) is not ornamental_inlay.Design:
            stock = tapered_inlay.PartStock(tuple(components))
        stocks.append(stock)
    fingerprints = tuple(s.motion_fingerprint if type(s) is replay.Trace else s.fingerprint
                         for s in stocks)
    assembly = composite_inlay.Assembly(design, *stocks, *fingerprints,
        registration_xy=registration_xy, slabs=slabs)
    return assembly, report


def facing_job(finish, trace, *, feed_mm_min, rpm):
    """Bind a replayed candidate, including partial work, to ordered output."""
    from ..cam_core import composite_inlay
    composite_inlay.verify(finish, trace, expected_motion=trace.motion_fingerprint)
    if len(trace.operations) != 1:
        raise ValueError("one facing operation required by this job adapter")
    op = trace.operations[0]
    moves = tuple(ordered_job.JobMove(m.role, m.start, m.end,
                  0 if m.role == "rapid" else feed_mm_min)
                  for m in trace.items if type(m) is replay.Motion)
    stage = ordered_job.Stage(op.name, op.tool.name, moves, rpm, operation=op,
                              source_revision=finish.fingerprint)
    return ordered_job.Job(finish.fingerprint, (stage,), trace.initial_position,
                           program_frame=finish.frame)


def audit_facing(finish, output, dialect):
    """Audit exact complete facing bytes and replay decoded composite removal.

    The ordered audit's bounding-prism residual is not a composite stock result;
    only the separate finishing report describes exposed assembled material.
    """
    from ..cam_core import composite_inlay
    if type(finish) is not composite_inlay.Finish or type(output) is not ComponentOutput:
        raise ValueError("Finish and ComponentOutput required")
    job = output.job
    if (type(job) is not ordered_job.Job or job.fingerprint != output.expected_job or
            job.source_kind != "direct" or not job.stock_present or
            job.source_fingerprint != finish.fingerprint or job.program_frame != finish.frame or
            type(output.expected_hashes) is not tuple or not output.expected_hashes or
            any(type(s.operation) is not replay.Operation for s in job.stages)):
        raise ValueError("stale or unsupported composite facing job/source binding")
    report = ordered_output.audit_files(job, dialect, output.files,
                                       expected_hashes=output.expected_hashes)
    decoded = ordered_dialects.decode(output.files, dialect,
        initial_work_tip=tuple(a+b for a, b in zip(job.initial_tip, job.translation_xyz_mm)))
    motions = tuple(ordered_job._observed_stage(s, d, job.translation_xyz_mm)
                    for s, d in zip(job.stages, decoded.stages))
    trace = ordered_job._replay_endmills(job, job.stages, motions)[-1][0]
    result = composite_inlay.verify(finish, trace, expected_motion=trace.motion_fingerprint)
    return {"finishing": result, "output": report,
            "scope": "decoded flat composite facing; declared cure/setup; no physical acceptance"}


def audit_tapered_pair(design, receiver_outputs, plug_outputs, dialect, *,
                       registration_xy=(0.0, 0.0), slabs=16):
    """Audit every component's complete bytes and assemble decoded V/mixed stock.

    Each component has its own complete job/setup; no inter-component low link
    or pre-cleared access is inferred. Caller lists are the supplied machining
    evidence, so omitted components leave their virgin blank material in place.
    """
    from ..cam_core import tapered_inlay, v_region

    if type(design) is not tapered_inlay.Design:
        raise ValueError("tapered Design required")
    stocks, reports = [], []
    for side, outputs in (("receiver", receiver_outputs), ("plug", plug_outputs)):
        if type(outputs) is not tuple or not outputs:
            raise ValueError("nonempty component output tuple required")
        components, part_reports = [], []
        targets = {t.fingerprint for t in design.targets(side)}
        for output in outputs:
            if type(output) is not ComponentOutput:
                raise ValueError("ComponentOutput required")
            job = output.job
            if (type(job) is not ordered_job.Job or
                    job.fingerprint != output.expected_job or
                    job.source_kind != "direct" or not job.stock_present or
                    job.source_fingerprint != design.source(side) or
                    job.program_frame != design.frame(side) or
                    type(output.expected_hashes) is not tuple or
                    not output.expected_hashes):
                raise ValueError("stale or unsupported tapered job/source binding")
            plans = [s.v_plan for s in job.stages if s.v_plan is not None]
            if not plans or any(p.target.fingerprint not in targets for p in plans):
                raise ValueError("tapered component needs a matching V design")
            report = ordered_output.audit_files(job, dialect, output.files,
                                                expected_hashes=output.expected_hashes)
            decoded = ordered_dialects.decode(output.files, dialect,
                initial_work_tip=tuple(a+b for a, b in zip(job.initial_tip,
                                                          job.translation_xyz_mm)))
            sweeps = []
            for stage, observed in zip(job.stages, decoded.stages):
                moves = ordered_job._observed_stage(stage, observed, job.translation_xyz_mm)
                if stage.v_plan is not None:
                    sweeps.append(ordered_job._decoded_v_plan(stage.v_plan, moves))
                else:
                    sweeps.append(ordered_job._replay_endmills(job, (stage,), (moves,),
                        initial_tip=moves[0].start)[0][0])
            components.append(v_region.VComposition(plans[0].target, tuple(sweeps)))
            part_reports.append(report)
        stocks.append(tapered_inlay.PartStock(tuple(components)))
        reports.append(tuple(part_reports))
    result = tapered_inlay.verify_pair(design, *stocks,
        expected_receiver_stock=stocks[0].fingerprint,
        expected_plug_stock=stocks[1].fingerprint,
        registration_xy=registration_xy, slabs=slabs)
    return {"assembly": asdict(result), "parts": tuple(reports),
            "scope": "decoded tapered stock and continuous rigid insertion; no physical fit"}
