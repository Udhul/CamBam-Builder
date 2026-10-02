"""Complete-byte assembly audit for independent straight-wall inlay jobs."""
from dataclasses import asdict

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
