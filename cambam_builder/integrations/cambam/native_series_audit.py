"""Linear native MOP-series stock audit over one planar cylindrical target.

Every stage reuses core motion replay and a bounded polygonal sweep oracle.
This bridge accepts a common straight-edge Region (with holes) or rectangle,
declared section depths and an overcut budget. It does not certify arc motion,
controller trajectories, physical cutter error or a rounded V profile.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from ...cam_core import polygon_rest, replay
from ...native.cad import Rect
from ...native.reader import read_cambam_bytes
from ...native.region import Region
from ...cam_extensions.strategy import (
    Alternative, REQUIRED_GATES, ResidualBounds, StageAudit,
)
from .native_series import NativeSeries


@dataclass(frozen=True)
class SectionMetric:
    depth_mm: float
    remaining_area_mm2: tuple
    protected_overcut_upper_mm2: float


@dataclass(frozen=True)
class AuditedStage:
    evidence: StageAudit
    sections: tuple


@dataclass(frozen=True)
class NativeSeriesAudit:
    trace: replay.Trace
    stock: replay.ReplayResult
    stages: tuple

    def as_alternative(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("alternative name required")
        return Alternative(name, tuple(stage.evidence for stage in self.stages))


def _polygon_target(target):
    if (type(target) is not replay.Target or target.cone_spine or target.polygon or
            target.inset_per_depth or target.island):
        raise ValueError("series audit supports one planar rectangle or Region")
    if target.region_shell:
        return target
    x0, y0, x1, y1 = target.bounds
    shell = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return replay.Target(target.name, target.bounds, target.depth,
                         region_shell=shell)


def _bind_source_target(source_path, candidate_path, series, target):
    """Require the replay Region to equal one original native target."""
    from shapely.geometry import Polygon

    if any(stage.target_ids != (target.name,) for stage in series.stages):
        raise ValueError("native MOP targets differ from common source target")
    source = read_cambam_bytes(Path(source_path).read_bytes(),
                               source_name="series audit source")
    candidate = read_cambam_bytes(Path(candidate_path).read_bytes(),
                                  source_name="series audit candidate")
    primitive = source.get_primitive(target.name)
    if type(primitive) is Rect:
        points = primitive.get_absolute_coordinates_xyz()
        if any(z != 0 for _, _, z in points):
            raise ValueError("native source target is not zero-Z planar")
        source_region = Polygon([(x, y) for x, y, _ in points])
    elif type(primitive) is Region:
        coordinates = primitive.get_absolute_coordinates_xyz()
        rings = (coordinates["outer_curve"],) + tuple(coordinates["hole_curves"])
        if any(z != 0 or bulge != 0 for ring in rings
               for _, _, z, bulge in ring):
            raise ValueError("curved native source needs separate bound adapter")
        source_region = Polygon([(x, y) for x, y, _, _ in rings[0]],
                                [[(x, y) for x, y, _, _ in ring]
                                 for ring in rings[1:]])
    else:
        raise ValueError("native source target must be Rect or straight Region")
    supplied_region = polygon_rest._geometry(target)
    if (not source_region.is_valid or source_region.is_empty or
            not source_region.equals(supplied_region) or
            target.depth > source.list_parts()[0].stock_thickness):
        raise ValueError("replay target differs from original native source")
    part = candidate.list_parts()[0]
    mops = {mop.name: mop for mop in candidate.get_mops_in_part(part) if mop.enabled}
    if any(mops[stage.name].target_depth != -target.depth
           for stage in series.stages):
        raise ValueError("replay target depth differs from native MOP floor")


def audit_linear_native_series(series, source_path, candidate_path, post_path, *,
                               targets, cutting_lengths_mm, entry_modes,
                               section_depths_mm, max_protected_overcut_mm2,
                               setup=None):
    """Recheck bytes, replay every emitted line and report per-prefix stock.

    A failed role/access/target/overcut check raises rather than creating a
    selectable StageAudit. The caller declares tool cutting lengths and whether
    each stage may enter virgin stock; posted MOP settings cannot grant access.
    """
    if type(series) is not NativeSeries:
        raise ValueError("native series observation required")
    series.check_freshness(source_path, candidate_path, post_path, setup=setup)
    depths = section_depths_mm
    if (type(depths) is not tuple or not depths or
            any(isinstance(d, bool) or not isinstance(d, (int, float)) or
                not math.isfinite(d) for d in depths) or
            list(depths) != sorted(set(depths))):
        raise ValueError("unique ordered finite section depths required")
    if (isinstance(max_protected_overcut_mm2, bool) or
            not isinstance(max_protected_overcut_mm2, (int, float)) or
            not math.isfinite(max_protected_overcut_mm2) or
            max_protected_overcut_mm2 < 0):
        raise ValueError("finite nonnegative overcut budget required")
    targets_in_order = tuple(targets[stage.name] for stage in series.stages)
    if not targets_in_order or any(target != targets_in_order[0]
                                   for target in targets_in_order[1:]):
        raise ValueError("series audit requires one common target")
    target = _polygon_target(targets_in_order[0])
    _bind_source_target(source_path, candidate_path, series, target)
    if (any(not 0 < depth <= target.depth for depth in depths) or
            depths[-1] != target.depth):
        raise ValueError("section depths must include the target floor")
    trace = series.to_trace(targets, cutting_lengths_mm, entry_modes)
    stock = replay.replay(trace, expected_source=series.evidence_fingerprint)
    if tuple(name for name, _ in stock.prefixes) != tuple(stage.name for stage in series.stages):
        raise ValueError("replayed operation prefixes differ from posted MOP order")
    region = polygon_rest._geometry(target)
    stages = []
    predecessor = series.source_semantic_sha256
    for stage, (_, cut_count) in zip(series.stages, stock.prefixes):
        cuts = stock.cuts[:cut_count]
        sections = []
        for depth in depths:
            remaining = polygon_rest._areas(target, cuts, depth)
            outer = polygon_rest._cut_polygon(cuts, depth, radial_error=1e-6)
            overcut = outer.difference(region).area
            if overcut > max_protected_overcut_mm2:
                raise ValueError(f"MOP {stage.name!r} exceeds protected overcut budget")
            sections.append(SectionMetric(depth, remaining, overcut))
        volume = polygon_rest._volume(target, cuts)
        final_area = sections[-1].remaining_area_mm2
        residual = ResidualBounds(final_area[0], final_area[1],
                                  volume[0], volume[1])
        stage_motion = hashlib.sha256(repr((series.candidate_sha256,
                                             series.motion_sha256, stage.name,
                                             cut_count)).encode("utf-8")).hexdigest()
        evidence = StageAudit(stage.name, "native", series.source_semantic_sha256,
                              predecessor, stage_motion, series.post_sha256,
                              residual, REQUIRED_GATES)
        stages.append(AuditedStage(evidence, tuple(sections)))
        predecessor = evidence.chain_fingerprint
    return NativeSeriesAudit(trace, stock, tuple(stages))
