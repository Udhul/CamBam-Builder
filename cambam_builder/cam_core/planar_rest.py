"""Feature-guided planar V candidates against verified composed stock.

Guides propose motion; full-profile verification and located residual enclosures
decide what that motion establishes. Clearance is cutting-profile clearance,
never permission for a rapid, shank, holder or fixture crossing.
"""
from dataclasses import dataclass
import math

from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import voronoi_diagram

from . import v_region as v


def _positive(value, name):
    value = v._finite(value, name)
    if value <= 0:
        raise ValueError(f"positive {name} required")
    return value


def _count(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"positive integer {name} required")
    return value


def _sample(line, step):
    """Densify every source segment, retaining corners and polygon topology."""
    result = []
    for a, b in zip(line.coords, tuple(line.coords)[1:]):
        count = max(1, math.ceil(math.dist(a, b) / step))
        if len(result) + count > 50000:
            raise ValueError("feature vertex budget exceeded")
        result.extend(tuple(round(a[j] + (b[j]-a[j])*i/count, 7)
                            for j in range(2)) for i in range(count))
    result.append(tuple(round(c, 7) for c in line.coords[-1]))
    return tuple(p for i, p in enumerate(result) if i == 0 or p != result[i-1])


class _Clearance:
    def __init__(self, prior, slabs):
        self.prior = v.VComposition(prior.target, prior.stages)
        self.levels = tuple(prior.target.cap_depth * i/slabs
                            for i in range(slabs)) + (prior.target.cap_depth,)
        self.free = {}

    def clear(self, tool, a, b):
        depth = max(a[2], b[2])
        # Exact same-profile linear retraces also prove clearance at a zero-
        # radius pointed floor, where polygonal inner buffers cannot do so.
        for stage in self.prior.stages:
            if type(stage) is not v.VPlan or stage.tool != tool:
                continue
            for path in stage.paths:
                for x, y in zip(path.points, path.points[1:]):
                    for p, q in ((x, y), (y, x)):
                        if (a[:2] == p[:2] and b[:2] == q[:2] and
                                a[2] <= p[2] and b[2] <= q[2]):
                            return True
        line = Point(a[:2]) if a[:2] == b[:2] else LineString((a[:2], b[:2]))
        # Removed sections shrink monotonically with depth. The upper candidate
        # disk at each slab's top must fit the prior inner union at its bottom.
        # This proves all intermediate heights, including variable-depth cuts.
        for top, bottom in zip(self.levels, self.levels[1:]):
            if top > depth:
                break
            if bottom not in self.free:
                self.free[bottom] = v.section_evidence(
                    self.prior, bottom).known_free_inner
            radius = (tool.radius(depth-top) + 1e-6) / math.cos(math.pi/128)
            if not self.free[bottom].covers(line.buffer(radius, quad_segs=32)):
                return False
            if bottom >= depth:
                break
        return True


def cutting_sweep_clear(prior, tool, a, b, *, slabs=8):
    """Conservative whole-height union-clearance of a linear cutting sweep.

    ``a``/``b`` are XY and positive penetration, including a stationary entry
    endpoint. False means unproved. Stock/source safety is re-established;
    whole-tool occupancy and process limits require the ordered-output gate.
    """
    if type(prior) is not v.VComposition or type(tool) is not v.VProfile:
        raise ValueError("composed stock and V profile required")
    slabs = _count(slabs, "clearance slabs")
    for point in (a, b):
        if (len(point) != 3 or any(not math.isfinite(v._finite(c, "point"))
                                   for c in point) or
                not 0 < point[2] <= min(prior.target.cap_depth, tool.cutting_length)):
            raise ValueError("cutting sweep outside depth/profile limits")
    return _Clearance(prior, slabs).clear(tool, a, b)


def _guides(target, tool, spacing, xy_step, margin, max_paths, max_sites):
    cap = min(target.cap_depth, tool.cutting_length)
    inset = v.contact_radius(target, tool, 0) + margin + spacing/2
    guides = []
    # Contact contours extend from shallow detail to the capped area. Preserve
    # all components and island rings rather than bridging their topology.
    for _ in range(max_paths):
        region = target.safe.buffer(-inset / math.cos(math.pi/128), quad_segs=32)
        if region.is_empty:
            break
        for polygon in v._polygons(region):
            guides.extend(("edge", LineString(ring.coords)) for ring in
                          (polygon.exterior,) + tuple(polygon.interiors))
        if len(guides) > max_paths:
            raise ValueError("feature guide budget exceeded")
        inset += spacing
    else:
        raise ValueError("feature contour budget exceeded")
    # A final cap-contact ring avoids depending on pitch alignment at the floor.
    deep = target.safe.buffer(-(v.contact_radius(target, tool, cap)+margin) /
                              math.cos(math.pi/128), quad_segs=32)
    for polygon in v._polygons(deep):
        guides.extend(("edge", LineString(ring.coords)) for ring in
                      (polygon.exterior,) + tuple(polygon.interiors))
    sites = []
    for ring in (target.safe.exterior,) + tuple(target.safe.interiors):
        sites.extend(_sample(LineString(ring.coords), xy_step)[:-1])
        if len(sites) > max_sites:
            raise ValueError("medial boundary-site budget exceeded")
    # This is a sampled-boundary Voronoi guide, not a certified exact medial
    # axis. Its local depths and continuous containment are checked separately.
    diagram = voronoi_diagram(MultiPoint(sites), edges=True)
    domain = target.safe.buffer(-(v.contact_radius(target, tool, 0)+margin),
                                quad_segs=32)
    for edge in v._segments(diagram):
        for line in v._segments(edge.intersection(domain)):
            if line.length > xy_step/10:
                guides.append(("fill", line))
    if len(guides) > max_paths:
        raise ValueError("feature guide budget exceeded")
    return guides


@dataclass(frozen=True)
class Candidate:
    plan: v.VPlan
    prior: v.VComposition
    proposed_length_mm: float
    omitted_air_length_mm: float
    max_cusp_mm: float
    guide_deviation_mm: float

    @property
    def composition(self):
        return v.VComposition(self.prior.target, self.prior.stages +
                              ((self.plan,) if self.plan.paths else ()))

    def section(self, depth):
        """Located residual and conditional new-removal/overlap intervals."""
        before = v.section_evidence(self.prior, depth)
        after = v.section_evidence(self.composition, depth)
        cut = v.section_evidence(self.plan, depth)
        return {"residual": after,
            "new_removal_mm2": (
                cut.known_free_inner.difference(before.known_free_outer).area,
                cut.known_free_outer.difference(before.known_free_inner).area),
            "overlap_mm2": (
                cut.known_free_inner.intersection(before.known_free_inner).area,
                cut.known_free_outer.intersection(before.known_free_outer).area)}

    def floor_cusp(self):
        """Locate capped-floor points where the requested cusp is unproved.

        Coverage at cap minus cusp proves the remaining axial thickness over
        the original capped floor is at most cusp. It does not certify walls
        or narrow features that have no capped floor.
        """
        target = self.prior.target
        cusp = _positive(self.max_cusp_mm, "maximum cusp")
        depth = max(0, target.cap_depth-cusp)
        section = v.section_evidence(self.composition, depth)
        floor = target.section(target.cap_depth, outer=True)
        unproved = floor.difference(section.known_free_inner)
        return {"maximum_cusp_mm": cusp, "depth_mm": depth,
                "floor": floor, "unproved": unproved,
                "status": "bounded" if unproved.is_empty else "partial"}


def generate(prior, tool, *, stepover_mm=1, max_cusp_mm=.25,
             xy_step_mm=1, margin_mm=.01, safe_z=3, clearance_slabs=8,
             max_paths=4000, max_sites=1500):
    """Generate verified contact/medial detail and conservatively pruned rest.

    Cusp controls contour pitch via the cutter's floor footprint. It is a
    scheduling criterion, not a global residual-thickness certificate; inspect
    located residuals, particularly at guide junctions and narrow features.
    Entries cut stock vertically and links retract above stock. Use explicit
    ``v_region.depth_passes`` and declared axial/setup limits before output.
    """
    if type(prior) is not v.VComposition or type(tool) is not v.VProfile:
        raise ValueError("composed stock and V profile required")
    prior = v.VComposition(prior.target, prior.stages)
    pitch = _positive(stepover_mm, "stepover")
    cusp = _positive(max_cusp_mm, "maximum cusp")
    step = _positive(xy_step_mm, "XY step")
    margin = _positive(margin_mm, "margin")
    safe_z = _positive(safe_z, "safe Z")
    if margin <= 1e-5 or cusp > tool.cutting_length:
        raise ValueError("feature error controls outside supported limits")
    slabs = _count(clearance_slabs, "clearance slabs")
    max_paths = _count(max_paths, "maximum paths")
    max_sites = _count(max_sites, "maximum sites")
    pitch = min(pitch, 2*tool.radius(cusp))
    guides = _guides(prior.target, tool, pitch, step, margin, max_paths, max_sites)
    clearance = _Clearance(prior, slabs)
    paths, proposed, omitted = [], 0.0, 0.0
    for role, line in guides:
        xy = _sample(line, step)
        if len(xy) < 2:
            continue
        path = v._depth_path(prior.target, tool, xy,
                            min(prior.target.cap_depth, tool.cutting_length), margin, role)
        if path is None:
            continue
        # Re-establish slope/containment before stock pruning. An invalid guide
        # must not disappear behind a claim that its stock was already removed.
        test = v.VPlan(prior.target, tool, (path,), v._motions((path,), safe_z),
                       safe_z, margin, pitch, "partial", "feature guide", "offset")
        v.verify(test)
        run = []
        for a, b in zip(path.points, path.points[1:]):
            length = math.dist(a[:2], b[:2])
            proposed += length
            if clearance.clear(tool, a, b):
                omitted += length
                if run:
                    paths.append(v.VPath(role, tuple(run)))
                    run = []
            else:
                if not run:
                    run.append(a)
                run.append(b)
        if run:
            paths.append(v.VPath(role, tuple(run)))
        if len(paths) > max_paths:
            raise ValueError("rest path budget exceeded")
    paths = tuple(paths)
    plan = v.verify(v.VPlan(prior.target, tool, paths, v._motions(paths, safe_z),
        safe_z, margin, pitch, "partial" if paths else "infeasible",
        "feature-guided residual candidate" if paths else
        "no unproved-clear admissible feature segments", "offset"))
    return Candidate(plan, prior, proposed, omitted, cusp,
                     prior.target.sagitta_mm + math.sqrt(2)*.5e-7)
