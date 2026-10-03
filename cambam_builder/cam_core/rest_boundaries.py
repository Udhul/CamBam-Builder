"""Derived planar Pocket targets from verified cylindrical predecessor stock.

Pure floor rest, useful cutter centers and native Pocket boundaries are distinct
sets. These GEOS constructions have the same conditional polygonal enclosure
limits as polygon_rest; they do not prescribe the native planner's motion.
"""

from dataclasses import dataclass
import hashlib
import math

from . import polygon_rest, replay


@dataclass(frozen=True)
class Boundary:
    shell: tuple
    holes: tuple

    @property
    def geometry(self):
        from shapely.geometry import Polygon
        return Polygon(self.shell, self.holes)


@dataclass(frozen=True)
class RestBoundaries:
    target: replay.Target
    prior_motion: str
    radius_mm: float
    overlap_mm: float
    margin_mm: float
    windows: tuple
    pure_rest_area_mm2: tuple
    reachable_rest_area_mm2: float

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


def _polygons(shape):
    if shape.is_empty:
        return ()
    if shape.geom_type == "Polygon":
        return (shape,)
    if hasattr(shape, "geoms"):
        return tuple(p for child in shape.geoms for p in _polygons(child))
    return ()


def derive(prior_trace, *, radius_mm, overlap_mm, margin_mm=0.01):
    """Return editable full-depth windows with deliberate cleared-space overlap.

    Only straight planar targets and complete, constant-depth cylinder sweeps
    are admitted. Unsupported variable-depth/rounded/V paths need the separate
    generated-output route. No cleared entry is inferred from these boundaries.
    """
    from shapely.ops import unary_union

    for value, positive in ((radius_mm, True), (overlap_mm, False),
                            (margin_mm, True)):
        if (type(value) not in (int, float) or not math.isfinite(value) or
                value < 0 or positive and value == 0):
            raise ValueError("finite positive radius/margin and nonnegative overlap required")
    if type(prior_trace) is not replay.Trace or not prior_trace.operations:
        raise ValueError("complete cylindrical predecessor required")
    target = prior_trace.operations[0].target
    if (not target.region_shell or target.cone_spine or target.polygon or
            target.inset_per_depth or target.island or
            any(op.target != target or op.tool.kind != "cylinder" or
                op.tool.radius <= radius_mm for op in prior_trace.operations) or
            type(prior_trace.items[-1]) is not replay.Event or
            prior_trace.items[-1].kind != "spindle_stop"):
        raise ValueError("one planar design and smaller cylindrical cleanup required")
    stock = replay.replay(prior_trace, expected_source=prior_trace.source_fingerprint)
    if (not stock.cuts or any(cut.bottom_start is not None for cut in stock.cuts) or
            min(cut.bottom for cut in stock.cuts) != -target.depth):
        raise ValueError("constant-depth predecessor sweeps reaching the floor required")
    design = polygon_rest._geometry(target)
    if not design.is_valid or design.is_empty:
        raise ValueError("valid planar design required")
    cleared = polygon_rest._cut_polygon(stock.cuts, target.depth, radial_error=-1e-6)
    # Inward compensation reserves a numerical margin. Buffering centers back
    # out produces a Pocket target, rather than asking Pocket to cut pure rest.
    feasible = design.buffer(-(radius_mm + margin_mm), quad_segs=128)
    reachable = feasible.buffer(radius_mm, quad_segs=128).intersection(design)
    useful = reachable.difference(cleared)
    pieces = []
    for component in _polygons(useful):
        centers = feasible.intersection(component.buffer(
            radius_mm + overlap_mm, quad_segs=128))
        window = centers.buffer(radius_mm, quad_segs=128).intersection(design)
        pieces.extend(_polygons(window))
    # Overlap can connect components: merge before emitting native selections.
    windows = []
    for window in sorted(_polygons(unary_union(pieces)),
                         key=lambda p: (*p.bounds, p.area)):
        # Region XML uses nine decimal places. Reserve the stated margin for
        # polygonization, bounded simplification and that interchange rounding.
        window = window.simplify(margin_mm / 4, preserve_topology=True)
        boundary = Boundary(
            tuple(tuple(round(v, 9) for v in p) for p in tuple(window.exterior.coords)[:-1]),
            tuple(tuple(tuple(round(v, 9) for v in p) for p in tuple(r.coords)[:-1])
                  for r in window.interiors))
        window = boundary.geometry
        if not window.is_valid:
            raise ValueError("derived Pocket topology cannot survive interchange precision")
        if (window.area <= 1e-10 or
                window.buffer(-radius_mm, quad_segs=128).is_empty):
            continue
        if not design.covers(window):
            raise ValueError("derived Pocket boundary escapes original design")
        windows.append(boundary)
    if not windows:
        raise ValueError("no useful representable native Pocket rest windows")
    return RestBoundaries(target, prior_trace.motion_fingerprint, float(radius_mm),
                          float(overlap_mm), float(margin_mm), tuple(windows),
                          polygon_rest._areas(target, stock.cuts, target.depth),
                          useful.area)
