"""Bounded ball-finishing policies over the shared SF01 height-field queries.

Strategies propose paths; decoded ordered output owns their stock evidence.
Guided flow chooses the less variable direction and samples the transverse
surface more densely on steeper slopes. It is a heuristic, not an
optimal-path or constant-scallop claim. All dimensions are millimetres.
"""

from dataclasses import replace
import math

from ..cam_core import relief, relief_stock
from ..cam_core.occupancy import Box, OccupancySetup, ToolBody
from ..cam_core.ordered_job import AxialLimits, Job, JobMove, Stage, Transition


def _positive(value: float, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {label} required")


def _axis(start: float, stop: float, step: float) -> list[float]:
    count = math.ceil((stop - start) / step)
    if count > 2000:
        raise ValueError("finishing sample budget exceeded")
    return [start + (stop - start) * i / count for i in range(count + 1)]


def _guided_stations(target: relief.CubicRelief | relief.HeightField,
                     cross_axis: int, step: float) -> list[float]:
    bounds = target.stock_xy
    low, high = bounds[cross_axis], bounds[cross_axis + 2]
    other = 1 - cross_axis
    locations = [bounds[other] + (bounds[other + 2] - bounds[other]) * t
                 for t in (0.25, 0.5, 0.75)]

    def slope(at: float, along: float) -> float:
        delta = min(step / 20, (high - low) / 1000)
        a, b = max(low, at - delta), min(high, at + delta)
        xy0, xy1 = [along, along], [along, along]
        xy0[cross_axis], xy1[cross_axis] = a, b
        return (target.height_at(*xy1) - target.height_at(*xy0)) / (b - a)

    stations = [low]
    while stations[-1] < high:
        at = stations[-1]
        slopes = [slope(at, along) for along in locations]
        # Surface distance, rather than XY distance alone, controls spacing.
        spacing = step / math.sqrt(1 + max(s * s for s in slopes))
        stations.append(min(high, at + spacing))
        if len(stations) > 2000:
            raise ValueError("guided station budget exceeded")
    if type(target) is relief.HeightField:
        # Grid creases are explicit features, not differentiable surface data.
        stations.extend(target.xs if cross_axis == 0 else target.ys)
    else:
        # Isolate transverse extrema using the normal's horizontal component.
        # A root adds a pencil pass through each valley/ridge feature.
        sample = _axis(low, high, min(step, (high - low) / 32))
        along = locations[1]
        for a, b in zip(sample, sample[1:]):
            sa, sb = slope(a, along), slope(b, along)
            if sa == 0:
                stations.append(a)
            if sa * sb < 0:
                for _ in range(32):
                    mid = (a + b) / 2
                    if slope(mid, along) * sa > 0:
                        a = mid
                    else:
                        b = mid
                stations.append((a + b) / 2)
    ordered = sorted(set(stations))
    return [value for i, value in enumerate(ordered)
            if i == 0 or value - ordered[i - 1] > 1e-7]


def _raster_paths(target: relief.CubicRelief | relief.HeightField,
                  strategy: str, stepover: float,
                  sample_step: float) -> list[list[tuple[float, float]]]:
    x0, y0, x1, y1 = target.stock_xy
    if strategy == "parallel":
        along_axis = 0
        stations = _axis(y0, y1, stepover)
    else:
        along_axis = 0 if target.gradient_bounds[0] <= target.gradient_bounds[1] else 1
        stations = _guided_stations(target, 1 - along_axis, stepover)
    along = _axis((x0, y0)[along_axis], (x1, y1)[along_axis], sample_step)
    if target.gradient_bounds[along_axis] == 0:
        along = [along[0], along[-1]]
    paths = []
    for index, station in enumerate(stations):
        values = along if index % 2 == 0 else list(reversed(along))
        paths.append([(value, station) if along_axis == 0 else (station, value)
                      for value in values])
    return paths


def _waterline_paths(target: relief.CubicRelief | relief.HeightField,
                     tip, grid_step: float, level_step: float
                     ) -> list[list[tuple[float, float]]]:
    """March contact-height levels; ambiguous saddles stay separate segments.

    Unambiguous matching cell fragments join into constant-Z paths below.
    Branching intersections stay separate; no contour topology claim is made.
    """
    x0, y0, x1, y1 = target.stock_xy
    xs, ys = _axis(x0, x1, grid_step), _axis(y0, y1, grid_step)
    if len(xs) * len(ys) > 15000:
        raise ValueError("waterline sampling budget exceeded")
    heights = {(x, y): tip(x, y) for x in xs for y in ys}
    low, high = min(heights.values()), max(heights.values())
    if high - low <= 1e-10:
        return []  # A flat floor needs a surface pass; report residual honestly.
    levels = _axis(low, high, level_step)
    paths = []
    for level in levels:
        fragments = []
        for left, right in zip(xs, xs[1:]):
            for bottom, top in zip(ys, ys[1:]):
                corners = [(left, bottom), (right, bottom),
                           (right, top), (left, top)]
                crossings = []
                for a, b in zip(corners, corners[1:] + corners[:1]):
                    ha, hb = heights[a], heights[b]
                    if (ha <= level < hb) or (hb <= level < ha):
                        t = (level - ha) / (hb - ha)
                        crossings.append((a[0] + t * (b[0] - a[0]),
                                          a[1] + t * (b[1] - a[1])))
                for i in range(0, len(crossings) - 1, 2):
                    if math.dist(crossings[i], crossings[i + 1]) > 1e-9:
                        fragments.append(crossings[i:i + 2])
        paths.extend(_join_fragments(fragments))
    return paths


def _join_fragments(fragments: list[list[tuple[float, float]]]
                    ) -> list[list[tuple[float, float]]]:
    """Join only degree-two shared contour endpoints at numerical precision."""
    def key(point):
        return tuple(round(value, 9) for value in point)

    incident = {}
    for index, fragment in enumerate(fragments):
        for point in fragment:
            incident.setdefault(key(point), []).append(index)
    unused = set(range(len(fragments)))
    paths = []
    while unused:
        index = min(unused)
        unused.remove(index)
        path = list(fragments[index])
        for reverse in (False, True):
            if reverse:
                path.reverse()
            while len(incident[key(path[-1])]) == 2:
                following = [i for i in incident[key(path[-1])] if i in unused]
                if len(following) != 1:
                    break
                next_index = following[0]
                a, b = fragments[next_index]
                path.append(b if key(a) == key(path[-1]) else a)
                unused.remove(next_index)
        paths.append(path)
    return paths


def generate(target: relief.CubicRelief | relief.HeightField, strategy: str, *,
             tool: ToolBody, stepover_mm: float, sample_step_mm: float,
             safe_z_mm: float, feed_mm_min: float, plunge_mm_min: float,
             rpm: float, axial_limits: AxialLimits,
             pitch_mm: float = 0.1, contact_tolerance_mm: float = 0.001,
             allowance_mm: float = 0.001,
             waterline_step_mm: float = 0.1,
             fixtures: tuple[Box, ...] = ()) -> Job:
    """Propose one complete high-link finishing job with explicit tool/process.

    ``stepover_mm`` is a maximum XY spacing, not a finish guarantee. Bilinear
    creases and analytic transverse extrema add guided pencil passes. Waterline
    fragments retain level Z individually and may leave broad shallow areas.
    Feed entries cut virgin stock under conservative absolute axial limits;
    no pre-roughing, cleared links, engagement or load is silently assumed.
    Call ``ordered_output.emit`` to obtain actual decoded stock/body evidence.
    """
    if type(target) not in (relief.CubicRelief, relief.HeightField):
        raise ValueError("supported relief target required")
    if strategy not in ("parallel", "waterline", "guided"):
        raise ValueError("unsupported relief finishing strategy")
    if type(tool) is not ToolBody or type(axial_limits) is not AxialLimits:
        raise ValueError("declared tool body and axial limits required")
    for value, label in ((stepover_mm, "stepover"), (sample_step_mm, "sample step"),
                         (safe_z_mm, "safe Z"), (feed_mm_min, "feed"),
                         (plunge_mm_min, "plunge feed"), (rpm, "spindle speed"),
                         (pitch_mm, "stock pitch"), (contact_tolerance_mm, "contact tolerance"),
                         (allowance_mm, "allowance"), (waterline_step_mm, "waterline step")):
        _positive(value, label)
    if allowance_mm < 0.0002 * (1 + sum(target.gradient_bounds)):
        raise ValueError("allowance does not cover four-decimal output rounding")
    radius, length = tool.bands[0].radius_mm, tool.bands[0].top_mm
    cache = {}

    def tip(x: float, y: float) -> float:
        key = x, y
        if key not in cache:
            cache[key] = relief.contact(target, radius, x, y,
                                       tolerance_mm=contact_tolerance_mm).upper_tip_z_mm
        return cache[key]

    if strategy == "waterline":
        paths = _waterline_paths(target, tip, stepover_mm, waterline_step_mm)
    else:
        paths = _raster_paths(target, strategy, stepover_mm, sample_step_mm)
    if not paths or sum(map(len, paths)) > 50000:
        raise ValueError("strategy has no paths or exceeds motion budget")
    x0, y0, x1, y1 = target.stock_xy
    safe = tuple(round(value, 4) for value in (x0, y0, safe_z_mm))
    if safe[2] <= 0:
        raise ValueError("safe Z vanishes at output precision")
    at = safe
    moves = []

    def add(role: str, end: tuple[float, float, float], feed: float = 0.0) -> None:
        nonlocal at
        # The caller receives the actual four-decimal proposal. Coalesce tiny
        # contour fragments/links before the writer's zero-motion refusal.
        end = tuple(round(value, 4) for value in end)
        if end != at:
            moves.append(JobMove(role, at, end, feed))
            at = end

    gx, gy = target.gradient_bounds
    for path in paths:
        margins = [0.0] * len(path)
        for index, (a, b) in enumerate(zip(path, path[1:])):
            margin = (gx * abs(b[0] - a[0]) + gy * abs(b[1] - a[1])) / 2
            margins[index] = max(margins[index], margin)
            margins[index + 1] = max(margins[index + 1], margin)
        points = [(x, y, tip(x, y) + margins[i] + allowance_mm)
                  for i, (x, y) in enumerate(path)]
        if strategy == "waterline":
            level = max(p[2] for p in points)
            points = [(x, y, level) for x, y, _ in points]
        if max(p[2] for p in points) >= safe_z_mm:
            raise ValueError("safe Z is below the proposed path")
        add("rapid", (points[0][0], points[0][1], safe_z_mm))
        add("entry", points[0], plunge_mm_min)
        for point in points[1:]:
            add("cut", point, feed_mm_min)
        add("retract", (at[0], at[1], safe_z_mm), plunge_mm_min)
    add("rapid", safe)
    operation = relief_stock.ReliefOperation(strategy, tool.tool_id, radius,
        length, target, strategy, pitch_mm, contact_tolerance_mm)
    stage = Stage(strategy, tool.tool_id, tuple(moves), rpm,
                  operation=operation, source_revision=target.fingerprint,
                  axial_limits=axial_limits)
    setup = OccupancySetup("program", Box("relief stock", (x0, y0,
        -target.stock_depth_mm, x1, y1, 0)), fixtures, (tool,))
    return Job(target.fingerprint, (stage,), safe, occupancy_setup=setup)


def sequence(*jobs: Job, boundary: str = "split") -> Job:
    """Compose finish/rest proposals on one design, preserving cumulative stock.

    Callers choose order and tool installations. Every input must start/end at
    the same safe pose and share physical stock/fixtures. Each subsequent stage
    declares an operator installation boundary, not observed machine telemetry.
    """
    if not jobs or boundary not in ("split", "pause"):
        raise ValueError("jobs and supported transition boundary required")
    if any(type(job) is not Job for job in jobs):
        raise ValueError("relief jobs required")
    first = jobs[0]
    stages, tools = [], {}
    for job in jobs:
        if (type(job) is not Job or len(job.stages) != 1 or
                job.source_fingerprint != first.source_fingerprint or
                job.translation_xyz_mm != first.translation_xyz_mm or
                job.program_frame != first.program_frame or
                job.work_frame != first.work_frame or
                job.stock_present != first.stock_present or
                job.source_kind != first.source_kind or
                job.units != first.units or
                job.initial_tip != first.initial_tip or
                job.stages[0].motions[-1].end != first.initial_tip or
                job.occupancy_setup is None or first.occupancy_setup is None or
                job.occupancy_setup.stock != first.occupancy_setup.stock or
                job.occupancy_setup.fixtures != first.occupancy_setup.fixtures):
            raise ValueError("incompatible relief sequence jobs")
        stage = job.stages[0]
        if stage.relief_operation is None:
            raise ValueError("relief-only sequence required")
        for tool in job.occupancy_setup.tools:
            if tool.tool_id in tools and tools[tool.tool_id] != tool:
                raise ValueError("tool identity has differing bodies")
            tools[tool.tool_id] = tool
        name = f"{len(stages) + 1}-{stage.id}"
        transition = (Transition("operator", boundary, stage.tool_id,
            first.initial_tip, f"install-{name}") if stages else None)
        stages.append(replace(stage, id=name,
            operation=replace(stage.operation, name=name), transition=transition))
    return replace(first, stages=tuple(stages), occupancy_setup=replace(
        first.occupancy_setup, tools=tuple(tools.values())))


def motion_cost(job: Job, *, rapid_mm_min: float) -> dict[str, float | int | str]:
    """Programmed distance/feed estimate; excludes acceleration and tool changes."""
    _positive(rapid_mm_min, "rapid speed")
    rapid = feed = minutes = 0.0
    count = 0
    for stage in job.stages:
        for move in stage.motions:
            distance = math.dist(move.start, move.end)
            if move.role in ("rapid", "rapid_retract"):
                rapid += distance
                minutes += distance / rapid_mm_min
            else:
                feed += distance
                minutes += distance / move.feed
            count += 1
    return {"feed_distance_mm": feed, "rapid_distance_mm": rapid,
            "estimated_minutes": minutes, "moves": count,
            "timing_scope": "constant programmed feed; no dynamics or installation time"}
