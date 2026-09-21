"""Pure milling and through-cut pass planning.

The planner composes caller-owned recommendation strategies with the dimensional
solver.  It does not contain cutting data, mutate CamBam documents, or establish
that a candidate is safe for production machining.
"""

from dataclasses import dataclass, fields
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Optional, Sequence, Tuple

from .machining_calculations import (
    ActiveConstraint,
    MillingConstraints,
    MillingSolution,
    solve_milling_constraints,
)
from .machining_recommendations import (
    RecommendationContext,
    RecommendationResult,
    RecommendationStrategy,
    recommend_milling,
)


@dataclass(frozen=True)
class PlanningDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class DepthPassPlan:
    units: str
    mode: str
    stock_thickness: float
    cut_through: float
    total_depth: float
    pass_count: int
    depth_increment: float
    rounding_increment: float
    nominal_overshoot: float
    pass_depths: Tuple[float, ...]
    final_pass_depth: float
    final_pass_stock: float
    final_pass_cut_through: float
    final_stock_fraction: float
    recommendation_met: bool
    diagnostics: Tuple[PlanningDiagnostic, ...]


@dataclass(frozen=True)
class MillingPlan:
    context: RecommendationContext
    recommendations: RecommendationResult
    solution: MillingSolution
    depth_plan: Optional[DepthPassPlan]
    safe_axial_depth: Optional[float]
    plunge_feed_rate: Optional[float]
    ramp_feed_rate: Optional[float]
    helical_feed_rate: Optional[float]
    stepover: Optional[float]
    stepover_fraction: Optional[float]
    missing_requirements: Tuple[str, ...]
    active_constraints: Tuple[ActiveConstraint, ...]
    diagnostics: Tuple[PlanningDiagnostic, ...]
    safety_notice: str = (
        "Starting recommendation only: verify the exact tool manufacturer's "
        "guidance, machine limits, workholding, rigidity and chip evacuation, "
        "then use supervised test cuts before production machining."
    )


def _positive_decimal(value, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def plan_depth_passes(
    *, units: str, stock_thickness: float, cut_through: float,
    pass_count: Optional[int] = None,
    max_depth_increment: Optional[float] = None,
    rounding_increment: Optional[float] = None,
) -> DepthPassPlan:
    """Balance a through-cut across passes under one caller-supplied constraint.

    ``max_depth_increment`` is already judged safe for the exact material, tool,
    setup and machine.  The one-third final-stock rule is advisory, not a veto.
    """
    if units not in ("mm", "in"):
        raise ValueError("units must be exactly 'mm' or 'in'")
    if (pass_count is None) == (max_depth_increment is None):
        raise ValueError(
            "exactly one of pass_count or max_depth_increment is required")
    stock = _positive_decimal(stock_thickness, "stock_thickness")
    through = _positive_decimal(cut_through, "cut_through")
    total = stock + through
    if total > Decimal("1000000000"):
        raise ValueError(
            "stock_thickness plus cut_through exceeds the supported numeric range")
    quantum = _positive_decimal(
        rounding_increment if rounding_increment is not None
        else (0.1 if units == "mm" else 0.001),
        "rounding_increment",
    )
    if pass_count is not None:
        if (isinstance(pass_count, bool) or not isinstance(pass_count, int)
                or not 1 <= pass_count <= 10000):
            raise ValueError("pass_count must be an integer from 1 to 10000")
    maximum = (None if max_depth_increment is None else
               _positive_decimal(max_depth_increment, "max_depth_increment"))

    def rounded(count: int) -> Decimal:
        equal = total / Decimal(count)
        steps = (equal / quantum).to_integral_value(rounding=ROUND_CEILING)
        candidate = steps * quantum
        return candidate + quantum if candidate <= equal else candidate

    diagnostics = []
    if pass_count is not None:
        mode = "pass_count"
        count = pass_count
        increment = rounded(count)
    else:
        mode = "max_depth_increment"
        count = int((total / maximum).to_integral_value(rounding=ROUND_FLOOR)) + 1
        if count > 10000:
            raise ValueError(
                "max_depth_increment would require more than 10000 passes")
        increment = rounded(count)
        while increment > maximum and count < 10000:
            count += 1
            increment = rounded(count)
        if increment > maximum:
            increment = total / Decimal(count)
            diagnostics.append(PlanningDiagnostic(
                "DEPTH_ROUNDING_RELAXED",
                "No upward-rounded increment fits the requested maximum; the "
                "unrounded equal increment is returned to honor that constraint.",
            ))
    if increment > Decimal("1000000000"):
        raise ValueError(
            "rounding_increment produces an unsupported depth increment")

    penultimate = Decimal(count - 1) * increment
    if penultimate >= total:
        increment = total / Decimal(count)
        penultimate = Decimal(count - 1) * increment
        diagnostics.append(PlanningDiagnostic(
            "DEPTH_ROUNDING_RELAXED",
            "Upward rounding would complete the cut in fewer than the requested "
            "passes; the unrounded equal increment preserves pass count.",
        ))
    final_stock = max(Decimal(0), stock - penultimate)
    final_pass = total - penultimate
    final_through = final_pass - final_stock
    fraction = final_stock / final_pass
    minimum_fraction = Decimal(1) / Decimal(3)
    required_stock = minimum_fraction * through / (Decimal(1) - minimum_fraction)
    met = final_stock >= required_stock
    if not met:
        diagnostics.append(PlanningDiagnostic(
            "FINAL_STOCK_ENGAGEMENT_LOW",
            "The requested constraint leaves less than one third of the final pass "
            "in stock. Review workholding and confirm the user's intent.",
        ))
    depths = tuple(float(min(Decimal(index) * increment, total))
                   for index in range(1, count + 1))
    return DepthPassPlan(
        units, mode, float(stock), float(through), float(total), count,
        float(increment), float(quantum),
        float(max(Decimal(0), Decimal(count) * increment - total)), depths,
        float(final_pass), float(final_stock), float(final_through),
        float(fraction), met, tuple(diagnostics),
    )


_SOLVER_FIELDS = tuple(
    item.name for item in fields(MillingConstraints) if item.name != "units")
_ENTRY_FIELDS = ("plunge_feed_rate", "ramp_feed_rate", "helical_feed_rate")


def plan_milling(
    context: RecommendationContext,
    strategies: Sequence[RecommendationStrategy],
    *,
    fixed_values: Optional[MillingConstraints] = None,
    stock_thickness: Optional[float] = None,
    cut_through: Optional[float] = None,
    rounding_increment: Optional[float] = None,
) -> MillingPlan:
    """Return one provenance-bearing candidate without mutating a document.

    Explicit ``fixed_values`` win over strategy suggestions.  When stock and
    cut-through are supplied, axial depth is treated as a safe maximum and the
    balanced actual increment is used for MRR/power diagnostics.
    """
    if not isinstance(context, RecommendationContext):
        raise TypeError("context must be RecommendationContext")
    selected = recommend_milling(context, strategies)
    if fixed_values is None:
        fixed_values = MillingConstraints(context.units)
    if not isinstance(fixed_values, MillingConstraints):
        raise TypeError("fixed_values must be MillingConstraints")
    if fixed_values.units != context.units:
        raise ValueError("fixed_values and recommendation context units must match")
    if (stock_thickness is None) != (cut_through is None):
        raise ValueError("stock_thickness and cut_through must be supplied together")

    supplied = {name: getattr(fixed_values, name) for name in _SOLVER_FIELDS
                if getattr(fixed_values, name) is not None}
    for name, expected in (("cutter_diameter", context.tool.cutter_diameter),
                           ("effective_flutes", context.tool.effective_flutes)):
        if name in supplied and supplied[name] != expected:
            raise ValueError(f"fixed {name} conflicts with the tool profile")
        supplied[name] = expected

    planning_constraints = []
    by_field = {item.field: item for item in selected.recommendations}
    fixed_fields = set(supplied)
    fixed_fields.update(item.field for item in selected.recommendations if item.fixed)
    superseded_pairs = {
        "surface_speed": "spindle_speed", "spindle_speed": "surface_speed",
        "chip_load": "feed_rate", "feed_rate": "chip_load",
    }
    for name in _SOLVER_FIELDS:
        item = by_field.get(name)
        if item is None or name in supplied:
            continue
        if not item.fixed and superseded_pairs.get(name) in fixed_fields:
            continue
        value = item.value
        limit = (context.machine.max_spindle_speed if name == "spindle_speed"
                 else context.machine.max_feed_rate if name == "feed_rate" else None)
        if limit is not None and value > limit and not item.fixed:
            planning_constraints.append(ActiveConstraint(name, value, limit, limit))
            value = limit
            # The capped operational setting is authoritative for achieved values;
            # do not leave its non-fixed, uncapped conjugate as an overconstraint.
            conjugate = superseded_pairs[name]
            if conjugate not in fixed_fields:
                supplied.pop(conjugate, None)
        supplied[name] = value

    safe_axial = supplied.get("axial_depth")
    depth_plan = None
    if stock_thickness is not None:
        if safe_axial is None:
            missing_depth = (
                "through-cut pass planning requires a caller-supplied safe axial_depth",
            )
        else:
            depth_plan = plan_depth_passes(
                units=context.units, stock_thickness=stock_thickness,
                cut_through=cut_through, max_depth_increment=safe_axial,
                rounding_increment=rounding_increment,
            )
            supplied["axial_depth"] = depth_plan.depth_increment
            missing_depth = ()
    else:
        missing_depth = ()

    solution = solve_milling_constraints(
        MillingConstraints(context.units, **supplied),
        context.machine.as_solver_limits(),
    )
    radial = solution.radial_engagement
    fraction = (None if radial is None else
                radial / context.tool.cutter_diameter)
    entry = {name: by_field.get(name) for name in _ENTRY_FIELDS}
    missing = tuple(dict.fromkeys(
        selected.missing_requirements + missing_depth + solution.missing_requirements))
    active = tuple(planning_constraints) + solution.active_constraints
    diagnostics = list(depth_plan.diagnostics if depth_plan else ())
    for field, value, limit in (
        ("cutting_power", solution.cutting_power,
         context.machine.max_cutting_power),
        ("torque", solution.torque, context.machine.max_torque),
    ):
        if value is not None and limit is not None and value > limit:
            diagnostics.append(PlanningDiagnostic(
                f"MACHINE_{field.upper()}_EXCEEDED",
                f"Candidate {field} {value:g} exceeds the declared machine "
                f"cap {limit:g}; no unevidenced derating was applied.",
            ))
    return MillingPlan(
        context=context, recommendations=selected, solution=solution,
        depth_plan=depth_plan, safe_axial_depth=safe_axial,
        plunge_feed_rate=entry["plunge_feed_rate"].value
        if entry["plunge_feed_rate"] else None,
        ramp_feed_rate=entry["ramp_feed_rate"].value
        if entry["ramp_feed_rate"] else None,
        helical_feed_rate=entry["helical_feed_rate"].value
        if entry["helical_feed_rate"] else None,
        stepover=radial, stepover_fraction=fraction,
        missing_requirements=missing, active_constraints=active,
        diagnostics=tuple(diagnostics),
    )
