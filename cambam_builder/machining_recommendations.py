"""Provenance-preserving milling recommendation profiles and extension API.

This module contains no built-in cutting-data catalog.  Callers provide fixed
values, diameter tables, or pure strategy callables.  The resulting suggestions
remain separate from the arithmetic solver and from CamBam document mutation.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .machining_calculations import MachineLimits


_PROVENANCE_KINDS = (
    "manufacturer_starting_point", "measured_shop_policy", "user_override",
)
_ENTRY_FIELDS = {
    "plunge_feed_rate": "plunge",
    "ramp_feed_rate": "ramp",
    "helical_feed_rate": "helical",
}


class RecommendationError(ValueError):
    """Raised when recommendation data is ambiguous, incompatible, or unsafe."""


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _optional_positive(value, name: str) -> Optional[float]:
    return None if value is None else _positive(value, name)


def _units(units: str) -> str:
    if units not in ("mm", "in"):
        raise ValueError("units must be exactly 'mm' or 'in'")
    return units


def _field_unit(field: str, units: str) -> str:
    metric = {
        "surface_speed": "m/min", "spindle_speed": "rpm",
        "chip_load": "mm/tooth", "effective_flutes": "count",
        "feed_rate": "mm/min", "plunge_feed_rate": "mm/min",
        "ramp_feed_rate": "mm/min", "helical_feed_rate": "mm/min",
        "axial_depth": "mm", "radial_engagement": "mm",
        "specific_cutting_force": "N/mm^2",
    }
    imperial = {
        "surface_speed": "ft/min", "spindle_speed": "rpm",
        "chip_load": "in/tooth", "effective_flutes": "count",
        "feed_rate": "in/min", "plunge_feed_rate": "in/min",
        "ramp_feed_rate": "in/min", "helical_feed_rate": "in/min",
        "axial_depth": "in", "radial_engagement": "in",
        "specific_cutting_force": "lbf/in^2",
    }
    try:
        return (metric if units == "mm" else imperial)[field]
    except KeyError as exc:
        raise ValueError(f"unsupported recommendation field: {field!r}") from exc


@dataclass(frozen=True)
class RecommendationProvenance:
    kind: str
    source: str
    reference: str
    version_or_date: str

    def __post_init__(self):
        if self.kind not in _PROVENANCE_KINDS:
            raise ValueError(
                "kind must be manufacturer_starting_point, measured_shop_policy, "
                "or user_override"
            )
        _text(self.source, "source")
        _text(self.reference, "reference")
        _text(self.version_or_date, "version_or_date")


@dataclass(frozen=True)
class ApplicableRange:
    parameter: str
    units: str
    minimum: float
    maximum: float

    def __post_init__(self):
        _text(self.parameter, "parameter")
        _text(self.units, "units")
        low = _positive(self.minimum, "minimum")
        high = _positive(self.maximum, "maximum")
        if low > high:
            raise ValueError("minimum cannot exceed maximum")

    def contains(self, value: float) -> bool:
        checked = _positive(value, self.parameter)
        return self.minimum <= checked <= self.maximum


@dataclass(frozen=True)
class ToolProfile:
    identifier: str
    units: str
    cutter_diameter: float
    effective_flutes: int
    tool_family: str
    substrate: str = "unspecified"
    coating: str = "unspecified"
    entry_modes: Tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.identifier, "tool identifier")
        _units(self.units)
        _positive(self.cutter_diameter, "cutter_diameter")
        if (isinstance(self.effective_flutes, bool)
                or not isinstance(self.effective_flutes, Integral)
                or self.effective_flutes <= 0):
            raise ValueError("effective_flutes must be a positive integer")
        _text(self.tool_family, "tool_family")
        _text(self.substrate, "substrate")
        _text(self.coating, "coating")
        modes = tuple(self.entry_modes)
        if len(set(modes)) != len(modes) or any(
                mode not in ("plunge", "ramp", "helical") for mode in modes):
            raise ValueError("entry_modes must be unique plunge, ramp, or helical values")
        object.__setattr__(self, "entry_modes", modes)


@dataclass(frozen=True)
class MaterialProfile:
    identifier: str
    material: str
    condition: str

    def __post_init__(self):
        _text(self.identifier, "material identifier")
        _text(self.material, "material")
        _text(self.condition, "condition")


@dataclass(frozen=True)
class MachineCapabilities:
    identifier: str
    units: str
    max_spindle_speed: Optional[float] = None
    max_feed_rate: Optional[float] = None
    max_cutting_power: Optional[float] = None
    max_torque: Optional[float] = None
    min_spindle_speed: Optional[float] = None
    min_feed_rate: Optional[float] = None

    def __post_init__(self):
        _text(self.identifier, "machine identifier")
        _units(self.units)
        for name in ("max_spindle_speed", "max_feed_rate", "max_cutting_power",
                     "max_torque", "min_spindle_speed", "min_feed_rate"):
            _optional_positive(getattr(self, name), name)
        _validate_bounds(self.min_spindle_speed, self.max_spindle_speed,
                         "spindle speed")
        _validate_bounds(self.min_feed_rate, self.max_feed_rate, "feed rate")

    def as_solver_limits(self) -> MachineLimits:
        """Return the subset of capabilities understood by the 9a solver."""
        return MachineLimits(
            max_spindle_speed=self.max_spindle_speed,
            max_feed_rate=self.max_feed_rate,
            min_spindle_speed=self.min_spindle_speed,
            min_feed_rate=self.min_feed_rate,
        )


def _validate_bounds(minimum: Optional[float], maximum: Optional[float],
                     name: str) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise RecommendationError(f"minimum {name} cannot exceed maximum {name}")


@dataclass(frozen=True)
class OperatingConstraints:
    """Optional setup/job restrictions intersected with machine capabilities."""

    units: str
    min_spindle_speed: Optional[float] = None
    max_spindle_speed: Optional[float] = None
    min_feed_rate: Optional[float] = None
    max_feed_rate: Optional[float] = None

    def __post_init__(self):
        _units(self.units)
        for name in ("min_spindle_speed", "max_spindle_speed",
                     "min_feed_rate", "max_feed_rate"):
            _optional_positive(getattr(self, name), name)
        _validate_bounds(self.min_spindle_speed, self.max_spindle_speed,
                         "spindle speed")
        _validate_bounds(self.min_feed_rate, self.max_feed_rate, "feed rate")


@dataclass(frozen=True)
class RecommendationContext:
    tool: ToolProfile
    material: MaterialProfile
    machine: MachineCapabilities
    operation: str = "milling"
    operating_constraints: Optional[OperatingConstraints] = None

    def __post_init__(self):
        if not isinstance(self.tool, ToolProfile):
            raise TypeError("tool must be a ToolProfile")
        if not isinstance(self.material, MaterialProfile):
            raise TypeError("material must be a MaterialProfile")
        if not isinstance(self.machine, MachineCapabilities):
            raise TypeError("machine must be MachineCapabilities")
        if self.tool.units != self.machine.units:
            raise RecommendationError("tool and machine units must match")
        if (self.operating_constraints is not None
                and not isinstance(self.operating_constraints, OperatingConstraints)):
            raise TypeError("operating_constraints must be OperatingConstraints")
        if (self.operating_constraints is not None
                and self.operating_constraints.units != self.tool.units):
            raise RecommendationError(
                "operating constraints and tool units must match")
        _text(self.operation, "operation")
        self.effective_limits()

    @property
    def units(self) -> str:
        return self.tool.units

    def effective_limits(self) -> MachineLimits:
        """Intersect machine capability with narrower setup/job restrictions."""
        job = self.operating_constraints

        def lower(machine_value, job_value):
            values = [value for value in (machine_value, job_value)
                      if value is not None]
            return max(values) if values else None

        def upper(machine_value, job_value):
            values = [value for value in (machine_value, job_value)
                      if value is not None]
            return min(values) if values else None

        limits = MachineLimits(
            max_spindle_speed=upper(
                self.machine.max_spindle_speed,
                job.max_spindle_speed if job else None),
            max_feed_rate=upper(
                self.machine.max_feed_rate,
                job.max_feed_rate if job else None),
            min_spindle_speed=lower(
                self.machine.min_spindle_speed,
                job.min_spindle_speed if job else None),
            min_feed_rate=lower(
                self.machine.min_feed_rate,
                job.min_feed_rate if job else None),
        )
        _validate_bounds(limits.min_spindle_speed, limits.max_spindle_speed,
                         "spindle speed after intersecting machine and job ranges")
        _validate_bounds(limits.min_feed_rate, limits.max_feed_rate,
                         "feed rate after intersecting machine and job ranges")
        return limits


@dataclass(frozen=True)
class Recommendation:
    field: str
    value: float
    units: str
    provenance: RecommendationProvenance
    applicable_ranges: Tuple[ApplicableRange, ...]
    fixed: bool = False
    entry_mode: Optional[str] = None
    rule: Optional[str] = None

    def __post_init__(self):
        _text(self.field, "field")
        _positive(self.value, self.field)
        _text(self.units, "units")
        if not isinstance(self.provenance, RecommendationProvenance):
            raise TypeError("provenance must be RecommendationProvenance")
        ranges = tuple(self.applicable_ranges)
        if not ranges or any(not isinstance(item, ApplicableRange) for item in ranges):
            raise ValueError("applicable_ranges must contain at least one ApplicableRange")
        object.__setattr__(self, "applicable_ranges", ranges)
        if not isinstance(self.fixed, bool):
            raise ValueError("fixed must be a boolean")
        expected_mode = _ENTRY_FIELDS.get(self.field)
        if expected_mode is None:
            if self.entry_mode is not None:
                raise ValueError("entry_mode is only valid for an entry-feed field")
        elif self.entry_mode != expected_mode or not self.rule or not self.rule.strip():
            raise ValueError(
                f"{self.field} requires entry_mode={expected_mode!r} and its own rule"
            )
        if self.fixed and self.provenance.kind != "user_override":
            raise ValueError("fixed recommendations must be user_override values")


@dataclass(frozen=True)
class StrategyResult:
    recommendations: Tuple[Recommendation, ...] = ()
    missing_requirements: Tuple[str, ...] = ()

    def __post_init__(self):
        recommendations = tuple(self.recommendations)
        missing = tuple(self.missing_requirements)
        if any(not isinstance(item, Recommendation) for item in recommendations):
            raise TypeError("recommendations must contain Recommendation values")
        if any(not isinstance(item, str) or not item for item in missing):
            raise ValueError("missing_requirements must contain non-empty strings")
        object.__setattr__(self, "recommendations", recommendations)
        object.__setattr__(self, "missing_requirements", missing)


@runtime_checkable
class RecommendationStrategy(Protocol):
    """Extension contract for deterministic, side-effect-free strategies."""

    @property
    def name(self) -> str:
        ...

    def recommend(self, context: RecommendationContext) -> StrategyResult:
        ...


@dataclass(frozen=True)
class ProfileRecommendationStrategy:
    name: str
    recommendations: Tuple[Recommendation, ...]

    def __post_init__(self):
        _text(self.name, "strategy name")
        values = tuple(self.recommendations)
        if any(not isinstance(item, Recommendation) for item in values):
            raise TypeError("recommendations must contain Recommendation values")
        object.__setattr__(self, "recommendations", values)

    def recommend(self, context: RecommendationContext) -> StrategyResult:
        return StrategyResult(self.recommendations)


@dataclass(frozen=True)
class FixedRecommendationStrategy(ProfileRecommendationStrategy):
    """A static strategy containing only explicit user-fixed values."""

    def __post_init__(self):
        super().__post_init__()
        if any(not item.fixed for item in self.recommendations):
            raise ValueError("FixedRecommendationStrategy values must have fixed=True")


@dataclass(frozen=True)
class DiameterRecommendationTable:
    """User-supplied chip-load or surface-speed values keyed by cutter diameter."""

    field: str
    diameter_units: str
    value_units: str
    points: Tuple[Tuple[float, float], ...]
    provenance: RecommendationProvenance
    tool_identifier: str
    material_identifier: str
    operation: str = "milling"

    def __post_init__(self):
        if self.field not in ("surface_speed", "chip_load"):
            raise ValueError("table field must be surface_speed or chip_load")
        _units(self.diameter_units)
        _text(self.value_units, "value_units")
        _text(self.tool_identifier, "tool_identifier")
        _text(self.material_identifier, "material_identifier")
        _text(self.operation, "operation")
        if not isinstance(self.provenance, RecommendationProvenance):
            raise TypeError("provenance must be RecommendationProvenance")
        points = tuple(tuple(point) for point in self.points)
        if not points:
            raise ValueError("points must contain at least one diameter/value pair")
        checked = []
        for point in points:
            if len(point) != 2:
                raise ValueError("each table point must be a diameter/value pair")
            checked.append((_positive(point[0], "diameter"),
                            _positive(point[1], self.field)))
        if any(left[0] >= right[0] for left, right in zip(checked, checked[1:])):
            raise ValueError("table diameters must be strictly increasing")
        object.__setattr__(self, "points", tuple(checked))

    @property
    def name(self) -> str:
        return f"{self.field} diameter table"

    def recommend(self, context: RecommendationContext) -> StrategyResult:
        identity = (context.tool.identifier == self.tool_identifier
                    and context.material.identifier == self.material_identifier
                    and context.operation == self.operation)
        if not identity:
            return StrategyResult(missing_requirements=(
                f"{self.field} table does not apply to this tool/material/operation",))
        if context.units != self.diameter_units:
            raise RecommendationError("table and context length units must match")
        expected = _field_unit(self.field, context.units)
        if self.value_units != expected:
            raise RecommendationError(
                f"{self.field} table units must be {expected!r} for {context.units!r}"
            )
        diameter = context.tool.cutter_diameter
        low, high = self.points[0][0], self.points[-1][0]
        if diameter < low or diameter > high:
            return StrategyResult(missing_requirements=(
                f"{self.field} table covers cutter_diameter {low}..{high} "
                f"{self.diameter_units}; requested {diameter}",))
        value = self.points[-1][1]
        for left, right in zip(self.points, self.points[1:]):
            if diameter <= right[0]:
                fraction = (diameter - left[0]) / (right[0] - left[0])
                value = left[1] + fraction * (right[1] - left[1])
                break
        if len(self.points) == 1:
            value = self.points[0][1]
        result = Recommendation(
            field=self.field, value=value, units=self.value_units,
            provenance=self.provenance,
            applicable_ranges=(ApplicableRange(
                "cutter_diameter", self.diameter_units, low, high),),
        )
        return StrategyResult((result,))


@dataclass(frozen=True)
class CallableRecommendationStrategy:
    """Adapter for a caller-owned pure calculation callable."""

    name: str
    calculate: Callable[[RecommendationContext], StrategyResult]

    def __post_init__(self):
        _text(self.name, "strategy name")
        if not callable(self.calculate):
            raise TypeError("calculate must be callable")

    def recommend(self, context: RecommendationContext) -> StrategyResult:
        result = self.calculate(context)
        if not isinstance(result, StrategyResult):
            raise TypeError("custom recommendation callable must return StrategyResult")
        return result


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: Tuple[Recommendation, ...]
    missing_requirements: Tuple[str, ...]
    applied_strategies: Tuple[str, ...]

    def __post_init__(self):
        recommendations = tuple(self.recommendations)
        missing = tuple(self.missing_requirements)
        applied = tuple(self.applied_strategies)
        if any(not isinstance(item, Recommendation) for item in recommendations):
            raise TypeError("recommendations must contain Recommendation values")
        if any(not isinstance(item, str) or not item for item in missing):
            raise ValueError("missing_requirements must contain non-empty strings")
        if any(not isinstance(item, str) or not item for item in applied):
            raise ValueError("applied_strategies must contain non-empty strings")
        object.__setattr__(self, "recommendations", recommendations)
        object.__setattr__(self, "missing_requirements", missing)
        object.__setattr__(self, "applied_strategies", applied)

    def get(self, field: str) -> Optional[Recommendation]:
        return next((item for item in self.recommendations if item.field == field), None)


def _range_context_value(context: RecommendationContext,
                         item: ApplicableRange) -> Optional[float]:
    values = {
        "cutter_diameter": (context.tool.cutter_diameter, context.units),
        "effective_flutes": (context.tool.effective_flutes, "count"),
        "max_spindle_speed": (context.machine.max_spindle_speed, "rpm"),
        "max_feed_rate": (context.machine.max_feed_rate,
                          "mm/min" if context.units == "mm" else "in/min"),
        "max_cutting_power": (context.machine.max_cutting_power,
                              "kW" if context.units == "mm" else "hp"),
        "max_torque": (context.machine.max_torque,
                       "N m" if context.units == "mm" else "lbf ft"),
    }
    if item.parameter not in values:
        raise RecommendationError(
            f"unsupported applicable-range parameter: {item.parameter!r}")
    value, expected_units = values[item.parameter]
    if item.units != expected_units:
        raise RecommendationError(
            f"range {item.parameter!r} units must be {expected_units!r}"
        )
    return value


def recommend_milling(
    context: RecommendationContext,
    strategies: Sequence[RecommendationStrategy],
) -> RecommendationResult:
    """Compose validated strategies; fixed user overrides win by field.

    Non-fixed strategies may fill different fields.  Conflicting suggestions for
    one field are rejected instead of being resolved by undocumented ordering.
    """
    if not isinstance(context, RecommendationContext):
        raise TypeError("context must be RecommendationContext")
    candidates = {}
    missing = []
    applied = []
    for strategy in strategies:
        if not isinstance(strategy, RecommendationStrategy):
            raise TypeError("strategies must implement RecommendationStrategy")
        result = strategy.recommend(context)
        if not isinstance(result, StrategyResult):
            raise TypeError("strategy recommend() must return StrategyResult")
        applied.append(strategy.name)
        missing.extend(result.missing_requirements)
        for item in result.recommendations:
            expected_units = _field_unit(item.field, context.units)
            if item.units != expected_units:
                raise RecommendationError(
                    f"{item.field} units must be {expected_units!r} for {context.units!r}"
                )
            applicable = True
            for input_range in item.applicable_ranges:
                value = _range_context_value(context, input_range)
                if value is None or not input_range.contains(value):
                    applicable = False
                    missing.append(
                        f"{item.field} is outside applicable {input_range.parameter} range"
                    )
                    break
            if not applicable:
                continue
            if item.entry_mode is not None and item.entry_mode not in context.tool.entry_modes:
                raise RecommendationError(
                    f"tool {context.tool.identifier!r} is not {item.entry_mode}-capable"
                )
            candidates.setdefault(item.field, []).append(item)
    selected = {}
    for field, values in candidates.items():
        fixed = [item for item in values if item.fixed]
        pool = fixed or values
        chosen = pool[0]
        if any(item != chosen for item in pool[1:]):
            qualifier = "fixed " if fixed else ""
            raise RecommendationError(
                f"conflicting {qualifier}recommendations for {field!r}; "
                "provide one compatible fixed user override"
            )
        selected[field] = chosen
    return RecommendationResult(
        tuple(selected.values()), tuple(dict.fromkeys(missing)), tuple(applied))
