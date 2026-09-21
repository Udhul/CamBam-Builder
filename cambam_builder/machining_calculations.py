"""Pure, unit-explicit milling calculations and constraint solving.

The module deliberately contains no tool/material recommendations and has no
dependency on CamBam documents.  ``units="mm"`` uses m/min surface speed,
mm/min feed, mm chip/depth/engagement, cm^3/min MRR, kW and N m.  ``units="in"``
uses ft/min, in/min, inches, in^3/min, horsepower and lbf ft.
"""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Dict, Optional, Tuple


_REL_TOL = 1e-9


class MachiningConstraintError(ValueError):
    """Raised when supplied milling facts or limits conflict."""


@dataclass(frozen=True)
class MillingConstraints:
    units: str
    cutter_diameter: Optional[float] = None
    surface_speed: Optional[float] = None
    spindle_speed: Optional[float] = None
    chip_load: Optional[float] = None
    effective_flutes: Optional[int] = None
    feed_rate: Optional[float] = None
    axial_depth: Optional[float] = None
    radial_engagement: Optional[float] = None
    material_removal_rate: Optional[float] = None
    specific_cutting_force: Optional[float] = None
    cutting_power: Optional[float] = None
    torque: Optional[float] = None


@dataclass(frozen=True)
class MachineLimits:
    max_spindle_speed: Optional[float] = None
    max_feed_rate: Optional[float] = None


@dataclass(frozen=True)
class ActiveConstraint:
    field: str
    requested: float
    applied: float
    limit: float
    code: str = "MACHINE_LIMIT_CAPPED"


@dataclass(frozen=True)
class MillingSolution:
    units: str
    cutter_diameter: Optional[float]
    surface_speed: Optional[float]
    spindle_speed: Optional[float]
    chip_load: Optional[float]
    effective_flutes: Optional[int]
    feed_rate: Optional[float]
    axial_depth: Optional[float]
    radial_engagement: Optional[float]
    material_removal_rate: Optional[float]
    specific_cutting_force: Optional[float]
    cutting_power: Optional[float]
    torque: Optional[float]
    requested_values: Tuple[Tuple[str, float], ...]
    solved_fields: Tuple[str, ...]
    missing_requirements: Tuple[str, ...]
    assumptions: Tuple[str, ...]
    active_constraints: Tuple[ActiveConstraint, ...]
    unit_labels: Tuple[Tuple[str, str], ...]


def _units(units: str) -> str:
    if units not in ("mm", "in"):
        raise ValueError("units must be exactly 'mm' or 'in'")
    return units


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _finite_result(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} calculation overflowed the supported numeric range")
    return value


def _calculated(name: str, operation) -> float:
    try:
        value = float(operation())
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} calculation overflowed the supported numeric range"
        ) from exc
    return _finite_result(value, name)


def rpm_from_surface_speed(surface_speed, cutter_diameter, *, units: str) -> float:
    """Convert m/min and mm, or ft/min and inches, to spindle RPM."""
    unit = _units(units)
    speed = _positive(surface_speed, "surface_speed")
    diameter = _positive(cutter_diameter, "cutter_diameter")
    factor = 1000.0 if unit == "mm" else 12.0
    return _calculated(
        "spindle_speed", lambda: factor * speed / (math.pi * diameter))


def surface_speed_from_rpm(spindle_speed, cutter_diameter, *, units: str) -> float:
    """Convert spindle RPM and diameter to m/min or ft/min."""
    unit = _units(units)
    rpm = _positive(spindle_speed, "spindle_speed")
    diameter = _positive(cutter_diameter, "cutter_diameter")
    factor = 1000.0 if unit == "mm" else 12.0
    return _calculated(
        "surface_speed", lambda: math.pi * diameter * rpm / factor)


def feed_from_chip_load(chip_load, spindle_speed, effective_flutes, *, units: str) -> float:
    """Return table feed in the selected length unit per minute."""
    _units(units)
    chip = _positive(chip_load, "chip_load")
    rpm = _positive(spindle_speed, "spindle_speed")
    flutes = _flutes(effective_flutes)
    return _calculated("feed_rate", lambda: chip * rpm * flutes)


def chip_load_from_feed(feed_rate, spindle_speed, effective_flutes, *, units: str) -> float:
    """Return chip load per effective flute in the selected length unit."""
    _units(units)
    feed = _positive(feed_rate, "feed_rate")
    rpm = _positive(spindle_speed, "spindle_speed")
    flutes = _flutes(effective_flutes)
    return _calculated("chip_load", lambda: feed / (rpm * flutes))


def material_removal_rate(feed_rate, axial_depth, radial_engagement, *, units: str) -> float:
    """Return cm^3/min for metric inputs or in^3/min for imperial inputs."""
    unit = _units(units)
    feed = _positive(feed_rate, "feed_rate")
    axial = _positive(axial_depth, "axial_depth")
    radial = _positive(radial_engagement, "radial_engagement")
    scale = 1000.0 if unit == "mm" else 1.0
    return _calculated(
        "material_removal_rate", lambda: feed * axial * radial / scale)


def cutting_power(material_removal_rate_value, specific_cutting_force, *, units: str) -> float:
    """Return kW (N/mm^2 input) or hp (lbf/in^2 input)."""
    unit = _units(units)
    removal = _positive(material_removal_rate_value, "material_removal_rate")
    force = _positive(specific_cutting_force, "specific_cutting_force")
    divisor = 60000.0 if unit == "mm" else 396000.0
    return _calculated("cutting_power", lambda: removal * force / divisor)


def cutting_torque(cutting_power_value, spindle_speed, *, units: str) -> float:
    """Return N m for kW or lbf ft for horsepower."""
    unit = _units(units)
    power = _positive(cutting_power_value, "cutting_power")
    rpm = _positive(spindle_speed, "spindle_speed")
    factor = 30000.0 if unit == "mm" else 16501.0
    return _calculated("torque", lambda: power * factor / (math.pi * rpm))


def _flutes(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("effective_flutes must be a positive integer")
    try:
        if not math.isfinite(float(value)):
            raise ValueError("effective_flutes must be a positive integer")
    except OverflowError as exc:
        raise ValueError("effective_flutes must be a positive integer") from exc
    return int(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=0.0)


def _set_or_check(values: Dict[str, float], name: str, candidate: float,
                  solved: list, equation: str) -> bool:
    if name == "effective_flutes":
        candidate = _flutes(candidate)
    else:
        candidate = _finite_result(float(candidate), name)
    current = values.get(name)
    if current is None:
        values[name] = candidate
        solved.append(name)
        return True
    if not _close(current, candidate):
        raise MachiningConstraintError(
            f"conflicting {equation} inputs: {name}={current!r}, expected {candidate!r}"
        )
    return False


def _unit_labels(units: str) -> Tuple[Tuple[str, str], ...]:
    if units == "mm":
        labels = {
            "cutter_diameter": "mm", "surface_speed": "m/min",
            "spindle_speed": "rpm", "chip_load": "mm/tooth",
            "effective_flutes": "count", "feed_rate": "mm/min",
            "axial_depth": "mm", "radial_engagement": "mm",
            "material_removal_rate": "cm^3/min",
            "specific_cutting_force": "N/mm^2", "cutting_power": "kW",
            "torque": "N m",
        }
    else:
        labels = {
            "cutter_diameter": "in", "surface_speed": "ft/min",
            "spindle_speed": "rpm", "chip_load": "in/tooth",
            "effective_flutes": "count", "feed_rate": "in/min",
            "axial_depth": "in", "radial_engagement": "in",
            "material_removal_rate": "in^3/min",
            "specific_cutting_force": "lbf/in^2", "cutting_power": "hp",
            "torque": "lbf ft",
        }
    return tuple(labels.items())


def solve_milling_constraints(
    constraints: MillingConstraints, limits: Optional[MachineLimits] = None
) -> MillingSolution:
    """Solve identifiable milling values and apply RPM/feed caps to derived settings.

    Supplied values are retained verbatim in ``requested_values`` and are checked
    before caps.  A supplied RPM or feed is a fixed machine setting and conflicts
    with a lower cap.  A derived RPM/feed may be capped; dependent achieved values
    are then recalculated and the target remains visible in ``requested_values``.
    """
    if not isinstance(constraints, MillingConstraints):
        raise TypeError("constraints must be a MillingConstraints instance")
    if limits is None:
        limits = MachineLimits()
    if not isinstance(limits, MachineLimits):
        raise TypeError("limits must be a MachineLimits instance")
    unit = _units(constraints.units)

    values: Dict[str, float] = {}
    requested = []
    for item in fields(constraints):
        if item.name == "units":
            continue
        value = getattr(constraints, item.name)
        if value is None:
            continue
        if item.name == "effective_flutes":
            checked = _flutes(value)
        else:
            checked = _positive(value, item.name)
        values[item.name] = checked
        requested.append((item.name, value))

    max_rpm = None if limits.max_spindle_speed is None else _positive(
        limits.max_spindle_speed, "max_spindle_speed")
    max_feed = None if limits.max_feed_rate is None else _positive(
        limits.max_feed_rate, "max_feed_rate")
    if (values.get("cutter_diameter") is not None
            and values.get("radial_engagement") is not None
            and values["radial_engagement"] > values["cutter_diameter"]):
        raise MachiningConstraintError(
            "radial_engagement cannot exceed cutter_diameter"
        )

    solved = []
    changed = True
    while changed:
        changed = False
        diameter, surface, rpm = (values.get(name) for name in (
            "cutter_diameter", "surface_speed", "spindle_speed"))
        if sum(value is None for value in (diameter, surface, rpm)) == 1:
            if rpm is None:
                changed |= _set_or_check(values, "spindle_speed",
                    rpm_from_surface_speed(surface, diameter, units=unit), solved,
                    "surface-speed")
            elif surface is None:
                changed |= _set_or_check(values, "surface_speed",
                    surface_speed_from_rpm(rpm, diameter, units=unit), solved,
                    "surface-speed")
            else:
                changed |= _set_or_check(values, "cutter_diameter",
                    _calculated("cutter_diameter", lambda: (
                        (1000.0 if unit == "mm" else 12.0)
                        * surface / (math.pi * rpm))), solved, "surface-speed")
        elif None not in (diameter, surface, rpm):
            _set_or_check(values, "surface_speed",
                surface_speed_from_rpm(rpm, diameter, units=unit), solved,
                "surface-speed")

        chip, rpm, flutes, feed = (values.get(name) for name in (
            "chip_load", "spindle_speed", "effective_flutes", "feed_rate"))
        if sum(value is None for value in (chip, rpm, flutes, feed)) == 1:
            if feed is None:
                candidate = feed_from_chip_load(chip, rpm, flutes, units=unit)
                changed |= _set_or_check(values, "feed_rate", candidate, solved, "feed")
            elif chip is None:
                candidate = chip_load_from_feed(feed, rpm, flutes, units=unit)
                changed |= _set_or_check(values, "chip_load", candidate, solved, "feed")
            elif rpm is None:
                candidate = _calculated(
                    "spindle_speed", lambda: feed / (chip * flutes))
                changed |= _set_or_check(values, "spindle_speed", candidate, solved, "feed")
            else:
                candidate = _calculated(
                    "effective_flutes", lambda: feed / (chip * rpm))
                nearest = round(candidate)
                if not _close(candidate, nearest) or nearest <= 0:
                    raise MachiningConstraintError(
                        "feed inputs imply a non-integral effective_flutes value"
                    )
                changed |= _set_or_check(values, "effective_flutes", nearest, solved, "feed")
        elif None not in (chip, rpm, flutes, feed):
            _set_or_check(values, "feed_rate",
                feed_from_chip_load(chip, rpm, flutes, units=unit), solved, "feed")

        feed, axial, radial, removal = (values.get(name) for name in (
            "feed_rate", "axial_depth", "radial_engagement", "material_removal_rate"))
        if sum(value is None for value in (feed, axial, radial, removal)) == 1:
            scale = 1000.0 if unit == "mm" else 1.0
            if removal is None:
                candidate = material_removal_rate(feed, axial, radial, units=unit)
                name = "material_removal_rate"
            elif feed is None:
                candidate, name = _calculated(
                    "feed_rate", lambda: removal * scale / (axial * radial)), "feed_rate"
            elif axial is None:
                candidate, name = _calculated(
                    "axial_depth", lambda: removal * scale / (feed * radial)), "axial_depth"
            else:
                candidate, name = _calculated(
                    "radial_engagement", lambda: removal * scale / (feed * axial)), "radial_engagement"
            changed |= _set_or_check(values, name, candidate, solved, "material-removal")
        elif None not in (feed, axial, radial, removal):
            _set_or_check(values, "material_removal_rate",
                material_removal_rate(feed, axial, radial, units=unit), solved,
                "material-removal")

        removal, force, power = (values.get(name) for name in (
            "material_removal_rate", "specific_cutting_force", "cutting_power"))
        if sum(value is None for value in (removal, force, power)) == 1:
            divisor = 60000.0 if unit == "mm" else 396000.0
            if power is None:
                candidate, name = cutting_power(removal, force, units=unit), "cutting_power"
            elif removal is None:
                candidate, name = _calculated(
                    "material_removal_rate", lambda: power * divisor / force), "material_removal_rate"
            else:
                candidate, name = _calculated(
                    "specific_cutting_force", lambda: power * divisor / removal), "specific_cutting_force"
            changed |= _set_or_check(values, name, candidate, solved, "cutting-power")
        elif None not in (removal, force, power):
            _set_or_check(values, "cutting_power",
                cutting_power(removal, force, units=unit), solved, "cutting-power")

        power, rpm, torque = (values.get(name) for name in (
            "cutting_power", "spindle_speed", "torque"))
        if sum(value is None for value in (power, rpm, torque)) == 1:
            factor = 30000.0 if unit == "mm" else 16501.0
            if torque is None:
                candidate, name = cutting_torque(power, rpm, units=unit), "torque"
            elif power is None:
                candidate, name = _calculated(
                    "cutting_power", lambda: torque * math.pi * rpm / factor), "cutting_power"
            else:
                candidate, name = _calculated(
                    "spindle_speed", lambda: power * factor / (math.pi * torque)), "spindle_speed"
            changed |= _set_or_check(values, name, candidate, solved, "torque")
        elif None not in (power, rpm, torque):
            _set_or_check(values, "torque",
                cutting_torque(power, rpm, units=unit), solved, "torque")

    if (values.get("cutter_diameter") is not None
            and values.get("radial_engagement") is not None
            and values["radial_engagement"] > values["cutter_diameter"]):
        raise MachiningConstraintError(
            "radial_engagement cannot exceed cutter_diameter"
        )

    active = []
    fixed_names = {name for name, unused in requested}
    if max_rpm is not None and values.get("spindle_speed", 0.0) > max_rpm:
        if "spindle_speed" in fixed_names:
            raise MachiningConstraintError(
                "fixed spindle_speed exceeds max_spindle_speed"
            )
        old = values["spindle_speed"]
        values["spindle_speed"] = max_rpm
        active.append(ActiveConstraint("spindle_speed", old, max_rpm, max_rpm))

    # Recalculate feed at an achieved capped RPM unless feed itself was fixed.
    if ("feed_rate" not in fixed_names and values.get("chip_load") is not None
            and values.get("spindle_speed") is not None
            and values.get("effective_flutes") is not None):
        values["feed_rate"] = feed_from_chip_load(
            values["chip_load"], values["spindle_speed"],
            values["effective_flutes"], units=unit)
    if max_feed is not None and values.get("feed_rate", 0.0) > max_feed:
        if "feed_rate" in fixed_names:
            raise MachiningConstraintError("fixed feed_rate exceeds max_feed_rate")
        old = values["feed_rate"]
        values["feed_rate"] = max_feed
        active.append(ActiveConstraint("feed_rate", old, max_feed, max_feed))

    # Recalculate achieved values only when a cap actually changed the solution.
    # Without an active cap, every supplied consistent value remains exact.
    if active:
        if values.get("cutter_diameter") is not None and values.get("spindle_speed") is not None:
            values["surface_speed"] = surface_speed_from_rpm(
                values["spindle_speed"], values["cutter_diameter"], units=unit)
        if (values.get("feed_rate") is not None and values.get("spindle_speed") is not None
                and values.get("effective_flutes") is not None):
            values["chip_load"] = chip_load_from_feed(
                values["feed_rate"], values["spindle_speed"],
                values["effective_flutes"], units=unit)
        if all(values.get(name) is not None for name in (
                "feed_rate", "axial_depth", "radial_engagement")):
            values["material_removal_rate"] = material_removal_rate(
                values["feed_rate"], values["axial_depth"],
                values["radial_engagement"], units=unit)
        if all(values.get(name) is not None for name in (
                "material_removal_rate", "specific_cutting_force")):
            values["cutting_power"] = cutting_power(
                values["material_removal_rate"], values["specific_cutting_force"], units=unit)
        if all(values.get(name) is not None for name in ("cutting_power", "spindle_speed")):
            values["torque"] = cutting_torque(
                values["cutting_power"], values["spindle_speed"], units=unit)

    missing = []
    groups = (
        (("cutter_diameter", "surface_speed", "spindle_speed"),
         "surface speed requires any two of cutter_diameter, surface_speed, spindle_speed"),
        (("chip_load", "spindle_speed", "effective_flutes", "feed_rate"),
         "feed requires any three of chip_load, spindle_speed, effective_flutes, feed_rate"),
        (("feed_rate", "axial_depth", "radial_engagement", "material_removal_rate"),
         "MRR requires any three of feed_rate, axial_depth, radial_engagement, material_removal_rate"),
        (("material_removal_rate", "specific_cutting_force", "cutting_power"),
         "power requires any two of material_removal_rate, specific_cutting_force, cutting_power"),
        (("cutting_power", "spindle_speed", "torque"),
         "torque requires any two of cutting_power, spindle_speed, torque"),
    )
    for names, message in groups:
        if any(values.get(name) is not None for name in names) and sum(
                values.get(name) is None for name in names) > 1:
            missing.append(message)

    assumptions = (
        "surface speed uses effective cutter diameter at the cutting depth",
        "effective_flutes counts cutting edges engaged per revolution",
        "MRR is rectangular engagement: feed_rate * axial_depth * radial_engagement",
        "specific cutting force is caller-supplied; no material or chip-thinning model is assumed",
    )
    result_names = [item.name for item in fields(MillingConstraints)
                    if item.name != "units"]
    return MillingSolution(
        units=unit,
        **{name: values.get(name) for name in result_names},
        requested_values=tuple(requested),
        solved_fields=tuple(dict.fromkeys(solved)),
        missing_requirements=tuple(missing),
        assumptions=assumptions,
        active_constraints=tuple(active),
        unit_labels=_unit_labels(unit),
    )
