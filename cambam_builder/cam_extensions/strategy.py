"""Deterministic selection among independently audited ordered CAM routes.

This policy consumes evidence; it does not infer stock from MOP settings or
manufacture an audit. A caller must refresh source/post fingerprints and supply
each stage's actual emitted-motion, safety gates and residual bounds. An unsafe
route cannot be selected, even by an explicit manual choice.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional


REQUIRED_GATES = frozenset({"motion", "tool", "entry", "link", "stock",
                            "target", "residual", "post"})
ROUTE_KINDS = frozenset({"native", "custom_region", "framework"})


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def _fingerprint(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ResidualBounds:
    area_lower_mm2: float
    area_upper_mm2: float
    volume_lower_mm3: float
    volume_upper_mm3: float

    def __post_init__(self):
        values = (self.area_lower_mm2, self.area_upper_mm2,
                  self.volume_lower_mm3, self.volume_upper_mm3)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in values) or
                self.area_lower_mm2 > self.area_upper_mm2 or
                self.volume_lower_mm3 > self.volume_upper_mm3):
            raise ValueError("residual area/volume bounds must be finite and ordered")


@dataclass(frozen=True)
class StageAudit:
    name: str
    route_kind: str
    source_fingerprint: str
    predecessor_fingerprint: str
    motion_fingerprint: str
    emitted_fingerprint: str
    residual: ResidualBounds
    passed_gates: frozenset
    freshness: str = "current"
    complete: bool = True

    @property
    def chain_fingerprint(self):
        return _digest((self.source_fingerprint, self.predecessor_fingerprint,
                        self.name, self.route_kind, self.motion_fingerprint,
                        self.emitted_fingerprint))


@dataclass(frozen=True)
class Alternative:
    name: str
    stages: tuple


@dataclass(frozen=True)
class Assessment:
    name: str
    status: str                 # feasible, partial, unsafe
    reasons: tuple
    residual: Optional[ResidualBounds]
    stage_count: int
    stage_residuals: tuple


@dataclass(frozen=True)
class Selection:
    status: str                 # selected, partial, infeasible
    chosen: Optional[str]
    assessments: tuple
    source_fingerprint: str


def _assess(source, alternative, area_budget, volume_budget):
    if not alternative.stages:
        return Assessment(alternative.name, "unsafe", ("no audited stages",), None, 0, ())
    predecessor = source
    reasons = []
    previous = None
    stage_residuals = []
    for index, stage in enumerate(alternative.stages):
        if type(stage) is not StageAudit:
            reasons.append(f"stage {index}: missing StageAudit")
            break
        if not stage.name or stage.route_kind not in ROUTE_KINDS:
            reasons.append(f"stage {index}: unsupported route identity")
        if (not _fingerprint(stage.source_fingerprint) or
                stage.source_fingerprint != source):
            reasons.append(f"stage {index}: stale source fingerprint")
        if stage.predecessor_fingerprint != predecessor:
            reasons.append(f"stage {index}: missing ordered predecessor evidence")
        if not _fingerprint(stage.motion_fingerprint) or not _fingerprint(stage.emitted_fingerprint):
            reasons.append(f"stage {index}: missing actual emitted-motion fingerprint")
        if stage.freshness != "current":
            reasons.append(f"stage {index}: stale evidence")
        if stage.complete is not True:
            reasons.append(f"stage {index}: incomplete emitted motion")
        if type(stage.passed_gates) is not frozenset or not REQUIRED_GATES <= stage.passed_gates:
            reasons.append(f"stage {index}: missing audit gates")
        if type(stage.residual) is not ResidualBounds:
            reasons.append(f"stage {index}: missing residual bounds")
        elif previous is not None and (
                stage.residual.area_lower_mm2 > previous.area_upper_mm2 + 1e-9 or
                stage.residual.volume_lower_mm3 > previous.volume_upper_mm3 + 1e-9):
            reasons.append(f"stage {index}: residual increases beyond prior bounds")
        if type(stage.residual) is ResidualBounds:
            stage_residuals.append((stage.name, stage.residual))
        previous = stage.residual if type(stage.residual) is ResidualBounds else None
        predecessor = stage.chain_fingerprint
    residual = alternative.stages[-1].residual if (
        type(alternative.stages[-1]) is StageAudit and
        type(alternative.stages[-1].residual) is ResidualBounds) else None
    if reasons:
        return Assessment(alternative.name, "unsafe", tuple(reasons), residual,
                          len(alternative.stages), tuple(stage_residuals))
    misses = []
    if residual.area_upper_mm2 > area_budget:
        misses.append("remaining area exceeds budget")
    if residual.volume_upper_mm3 > volume_budget:
        misses.append("remaining volume exceeds budget")
    return Assessment(alternative.name, "partial" if misses else "feasible",
                      tuple(misses), residual, len(alternative.stages),
                      tuple(stage_residuals))


def select_strategy(source_fingerprint, alternatives, *, max_area_mm2,
                    max_volume_mm3, tie_order, manual_choice=None):
    """Choose a feasible route, then least-residual safe partial route.

    Rank by final upper area, then upper volume, then caller-declared tie order.
    Manual choice keeps the named safe route, including a partial route, and
    reports its actual status. Unsafe choices return infeasible diagnostics.
    """
    if not _fingerprint(source_fingerprint):
        raise ValueError("source fingerprint must be SHA-256")
    for value in (max_area_mm2, max_volume_mm3):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative area and volume budgets required")
    if (not isinstance(alternatives, tuple) or
            any(type(item) is not Alternative or not item.name or
                type(item.stages) is not tuple for item in alternatives)):
        raise ValueError("alternatives must be a tuple of named ordered routes")
    names = tuple(item.name for item in alternatives)
    if len(set(names)) != len(names) or type(tie_order) is not tuple or set(tie_order) != set(names) or len(tie_order) != len(names):
        raise ValueError("tie order must name every distinct alternative once")
    if manual_choice is not None and manual_choice not in names:
        raise ValueError("manual choice is not an alternative")
    assessments = tuple(_assess(source_fingerprint, item, max_area_mm2,
                                max_volume_mm3) for item in alternatives)
    by_name = {item.name: item for item in assessments}
    if manual_choice is not None:
        chosen = by_name[manual_choice]
        return Selection("infeasible" if chosen.status == "unsafe" else
                         "partial" if chosen.status == "partial" else "selected",
                         None if chosen.status == "unsafe" else chosen.name,
                         assessments, source_fingerprint)
    order = {name: index for index, name in enumerate(tie_order)}
    for status in ("feasible", "partial"):
        eligible = [item for item in assessments if item.status == status]
        if eligible:
            chosen = min(eligible, key=lambda item: (item.residual.area_upper_mm2,
                                                      item.residual.volume_upper_mm3,
                                                      order[item.name]))
            return Selection("selected" if status == "feasible" else "partial",
                             chosen.name, assessments, source_fingerprint)
    return Selection("infeasible", None, assessments, source_fingerprint)
