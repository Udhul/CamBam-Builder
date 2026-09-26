"""Controller-neutral decoded stage values for bounded M5 output audits."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedMove:
    g: int
    feed: float
    start: tuple
    end: tuple


@dataclass(frozen=True)
class DecodedProgram:
    tool_label: str
    startup: tuple
    rpm: float
    moves: tuple
    end_position: tuple
    spindle_stopped: bool
    program_ended: bool
