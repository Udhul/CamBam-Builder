"""Independent strict reader for the bounded UCCNC split-file profile.

Only the declared absolute metric XYZ G0/G1 subset has modeled effects.
Unknown words, modal changes, macros and tool commands fail closed.
"""

from dataclasses import dataclass
import math
import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_MOVE = re.compile(
    rf"^(G0|G1)(?: F({_NUMBER}))? X({_NUMBER}) Y({_NUMBER}) Z({_NUMBER})$")
_SPINDLE = re.compile(rf"^M3 S({_NUMBER})$")
_HEADER = re.compile(r"^\( M5 UCCNC split-file v1 TOOL (T[1-9][0-9]*) G54 \)$")
_STARTUP = ("G21", "G90", "G17", "G61", "G40", "G49", "G54")


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


def _value(word):
    value = float(word)
    if not math.isfinite(value) or abs(value) > 1_000_000:
        raise ValueError("UCCNC nonfinite or excessive number")
    return value


def decode_program(data, *, initial_tip):
    """Decode all bytes, including startup and end roles, without an emitter."""
    if type(data) is not bytes:
        raise ValueError("UCCNC program bytes required")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("UCCNC program must be ASCII") from exc
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("UCCNC program line encoding changed")
    lines = text.splitlines()
    header = _HEADER.fullmatch(lines[0]) if lines else None
    if len(lines) < 12 or header is None:
        raise ValueError("unsupported UCCNC header")
    if tuple(lines[1:8]) != _STARTUP:
        raise ValueError("unsupported UCCNC startup or offset state")
    match = _SPINDLE.fullmatch(lines[8])
    if match is None:
        raise ValueError("unsupported UCCNC spindle start")
    rpm = _value(match.group(1))
    if rpm <= 0:
        raise ValueError("UCCNC spindle speed must be positive")
    if lines[-2:] != ["M5", "M30"]:
        raise ValueError("UCCNC spindle stop or end changed")
    if len(initial_tip) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and
            math.isfinite(v) for v in initial_tip):
        raise ValueError("declared initial tip required")
    at, feed, moves = tuple(initial_tip), None, []
    for index, line in enumerate(lines[9:-2], 10):
        match = _MOVE.fullmatch(line)
        if match is None:
            raise ValueError(f"unsupported UCCNC command at line {index}")
        g = 0 if match.group(1) == "G0" else 1
        supplied_feed = match.group(2)
        if g == 0 and supplied_feed is not None:
            raise ValueError("rapid move carries feed")
        if g == 1:
            if supplied_feed is not None:
                feed = _value(supplied_feed)
            if feed is None or feed <= 0:
                raise ValueError("feed move has no positive feed")
        end = tuple(_value(match.group(i)) for i in (3, 4, 5))
        if end == at:
            raise ValueError("zero-length UCCNC motion")
        moves.append(DecodedMove(g, 0 if g == 0 else feed, at, end))
        at = end
    if not moves:
        raise ValueError("empty UCCNC motion")
    return DecodedProgram(header.group(1), _STARTUP, rpm, tuple(moves), at,
                          True, True)
