"""Independent strict reader for the bounded Grbl v1.1 M5 fixture.

Stage comments identify expected installed tools but have no controller effect.
M0 is only a pause; external transition effects are audited separately.
"""

from dataclasses import dataclass
import math
import re

from .m5_decoded import DecodedMove, DecodedProgram


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_MOVE = re.compile(
    rf"^(G0|G1)(?: F({_NUMBER}))? X({_NUMBER}) Y({_NUMBER}) Z({_NUMBER})$")
_SPINDLE = re.compile(rf"^M3 S({_NUMBER})$")
_OFFSET = re.compile(rf"^G43\.1 Z({_NUMBER})$")
_STAGE = re.compile(r"^\( STAGE (T[1-9][0-9]*) \)$")
_STARTUP = ("G21", "G90", "G17", "G94", "G61", "G40", "G49", "G54")


@dataclass(frozen=True)
class DecodedJob:
    stages: tuple
    length_offsets_mm: tuple
    stops: tuple


def _value(raw):
    value = float(raw)
    if not math.isfinite(value) or abs(value) > 1_000_000:
        raise ValueError("Grbl nonfinite or excessive number")
    return value


def decode_job(data, *, initial_tip):
    """Consume the complete byte stream and expose each pause and offset."""
    if type(data) is not bytes:
        raise ValueError("Grbl program bytes required")
    try:
        contents = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Grbl program must be ASCII") from exc
    if not contents.endswith("\n") or "\r" in contents:
        raise ValueError("Grbl program line encoding changed")
    lines = contents.splitlines()
    if (tuple(lines[:9]) != ("( M5 GRBL v1.1 PORTABILITY 1 )", *_STARTUP)
            or len(initial_tip) != 3 or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in initial_tip)):
        raise ValueError("unsupported Grbl startup or initial tip")
    index, stages, offsets, stops = 9, [], [], []
    while index < len(lines):
        match = _STAGE.fullmatch(lines[index])
        if match is None or index + 2 >= len(lines):
            raise ValueError("unsupported Grbl stage or transition")
        tool = match.group(1)
        index += 1
        offset_match = _OFFSET.fullmatch(lines[index])
        if lines[index] == "G49":
            offset = 0.0
        elif offset_match:
            offset = _value(offset_match.group(1))
        else:
            raise ValueError("unsupported Grbl tool length state")
        offsets.append(offset)
        index += 1
        match = _SPINDLE.fullmatch(lines[index])
        if match is None or _value(match.group(1)) <= 0:
            raise ValueError("unsupported Grbl spindle start")
        rpm = _value(match.group(1))
        index += 1
        at, feed, moves = tuple(initial_tip), None, []
        while index < len(lines) and lines[index] != "M5":
            match = _MOVE.fullmatch(lines[index])
            if match is None:
                raise ValueError(f"unsupported Grbl command at line {index + 1}")
            g = 0 if match.group(1) == "G0" else 1
            supplied = match.group(2)
            if g == 0 and supplied is not None:
                raise ValueError("Grbl rapid carries feed")
            if g == 1:
                if supplied is not None:
                    feed = _value(supplied)
                if feed is None or feed <= 0:
                    raise ValueError("Grbl feed move has no positive feed")
            end = tuple(_value(match.group(j)) for j in (3, 4, 5))
            if end == at:
                raise ValueError("zero-length Grbl motion")
            moves.append(DecodedMove(g, 0 if g == 0 else feed, at, end))
            at = end
            index += 1
        if not moves or index >= len(lines):
            raise ValueError("incomplete Grbl stage")
        index += 1  # M5
        ended = index == len(lines) - 1 and lines[index] == "M30"
        stages.append(DecodedProgram(tool, _STARTUP, rpm, tuple(moves), at,
                                     True, ended))
        if ended:
            break
        if index >= len(lines) - 1 or lines[index] != "M0":
            raise ValueError("unsupported Grbl stop or end")
        stops.append(len(stages) - 1)
        index += 1
    if index != len(lines) - 1 or len(stops) != len(stages) - 1:
        raise ValueError("Grbl trailing command or missing end")
    return DecodedJob(tuple(stages), tuple(offsets), tuple(stops))
