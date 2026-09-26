"""Strict UCCNC and Grbl output for caller-supplied ordered jobs.

The readers consume complete NC byte streams without using the writer or the
source job.  A Grbl offset command changes the displayed work Z at a stationary
machine.  Any following return to the declared safe tip is emitted and decoded
as a separate, real rapid motion.
"""

from dataclasses import dataclass
import math
import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_MOVE = re.compile(
    rf"^(G0|G1)(?: F({_NUMBER}))? X({_NUMBER}) Y({_NUMBER}) Z({_NUMBER})$")
_SPINDLE = re.compile(rf"^M3 S({_NUMBER})$")
_OFFSET = re.compile(rf"^G43\.1 Z({_NUMBER})$")
_TOOL = r"T[1-9][0-9]*"
_UCCNC_HEADER = re.compile(rf"^\( ORDERED UCCNC v1 TOOL ({_TOOL}) \)$")
_GRBL_STAGE = re.compile(rf"^\( STAGE ({_TOOL}) \)$")
_UCCNC_STARTUP = ("G21", "G90", "G17", "G61", "G40", "G49", "G54")
_GRBL_STARTUP = ("G21", "G90", "G17", "G94", "G61", "G40", "G49", "G54")
_GRBL_HEADER = "( ORDERED GRBL v1.1 JOB 1 )"


@dataclass(frozen=True)
class DecodedMove:
    g: int
    feed: float
    start: tuple
    end: tuple


@dataclass(frozen=True)
class DecodedStage:
    tool_id: str
    offset_mm: float
    rpm: float
    moves: tuple
    transition_moves: tuple
    pause_after: bool
    end_position: tuple
    program_ended: bool
    offset_command: str


@dataclass(frozen=True)
class DecodedJob:
    stages: tuple
    final_position: tuple
    startup: tuple


def _dialect(dialect):
    if dialect in ("uccnc", "grbl", "grbl-v1.1"):
        return "grbl" if dialect == "grbl-v1.1" else dialect
    raise ValueError("unsupported ordered-job dialect")


def _value(raw):
    value = float(raw)
    if not math.isfinite(value) or abs(value) > 1_000_000:
        raise ValueError("nonfinite or excessive NC number")
    return value


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("numeric NC value required")
    value = _value(value)
    rounded = round(value, 4)
    if not rounded:
        return "0"
    return format(rounded, ".4f").rstrip("0").rstrip(".")


def _tip(tip):
    if len(tip) != 3 or any(type(v) not in (int, float) for v in tip):
        raise ValueError("finite XYZ tip required")
    return tuple(_value(v) for v in tip)


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _xyz(point):
    return " ".join(axis + _number(value) for axis, value in zip("XYZ", point))


def _tool(tool):
    if type(tool) is not str or re.fullmatch(_TOOL, tool) is None:
        raise ValueError("positive T-number tool ID required")
    return tool


def _motion_lines(stage, translation, initial_tip):
    motions = stage.motions
    if not motions:
        raise ValueError("empty controller stage")
    at = initial_tip
    lines = []
    rounded_at = _add(initial_tip, translation)
    for motion in motions:
        if _tip(motion.start) != at or motion.role not in (
                "rapid", "rapid_retract", "approach", "entry", "cleared_descent",
                "cut", "retract"):
            raise ValueError("noncontinuous or unsupported controller motion")
        end = _tip(motion.end)
        rounded_end = tuple(float(_number(v)) for v in _add(end, translation))
        if rounded_end == rounded_at:
            raise ValueError("controller motion vanishes after rounding")
        if motion.role in ("rapid", "rapid_retract"):
            if motion.feed not in (0, 0.0, None):
                raise ValueError("rapid cannot carry feed")
            lines.append("G0 " + _xyz(_add(end, translation)))
        else:
            feed = _value(motion.feed)
            if feed <= 0 or float(_number(feed)) <= 0:
                raise ValueError("positive feed required")
            lines.append("G1 F" + _number(feed) + " " + _xyz(_add(end, translation)))
        at, rounded_at = end, rounded_end
    return lines


def render(job, dialect):
    """Return complete program bytes, one per stage for UCCNC, one for Grbl."""
    dialect = _dialect(dialect)
    start = _tip(job.initial_tip)
    translation = _tip(job.translation_xyz_mm)
    if not job.stages:
        raise ValueError("empty ordered job")
    files = []
    if dialect == "grbl":
        lines = [_GRBL_HEADER, *_GRBL_STARTUP]
        previous_offset = 0.0
    previous_end = start
    for index, stage in enumerate(job.stages):
        boundary = getattr(getattr(stage, "transition", None), "boundary", None)
        if index == 0 and boundary is not None:
            raise ValueError("first stage cannot have a transition boundary")
        if index and boundary != ("split" if dialect == "uccnc" else "pause"):
            raise ValueError("stage transition incompatible with controller dialect")
        tool = _tool(stage.tool_id)
        rpm = _value(stage.rpm)
        if rpm <= 0 or float(_number(rpm)) <= 0:
            raise ValueError("positive spindle RPM required")
        offset = _value(stage.offset_mm)
        stage_start = _tip(stage.motions[0].start)
        if stage_start != previous_end:
            raise ValueError("ordered stage continuity differs")
        if dialect == "uccnc":
            if offset != 0:
                raise ValueError("UCCNC split profile requires G49")
            lines = [f"( ORDERED UCCNC v1 TOOL {tool} )", *_UCCNC_STARTUP,
                     "M3 S" + _number(rpm)]
        else:
            lines.append(f"( STAGE {tool} )")
            lines.append("G49" if offset == 0 else "G43.1 Z" + _number(offset))
            # At stationary machine coordinates, changing the active length
            # offset changes displayed work Z by old_offset - new_offset.
            delta = float(_number(offset)) - previous_offset
            if delta:
                safe = _add(stage_start, translation)
                shifted = (safe[0], safe[1], safe[2] - delta)
                if tuple(float(_number(v)) for v in shifted) != tuple(
                        float(_number(v)) for v in safe):
                    lines.append("G0 " + _xyz(safe))
            lines.append("M3 S" + _number(rpm))
            previous_offset = float(_number(offset))
        lines.extend(_motion_lines(stage, translation, stage_start))
        previous_end = _tip(stage.motions[-1].end)
        lines.append("M5")
        if dialect == "uccnc":
            lines.append("M30")
            files.append(("\n".join(lines) + "\n").encode("ascii"))
        elif index < len(job.stages) - 1:
            lines.append("M0")
    if dialect == "grbl":
        lines.append("M30")
        files.append(("\n".join(lines) + "\n").encode("ascii"))
    return tuple(files)


def _lines(data):
    if type(data) is not bytes:
        raise ValueError("NC program bytes required")
    try:
        content = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("NC program must be ASCII") from exc
    if not content.endswith("\n") or "\r" in content:
        raise ValueError("NC program line encoding changed")
    return content.splitlines()


def _read_motion(line, at, feed):
    match = _MOVE.fullmatch(line)
    if match is None:
        raise ValueError("unsupported NC motion word")
    g = 0 if match.group(1) == "G0" else 1
    supplied = match.group(2)
    if g == 0 and supplied is not None:
        raise ValueError("rapid carries feed")
    if g == 1:
        feed = _value(supplied) if supplied is not None else feed
        if feed is None or feed <= 0:
            raise ValueError("feed move has no positive feed")
    end = tuple(_value(match.group(i)) for i in (3, 4, 5))
    if end == at:
        raise ValueError("zero-length NC motion")
    return DecodedMove(g, 0 if g == 0 else feed, at, end), feed


def _read_spindle(line):
    match = _SPINDLE.fullmatch(line)
    if match is None:
        raise ValueError("unsupported spindle command")
    rpm = _value(match.group(1))
    if rpm <= 0:
        raise ValueError("nonpositive spindle RPM")
    return rpm


def _decode_uccnc(files, initial):
    stages = []
    next_initial = initial
    for data in files:
        lines = _lines(data)
        header = _UCCNC_HEADER.fullmatch(lines[0]) if lines else None
        if (header is None or len(lines) < 12 or
                tuple(lines[1:8]) != _UCCNC_STARTUP or
                lines[-2:] != ["M5", "M30"]):
            raise ValueError("unsupported UCCNC program structure")
        rpm = _read_spindle(lines[8])
        at, feed, moves = next_initial, None, []
        for line in lines[9:-2]:
            move, feed = _read_motion(line, at, feed)
            moves.append(move)
            at = move.end
        if not moves:
            raise ValueError("empty UCCNC stage")
        stages.append(DecodedStage(header.group(1), 0.0, rpm, tuple(moves),
                                   (), False, at, True, "G49"))
        next_initial = at
    return DecodedJob(tuple(stages), stages[-1].end_position, _UCCNC_STARTUP)


def _decode_grbl(data, initial):
    lines = _lines(data)
    if tuple(lines[:9]) != (_GRBL_HEADER, *_GRBL_STARTUP):
        raise ValueError("unsupported Grbl startup")
    index, at, old_offset, stages = 9, initial, 0.0, []
    while index < len(lines):
        stage = _GRBL_STAGE.fullmatch(lines[index])
        if stage is None or index + 2 >= len(lines):
            raise ValueError("unsupported Grbl stage")
        index += 1
        command = lines[index]
        offset_match = _OFFSET.fullmatch(command)
        if command == "G49":
            offset = 0.0
        elif offset_match is not None:
            offset = _value(offset_match.group(1))
        else:
            raise ValueError("unsupported Grbl length state")
        # The machine has not moved, but the active work coordinate has.
        before_offset = at
        changed = old_offset != offset
        at = (at[0], at[1], at[2] + old_offset - offset)
        old_offset = offset
        index += 1
        transitions = []
        if changed:
            if index >= len(lines) or not lines[index].startswith("G0 "):
                raise ValueError("missing Grbl offset compensation motion")
            move, _ = _read_motion(lines[index], at, None)
            if any(abs(a - b) > 0.000051 for a, b in zip(
                    move.end, before_offset)):
                raise ValueError("Grbl offset compensation missed safe tip")
            transitions.append(move)
            at = move.end
            index += 1
        if index >= len(lines):
            raise ValueError("incomplete Grbl stage")
        rpm = _read_spindle(lines[index])
        index += 1
        feed, moves = None, []
        while index < len(lines) and lines[index] != "M5":
            move, feed = _read_motion(lines[index], at, feed)
            moves.append(move)
            at = move.end
            index += 1
        if not moves or index >= len(lines):
            raise ValueError("incomplete Grbl stage motion")
        index += 1
        ended = index == len(lines) - 1 and lines[index] == "M30"
        pause = not ended and index < len(lines) - 1 and lines[index] == "M0"
        if not ended and not pause:
            raise ValueError("unsupported Grbl stop or end")
        stages.append(DecodedStage(stage.group(1), offset, rpm, tuple(moves),
                                   tuple(transitions), pause, at, ended, command))
        if ended:
            break
        index += 1
    if (not stages or index != len(lines) - 1 or
            any(not stage.pause_after for stage in stages[:-1])):
        raise ValueError("Grbl trailing words or missing end")
    return DecodedJob(tuple(stages), at, _GRBL_STARTUP)


def decode(files, dialect, *, initial_work_tip):
    """Decode every byte, retaining ordered stages and real offset travel."""
    dialect = _dialect(dialect)
    initial = _tip(initial_work_tip)
    if type(files) not in (tuple, list) or not files:
        raise ValueError("nonempty ordered NC file tuple required")
    if dialect == "uccnc":
        return _decode_uccnc(files, initial)
    if len(files) != 1:
        raise ValueError("Grbl job requires one program")
    return _decode_grbl(files[0], initial)
