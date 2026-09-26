"""Independent ordered-job controller byte and state checks."""

from dataclasses import dataclass
import unittest

from cambam_builder.integrations.ordered_dialects import decode, render


@dataclass(frozen=True)
class Move:
    role: str
    start: tuple
    end: tuple
    feed: float


@dataclass(frozen=True)
class Stage:
    tool_id: str
    rpm: float
    offset_mm: float
    motions: tuple
    transition: object = None


@dataclass(frozen=True)
class Transition:
    boundary: str


@dataclass(frozen=True)
class Job:
    initial_tip: tuple
    translation_xyz_mm: tuple
    stages: tuple


def _stage(tool, offset, x, rpm, boundary=None):
    safe = (0, 0, 5)
    a, b, c = (x, 0, 5), (x, 0, -1), (x + 1, 0, -1)
    d = (x + 1, 0, 5)
    return Stage(tool, rpm, offset, (
        Move("rapid", safe, a, 0),
        Move("entry", a, b, 42.5),
        Move("cut", b, c, 375.25),
        Move("retract", c, d, 125),
        Move("rapid", d, safe, 0),
    ), Transition(boundary) if boundary else None)


class OrderedDialectTests(unittest.TestCase):
    def test_uccnc_split_arbitrary_tools_feeds_and_translation(self):
        job = Job((0, 0, 5), (12, -3, 1), (
            _stage("T17", 0, 2, 12250.5),
            _stage("T17", 0, 4, 10800, "split")))
        files = render(job, "uccnc")
        self.assertEqual(len(files), 2)
        decoded = decode(files, "uccnc", initial_work_tip=(12, -3, 6))
        self.assertEqual([stage.tool_id for stage in decoded.stages],
                         ["T17", "T17"])
        self.assertEqual(decoded.stages[0].rpm, 12250.5)
        self.assertEqual(decoded.stages[0].moves[1].feed, 42.5)
        self.assertEqual(decoded.stages[0].moves[2].end, (15, -3, 0))
        self.assertEqual(decoded.stages[1].moves[0].start, (12, -3, 6))
        self.assertEqual(decoded.stages[1].transition_moves, ())
        self.assertEqual(decoded.final_position, (12, -3, 6))

    def test_grbl_carries_offset_state_and_decodes_real_compensation(self):
        job = Job((0, 0, 5), (12, -3, 1), (
            _stage("T17", 2, 2, 12250),
            _stage("T8", 3.5, 4, 10800, "pause"),
            _stage("T17", 0, 6, 12250, "pause")))
        files = render(job, "grbl")
        self.assertEqual(len(files), 1)
        decoded = decode(files, "grbl", initial_work_tip=(12, -3, 6))
        self.assertEqual([s.tool_id for s in decoded.stages],
                         ["T17", "T8", "T17"])
        self.assertEqual([s.offset_mm for s in decoded.stages],
                         [2, 3.5, 0])
        self.assertEqual([s.pause_after for s in decoded.stages],
                         [True, True, False])
        self.assertEqual([m.start[2] for s in decoded.stages
                          for m in s.transition_moves], [4, 4.5, 9.5])
        self.assertEqual([m.end[2] for s in decoded.stages
                          for m in s.transition_moves], [6, 6, 6])
        self.assertEqual(decoded.stages[1].moves[0].start, (12, -3, 6))
        self.assertEqual(decoded.final_position, (12, -3, 6))

    def test_rejects_unknown_and_incomplete_bytes(self):
        job = Job((0, 0, 5), (0, 0, 0), (_stage("T42", 1, 2, 12000),))
        original = render(job, "grbl")[0]
        for altered in (
                original.replace(b"M30\n", b"G91\nM30\n"),
                original.replace(b"G43.1 Z1\n", b"G43.1 Q1\n"),
                original.replace(b"M5\n", b"M6\n"),
                original.replace(b"\n", b"\r\n"),
                original[:-1]):
            with self.subTest(altered=altered[-40:]):
                with self.assertRaises(ValueError):
                    decode((altered,), "grbl", initial_work_tip=(0, 0, 5))
        changed_effect = original.replace(b"G43.1 Z1\n", b"G43.1 Z2\n")
        self.assertEqual(decode((changed_effect,), "grbl",
                                initial_work_tip=(0, 0, 5)).stages[0].offset_mm, 2)
        uccnc = render(Job((0, 0, 5), (0, 0, 0),
                           (_stage("T42", 0, 2, 12000),)), "uccnc")[0]
        with self.assertRaises(ValueError):
            decode((uccnc.replace(b"G49\n", b"G43.1 Z1\n"),),
                   "uccnc", initial_work_tip=(0, 0, 5))

    def test_rejects_unsupported_source_values(self):
        job = Job((0, 0, 5), (0, 0, 0), (_stage("T42", 2, 2, 12000),))
        with self.assertRaisesRegex(ValueError, "requires G49"):
            render(job, "uccnc")
        with self.assertRaisesRegex(ValueError, "tool ID"):
            render(Job((0, 0, 5), (0, 0, 0),
                       (_stage("tool42", 0, 2, 12000),)), "grbl")


if __name__ == "__main__":
    unittest.main()
