"""Native triangle source, separate preview and explicit posted-motion gate."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from cambam_builder.integrations.cambam import native_convex_rest as native
from cambam_builder.native.cad import Vertex
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


class NativeConvexRestTests(unittest.TestCase):
    def setUp(self):
        Path("output").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="native-convex-rest-tests-",
                                                dir="output")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "triangle"
        self.manifest = native.build_workflow(self.root)

    def _post(self):
        script = (self.root / "explicit" / "triangle-explicit.cb")
        project = read_cambam_bytes(script.read_bytes())
        lines = [
            "( Made using CamBam - test fixture only )",
            "( triangle-explicit 1/1/2000 1:00:00 PM )",
            "( Post processor: Default )",
            "G21 G90 G61 G40", "G0 Z5", "T3 M6", "G17", "M3 S12000",
            project.list_mops()[0].custom_script, "M5", "M30", "",
        ]
        path = self.root / "synthetic-post.nc"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_distinct_source_preview_and_literal_candidate(self):
        root = self.root
        source = read_cambam_bytes((root / "source.cb").read_bytes())
        preview = read_cambam_bytes((root / "preview/triangle-preview.cb").read_bytes())
        explicit = read_cambam_bytes((root / "explicit/triangle-explicit.cb").read_bytes())
        self.assertEqual(len(source.list_primitives()), 1)
        self.assertEqual(source.list_mops(), [])
        self.assertEqual(len(preview.list_primitives()), 2)
        self.assertEqual(len(explicit.list_primitives()), 2)
        self.assertEqual(len([m for m in preview.list_mops() if m.enabled]), 1)
        self.assertEqual(len([m for m in explicit.list_mops() if m.enabled]), 1)
        self.assertEqual(self.manifest["execution_status"], "pending_actual_CamBam_post")
        self.assertAlmostEqual(self.manifest["pure_rest_mm2"]["0"],
                               48 - 1.44 * math.pi, delta=1e-9)
        self.assertAlmostEqual(self.manifest["pure_rest_mm2"]["1"],
                               64 / 3 - .04 * math.pi, delta=1e-9)
        self.assertEqual(self.manifest["completion"], "partial_target_completion")

    def test_supplied_prior_and_source_freshness(self):
        prior = json.loads((self.root / "prior.json").read_text(encoding="utf-8"))
        source = (self.root / "source.cb").read_bytes()
        prior["items"][2]["end"][2] = -1.1
        prior["items"][3]["start"][2] = -1.1
        with self.assertRaisesRegex(ValueError, "original clearance"):
            native._plan(source, prior)
        prior = native.synthetic_prior(source)
        prior["source_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "stale"):
            native._plan(source, prior)
        edited = native.synthetic_source()
        edited.get_primitive("triangle-finish").outer_curve.vertices[1] = Vertex(13, 0)
        changed_source = serialize_cambam_bytes(edited)
        with self.assertRaisesRegex(ValueError, "unsupported triangle Region"):
            native._plan(changed_source, native.synthetic_prior(changed_source))
        with self.assertRaisesRegex(ValueError, "supplied prior trace"):
            native.build_workflow(Path(self.temp.name) / "missing-prior",
                                  source_path=self.root / "source.cb")
        rebuilt = native.build_workflow(
            Path(self.temp.name) / "from-existing-source",
            source_path=self.root / "source.cb",
            prior_path=self.root / "prior.json")
        self.assertEqual(rebuilt["pure_rest_mm2"], self.manifest["pure_rest_mm2"])

    def test_synthetic_post_audit_and_tamper_rejection(self):
        # A constructed post exercises the parser; only a real CamBam post can
        # establish the user acceptance gate for a delivered candidate.
        post = self._post()
        manifest = self.root / "expected-motion.json"
        report = native.audit_post(manifest, post)
        self.assertEqual(report["status"], "bounded_triangle_post_pass")
        self.assertEqual(report["stock_prefixes"], (("prior", 1), ("cleanup", 2)))
        self.assertAlmostEqual(report["final_rest_mm2"]["0"],
                               self.manifest["final_rest_mm2"]["0"], delta=1e-9)
        text = post.read_text(encoding="utf-8")
        post.write_text(text.replace("X4 Y2 Z-2", "X4 Y2 Z-2.1"),
                        encoding="utf-8")
        self.assertNotEqual(native.audit_post(manifest, post)["status"],
                            "bounded_triangle_post_pass")
        candidate = self.root / "explicit" / "triangle-explicit.cb"
        candidate.write_bytes(candidate.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            native.audit_post(manifest, post)


if __name__ == "__main__":
    unittest.main()
