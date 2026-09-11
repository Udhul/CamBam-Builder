"""Regression coverage for the 4e external-client artifact verifier."""

from pathlib import Path
import shutil
import unittest
import uuid

from cambam_builder import CBProject
from demos.mcp_client_acceptance_verify import verify_pair


class MCPAcceptanceVerifierTests(unittest.TestCase):
    def test_accepts_contract_pair_and_rejects_untranslated_b(self):
        root = Path("output") / f"mcp-acceptance-verifier-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, root, True)
        try:
            project = CBProject("slice")
            layer = project.add_layer("Geometry")
            rect = project.add_rect(layer, corner=(0, 0), width=20, height=10,
                                    identifier="outline")
            part = project.add_part("Part", stock_width=0, stock_height=0,
                                    stock_thickness=0, stock_material="")
            project.add_profile_mop(
                part, [rect], name="profile", identifier="profile", profile_side="Outside",
                target_depth=-1, depth_increment=0.5, tool_diameter=3,
                cut_feedrate=300, plunge_feedrate=100, spindle_speed=12000,
                stock_surface=0, clearance_plane=5,
            )
            a_path = root / "A.cb"
            b_path = root / "B.cb"
            project.save(str(a_path))
            project.save(str(b_path))
            with self.assertRaisesRegex(ValueError, "must differ"):
                verify_pair(a_path, b_path)

            project.translate_primitive(rect, 5, 2, bake=False)
            project.save(str(b_path))
            result = verify_pair(a_path, b_path)
            self.assertIn("artifact contract passed", result["status"])
            self.assertEqual(result["B"]["world_xyz"], (
                (5.0, 2.0, 0.0), (25.0, 2.0, 0.0),
                (25.0, 12.0, 0.0), (5.0, 12.0, 0.0),
            ))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
