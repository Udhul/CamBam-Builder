"""Native V input normalizes to the detached standalone request."""

import json
import tempfile
import unittest
from pathlib import Path

from cambam_builder.cam_core import tapered_vcarve
from cambam_builder.cambam_reader import read_cambam_bytes
from cambam_builder.integrations.cambam.native_variable_v import (
    build_native_workflow, normalize, normalize_bytes, plan_native_input, synthetic_setup,
    synthetic_source,
)
from cambam_builder.integrations.cambam.variable_cone_script import (
    audit_variable_post, build_variable_carrier,
)
from cambam_builder.integrations.cambam.variable_cone_engrave import build_engrave_candidate
from cambam_builder.integrations.direct_variable_v import build_program


class NativeVariableVTests(unittest.TestCase):
    def test_native_input_and_separate_derived_carriers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / "authored.cb"
            synthetic_source().save(str(original))
            source_bytes = original.read_bytes()
            request = normalize_bytes(source_bytes, synthetic_setup())
            self.assertEqual(request, tapered_vcarve.standalone_request())
            self.assertEqual(tapered_vcarve.generate(request),
                             tapered_vcarve.generate())

            result = build_native_workflow(directory / "native",
                                           source_path=original,
                                           setup=synthetic_setup())
            self.assertEqual(result["status"], "native_V_input_normalized")
            source = Path(result["original"])
            preview = Path(result["preview"])
            explicit = Path(result["explicit"])
            self.assertEqual(source.read_bytes(), source_bytes)
            reference = build_variable_carrier(directory / "standalone")
            native_motion = json.loads((explicit.parent / "expected-motion.json")
                                       .read_text(encoding="utf-8"))
            old_motion = json.loads((directory / "standalone" / "expected-motion.json")
                                    .read_text(encoding="utf-8"))
            self.assertEqual(native_motion["expected_items"],
                             old_motion["expected_items"])
            self.assertEqual(native_motion["script_lines"],
                             old_motion["script_lines"])
            self.assertEqual(result["plan_fingerprint"],
                             reference["plan_fingerprint"])

            source_project = read_cambam_bytes(source.read_bytes())
            source_guide = source_project.get_primitive("tapered-target-spine")
            source_mop = source_project.list_mops()[0]
            self.assertFalse(source_mop.enabled)
            self.assertEqual(source_project.get_mop_targets(source_mop),
                             [source_guide.internal_id])
            preview_project = read_cambam_bytes(preview.read_bytes())
            preview_guide = preview_project.get_primitive("tapered-target-spine")
            cut = preview_project.get_primitive("generated-variable-v-cut")
            self.assertEqual(preview_guide.internal_id, source_guide.internal_id)
            self.assertNotEqual(cut.internal_id, source_guide.internal_id)
            enabled = [m for m in preview_project.list_mops() if m.enabled]
            self.assertEqual(len(enabled), 1)
            self.assertEqual(preview_project.get_mop_targets(enabled[0]),
                             [cut.internal_id])
            explicit_project = read_cambam_bytes(explicit.read_bytes())
            explicit_guide = explicit_project.get_primitive("tapered-target-spine")
            anchor = explicit_project.get_primitive("variable-script-anchor")
            self.assertEqual(explicit_guide.internal_id, source_guide.internal_id)
            enabled = [m for m in explicit_project.list_mops() if m.enabled]
            self.assertEqual(len(enabled), 1)
            self.assertEqual(explicit_project.get_mop_targets(enabled[0]),
                             [anchor.internal_id])
            self.assertNotIn(source_guide.internal_id,
                             explicit_project.get_mop_targets(enabled[0]))
            synthetic_post = explicit.parent / "V-variable.nc"
            synthetic_post.write_text(
                "( V-variable synthetic reference )\n"
                "( Post processor: Default )\n"
                "G21 G90 G61 G40\nG0 Z5\nT3 M6\nG17\n"
                "M3 S12000\nG0 X-10 Y-10\nG98\n" +
                enabled[0].custom_script + "\nG80\nG0 Z5\nM5\nM30\n",
                encoding="utf-8")
            audit = audit_variable_post(explicit.parent / "expected-motion.json",
                                        synthetic_post)
            self.assertEqual(audit["status"],
                             "bounded_emitted_variable_v_motion_pass")

    def test_relevant_edits_and_inheritance_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.cb"
            synthetic_source().save(str(path))

            def read():
                return read_cambam_bytes(path.read_bytes())

            project = read()
            project.list_mops()[0].name = "renamed source intent"
            self.assertEqual(normalize(project, synthetic_setup()),
                             tapered_vcarve.standalone_request())
            project = read()
            project.get_primitive("tapered-target-spine").vertices[1].z = -0.5
            with self.assertRaisesRegex(ValueError, "bounded family"):
                normalize(project, synthetic_setup())
            project = read()
            project.list_parts()[0].stock_width = 15
            with self.assertRaisesRegex(ValueError, "bounded family"):
                normalize(project, synthetic_setup())
            project = read()
            project.list_mops()[0].tool_diameter = 5
            with self.assertRaisesRegex(ValueError, "tool_diameter"):
                normalize(project, synthetic_setup())
            project = read()
            project.list_mops()[0].set_parameter_state("tool_diameter", "Default")
            project.save(str(path))
            with self.assertRaisesRegex(ValueError, "inherited"):
                normalize(read(), synthetic_setup())
            setup = synthetic_setup()
            setup["cone_conical_length_mm"] = 2
            with self.assertRaisesRegex(ValueError, "90-degree"):
                normalize_bytes(path.read_bytes(), setup)
            synthetic_source().save(str(path))
            unknown = path.read_bytes().replace(
                b"</engrave>", b"<LeadInMove>Spiral</LeadInMove></engrave>")
            with self.assertRaisesRegex(ValueError, "XML fields"):
                normalize_bytes(unknown, synthetic_setup())

    def test_edited_native_geometry_stock_and_tool_plan_without_carriers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "edited.cb"
            project = synthetic_source()
            guide = project.get_primitive("tapered-target-spine")
            guide.vertices[0].x = 3
            guide.vertices[0].z = -0.8
            guide.vertices[1].x = 17
            guide.vertices[1].z = -2.4
            project.list_parts()[0].stock_width = 22
            project.list_mops()[0].tool_diameter = 8
            project.save(str(source))
            setup = synthetic_setup()
            setup["cone_maximum_radius_mm"] = 4
            setup["cone_conical_length_mm"] = 4
            setup["cut_x_interval_mm"] = [5, 15]
            request = normalize_bytes(source.read_bytes(), setup)
            self.assertEqual(request, tapered_vcarve.TaperedRequest(
                target_spine=(3, 17, 2, 0.8, 2.4),
                stock_bounds=(-2, -2, 20, 6), stock_bottom=-3,
                tool=request.tool, cut_interval=(5, 15)))
            evidence = plan_native_input(source.read_bytes(), setup)
            self.assertEqual(evidence["status"], "straight_variable_v_plan_verified")
            self.assertEqual(evidence["output_state"], "planning_only")
            self.assertEqual(evidence["cone_radius_mm"], 4)
            self.assertGreater(evidence["section_rest_mm2"]["depth_0"], 0)
            self.assertNotEqual(evidence["plan_fingerprint"],
                                tapered_vcarve.generate().fingerprint)
            with self.assertRaisesRegex(ValueError, "planning-only"):
                build_native_workflow(directory / "carrier", source_path=source,
                                      setup=setup)
            self.assertFalse((directory / "carrier").exists())
            for name, build in (("literal", build_variable_carrier),
                                ("preview", build_engrave_candidate)):
                with self.subTest(name=name), self.assertRaisesRegex(
                        ValueError, "planning-only"):
                    build(directory / name, native_source_bytes=source.read_bytes(),
                          native_setup=setup)
                self.assertFalse((directory / name).exists())
            with self.assertRaisesRegex(ValueError, "unsupported direct V plan"):
                build_program(directory / "direct", source_path=source, setup=setup)
            self.assertFalse((directory / "direct").exists())

            wrong = dict(setup, cone_maximum_radius_mm=3)
            with self.assertRaises(ValueError):
                normalize_bytes(source.read_bytes(), wrong)
            wrong = dict(setup, cut_x_interval_mm=[3, 15])
            with self.assertRaises(ValueError):
                normalize_bytes(source.read_bytes(), wrong)


if __name__ == "__main__":
    unittest.main()
