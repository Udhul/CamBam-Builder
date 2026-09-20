"""Native MOP parameter state and template fidelity regressions."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cambam_entities import (
    MOP_COMMON_FIELD_POLICIES,
    MOP_DRILL_FIELD_POLICIES,
    MOP_ENGRAVE_FIELD_POLICIES,
    MOP_POCKET_FIELD_POLICIES,
    MOP_PROFILE_FIELD_POLICIES,
)
from cambam_builder.cambam_reader import read_cambam_file
from cambam_builder.cambam_writer import save_cambam_file


class MopParameterTests(unittest.TestCase):
    def setUp(self):
        self.output = Path("output").resolve()
        self.output.mkdir(exist_ok=True)

    def make_project(self):
        project = CBProject("parameter-regression")
        layer = project.add_layer("Geometry")
        part = project.add_part("Machining")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        mops = [
            project.add_profile_mop(part, targets=[target], identifier="profile",
                                    target_depth=-1.25, stepover=0.2,
                                    corner_overcut=True,
                                    custom_mop_header="  G0 X0\n",
                                    tab_method="Automatic", tab_style="Triangle"),
            project.add_pocket_mop(part, targets=[target], identifier="pocket",
                                   target_depth=-2.0, region_fill_style="HorizontalScanline"),
            project.add_engrave_mop(part, targets=[target], identifier="engrave",
                                    target_depth=-0.4, cut_ordering="LevelFirst"),
            project.add_drill_mop(part, targets=[target], identifier="drill",
                                  target_depth=-3.0, drilling_method="SpiralMill_CW",
                                  tool_diameter=1.0, hole_diameter=2.2,
                                  drill_lead_out=True, lead_out_length=1.0),
        ]
        return project, mops

    @staticmethod
    def mop(tree, identifier):
        for element in tree.getroot().findall("./parts/part/machineops/*"):
            if identifier in (element.findtext("Tag") or ""):
                return element
        raise AssertionError(identifier)

    def save(self, project, directory, name):
        path = Path(directory) / f"{name}.cb"
        save_cambam_file(project, str(path))
        return path

    def test_supported_fields_and_unknown_native_children_round_trip(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            profile = self.mop(tree, "profile")
            ET.SubElement(profile, "NativeExtension").text = "retained"
            lead_out = ET.SubElement(profile, "LeadOutMove", {"state": "Default"})
            ET.SubElement(lead_out, "NativeLeadOutField").text = "keep"
            profile.find("TargetDepth").set("state", "Default")
            profile.find("TargetDepth").text = "-99.5"
            tree.write(first, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(first))
            self.assertIsNotNone(loaded)
            self.assertEqual(-99.5, loaded.get_mop("profile").target_depth)
            self.assertEqual("Default", loaded.get_mop("profile")._xml_parameter_states["target_depth"])
            self.assertEqual("Triangle", loaded.get_mop("profile").tab_style)
            self.assertTrue(loaded.get_mop("profile").corner_overcut)
            self.assertEqual("  G0 X0\n", loaded.get_mop("profile").custom_mop_header)
            self.assertEqual("HorizontalScanline", loaded.get_mop("pocket").region_fill_style)
            self.assertEqual("LevelFirst", loaded.get_mop("engrave").cut_ordering)
            self.assertEqual(2.2, loaded.get_mop("drill").hole_diameter)

            second = self.save(loaded, directory, "second")
            round_trip = ET.parse(second)
            profile = self.mop(round_trip, "profile")
            self.assertEqual("Value", profile.find("CornerOvercut").get("state"))
            self.assertEqual("true", profile.findtext("CornerOvercut"))
            self.assertEqual("retained", profile.findtext("NativeExtension"))
            self.assertEqual("keep", profile.findtext("LeadOutMove/NativeLeadOutField"))
            self.assertEqual("Default", profile.find("TargetDepth").get("state"))
            self.assertEqual("-99.5", profile.find("TargetDepth").text)

    def test_direct_edit_overrides_imported_state_and_explicit_state_api(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            loaded = read_cambam_file(str(first))
            mop = loaded.get_mop("profile")
            mop.target_depth = -7.0
            second = self.save(loaded, directory, "edited")
            element = self.mop(ET.parse(second), "profile")
            self.assertEqual("Value", element.find("TargetDepth").get("state"))
            self.assertEqual("-7.0", element.find("TargetDepth").text)

            mop.set_parameter_state("target_depth", "Default")
            third = self.save(loaded, directory, "default")
            element = self.mop(ET.parse(third), "profile")
            self.assertEqual("Default", element.find("TargetDepth").get("state"))
            mop.target_depth = -7.0  # Repeating a cached value intentionally makes it explicit.
            element = mop.to_xml_element(loaded, [1])
            self.assertEqual("Value", element.find("TargetDepth").get("state"))
            with self.assertRaises(ValueError):
                mop.set_parameter_state("no_such_parameter", "Value")
            with self.assertRaises(ValueError):
                mop.set_parameter_state("target_depth", "Inherited")
            with self.assertRaises(ValueError):
                mop.set_parameter_state("tab_width", "Default")

    def test_explicit_state_api_applies_to_new_mops(self):
        project, mops = self.make_project()
        mop = mops[0]
        mop.set_parameter_state("target_depth", "Default")
        element = mop.to_xml_element(project, [1])
        self.assertEqual("Default", element.find("TargetDepth").get("state"))

    def test_common_fresh_export_policy_is_shared_by_all_mop_families(self):
        project = CBProject("common-policy", default_tool_diameter=6.5)
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part", default_spindle_speed=12000)
        mops = (
            project.add_profile_mop(part, [target], identifier="profile"),
            project.add_pocket_mop(part, [target], identifier="pocket"),
            project.add_engrave_mop(part, [target], identifier="engrave"),
            project.add_drill_mop(part, [target], identifier="drill"),
        )

        omitted = {
            "TargetDepth", "DepthIncrement", "CutFeedrate",
            "CustomMOPHeader", "CustomMOPFooter",
        }
        for mop in mops:
            with self.subTest(mop=type(mop).__name__):
                element = mop.to_xml_element(project, [1])
                for field_name, policy in MOP_COMMON_FIELD_POLICIES.items():
                    child = element.find(policy.xml_tag)
                    if policy.xml_tag in omitted:
                        self.assertIsNone(child, field_name)
                    else:
                        self.assertIsNotNone(child, field_name)
                        self.assertEqual("Value", child.get("state"), field_name)
                self.assertEqual("6.5", element.findtext("ToolDiameter"))
                self.assertEqual("12000", element.findtext("SpindleSpeed"))
                for unmodeled in ("SpindleRange", "StartPoint"):
                    self.assertIsNone(element.find(unmodeled))
        self.assertIsNone(mops[-1].to_xml_element(project, [1]).find("RoughingFinishing"))

    def test_common_policy_does_not_invent_depth_or_feed_fallbacks(self):
        project = CBProject("no-fallbacks")
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part")
        mop = project.add_profile_mop(
            part, [target], identifier="profile", target_depth=-2.0,
            custom_mop_header="G90", custom_mop_footer="M5",
        )
        element = mop.to_xml_element(project, [1])

        self.assertEqual("-2.0", element.findtext("TargetDepth"))
        self.assertIsNone(element.find("DepthIncrement"))
        self.assertIsNone(element.find("CutFeedrate"))
        self.assertEqual("G90", element.findtext("CustomMOPHeader"))
        self.assertEqual("M5", element.findtext("CustomMOPFooter"))
        for tag in ("TargetDepth", "CustomMOPHeader", "CustomMOPFooter"):
            self.assertEqual("Value", element.find(tag).get("state"), tag)

    def test_profile_and_pocket_fresh_subtype_policies_are_mode_aware(self):
        project = CBProject("subtype-policy")
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part")
        profile = project.add_profile_mop(
            part, [target], identifier="profile", lead_in_type="None",
            final_depth_increment=None, tab_method="None",
        )
        pocket = project.add_pocket_mop(
            part, [target], identifier="pocket", lead_in_type="None",
            final_depth_increment=None,
        )

        for mop, policies in (
                (profile, MOP_PROFILE_FIELD_POLICIES),
                (pocket, MOP_POCKET_FIELD_POLICIES)):
            element = mop.to_xml_element(project, [1])
            for field_name, policy in policies.items():
                node = element.find("/".join(policy.xml_path))
                applicable = mop._policy_applies(policy)
                omitted = (policy.omit_when_none
                           and getattr(mop, field_name) is None)
                if not applicable or omitted:
                    self.assertIsNone(node, field_name)
                else:
                    self.assertIsNotNone(node, field_name)
                    if policy.leaf_state:
                        self.assertEqual("Value", node.get("state"), field_name)
            self.assertEqual("Value", element.find("LeadInMove").get("state"))
            self.assertEqual("None", element.findtext("LeadInMove/LeadInType"))
            self.assertIsNone(element.find("LeadInMove/SpiralAngle"))
            self.assertIsNone(element.find("LeadOutMove"))
            self.assertIsNone(element.find("LeadInMove/TangentRadius"))
            self.assertIsNone(element.find("LeadInMove/LeadInFeedrate"))

        tabs = profile.to_xml_element(project, [1]).find("HoldingTabs")
        self.assertEqual("Value", tabs.get("state"))
        self.assertEqual(["TabMethod"], [child.tag for child in tabs])

    def test_spiral_lead_and_automatic_tabs_emit_only_mode_dependencies(self):
        project = CBProject("nested-policy")
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part")
        profile = project.add_profile_mop(
            part, [target], identifier="profile", lead_in_type="Spiral",
            lead_in_spiral_angle=17.5, tab_method="Automatic",
            tab_width=7, tab_height=2, tab_min_tabs=2, tab_max_tabs=5,
            tab_distance=0, tab_size_threshold=3, tab_use_leadins=False,
            tab_style="Triangle",
        )
        element = profile.to_xml_element(project, [1])

        lead = element.find("LeadInMove")
        self.assertEqual("Value", lead.get("state"))
        self.assertEqual(["LeadInType", "SpiralAngle"],
                         [child.tag for child in lead])
        self.assertTrue(all(child.get("state") == "Value" for child in lead))
        tabs = element.find("HoldingTabs")
        self.assertEqual("Value", tabs.get("state"))
        self.assertEqual(
            ["TabMethod", "Width", "Height", "MinimumTabs", "MaximumTabs",
             "TabDistance", "SizeThreshold", "UseLeadIns", "TabStyle"],
            [child.tag for child in tabs],
        )
        self.assertTrue(all("state" not in child.attrib for child in tabs))

    def test_engrave_fresh_subtype_policy_omits_unset_final_increment(self):
        project = CBProject("engrave-policy")
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part")
        engrave = project.add_engrave_mop(
            part, [target], identifier="engrave",
            roughing_finishing="Finishing", final_depth_increment=None,
            cut_ordering="LevelFirst",
        )

        element = engrave.to_xml_element(project, [1])
        self.assertEqual(
            {"roughing_finishing", "final_depth_increment", "cut_ordering"},
            set(MOP_ENGRAVE_FIELD_POLICIES),
        )
        self.assertEqual("Value", element.find("RoughingFinishing").get("state"))
        self.assertEqual("Finishing", element.findtext("RoughingFinishing"))
        self.assertIsNone(element.find("FinalDepthIncrement"))
        self.assertEqual("Value", element.find("CutOrdering").get("state"))
        self.assertEqual("LevelFirst", element.findtext("CutOrdering"))

    def test_drill_fresh_policy_is_exhaustive_and_method_aware(self):
        project = CBProject("drill-policy")
        layer = project.add_layer("Geometry")
        target = project.add_circle(
            layer, center=(0, 0), identifier="target", diameter=8)
        part = project.add_part("Part")
        drills = (
            project.add_drill_mop(
                part, [target], identifier="canned", drilling_method="CannedCycle",
                peck_distance=1.5, retract_height=3, dwell=25,
                custom_script="ignored inactive script"),
            project.add_drill_mop(
                part, [target], identifier="spiral", drilling_method="SpiralMill_CW",
                tool_diameter=2, hole_diameter=8, drill_lead_out=True,
                spiral_flat_base=False, lead_out_length=-0.5),
            project.add_drill_mop(
                part, [target], identifier="auto", drilling_method="SpiralMill_CCW",
                tool_diameter=2, hole_diameter=None),
            project.add_drill_mop(
                part, [target], identifier="script", drilling_method="CustomScript",
                custom_script="$x,$y|G1 Z$z F$f"),
        )

        method_fields = {policy.xml_path[0]
                         for policy in MOP_DRILL_FIELD_POLICIES.values()}
        expected = (
            {"DrillingMethod", "PeckDistance", "RetractHeight", "Dwell"},
            {"DrillingMethod", "HoleDiameter", "DrillLeadOut",
             "SpiralFlatBase", "LeadOutLength"},
            {"DrillingMethod", "HoleDiameter", "DrillLeadOut",
             "SpiralFlatBase", "LeadOutLength"},
            {"DrillingMethod", "CustomScript"},
        )
        for drill, expected_fields in zip(drills, expected):
            element = drill.to_xml_element(project, [1])
            present = {child.tag for child in element if child.tag in method_fields}
            self.assertEqual(expected_fields, present, drill.user_identifier)
            for field_name, policy in MOP_DRILL_FIELD_POLICIES.items():
                node = element.find(policy.xml_path[0])
                applicable = drill._policy_applies(policy)
                omitted = drill._policy_omits(policy, getattr(drill, field_name))
                self.assertEqual(applicable and not omitted, node is not None,
                                 field_name)
                if node is not None:
                    expected_state = drill._policy_state(
                        policy, getattr(drill, field_name))
                    self.assertEqual(expected_state, node.get("state"), field_name)
        self.assertEqual("Default", drills[2].to_xml_element(
            project, [1]).find("HoleDiameter").get("state"))
        self.assertEqual("$x,$y|G1 Z$z F$f", drills[3].to_xml_element(
            project, [1]).findtext("CustomScript"))

    def test_drill_fresh_unsupported_and_empty_custom_methods_are_rejected(self):
        project = CBProject("drill-method-validation")
        layer = project.add_layer("Geometry")
        target = project.add_circle(
            layer, center=(0, 0), identifier="target", diameter=8)
        part = project.add_part("Part")
        unsupported = project.add_drill_mop(
            part, [target], identifier="plugin", drilling_method="PluginMethod")
        with self.assertRaisesRegex(ValueError, "preserve-only"):
            unsupported.to_xml_element(project, [1])
        empty_script = project.add_drill_mop(
            part, [target], identifier="script", drilling_method="CustomScript")
        with self.assertRaisesRegex(ValueError, "nonempty custom_script"):
            empty_script.to_xml_element(project, [1])

    def test_imported_drill_method_switch_reconciles_modeled_fields(self):
        project = CBProject("drill-method-switch")
        layer = project.add_layer("Geometry")
        target = project.add_circle(
            layer, center=(0, 0), identifier="target", diameter=8)
        part = project.add_part("Part")
        project.add_drill_mop(
            part, [target], identifier="drill", drilling_method="CannedCycle",
            tool_diameter=2, peck_distance=1, retract_height=3, dwell=10)
        with tempfile.TemporaryDirectory(
                prefix="mop-parameters-", dir=self.output) as directory:
            source = self.save(project, directory, "canned")
            tree = ET.parse(source)
            ET.SubElement(self.mop(tree, "drill"), "NativeMethodField").text = "keep"
            tree.write(source, encoding="utf-8", xml_declaration=True)

            invalid = read_cambam_file(str(source))
            invalid_drill = invalid.get_mop("drill")
            invalid_drill.drilling_method = "SpiralMill_CW"
            invalid_drill.hole_diameter = 2
            with self.assertRaisesRegex(ValueError, "greater than tool_diameter"):
                invalid_drill.to_xml_element(invalid, [1])

            loaded = read_cambam_file(str(source))
            drill = loaded.get_mop("drill")
            drill.drilling_method = "SpiralMill_CW"
            drill.hole_diameter = None
            spiral = drill.to_xml_element(loaded, [1])
            self.assertEqual("keep", spiral.findtext("NativeMethodField"))
            self.assertIsNone(spiral.find("PeckDistance"))
            self.assertIsNone(spiral.find("RetractHeight"))
            self.assertIsNone(spiral.find("Dwell"))
            self.assertEqual("Default", spiral.find("HoleDiameter").get("state"))
            for tag in ("DrillLeadOut", "SpiralFlatBase", "LeadOutLength"):
                self.assertEqual("Value", spiral.find(tag).get("state"), tag)

            spiral_path = self.save(loaded, directory, "spiral")
            scripted = read_cambam_file(str(spiral_path))
            drill = scripted.get_mop("drill")
            drill.drilling_method = "CustomScript"
            drill.custom_script = "$x,$y|G1 Z$z"
            custom = drill.to_xml_element(scripted, [1])
            self.assertEqual("$x,$y|G1 Z$z", custom.findtext("CustomScript"))
            for tag in ("PeckDistance", "RetractHeight", "Dwell", "HoleDiameter",
                        "DrillLeadOut", "SpiralFlatBase", "LeadOutLength"):
                self.assertIsNone(custom.find(tag), tag)
            self.assertEqual("keep", custom.findtext("NativeMethodField"))

    def test_unknown_imported_drill_method_is_preserved_but_not_switched(self):
        project = CBProject("unknown-drill-method")
        layer = project.add_layer("Geometry")
        target = project.add_circle(
            layer, center=(0, 0), identifier="target", diameter=8)
        part = project.add_part("Part")
        project.add_drill_mop(part, [target], identifier="drill")
        with tempfile.TemporaryDirectory(
                prefix="mop-parameters-", dir=self.output) as directory:
            source = self.save(project, directory, "native")
            tree = ET.parse(source)
            native = self.mop(tree, "drill")
            native.find("DrillingMethod").text = "PluginMethod"
            ET.SubElement(native, "PluginMethodData").text = "keep"
            tree.write(source, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(source))
            preserved = loaded.get_mop("drill").to_xml_element(loaded, [1])
            self.assertEqual("PluginMethod", preserved.findtext("DrillingMethod"))
            self.assertEqual("keep", preserved.findtext("PluginMethodData"))
            loaded.get_mop("drill").drilling_method = "CannedCycle"
            with self.assertRaisesRegex(ValueError, "unsupported imported"):
                loaded.get_mop("drill").to_xml_element(loaded, [1])

    def test_imported_nested_mode_switches_reconcile_modeled_children(self):
        project = CBProject("nested-switch")
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part")
        project.add_profile_mop(
            part, [target], identifier="profile", lead_in_type="None",
            tab_method="None",
        )
        project.add_pocket_mop(
            part, [target], identifier="pocket", lead_in_type="None")
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(project, directory, "none")
            loaded = read_cambam_file(str(first))
            mop = loaded.get_mop("profile")
            mop.lead_in_type = "Spiral"
            mop.tab_method = "Automatic"
            enabled = mop.to_xml_element(loaded, [1])
            self.assertEqual("30.0", enabled.findtext("LeadInMove/SpiralAngle"))
            self.assertEqual("6.0", enabled.findtext("HoldingTabs/Width"))
            self.assertEqual("Square", enabled.findtext("HoldingTabs/TabStyle"))
            pocket = loaded.get_mop("pocket")
            pocket.lead_in_type = "Spiral"
            self.assertEqual("30.0", pocket.to_xml_element(
                loaded, [1]).findtext("LeadInMove/SpiralAngle"))

            second = self.save(loaded, directory, "automatic")
            tree = ET.parse(second)
            profile = self.mop(tree, "profile")
            ET.SubElement(profile.find("LeadInMove"), "NativeLeadField").text = "keep"
            ET.SubElement(profile.find("HoldingTabs"), "NativeTabField").text = "keep"
            lead_out = ET.SubElement(profile, "LeadOutMove", {"state": "Default"})
            ET.SubElement(lead_out, "NativeLeadOutField").text = "keep"
            tree.write(second, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(second))
            mop = loaded.get_mop("profile")
            mop.lead_in_type = "None"
            mop.tab_method = "None"
            disabled = mop.to_xml_element(loaded, [1])
            self.assertIsNone(disabled.find("LeadInMove/SpiralAngle"))
            self.assertEqual("keep", disabled.findtext("LeadInMove/NativeLeadField"))
            self.assertEqual(["TabMethod", "NativeTabField"],
                             [child.tag for child in disabled.find("HoldingTabs")])
            self.assertEqual("keep", disabled.findtext(
                "LeadOutMove/NativeLeadOutField"))

    def test_unsupported_nested_authoring_is_rejected_but_import_preserves_it(self):
        project = CBProject("unsupported-nested")
        layer = project.add_layer("Geometry")
        target = project.add_rect(layer, identifier="target", width=4, height=2)
        part = project.add_part("Part")
        tangent = project.add_profile_mop(
            part, [target], identifier="tangent", lead_in_type="Tangent")
        with self.assertRaisesRegex(ValueError, "preserve-only"):
            tangent.to_xml_element(project, [1])
        invalid_tabs = project.add_profile_mop(
            part, [target], identifier="tabs", lead_in_type="None",
            tab_method="Automatic", tab_style="Square", tab_use_leadins=True)
        with self.assertRaisesRegex(ValueError, "active lead-in"):
            invalid_tabs.to_xml_element(project, [1])

        native_project = CBProject("preserve-native-nested")
        layer = native_project.add_layer("Geometry")
        target = native_project.add_rect(
            layer, identifier="target", width=4, height=2)
        part = native_project.add_part("Part")
        native_project.add_profile_mop(
            part, [target], identifier="native", lead_in_type="None",
            tab_method="None")
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            source = self.save(native_project, directory, "native")
            tree = ET.parse(source)
            profile = self.mop(tree, "native")
            profile.find("LeadInMove/LeadInType").text = "Tangent"
            profile.find("HoldingTabs/TabMethod").text = "Manual"
            ET.SubElement(profile.find("HoldingTabs"), "ManualTabPoints").text = "native"
            tree.write(source, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(source))
            preserved = loaded.get_mop("native").to_xml_element(loaded, [1])
            self.assertEqual("Tangent", preserved.findtext(
                "LeadInMove/LeadInType"))
            self.assertEqual("Manual", preserved.findtext(
                "HoldingTabs/TabMethod"))
            self.assertEqual("native", preserved.findtext(
                "HoldingTabs/ManualTabPoints"))

            lead_switch = read_cambam_file(str(source))
            lead_switch.get_mop("native").lead_in_type = "None"
            with self.assertRaisesRegex(ValueError, "unsupported native lead mode"):
                lead_switch.get_mop("native").to_xml_element(lead_switch, [1])
            tab_switch = read_cambam_file(str(source))
            tab_switch.get_mop("native").tab_method = "Automatic"
            with self.assertRaisesRegex(ValueError, "Manual holding-tab"):
                tab_switch.get_mop("native").to_xml_element(tab_switch, [1])

    def test_spiral_drill_signed_clearance_geometry_and_tool_fit(self):
        project = CBProject("spiral-clearance")
        layer = project.add_layer("Geometry")
        target = project.add_circle(layer, center=(0, 0), identifier="hole", diameter=6)
        part = project.add_part("Part")
        drill = project.add_drill_mop(
            part, targets=[target], identifier="spiral",
            drilling_method="SpiralMill_CW", tool_diameter=4,
            hole_diameter=6, roughing_clearance=-1,
        )

        self.assertEqual(8, drill.effective_spiral_hole_diameter())
        element = drill.to_xml_element(project, [1])
        self.assertEqual("6", element.findtext("HoleDiameter"))
        self.assertEqual("-1", element.findtext("RoughingClearance"))
        for tag in ("PeckDistance", "RetractHeight", "Dwell", "CustomScript"):
            self.assertIsNone(element.find(tag), tag)

        drill.lead_out_length = 1
        with self.assertRaisesRegex(ValueError, "drill_lead_out is false"):
            drill.to_xml_element(project, [1])
        drill.drill_lead_out = True
        drill.lead_out_length = 4.1
        with self.assertRaisesRegex(ValueError, "effective hole radius"):
            drill.to_xml_element(project, [1])
        drill.lead_out_length = 4
        drill.to_xml_element(project, [1])

        drill.roughing_clearance = 1
        drill.lead_out_length = 0
        self.assertEqual(4, drill.effective_spiral_hole_diameter())
        with self.assertRaisesRegex(ValueError, "greater than tool_diameter"):
            drill.to_xml_element(project, [1])

        drill.roughing_clearance = 0.9
        self.assertAlmostEqual(4.2, drill.effective_spiral_hole_diameter())
        drill.to_xml_element(project, [1])

    def test_missing_optional_elements_remain_absent_until_edited(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            drill = self.mop(tree, "drill")
            drill.remove(drill.find("HoleDiameter"))
            tree.write(first, encoding="utf-8", xml_declaration=True)
            loaded = read_cambam_file(str(first))
            self.assertIsNone(loaded.get_mop("drill").hole_diameter)
            second = self.save(loaded, directory, "second")
            self.assertIsNone(self.mop(ET.parse(second), "drill").find("HoleDiameter"))
            loaded.get_mop("drill").hole_diameter = 3.0
            third = self.save(loaded, directory, "third")
            hole = self.mop(ET.parse(third), "drill").find("HoleDiameter")
            self.assertIsNotNone(hole)
            self.assertEqual("Value", hole.get("state"))
            self.assertEqual("3.0", hole.text)

    def test_imported_spiral_irrelevant_default_fields_are_preserved(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            drill = self.mop(tree, "drill")
            for tag, value in (("PeckDistance", "0"), ("RetractHeight", "3"),
                               ("Dwell", "0"), ("CustomScript", None)):
                element = ET.SubElement(drill, tag, {"state": "Default"})
                element.text = value
            tree.write(first, encoding="utf-8", xml_declaration=True)

            loaded = read_cambam_file(str(first))
            second = self.save(loaded, directory, "second")
            preserved = self.mop(ET.parse(second), "drill")
            for tag, value in (("PeckDistance", "0"), ("RetractHeight", "3"),
                               ("Dwell", "0"), ("CustomScript", None)):
                element = preserved.find(tag)
                self.assertIsNotNone(element, tag)
                self.assertEqual("Default", element.get("state"), tag)
                self.assertEqual(value, element.text, tag)

    def test_nested_edits_activate_container_without_changing_plain_scalar_format(self):
        source, _ = self.make_project()
        with tempfile.TemporaryDirectory(prefix="mop-parameters-", dir=self.output) as directory:
            first = self.save(source, directory, "first")
            tree = ET.parse(first)
            profile = self.mop(tree, "profile")
            profile.find("HoldingTabs").set("state", "Default")
            profile.find("LeadInMove").set("state", "Default")
            tree.write(first, encoding="utf-8")
            loaded = read_cambam_file(str(first))
            mop = loaded.get_mop("profile")
            self.assertIsNone(mop._xml_template.find("primitive"))
            mop.tab_width = 9.0
            mop.lead_in_spiral_angle = 22.0
            element = mop.to_xml_element(loaded, [1])
            self.assertEqual("Value", element.find("HoldingTabs").get("state"))
            self.assertEqual("9.0", element.findtext("HoldingTabs/Width"))
            self.assertNotIn("state", element.find("HoldingTabs/Width").attrib)
            self.assertEqual("Value", element.find("LeadInMove").get("state"))
            self.assertEqual("22.0", element.findtext("LeadInMove/SpiralAngle"))


if __name__ == "__main__":
    unittest.main()
