"""Executable coverage map for the core/MCP MOP parity contract."""

import json
import unittest
from dataclasses import fields
from pathlib import Path

from cambam_builder.cambam_entities import (
    Arc,
    Circle,
    DrillMop,
    EngraveMop,
    MOP_COMMON_FIELD_POLICIES,
    MOP_DRILL_FIELD_POLICIES,
    MOP_ENGRAVE_FIELD_POLICIES,
    MOP_POCKET_FIELD_POLICIES,
    MOP_PROFILE_FIELD_POLICIES,
    Pline,
    PocketMop,
    Points,
    ProfileMop,
    Rect,
    Text,
)
from cambam_builder.mcp_adapter.service import DocumentService
from cambam_builder.region import Region


IDENTITY_FIELDS = {"internal_id", "user_identifier", "name"}
PROTOCOL_FIELDS = {
    "workspace_id", "document", "expected_revision", "request_id",
    "identifier", "part", "targets",
}

# Each modeled field must have exactly one authoring disposition and one
# inspection disposition.  This data backs the human-readable matrix in
# docs/MCP_CONTRACT.md and deliberately says nothing about CamBam fields the
# core does not model.
PARITY = {
    "profile": {
        "class": ProfileMop,
        "author": {
            "enabled", "target_depth", "depth_increment", "stock_surface",
            "roughing_clearance", "clearance_plane", "spindle_speed",
            "tool_diameter", "plunge_feedrate", "cut_feedrate",
            "profile_side", "corner_overcut", "tab_method", "tab_width",
            "tab_height", "tab_min_tabs", "tab_max_tabs", "tab_distance",
            "tab_size_threshold", "tab_use_leadins", "tab_style",
        },
        "pinned": {
            "spindle_direction", "velocity_mode", "work_plane",
            "optimisation_mode", "tool_number", "tool_profile",
            "max_crossover_distance", "custom_mop_header",
            "custom_mop_footer", "stepover", "milling_direction",
            "collision_detection", "lead_in_type", "lead_in_spiral_angle",
            "final_depth_increment", "cut_ordering",
        },
        "excluded": set(),
        "inspection_hidden": {
            "optimisation_mode", "max_crossover_distance",
            "lead_in_spiral_angle",
        },
    },
    "pocket": {
        "class": PocketMop,
        "author": {
            "enabled", "target_depth", "depth_increment", "stock_surface",
            "roughing_clearance", "clearance_plane", "spindle_speed",
            "tool_diameter", "plunge_feedrate", "cut_feedrate",
        },
        "pinned": {
            "spindle_direction", "velocity_mode", "work_plane",
            "optimisation_mode", "tool_number", "tool_profile",
            "max_crossover_distance", "custom_mop_header",
            "custom_mop_footer", "stepover", "stepover_feedrate",
            "milling_direction", "collision_detection", "lead_in_type",
            "lead_in_spiral_angle", "final_depth_increment", "cut_ordering",
            "region_fill_style", "finish_stepover",
            "finish_stepover_at_target_depth", "roughing_finishing",
        },
        "excluded": set(),
        "inspection_hidden": {
            "optimisation_mode", "max_crossover_distance",
            "lead_in_spiral_angle",
        },
    },
    "engrave": {
        "class": EngraveMop,
        "author": {
            "enabled", "target_depth", "depth_increment", "stock_surface",
            "roughing_clearance", "clearance_plane", "spindle_speed",
            "tool_diameter", "tool_profile", "plunge_feedrate",
            "cut_feedrate",
        },
        "pinned": {
            "spindle_direction", "velocity_mode", "work_plane",
            "optimisation_mode", "tool_number", "max_crossover_distance",
            "custom_mop_header", "custom_mop_footer", "roughing_finishing",
            "final_depth_increment", "cut_ordering",
        },
        "excluded": set(),
        "inspection_hidden": {
            "optimisation_mode", "max_crossover_distance",
        },
    },
    "drill": {
        "class": DrillMop,
        "author": {
            "enabled", "target_depth", "depth_increment", "stock_surface",
            "roughing_clearance", "clearance_plane", "spindle_speed",
            "tool_diameter", "tool_profile", "plunge_feedrate",
            "cut_feedrate", "drilling_method", "peck_distance",
            "retract_height", "dwell", "hole_diameter", "drill_lead_out",
            "spiral_flat_base", "lead_out_length",
        },
        "pinned": {
            "spindle_direction", "velocity_mode", "work_plane",
            "optimisation_mode", "tool_number", "max_crossover_distance",
            "custom_mop_header", "custom_mop_footer",
        },
        "excluded": {"custom_script"},
        "inspection_hidden": {
            "optimisation_mode", "max_crossover_distance", "custom_script",
        },
    },
}


class McpMopParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = (
            Path(__file__).parents[1]
            / "cambam_builder" / "mcp_adapter" / "contract_v1.schema.json"
        )
        cls.definitions = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]

    def test_every_modeled_field_has_one_author_and_inspection_disposition(self):
        for family, contract in PARITY.items():
            with self.subTest(family=family):
                modeled = {
                    field.name for field in fields(contract["class"])
                } - IDENTITY_FIELDS
                author_sets = (
                    contract["author"], contract["pinned"], contract["excluded"]
                )
                self.assertEqual(modeled, set().union(*author_sets))
                self.assertEqual(sum(map(len, author_sets)), len(modeled))

                inspected = set(
                    self.definitions[f"{family.capitalize()}Parameters"]
                    ["properties"]
                )
                self.assertEqual(modeled, inspected | contract["inspection_hidden"])
                self.assertFalse(inspected & contract["inspection_hidden"])

    def test_executable_field_policies_cover_the_documented_sixty_slots(self):
        common = set(MOP_COMMON_FIELD_POLICIES)
        subtype_policies = {
            "profile": set(MOP_PROFILE_FIELD_POLICIES),
            "pocket": set(MOP_POCKET_FIELD_POLICIES),
            "engrave": set(MOP_ENGRAVE_FIELD_POLICIES),
            "drill": set(MOP_DRILL_FIELD_POLICIES),
        }
        self.assertEqual(18, len(common))
        self.assertEqual(
            {"profile": 18, "pocket": 12, "engrave": 3, "drill": 9},
            {family: len(policy) for family, policy in subtype_policies.items()},
        )
        self.assertEqual(60, len(common) + sum(map(len, subtype_policies.values())))
        for family, contract in PARITY.items():
            with self.subTest(family=family):
                modeled_parameters = (
                    contract["author"] | contract["pinned"] | contract["excluded"]
                ) - {"enabled"}
                self.assertEqual(common | subtype_policies[family], modeled_parameters)

    def test_author_classification_matches_closed_tool_inputs(self):
        aliases = {"profile": {"side": "profile_side"}}
        for family, contract in PARITY.items():
            with self.subTest(family=family):
                properties = set(
                    self.definitions[f"machining_add_{family}_input"]["properties"]
                ) - PROTOCOL_FIELDS
                for source, target in aliases.get(family, {}).items():
                    properties.remove(source)
                    properties.add(target)
                self.assertEqual(contract["author"], properties)

    def test_mcp_has_target_mutation_but_no_parameter_mutation(self):
        mop_edits = {
            name for name in DocumentService.EDIT_TOOLS
            if name.startswith("machining_")
        }
        self.assertEqual(
            {
                "machining_add_profile", "machining_add_pocket",
                "machining_add_engrave", "machining_add_drill",
                "machining_set_mop_targets", "machining_configure_part",
            },
            mop_edits,
        )

    def test_target_kind_matrix_matches_service_rules(self):
        samples = {
            "rect": Rect(),
            "circle": Circle(),
            "arc": Arc(),
            "open_pline": Pline(vertices=[(0, 0), (1, 0)], closed=False),
            "closed_pline": Pline(
                vertices=[(0, 0), (2, 0), (2, 2), (0, 2)], closed=True
            ),
            "points": Points(vertices=[(0, 0)]),
            "text": Text(),
            "region": Region(
                outer_curve=Pline(
                    vertices=[(0, 0), (3, 0), (3, 3), (0, 3)], closed=True
                )
            ),
        }
        expected = {
            "profile": {
                "rect", "circle", "open_pline", "closed_pline", "text", "region"
            },
            "pocket": {"rect", "circle", "closed_pline", "text", "region"},
            "engrave": {
                "rect", "circle", "arc", "open_pline", "closed_pline", "text"
            },
            "drill": {"circle", "points"},
        }
        for family, (_, allowed) in DocumentService.MOP_TARGET_RULES.items():
            with self.subTest(family=family):
                self.assertEqual(
                    expected[family],
                    {name for name, primitive in samples.items() if allowed(primitive)},
                )


if __name__ == "__main__":
    unittest.main()
