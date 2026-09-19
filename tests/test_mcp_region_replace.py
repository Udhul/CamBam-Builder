"""Focused adapter regressions for atomic contour-to-Region replacement."""

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

if find_spec("mcp") is not None:
    import anyio
    from cambam_builder.mcp_adapter.schema import OUTPUTS
    from cambam_builder.mcp_adapter.service import DocumentService
else:  # pragma: no cover - base-library environment
    DocumentService = None
    OUTPUTS = {}


@unittest.skipIf(DocumentService is None, "Install .[mcp] for adapter checks")
class RegionReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_async(self, test):
        async def run():
            self.service = DocumentService(self.root)
            await test()
        anyio.run(run)

    def args(self, **kwargs):
        result = {"workspace_id": self.service.workspace.id, "request_id": str(uuid4())}
        result.update(kwargs)
        return result

    async def call(self, name, arguments):
        result = await self.service.call_tool(name, arguments)
        OUTPUTS[name].validate(result)
        return result

    async def create(self):
        result = await self.call("document_create", self.args(name="replace", units="mm"))
        self.assertTrue(result["ok"], result)
        return result["document"]

    async def rectangle(self, handle, revision, identifier, x, y, width, height, layer="Geometry"):
        result = await self.call("geometry_add_rectangle", self.args(
            document=handle, expected_revision=revision, identifier=identifier,
            layer=layer, x=x, y=y, width=width, height=height,
        ))
        self.assertTrue(result["ok"], result)
        return result["data"]["entity_id"]

    async def inspect(self, handle, revision):
        result = await self.call("document_inspect", {
            "workspace_id": self.service.workspace.id,
            "document": handle, "expected_revision": revision,
        })
        self.assertTrue(result["ok"], result)
        return result

    def test_two_rectangles_become_one_region_and_sources_disappear(self):
        async def test():
            handle = await self.create()
            outer = await self.rectangle(handle, 0, "outer", 0, 0, 20, 10)
            hole = await self.rectangle(handle, 1, "hole", 5, 2, 4, 3)
            result = await self.call("geometry_replace_with_region", self.args(
                document=handle, expected_revision=2, identifier="plate",
                outer_id=outer, hole_ids=[hole],
            ))
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["revision"], 3)
            self.assertEqual(result["data"]["layer"], "Geometry")
            self.assertEqual(result["data"]["removed_entity_ids"], [outer, hole])

            inspected = await self.inspect(handle, 3)
            primitives = [item for item in inspected["data"]["entities"]
                          if item["kind"] == "primitive"]
            self.assertEqual(len(primitives), 1)
            self.assertEqual(primitives[0]["id"], result["data"]["entity_id"])
            self.assertEqual(primitives[0]["identifier"], "plate")
            self.assertEqual(primitives[0]["geometry"]["kind"], "region")
            self.assertEqual(len(primitives[0]["geometry"]["hole_curves"]), 1)
        self.run_async(test)

    def test_invalid_topology_and_cross_layer_requests_are_atomic(self):
        async def test():
            handle = await self.create()
            outer = await self.rectangle(handle, 0, "outer", 0, 0, 10, 10)
            outside = await self.rectangle(handle, 1, "outside", 20, 20, 2, 2)
            invalid = await self.call("geometry_replace_with_region", self.args(
                document=handle, expected_revision=2, identifier="bad",
                outer_id=outer, hole_ids=[outside],
            ))
            self.assertFalse(invalid["ok"])
            self.assertEqual(invalid["error"]["code"], "INVALID_ARGUMENT")
            inspected = await self.inspect(handle, 2)
            self.assertEqual(inspected["data"]["summary"]["counts"]["primitives"], 2)

            duplicate = await self.call("geometry_replace_with_region", self.args(
                document=handle, expected_revision=2, identifier="duplicate",
                outer_id=outer, hole_ids=[outer],
            ))
            self.assertFalse(duplicate["ok"])
            self.assertEqual(duplicate["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual((await self.inspect(handle, 2))["data"]["summary"]
                             ["counts"]["primitives"], 2)

            other = await self.rectangle(handle, 2, "other", 2, 2, 2, 2, layer="Other")
            cross_layer = await self.call("geometry_replace_with_region", self.args(
                document=handle, expected_revision=3, identifier="bad-layer",
                outer_id=outer, hole_ids=[other],
            ))
            self.assertFalse(cross_layer["ok"])
            self.assertEqual(cross_layer["error"]["code"], "INVALID_ARGUMENT")
            inspected = await self.inspect(handle, 3)
            self.assertEqual(inspected["data"]["summary"]["counts"]["primitives"], 3)
        self.run_async(test)

    def test_profile_target_is_replaced_with_region_target(self):
        async def test():
            handle = await self.create()
            outer = await self.rectangle(handle, 0, "outer", 0, 0, 20, 10)
            hole = await self.rectangle(handle, 1, "hole", 5, 2, 4, 3)
            profile = await self.call("machining_add_profile", self.args(
                document=handle, expected_revision=2, identifier="profile",
                part="Part", targets=[outer], side="Outside", target_depth=-1,
                depth_increment=0.5, tool_diameter=3, cut_feedrate=300,
                plunge_feedrate=100, spindle_speed=12000,
                clearance_plane=3,
            ))
            self.assertTrue(profile["ok"], profile)
            result = await self.call("geometry_replace_with_region", self.args(
                document=handle, expected_revision=3, identifier="plate",
                outer_id=outer, hole_ids=[hole],
            ))
            self.assertTrue(result["ok"], result)
            records = (await self.inspect(handle, 4))["data"]["entities"]
            mop = next(item for item in records if item["kind"] == "mop")
            self.assertEqual(mop["targets"], [result["data"]["entity_id"]])
        self.run_async(test)

    def test_reflected_circle_hole_uses_canonical_minor_quarter_arcs(self):
        async def test():
            handle = await self.create()
            outer = await self.rectangle(handle, 0, "outer", 0, 0, 20, 20)
            circle = await self.call("geometry_add_circle", self.args(
                document=handle, expected_revision=1, identifier="hole",
                layer="Geometry", x=10, y=10, diameter=4,
            ))
            self.assertTrue(circle["ok"], circle)
            circle_id = circle["data"]["entity_id"]
            reflected = await self.call("geometry_mirror", self.args(
                document=handle, expected_revision=2, entity_id=circle_id,
                axis="x", position=10,
            ))
            self.assertTrue(reflected["ok"], reflected)
            result = await self.call("geometry_replace_with_region", self.args(
                document=handle, expected_revision=3, identifier="plate",
                outer_id=outer, hole_ids=[circle_id],
            ))
            self.assertTrue(result["ok"], result)
            records = (await self.inspect(handle, 4))["data"]["entities"]
            region = next(item for item in records if item["kind"] == "primitive")
            bulges = region["geometry"]["hole_curves"][0]["bulges"]
            self.assertEqual(len(bulges), 4)
            self.assertTrue(all(0 < bulge < 1 for bulge in bulges))
        self.run_async(test)

    def test_existing_region_update_preserves_identity_layer_and_pocket_target(self):
        async def test():
            handle = await self.create()
            region = await self.call("geometry_add_region", self.args(
                document=handle, expected_revision=0, identifier="plate",
                layer="Geometry",
                outer={"points": [
                    {"x": 0, "y": 0}, {"x": 20, "y": 0},
                    {"x": 20, "y": 10}, {"x": 0, "y": 10},
                ]},
            ))
            self.assertTrue(region["ok"], region)
            region_id = region["data"]["entity_id"]
            translated = await self.call("geometry_translate", self.args(
                document=handle, expected_revision=1, entity_id=region_id,
                dx=100, dy=50,
            ))
            self.assertTrue(translated["ok"], translated)
            pocket = await self.call("machining_add_pocket", self.args(
                document=handle, expected_revision=2, identifier="pocket",
                part="Part", targets=[region_id], target_depth=-2,
                depth_increment=1, tool_diameter=3, cut_feedrate=300,
                plunge_feedrate=100, spindle_speed=12000,
                clearance_plane=3,
            ))
            self.assertTrue(pocket["ok"], pocket)

            updated = await self.call("geometry_update_region", self.args(
                document=handle, expected_revision=3, entity_id=region_id,
                outer={"points": [
                    {"x": 2, "y": 1}, {"x": 32, "y": 1},
                    {"x": 32, "y": 16}, {"x": 2, "y": 16},
                ]},
                holes=[{"points": [
                    {"x": 10, "y": 5}, {"x": 14, "y": 5},
                    {"x": 14, "y": 9}, {"x": 10, "y": 9},
                ]}],
            ))
            self.assertTrue(updated["ok"], updated)
            self.assertEqual(updated["revision"], 4)
            self.assertEqual(updated["data"]["entity_id"], region_id)
            self.assertEqual(updated["data"]["layer"], "Geometry")
            self.assertEqual(updated["data"]["geometry"]["bounds"], [2.0, 1.0, 32.0, 16.0])
            self.assertEqual(len(updated["data"]["geometry"]["hole_curves"]), 1)

            records = (await self.inspect(handle, 4))["data"]["entities"]
            primitives = [record for record in records if record["kind"] == "primitive"]
            self.assertEqual(len(primitives), 1)
            self.assertEqual(primitives[0]["id"], region_id)
            mop = next(record for record in records if record["kind"] == "mop")
            self.assertEqual(mop["targets"], [region_id])

            exported = await self.call("document_export", {
                "workspace_id": self.service.workspace.id,
                "document": handle, "expected_revision": 4,
                "suggested_filename": "updated.cb",
            })
            self.assertTrue(exported["ok"], exported)
            reimported = await self.call("document_import", self.args(
                units="mm", source_name="updated.cb",
                content=exported["data"]["content"],
                expected_sha256=exported["data"]["sha256"],
            ))
            self.assertTrue(reimported["ok"], reimported)
            reopened = (await self.inspect(reimported["document"], 0))["data"]["entities"]
            reopened_region = next(record for record in reopened
                                   if record.get("id") == region_id)
            self.assertEqual(reopened_region["geometry"]["bounds"], [2.0, 1.0, 32.0, 16.0])
            reopened_mop = next(record for record in reopened if record["kind"] == "mop")
            self.assertEqual(reopened_mop["targets"], [region_id])

        self.run_async(test)

    def test_existing_region_update_rejects_invalid_topology_atomically(self):
        async def test():
            handle = await self.create()
            region = await self.call("geometry_add_region", self.args(
                document=handle, expected_revision=0, identifier="plate",
                layer="Geometry",
                outer={"points": [
                    {"x": 0, "y": 0}, {"x": 20, "y": 0},
                    {"x": 20, "y": 10}, {"x": 0, "y": 10},
                ]},
            ))
            region_id = region["data"]["entity_id"]
            invalid = await self.call("geometry_update_region", self.args(
                document=handle, expected_revision=1, entity_id=region_id,
                outer={"points": [
                    {"x": 0, "y": 0}, {"x": 20, "y": 0},
                    {"x": 20, "y": 10}, {"x": 0, "y": 10},
                ]},
                holes=[{"points": [
                    {"x": 30, "y": 30}, {"x": 32, "y": 30},
                    {"x": 32, "y": 32}, {"x": 30, "y": 32},
                ]}],
            ))
            self.assertFalse(invalid["ok"], invalid)
            self.assertEqual(invalid["error"]["code"], "INVALID_ARGUMENT")
            self.assertEqual(invalid["revision"], 1)
            records = (await self.inspect(handle, 1))["data"]["entities"]
            unchanged = next(record for record in records if record.get("id") == region_id)
            self.assertEqual(unchanged["geometry"]["bounds"], [0.0, 0.0, 20.0, 10.0])
            self.assertEqual(unchanged["geometry"]["hole_curves"], [])

        self.run_async(test)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
