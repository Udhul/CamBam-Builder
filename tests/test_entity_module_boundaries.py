from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EntityModuleBoundaryTests(unittest.TestCase):
    def test_facade_reexports_canonical_owner_classes(self):
        from cambam_builder import cambam_entities as facade
        from cambam_builder import cad_entities, cam_entities, entity_core
        from cambam_builder.native import cad, cam, core, region as native_region
        from cambam_builder.region import Region

        owner_exports = {
            core: (
                "Vertex",
                "BoundingBox",
                "CamBamEntity",
                "Primitive",
            ),
            cad: (
                "Layer",
                "Pline",
                "Circle",
                "Rect",
                "Arc",
                "Points",
                "Text",
            ),
            cam: (
                "Part",
                "Mop",
                "ProfileMop",
                "PocketMop",
                "EngraveMop",
                "DrillMop",
            ),
        }
        for owner, names in owner_exports.items():
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(facade, name), getattr(owner, name))
                    legacy = {
                        core: entity_core,
                        cad: cad_entities,
                        cam: cam_entities,
                    }[owner]
                    self.assertIs(getattr(legacy, name), getattr(owner, name))
                    self.assertEqual(getattr(owner, name).__module__, owner.__name__)

        self.assertIs(facade.Region, Region)
        self.assertIs(Region, native_region.Region)
        self.assertIs(Region.__mro__[1], core.Primitive)
        self.assertEqual(Region.__module__, "cambam_builder.native.region")

    def test_owner_and_facade_construction_use_canonical_types(self):
        from cambam_builder import cambam_entities as facade
        from cambam_builder import cad_entities, cam_entities, entity_core

        owner_vertex = entity_core.Vertex(1, 2, 3)
        facade_vertex = facade.Vertex(4, 5, 6)
        owner_pline = cad_entities.Pline(vertices=[owner_vertex, facade_vertex])
        facade_pline = facade.Pline(vertices=[(0, 0), (1, 0)])
        owner_mop = cam_entities.ProfileMop()
        facade_mop = facade.ProfileMop()

        self.assertIs(type(owner_vertex), type(facade_vertex))
        self.assertIs(type(owner_pline), type(facade_pline))
        self.assertIs(type(owner_mop), type(facade_mop))

    def test_clean_process_imports_succeed_in_varied_orders(self):
        modules = (
            "cambam_builder.cambam_entities",
            "cambam_builder.entity_core",
            "cambam_builder.cad_entities",
            "cambam_builder.region",
            "cambam_builder.cam_entities",
            "cambam_builder.cambam_project",
            "cambam_builder.cambam_reader",
            "cambam_builder.cambam_writer",
            "cambam_builder.native.core",
            "cambam_builder.native.cad",
            "cambam_builder.native.region",
            "cambam_builder.native.cam",
            "cambam_builder.native.project",
            "cambam_builder.native.reader",
            "cambam_builder.native.writer",
        )
        orders = [
            modules,
            tuple(reversed(modules)),
            (modules[3], modules[2], modules[1], modules[4], modules[0]),
            (modules[4], modules[1], modules[0], modules[3], modules[2]),
            (modules[1], modules[3], modules[0], modules[2], modules[4]),
        ]
        script_template = "\n".join(
            (
                "import importlib",
                "for name in {order!r}:",
                "    importlib.import_module(name)",
                "from cambam_builder import cambam_entities as facade",
                "from cambam_builder.entity_core import Primitive",
                "from cambam_builder.region import Region",
                "from cambam_builder.native.core import Primitive as NativePrimitive",
                "from cambam_builder.native.region import Region as NativeRegion",
                "from cambam_builder.native.project import CamBamProject as NativeProject",
                "from cambam_builder import CBProject",
                "assert facade.Region is Region",
                "assert Region is NativeRegion",
                "assert Primitive is NativePrimitive",
                "assert CBProject is NativeProject",
                "assert issubclass(Region, Primitive)",
            )
        )
        for order in orders:
            with self.subTest(order=order):
                result = subprocess.run(
                    [sys.executable, "-c", script_template.format(order=order)],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
