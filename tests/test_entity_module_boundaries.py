from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EntityModuleBoundaryTests(unittest.TestCase):
    def test_facade_reexports_canonical_owner_classes(self):
        from cambam_builder import cambam_entities as facade
        from cambam_builder import cad_entities, cam_entities, entity_core
        from cambam_builder.region import Region

        owner_exports = {
            entity_core: (
                "Vertex",
                "BoundingBox",
                "CamBamEntity",
                "Primitive",
            ),
            cad_entities: (
                "Layer",
                "Pline",
                "Circle",
                "Rect",
                "Arc",
                "Points",
                "Text",
            ),
            cam_entities: (
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
                    self.assertEqual(getattr(owner, name).__module__, owner.__name__)

        self.assertIs(facade.Region, Region)
        self.assertIs(Region.__mro__[1], entity_core.Primitive)
        self.assertEqual(Region.__module__, "cambam_builder.region")

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
                "assert facade.Region is Region",
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
