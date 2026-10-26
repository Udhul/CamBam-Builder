"""Native source identity and tessellation binding for constructive geometry."""

from dataclasses import replace
import json
import unittest
import uuid
import xml.etree.ElementTree as ET

from cambam_builder.cam_core.constructive import ConstructiveSource, SweptRidge
from cambam_builder.integrations.cambam.native_constructive import (
    create_document, read_document, regenerate,
)
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes


def source() -> ConstructiveSource:
    return ConstructiveSource("frieze-unit", 1.2, 0.8, 0.6, -0.24,
        (SweptRidge("wave", (0, 1.2), (0.4, 0.4), (0.22, 0.28),
                    (0.08, 0.10), wave_amplitude_mm=0.07),),
        repeat_count=2, provenance=("native constructive acceptance test",))


class NativeConstructiveTests(unittest.TestCase):
    def test_source_identity_and_surface_survive_native_save_load(self):
        original = source()
        document = create_document(original, x_samples=25, y_samples=17)
        loaded_project = read_cambam_bytes(document.xml_bytes)
        loaded_surface = loaded_project.get_primitive(document.surface_identifier)
        self.assertIsNotNone(loaded_surface.native_tag)
        self.assertEqual(loaded_project.list_mops(), [])
        surface_id = loaded_surface.internal_id
        saved = serialize_cambam_bytes(loaded_project)
        reopened = read_document(saved)
        self.assertEqual(reopened.source.canonical_json(), original.canonical_json())
        self.assertEqual(reopened.source.fingerprint, original.fingerprint)
        self.assertEqual(read_cambam_bytes(saved)
                         .get_primitive(document.surface_identifier).internal_id,
                         surface_id)
        self.assertEqual(reopened.check_document()["status"], "pass")
        self.assertLess(reopened.source_deviation_bound_mm,
                        document.source_deviation_bound_mm + 1e-14)

    def test_increased_sampling_reduces_declared_fidelity_error(self):
        coarse = create_document(source(), x_samples=9, y_samples=7)
        fine = create_document(source(), x_samples=33, y_samples=25)
        self.assertLess(fine.source_deviation_bound_mm,
                        coarse.source_deviation_bound_mm)
        self.assertEqual(coarse.source.fingerprint, fine.source.fingerprint)

    def test_source_modifiers_require_explicit_regeneration(self):
        original = source()
        document = create_document(original, x_samples=17, y_samples=13)
        edited = replace(original, repeat_count=3)
        with self.assertRaisesRegex(ValueError, "explicitly regenerate"):
            document.check_document(current_source=edited)
        fresh = regenerate(edited, x_samples=17, y_samples=13)
        self.assertEqual(fresh.source.fingerprint, edited.fingerprint)
        self.assertNotEqual(fresh.geometry_sha256, document.geometry_sha256)
        self.assertEqual(fresh.check_document()["status"], "pass")

    def test_edited_native_surface_vertices_are_rejected(self):
        document = create_document(source(), x_samples=9, y_samples=7)
        project = read_cambam_bytes(document.xml_bytes)
        surface = project.get_primitive(document.surface_identifier)
        surface.vertices[0].z += 0.01
        edited_bytes = serialize_cambam_bytes(project)
        with self.assertRaisesRegex(ValueError, "Surface geometry changed"):
            document.check_document(edited_bytes)

    def test_source_envelope_edits_and_missing_surface_reject(self):
        document = create_document(source(), x_samples=9, y_samples=7)
        project = read_cambam_bytes(document.xml_bytes)
        surface = project.get_primitive(document.surface_identifier)
        envelope = json.loads(surface.native_tag)
        envelope["source_fingerprint"] = "0" * 64
        surface.native_tag = json.dumps(envelope)
        with self.assertRaisesRegex(ValueError, "provenance"):
            read_document(serialize_cambam_bytes(project))

    def test_native_units_and_entity_identity_cannot_rebind_source(self):
        document = create_document(source(), x_samples=9, y_samples=7)
        project = read_cambam_bytes(document.xml_bytes)
        project.units = "Inches"
        with self.assertRaisesRegex(ValueError, "Millimeters"):
            document.check_document(serialize_cambam_bytes(project))
        project.units = None  # Installed native XML default is Millimeters.
        self.assertEqual(document.check_document(serialize_cambam_bytes(project))["status"], "pass")
        root = ET.fromstring(document.xml_bytes)
        tag = root.find("./layers/layer/objects/surface/Tag")
        data = json.loads(tag.text)
        data["internal_id"] = str(uuid.uuid4())
        tag.text = json.dumps(data)
        with self.assertRaisesRegex(ValueError, "identity"):
            read_document(ET.tostring(root))

    def test_reduced_fidelity_bound_cannot_hide_native_rounding(self):
        flat = ConstructiveSource("rounding", 1, 1, 1, -0.12345678949, ())
        document = create_document(flat, x_samples=2, y_samples=2)
        project = read_cambam_bytes(document.xml_bytes)
        surface = project.get_primitive(document.surface_identifier)
        self.assertGreater(abs(surface.vertices[0].z - flat.base_height_mm), 4e-10)
        envelope = json.loads(surface.native_tag)
        envelope["source_deviation_bound_mm"] = flat.sample_heightfield(
            x_samples=2, y_samples=2).deviation_bound_mm
        surface.native_tag = json.dumps(envelope)
        with self.assertRaisesRegex(ValueError, "deviation bound"):
            read_document(serialize_cambam_bytes(project))

        # The input may carry more decimal places than our own serializer.
        # Re-rounding those values during validation would hide an added error.
        root = ET.fromstring(document.xml_bytes)
        for vertex in root.iter("v"):
            vertex.text = vertex.text.replace('-0.123456789', '-0.12345678851')
        altered = ET.tostring(root)
        self.assertNotEqual(altered, document.xml_bytes)
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            document.check_document(altered)

        project = read_cambam_bytes(document.xml_bytes)
        project.remove_primitive(document.surface_identifier)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            read_document(serialize_cambam_bytes(project))


if __name__ == "__main__":
    unittest.main()
