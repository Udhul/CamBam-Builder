"""Source-bound editable BO01 authoring and complete XML fidelity regressions."""

from dataclasses import replace
import hashlib
import unittest
import xml.etree.ElementTree as ET

from cambam_builder import CBProject
from cambam_builder.native.cad import Pline
from cambam_builder.native.core import Vertex
from cambam_builder.native.reader import read_cambam_bytes
from cambam_builder.native.writer import serialize_cambam_bytes

try:
    from shapely.geometry import LineString
    from cambam_builder.cam_core import ordered_job as oj, replay, v_region as v
    from cambam_builder.cam_core.occupancy import Box, OccupancySetup, ToolBand, ToolBody
    from cambam_builder.cam_extensions import bundle_search as b
    from cambam_builder.integrations import ordered_output
    from cambam_builder.integrations.cambam import bundle_document as native
    HAS_PLANAR = True
except ModuleNotFoundError as exc:
    if exc.name != "shapely":
        raise
    HAS_PLANAR = False


def _source(kind="rect", *, disabled=False):
    project = CBProject("bound source")
    layer = project.add_layer("Source CAD")
    shell = [(0, 0), (10, 0), (10, 8), (0, 8)]
    if kind == "rect":
        primitive = project.add_rect(layer, identifier="design", width=10, height=8)
    elif kind == "pline":
        primitive = project.add_pline(layer, shell, closed=True, identifier="design")
    else:
        outer = Pline(vertices=shell, closed=True)
        hole = Pline(vertices=[(4, 3), (6, 3), (6, 5), (4, 5)], closed=True)
        primitive = project.add_region(layer, outer, [hole], identifier="design")
        project.rotate_primitive_deg(primitive, 90)
        project.translate_primitive(primitive, 9, 2)
    primitive.native_tag = "operator source note"
    part = project.add_part("Stock", stock_offset=(-2, -2), stock_width=20,
                            stock_height=20, stock_thickness=2)
    if disabled:
        project.add_profile_mop(part, [primitive], identifier="old disabled",
                                enabled=False, target_depth=-1, depth_increment=.5)
    return serialize_cambam_bytes(project)


def _bundle(design, kinds=("v",), *, tools=None):
    """Small genuine decoded-stock, occupancy and axial audited stages."""
    target = design.target
    safe_z = 3
    start = (-1, -1, safe_z)
    stages = []
    bodies = {}
    for index, kind in enumerate(kinds):
        tool_id = (tools or tuple(f"T{i + 1}" for i in range(len(kinds))))[index]
        y = target.safe.bounds[1] + 1.25 + index * .8
        center = target.safe.buffer(-1).intersection(LineString((
            (target.safe.bounds[0] - 1, y), (target.safe.bounds[2] + 1, y))))
        segments = v._segments(center)
        line = max(segments, key=lambda item: item.length)
        a, z = tuple(line.coords[0]), tuple(line.coords[-1])
        # Keep a generous wall margin, including at taper sections.
        points = ((a[0] + .1, a[1], target.cap_depth),
                  (z[0] - .1, z[1], target.cap_depth))
        transition = None if not stages else oj.Transition(
            "operator", "split", tool_id, start, "declared-resume")
        stage_id = f"stage-{index + 1}"
        if kind == "v":
            tool = v.VProfile("pointed", 90, 0, 2, 2)
            paths = (v.VPath("fill", points),)
            plan = v.VPlan(target, tool, paths, v._motions(paths, safe_z),
                           safe_z, .02, 1, "partial", "synthetic safe line")
            moves = tuple(oj.JobMove(move.role, move.start, move.end,
                0 if move.role == "rapid" else 60 if move.role == "entry" else 200)
                for move in v.complete_motion(plan, start))
            stage = oj.Stage(stage_id, tool_id, moves, 12000, v_plan=plan,
                source_revision=plan.fingerprint, transition=transition,
                axial_limits=oj.AxialLimits(.5, .5))
            radius = tool.radius(tool.cutting_length)
        else:
            tool = replay.ToolProfile(tool_id, "cylinder", .2, 2)
            shape = target.safe
            operation = replay.Operation(stage_id, tool, replay.Target(
                "original-design", shape.bounds, target.cap_depth,
                region_shell=tuple(shape.exterior.coords)[:-1],
                region_holes=tuple(tuple(ring.coords)[:-1] for ring in shape.interiors)))
            high = (points[0][0], points[0][1], safe_z)
            low = (*high[:2], -target.cap_depth)
            end = (points[-1][0], points[-1][1], -target.cap_depth)
            back = (*end[:2], safe_z)
            moves = (oj.JobMove("rapid", start, high),
                oj.JobMove("entry", high, low, 60),
                oj.JobMove("cut", low, end, 200),
                oj.JobMove("retract", end, back, 200),
                oj.JobMove("rapid", back, start))
            stage = oj.Stage(stage_id, tool_id, moves, 12000, operation=operation,
                source_revision="synthetic-cylinder", transition=transition,
                axial_limits=oj.AxialLimits(.5, .5))
            radius = tool.radius
        stages.append(stage)
        bodies[tool_id] = ToolBody(tool_id, (
            ToolBand("cutter", 0, 2, radius),
            ToolBand("shank", 2, 3, 2), ToolBand("holder", 3, 5, 3)))
    part = read_cambam_bytes(design.source_bytes).get_part(design.part_identifier)
    x, y, top = part.stock_drawing_origin
    setup = OccupancySetup(target.frame, Box("stock", (
        x, y, top - part.stock_thickness, x + part.stock_width,
        y + part.stock_height, top)), (), tuple(bodies.values()))
    job = oj.Job(design.source_sha256, tuple(stages), start,
                  program_frame=target.frame, occupancy_setup=setup)
    files, _ = ordered_output.emit(job, "uccnc", coordinate_decimals=6)
    return b.Bundle(job, files, "uccnc", (0, 1), (0, 1), target.cap_depth,
        None, b.Cost(1, 1, 1, len(bodies) - 1, len(stages) - 1),
        tuple(hashlib.sha256(data).hexdigest() for data in files),
        design_fingerprint=target.fingerprint)


@unittest.skipUnless(HAS_PLANAR, "optional planar backend is absent")
class NativeBundleDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _source(disabled=True)
        cls.design = native.bind_design(cls.source, "design", "Stock",
            cap_depth_mm=.5, design_angle_degrees=60)
        cls.bundle = _bundle(cls.design, ("cylinder", "v"))
        cls.document = native.author(cls.design, cls.bundle)

    def _mutate(self, callback):
        root = ET.fromstring(self.document.xml_bytes)
        callback(root)
        return ET.tostring(root)

    def test_rect_binding_keeps_angle_and_source_independent_of_cutter(self):
        self.assertEqual(self.design.target.design_angle_degrees, 60)
        self.assertEqual(self.bundle.job.stages[-1].v_plan.tool.angle_degrees, 90)
        self.assertEqual(self.design.target.source_id, hashlib.sha256(self.source).hexdigest())
        self.design.check_source(self.source)
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            self.design.check_source(self.source + b"\n")
        forged = replace(self.design, target=replace(self.design.target,
            safe=self.design.target.safe.buffer(-.1)))
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            forged.check_source(self.source)

    def test_editable_stages_settings_world_paths_and_source_xml_preserved(self):
        document = self.document
        self.assertEqual(document.check_document()["status"], "pass")
        self.assertEqual(document.check_document()["actual_post"], "not_evaluated")
        root = ET.fromstring(document.xml_bytes)
        original = ET.fromstring(self.source)
        source_layer = original.find("./layers/layer")
        authored_source_layer = root.find("./layers/layer")
        source_layer.tail = authored_source_layer.tail = None
        self.assertEqual(ET.tostring(authored_source_layer), ET.tostring(source_layer))
        self.assertEqual(ET.tostring(root.find("./parts/part/Stock")),
                         ET.tostring(original.find("./parts/part/Stock")))
        self.assertEqual(root.findtext("./MachiningOptions/FastPlungeHeight"), "0")
        self.assertEqual(root.find("./MachiningOptions/FastPlungeHeight").attrib, {})
        project = read_cambam_bytes(document.xml_bytes)
        mops = [mop for mop in project.list_mops() if mop.enabled]
        self.assertEqual(tuple(mop.user_identifier for mop in mops), document.stage_mop_identifiers)
        self.assertFalse(project.get_mop("old disabled").enabled)
        self.assertEqual([mop.tool_profile for mop in mops], ["EndMill", "VCutter"])
        for stage, mop in zip(self.bundle.job.stages, mops):
            self.assertEqual((mop.target_depth, mop.stock_surface, mop.roughing_clearance,
                mop.max_crossover_distance), (0, 0, 0, 0))
            self.assertEqual((mop.clearance_plane, mop.depth_increment,
                mop.plunge_feedrate, mop.cut_feedrate), (3, .5, 60, 200))
            self.assertEqual((mop.velocity_mode, mop.optimisation_mode), ("ExactStop", "None"))
            targets = [project.get_primitive(uid) for uid in project.get_mop_targets(mop)]
            self.assertTrue(all(type(path) is Pline and not path.closed for path in targets))
            self.assertTrue(all(project.get_parent_of_primitive(path) is None for path in targets))
            self.assertEqual(tuple(point[:3] for point in targets[0].get_absolute_coordinates_xyz()),
                (next(move.start for move in stage.motions if move.role == "cut"),
                 next(move.end for move in stage.motions if move.role == "cut")))
            self.assertIn(document.design.source_sha256, targets[0].description)
            self.assertIn(self.bundle.job.fingerprint, mop.native_tag)
        self.assertEqual(mops[0].custom_mop_footer, "G0 Z3\nM5")
        self.assertEqual(mops[1].custom_mop_footer, "G0 Z3")
        self.assertNotIn(b"CustomScript", document.xml_bytes)

    def test_two_library_save_reopen_cycles_and_id_renumbering(self):
        data = self.document.xml_bytes
        for _ in range(2):
            data = serialize_cambam_bytes(read_cambam_bytes(data))
            self.assertEqual(self.document.check_document(data)["status"], "pass")

    def test_transformed_holed_region_and_closed_pline(self):
        for kind in ("region", "pline"):
            with self.subTest(kind=kind):
                source = _source(kind)
                design = native.bind_design(source, "design", "Stock",
                    cap_depth_mm=.5, design_angle_degrees=60)
                document = native.author(design, _bundle(design))
                document.check_document()
                data = document.xml_bytes
                for _ in range(2):
                    data = serialize_cambam_bytes(read_cambam_bytes(data))
                    document.check_document(data)
                self.assertEqual(len(design.target.safe.interiors), int(kind == "region"))

    def test_document_rejects_geometry_settings_extra_xml_and_order_edits(self):
        mutations = {
            "target depth": lambda root: setattr(root.find(".//engrave/TargetDepth"), "text", "-1"),
            "clearance": lambda root: setattr(root.find(".//engrave/ClearancePlane"), "text", "2"),
            "feed": lambda root: setattr(root.find(".//engrave/CutFeedrate"), "text", "201"),
            "fast plunge": lambda root: setattr(root.find("./MachiningOptions/FastPlungeHeight"), "text", "1"),
            "units": lambda root: root.set("units", "Inches"),
            "unknown MOP": lambda root: ET.SubElement(root.find(".//engrave"), "UnknownBehavior", state="Value"),
            "unknown root": lambda root: ET.SubElement(root, "UnknownBehavior"),
            "source attr": lambda root: root.find(".//rect").set("unknown", "value"),
            "source geometry": lambda root: root.find(".//rect").set("w", "9"),
            "generated geometry": lambda root: setattr(root.find("./layers/layer[2]/objects/pline/pts/p"), "text", "2,2,-.4"),
            "extra enabled MOP": lambda root: ET.SubElement(root.find("./parts/part/machineops"), "engrave", Enabled="true"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.document.check_document(self._mutate(mutate))
        root = ET.fromstring(self.document.xml_bytes)
        container = root.find("./parts/part/machineops")
        stages = container.findall("engrave")
        for mop in stages:
            container.remove(mop)
        container.extend(reversed(stages))
        with self.assertRaisesRegex(ValueError, "fidelity changed"):
            self.document.check_document(ET.tostring(root))
        forged = replace(self.document, xml_bytes=self._mutate(mutations["feed"]))
        with self.assertRaises(ValueError):
            forged.check_document()

    def test_bundle_bytes_source_frame_target_and_evidence_cannot_be_rebound(self):
        bundle = self.bundle
        cases = [replace(bundle, files=(bundle.files[0] + b"\n", *bundle.files[1:])),
            replace(bundle, design_fingerprint=""),
            replace(bundle, design_fingerprint="changed"),
            replace(bundle, program_sha256=("0" * 64,) * len(bundle.files)),
            replace(bundle, job=replace(bundle.job, source_fingerprint="changed")),
            replace(bundle, job=replace(bundle.job, translation_xyz_mm=(1, 0, 0))),
            replace(bundle, job=replace(bundle.job, occupancy_setup=None))]
        stage = bundle.job.stages[-1]
        changed_target = replace(stage.v_plan.target, cap_depth=.4)
        changed = replace(stage, v_plan=replace(stage.v_plan, target=changed_target))
        cases.append(replace(bundle, job=replace(bundle.job,
            stages=(*bundle.job.stages[:-1], changed))))
        no_axial = replace(bundle.job.stages[0], axial_limits=None)
        cases.append(replace(bundle, job=replace(bundle.job,
            stages=(no_axial, *bundle.job.stages[1:]))))
        for candidate in cases:
            with self.subTest(candidate=candidate.job.source_fingerprint), self.assertRaises(ValueError):
                native.author(self.design, candidate)

    def test_cylinder_only_requires_fingerprint_and_bound_taper_safety(self):
        bundle = _bundle(self.design, ("cylinder",))
        native.author(self.design, bundle).check_document()
        with self.assertRaisesRegex(ValueError, "design fingerprint"):
            native.author(self.design, replace(bundle, design_fingerprint=""))
        # Same exact source and planar shell do not prove a much steeper taper.
        steeper = native.bind_design(self.source, "design", "Stock",
            cap_depth_mm=.5, design_angle_degrees=150)
        with self.assertRaisesRegex(ValueError, "capped finish target"):
            native.author(steeper, replace(bundle,
                design_fingerprint=steeper.target.fingerprint))

    def test_contiguous_cuts_form_paths_and_feed_changes_are_rejected(self):
        bundle = _bundle(self.design)
        stage = bundle.job.stages[0]
        a, z = stage.v_plan.paths[0].points
        middle = tuple((x + y) / 2 for x, y in zip(a, z))
        shifted = tuple((x, y + 1, depth) for x, y, depth in (a, z))
        paths = (v.VPath("fill", (a, middle, z)), v.VPath("fill", shifted))
        plan = replace(stage.v_plan, paths=paths, motions=v._motions(paths, 3))
        moves = tuple(oj.JobMove(move.role, move.start, move.end,
            0 if move.role == "rapid" else 60 if move.role == "entry" else 200)
            for move in v.complete_motion(plan, bundle.job.initial_tip))
        stage = replace(stage, v_plan=plan, source_revision=plan.fingerprint,
                        motions=moves)
        job = replace(bundle.job, stages=(stage,))
        files, _ = ordered_output.emit(job, "uccnc", coordinate_decimals=6)
        bundle = replace(bundle, job=job, files=files,
            program_sha256=tuple(hashlib.sha256(data).hexdigest() for data in files))
        document = native.author(self.design, bundle)
        project = read_cambam_bytes(document.xml_bytes)
        mop = project.get_mop(document.stage_mop_identifiers[0])
        targets = [project.get_primitive(uid) for uid in project.get_mop_targets(mop)]
        self.assertEqual(sorted(len(path.vertices) for path in targets), [2, 3])
        root = ET.fromstring(document.xml_bytes)
        references = root.find(".//engrave/primitive")
        references[:] = list(reversed(references))
        document.check_document(ET.tostring(root))
        for role in ("cut", "entry"):
            changed = list(moves)
            index = next(i for i, move in enumerate(changed) if move.role == role)
            changed[index] = replace(changed[index], feed=changed[index].feed + 1)
            bad = replace(stage, motions=tuple(changed))
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "uniform"):
                native.author(self.design, replace(bundle,
                    job=replace(job, stages=(bad,))))

    def test_source_rejects_units_open_curves_z_stock_nesting_and_enabled_mops(self):
        def bind(data):
            return native.bind_design(data, "design", "Stock",
                cap_depth_mm=.5, design_angle_degrees=60)
        for field, value in (("units", None), ("units", "Inches")):
            project = read_cambam_bytes(self.source)
            setattr(project, field, value)
            with self.assertRaises(ValueError):
                bind(serialize_cambam_bytes(project))
        for attribute, value in (("nesting_method", "Grid"),
                ("machining_origin", (1, 0)), ("stock_surface", 1),
                ("stock_thickness", .2), ("stock_width", 1),
                ("stock_present", False), ("enabled", False)):
            project = read_cambam_bytes(self.source)
            setattr(project.get_part("Stock"), attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                bind(serialize_cambam_bytes(project))
        project = read_cambam_bytes(self.source)
        project.get_mop("old disabled").enabled = True
        with self.assertRaises(ValueError):
            bind(serialize_cambam_bytes(project))
        project = read_cambam_bytes(self.source)
        other = project.add_part("Other", enabled=False)
        project.add_engrave_mop(other, [project.get_primitive("design")], enabled=True)
        with self.assertRaisesRegex(ValueError, "disabled pre-existing"):
            bind(serialize_cambam_bytes(project))
        for closed, bulge, elevation in ((False, 0, 0), (True, .1, 0), (True, 0, .1)):
            project = CBProject("bad")
            layer = project.add_layer("cad")
            project.add_pline(layer, [Vertex(0, 0, elevation, bulge=bulge), (10, 0), (10, 8), (0, 8)],
                closed=closed, identifier="design")
            project.add_part("Stock", stock_width=10, stock_height=8, stock_thickness=2)
            with self.subTest(closed=closed, bulge=bulge, z=elevation), self.assertRaises(ValueError):
                bind(serialize_cambam_bytes(project))
        project = CBProject("self intersection")
        layer = project.add_layer("CAD")
        project.add_pline(layer, [(0, 0), (10, 8), (10, 0), (0, 8)],
            closed=True, identifier="design")
        project.add_part("Stock", stock_width=10, stock_height=8, stock_thickness=2)
        with self.assertRaisesRegex(ValueError, "invalid V Region"):
            bind(serialize_cambam_bytes(project))

    def test_same_tool_stages_keep_spindle_running(self):
        bundle = _bundle(self.design, ("v", "v"), tools=("T1", "T1"))
        document = native.author(self.design, bundle, name="Same tool")
        project = read_cambam_bytes(document.xml_bytes)
        self.assertTrue(all(project.get_mop(identifier).custom_mop_footer == "G0 Z3"
                            for identifier in document.stage_mop_identifiers))
        document.check_document()

    def test_selected_stock_requires_explicit_unambiguous_native_coordinates(self):
        for path, size in (("Stock/PMin", 3), ("Stock/PMax", 3), ("MachiningOrigin", 2)):
            for mutation in ("missing", "duplicate", "empty", "malformed", "short",
                             "long", "nan", "infinite", "underscores", "nested", "attribute"):
                root = ET.fromstring(self.source)
                part = root.find("./parts/part")
                node = part.find(path)
                parent = part if "/" not in path else part.find(path.rsplit("/", 1)[0])
                if mutation == "missing":
                    parent.remove(node)
                elif mutation == "duplicate":
                    parent.append(ET.fromstring(ET.tostring(node)))
                elif mutation == "nested":
                    ET.SubElement(node, "value").text = "0"
                elif mutation == "attribute":
                    node.set("state", "Default")
                else:
                    node.text = {
                        "empty": "", "malformed": "not-a-point", "short": "0",
                        "long": ",".join("0" for _ in range(size + 1)),
                        "nan": ",".join(["NaN"] + ["0"] * (size - 1)),
                        "infinite": ",".join(["1e999"] + ["0"] * (size - 1)),
                        "underscores": ",".join(["0_0"] + ["0"] * (size - 1)),
                    }[mutation]
                with self.subTest(path=path, mutation=mutation), self.assertRaises(ValueError):
                    native.bind_design(ET.tostring(root), "design", "Stock",
                        cap_depth_mm=.5, design_angle_degrees=60)
        for mutation in ("duplicate Stock", "unknown Stock geometry", "reversed bounds",
                         "duplicate parts", "duplicate layers", "duplicate machineops"):
            root = ET.fromstring(self.source)
            part = root.find("./parts/part")
            stock = part.find("Stock")
            if mutation == "duplicate Stock":
                part.append(ET.fromstring(ET.tostring(stock)))
            elif mutation == "unknown Stock geometry":
                ET.SubElement(stock, "Offset").text = "1,0,0"
            elif mutation == "reversed bounds":
                stock.find("PMin").text, stock.find("PMax").text = (
                    stock.findtext("PMax"), stock.findtext("PMin"))
            elif mutation == "duplicate machineops":
                ET.SubElement(part, "machineops")
            else:
                container = root.find("parts" if mutation == "duplicate parts" else "layers")
                root.append(ET.fromstring(ET.tostring(container)))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.bind_design(ET.tostring(root), "design", "Stock",
                    cap_depth_mm=.5, design_angle_degrees=60)

    def test_selected_geometry_and_matrix_records_do_not_use_silent_fallbacks(self):
        for kind in ("rect", "pline", "region"):
            source = _source(kind)
            for mutation in ("missing matrix value", "duplicate matrix", "bad matrix",
                             "bad geometry", "ambiguous geometry"):
                root = ET.fromstring(source)
                shape = root.find("./layers/layer/objects/*")
                matrix = shape.find("mat")
                if mutation == "missing matrix value":
                    matrix.attrib.clear()
                elif mutation == "duplicate matrix":
                    shape.append(ET.fromstring(ET.tostring(matrix)))
                elif mutation == "bad matrix":
                    matrix.set("m", "not-a-matrix")
                elif kind == "rect":
                    if mutation == "bad geometry":
                        shape.set("w", "not-a-number")
                    else:
                        del shape.attrib["w"]
                else:
                    points = shape.find("pts" if kind == "pline" else "./OuterCurve/pts")
                    if mutation == "bad geometry":
                        points.find("p").text = "NaN,0,0"
                    else:
                        parent = shape if kind == "pline" else shape.find("OuterCurve")
                        parent.append(ET.fromstring(ET.tostring(points)))
                with self.subTest(kind=kind, mutation=mutation), self.assertRaises(ValueError):
                    native.bind_design(ET.tostring(root), "design", "Stock",
                        cap_depth_mm=.5, design_angle_degrees=60)
        # These are established native identity defaults, unlike a malformed
        # present matrix or missing explicit stock/origin/dimensions.
        root = ET.fromstring(_source())
        shape = root.find("./layers/layer/objects/rect")
        del shape.attrib["p"]
        shape.remove(shape.find("mat"))
        design = native.bind_design(ET.tostring(root), "design", "Stock",
            cap_depth_mm=.5, design_angle_degrees=60)
        self.assertTrue(design.target.safe.equals(self.design.target.safe))

    def test_parent_matrix_schema_is_checked_without_discarding_world_placement(self):
        project = read_cambam_bytes(self.source)
        layer = project.list_layers()[0]
        parent = project.add_rect(layer, identifier="anchor")
        project.link_primitive_parent(project.get_primitive("design"), parent)
        project.translate_primitive(parent, 1, 0)
        source = serialize_cambam_bytes(project)
        design = native.bind_design(source, "design", "Stock",
            cap_depth_mm=.5, design_angle_degrees=60)
        self.assertEqual(design.target.safe.bounds, (1, 0, 11, 8))
        root = ET.fromstring(source)
        for element in root.findall("./layers/layer/objects/*"):
            if '"user_id":"anchor"' in element.findtext("Tag", ""):
                element.find("mat").attrib.clear()
        with self.assertRaisesRegex(ValueError, "matrix record"):
            native.bind_design(ET.tostring(root), "design", "Stock",
                cap_depth_mm=.5, design_angle_degrees=60)

    def test_occupancy_stock_must_match_entire_native_part_including_extra_material(self):
        setup = self.bundle.job.occupancy_setup
        self.assertEqual(setup.stock.bounds, (-2, -2, -2, 18, 18, 0))
        self.document.check_document()
        for bounds in ((0, 0, -.5, 10, 8, 0), (-1.9, -2, -2, 18.1, 18, 0),
                       (-2, -2, -1, 18, 18, 0), (-3, -3, -3, 19, 19, 0)):
            changed_setup = replace(setup, stock=Box("incorrect stock", bounds))
            bundle = replace(self.bundle, job=replace(self.bundle.job,
                occupancy_setup=changed_setup))
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "complete source Part stock"):
                native.author(self.design, bundle)


if __name__ == "__main__":
    unittest.main()
