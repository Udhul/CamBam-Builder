"""Strict native input for the one accepted bounded variable-depth V request."""

import json
from pathlib import Path

from ... import CBProject
from ...cad_entities import Pline
from ...cam_core import tapered_vcarve
from ...cam_core.vcarve import PointedCone
from ...cam_entities import EngraveMop
from ...cambam_reader import read_cambam_bytes


SOURCE_FIELDS = {
    "tool_number": 3, "tool_diameter": 6, "tool_profile": "VCutter",
    "stock_surface": 0, "target_depth": 0, "depth_increment": 3,
    "roughing_clearance": 0, "clearance_plane": 5,
    "plunge_feedrate": 60, "cut_feedrate": 300,
    "spindle_direction": "CW", "spindle_speed": 12000,
    "work_plane": "XY", "velocity_mode": "ExactStop",
    "optimisation_mode": "None",
    "max_crossover_distance": 0.7,
    "roughing_finishing": "Roughing",
    "final_depth_increment": 0, "cut_ordering": "DepthFirst",
}
SOURCE_XML_TAGS = {
    "TargetDepth", "DepthIncrement", "StockSurface",
    "RoughingClearance", "ClearancePlane",
    "SpindleDirection", "SpindleSpeed", "VelocityMode", "WorkPlane",
    "OptimisationMode", "ToolDiameter", "ToolNumber", "ToolProfile",
    "PlungeFeedrate", "CutFeedrate", "MaxCrossoverDistance",
    "RoughingFinishing", "FinalDepthIncrement", "CutOrdering",
}


def synthetic_setup():
    """Values absent from the native Pline/MOP; no inherited tool is assumed."""
    return {
        "units": "mm", "frame": "tapered-groove-drawing",
        "tip_datum": "pointed_cone_tip", "cone_maximum_radius_mm": 3,
        "cone_conical_length_mm": 3, "cut_x_interval_mm": [2, 10],
        "safe_z_mm": 1, "approach_feed_mm_min": 120,
        "retract_feed_mm_min": 300, "setup_tip_xyz_mm": [-10, -10, 5],
    }


def synthetic_source():
    """Editable CamBam geometry and disabled, non-executable V intent."""
    project = CBProject("Native bounded variable-depth V input")
    layer = project.add_layer("Original finish geometry")
    guide = project.add_pline(layer, [(0, 2, -1), (12, 2, -2.5)],
                              identifier="tapered-target-spine")
    part = project.add_part("Tapered V part", stock_thickness=3,
                            stock_width=18, stock_height=8,
                            stock_offset=(-2, -2), stock_surface=0,
                            machining_origin=(0, 0), nesting_method="None")
    mop = project.add_engrave_mop(
        part, targets=[guide], name="SOURCE V groove intent (disabled)",
        identifier="source-v-intent", enabled=False, **SOURCE_FIELDS)
    if None in (layer, guide, part, mop):
        raise RuntimeError("could not create native V source")
    return project


def normalize(project, setup, *, allow_attachments=False):
    """Return only the detached bounded request; reject unsupported edits."""
    if setup != synthetic_setup():
        raise ValueError("unsupported or incomplete native V setup/tool components")
    parts = project.list_parts()
    shapes = project.list_primitives()
    mops = project.list_mops()
    if (len(parts) != 1 or
            (not allow_attachments and (len(shapes) != 1 or len(mops) != 1))):
        raise ValueError("native V requires one Part, target Pline and source Engrave")
    part = parts[0]
    guides = [shape for shape in shapes
              if shape.user_identifier == "tapered-target-spine"]
    source_mops = [item for item in mops
                   if item.user_identifier == "source-v-intent"]
    if len(guides) != 1 or len(source_mops) != 1:
        raise ValueError("native V finish spine or source Engrave missing")
    guide, mop = guides[0], source_mops[0]
    if (not part.enabled or part.nesting_method != "None" or
            tuple(part.machining_origin) != (0, 0) or
            part.stock_drawing_origin != (-2, -2, 0) or
            (part.stock_width, part.stock_height, part.stock_thickness) !=
            (18, 8, 3)):
        raise ValueError("unsupported native V Part stock or placement")
    if (type(guide) is not Pline or guide.user_identifier != "tapered-target-spine" or
            guide.closed or len(guide.vertices) != 2 or
            tuple(guide.get_absolute_coordinates_xyz()) !=
            ((0, 2, -1, 0), (12, 2, -2.5, 0))):
        raise ValueError("unsupported native V finish spine or transform")
    if (type(mop) is not EngraveMop or mop.enabled or
            mop not in project.get_mops_in_part(part) or
            project.get_mop_targets(mop) != [guide.internal_id]):
        raise ValueError("native V source Engrave must be disabled and target finish spine")
    if set(getattr(mop, "_xml_parameter_states", {})) != set(SOURCE_FIELDS):
        raise ValueError("unsupported native V Engrave parameter set")
    template = getattr(mop, "_xml_template", None)
    if (template is None or set(template.attrib) != {"Enabled"} or
            {child.tag for child in template} != SOURCE_XML_TAGS or
            len(template) != len(SOURCE_XML_TAGS)):
        raise ValueError("unsupported native V Engrave XML fields")
    for field, value in SOURCE_FIELDS.items():
        if getattr(mop, "_xml_parameter_states", {}).get(field) != "Value":
            raise ValueError(f"native V {field} is inherited or absent")
        if getattr(mop, field) != value:
            raise ValueError(f"unsupported native V {field} change")
    request = tapered_vcarve.TaperedRequest(
        target_spine=(0, 12, 2, 1, 2.5),
        stock_bounds=(-2, -2, 16, 6), stock_bottom=-3,
        tool=PointedCone(setup["cone_maximum_radius_mm"],
                         setup["cone_conical_length_mm"]),
        cut_interval=tuple(setup["cut_x_interval_mm"]),
        safe_z=setup["safe_z_mm"])
    if request != tapered_vcarve.standalone_request():
        raise ValueError("unsupported native V target or tool request")
    return request


def normalize_bytes(data, setup):
    return normalize(read_cambam_bytes(data, source_name="native V input"), setup)


def build_native_workflow(directory, *, source_path=None, setup=None):
    """Prepare separate original, preview and literal-motion files from XML."""
    from .variable_cone_engrave import build_engrave_candidate
    from .variable_cone_script import build_variable_carrier

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("native V output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    setup = synthetic_setup() if setup is None else setup
    source = directory / "source.cb"
    if source_path is None:
        synthetic_source().save(str(source))
    else:
        source.write_bytes(Path(source_path).read_bytes())
    data = source.read_bytes()
    request = normalize_bytes(data, setup)
    plan = tapered_vcarve.generate(request)
    (directory / "setup.json").write_text(json.dumps(setup, indent=2) + "\n",
                                          encoding="utf-8")
    preview = build_engrave_candidate(directory / "preview",
                                      native_source_bytes=data, native_setup=setup)
    explicit = build_variable_carrier(directory / "explicit",
                                      native_source_bytes=data, native_setup=setup)
    if (preview["plan_fingerprint"] != plan.fingerprint or
            explicit["plan_fingerprint"] != plan.fingerprint):
        raise ValueError("native V candidate request differs from standalone plan")
    result = {
        "status": "native_V_input_normalized",
        "request_equals_standalone": request == tapered_vcarve.standalone_request(),
        "plan_fingerprint": plan.fingerprint,
        "original": str(source),
        "preview": str(directory / "preview" / "V-variable-engrave.cb"),
        "explicit": str(directory / "explicit" / "V-variable.cb"),
        "preview_execution": "unaccepted_native_post",
        "explicit_execution": "pending_post_of_this_candidate",
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare bounded native V input")
    parser.add_argument("output", help="new output directory")
    parser.add_argument("source", nargs="?", help="existing native source.cb")
    parser.add_argument("--setup", help="explicit setup.json for existing source")
    args = parser.parse_args()
    setup = json.loads(Path(args.setup).read_text(encoding="utf-8")) if args.setup else None
    print(json.dumps(build_native_workflow(args.output, source_path=args.source,
                                           setup=setup), indent=2))
