"""Bind source CAD to finite BO01 search results and editable CamBam jobs.

Run from a checkout with the optional planar extra installed:
    python demos/native_search_workflow.py output/native-search-example

The search bounds and estimated costs are planning evidence. CamBam must still
generate and pass the actual native post audit before execution is considered.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cambam_builder import CBProject
from cambam_builder.cam_core import ordered_job as ordered, replay, v_region
from cambam_builder.cam_core.occupancy import Box, OccupancySetup, ToolBand, ToolBody
from cambam_builder.cam_extensions import bundle_search
from cambam_builder.integrations.cambam import bundle_document
from cambam_builder.native.manufacturing import (
    ComponentPlacement,
    SetupFrame,
    SetupStock,
    compose_setup,
)
from cambam_builder.native.reader import read_cambam_bytes, read_cambam_file


def _outline(project: CBProject, layer: str, identifier: str,
             points: tuple[tuple[float, float], ...], *, parent: str | None = None):
    return project.add_pline(layer, points, closed=True, identifier=identifier,
                             parent=parent)


def _mixed_region_source() -> CBProject:
    """Create a small ornamental Region with a protected square island."""
    project = CBProject("native-search-ornament")
    project.units = "Millimeters"
    layer = project.add_layer("Ornamental geometry")
    outer = _outline(project, "Ornamental geometry", "ornament-shell", (
        (0, 0), (4, 0), (4, 2), (3.2, 2), (3.2, 3), (2.4, 3),
        (2.4, 2), (1.6, 2), (1.6, 2.6), (0.8, 2.6), (0.8, 2), (0, 2),
    ))
    hole = _outline(project, "Ornamental geometry", "ornament-island", (
        (1.6, 0.7), (1.6, 1.3), (2.2, 1.3), (2.2, 0.7),
    ))
    region = project.add_region(
        layer, outer, [hole], identifier="ornamental-region",
        groups=["ornament", "protected-island"],
        description="Connected lobed ornament around a protected square island",
    )
    if region is None:
        raise RuntimeError("could not create ornamental Region")
    part = project.add_part(
        "ornament-stock", stock_width=4, stock_height=3, stock_thickness=3,
        stock_material="Aluminum",
    )
    context = project.add_profile_mop(
        part, [region], identifier="ornament-context-profile",
        name="Disabled source design context", target_depth=-0.6,
        depth_increment=0.3, tool_diameter=1.2,
    )
    if context is None:
        raise RuntimeError("could not create source context MOP")
    context.enabled = False
    return project


def _component_design() -> CBProject:
    """Create one asymmetric insert component and two disabled face contexts."""
    project = CBProject("native-search-insert-design")
    project.units = "Millimeters"
    layer = project.add_layer("Insert design")
    root = project.add_rect(
        layer, width=6, height=4, identifier="insert-root",
        groups=["insert", "component"],
        description="Six by four asymmetric component envelope",
    )
    shell = _outline(project, "Insert design", "insert-region-shell", (
        (1, 0.6), (4.8, 0.6), (4.8, 2.4), (3.7, 2.4),
        (3.7, 3.1), (1, 3.1),
    ), parent="insert-root")
    hole = _outline(project, "Insert design", "insert-region-hole", (
        (1.8, 1.2), (1.8, 1.7), (2.5, 1.7), (2.5, 1.2),
    ), parent="insert-root")
    region = project.add_region(
        layer, shell, [hole], identifier="insert-region", parent="insert-root",
        groups=["insert", "machined-feature"],
        description="Offset asymmetric holed feature used by both face jobs",
    )
    if root is None or region is None:
        raise RuntimeError("could not create insert component geometry")

    for identifier in ("insert-front", "insert-back"):
        part = project.add_part(
            identifier, stock_width=6, stock_height=4, stock_thickness=3,
            stock_material="Aluminum",
        )
        context = project.add_profile_mop(
            part, [root], identifier=f"{identifier}-context-profile",
            name=f"Disabled {identifier} context", target_depth=-0.4,
            depth_increment=0.2, tool_diameter=0.8,
        )
        if context is None:
            raise RuntimeError(f"could not create {identifier} source MOP")
        context.enabled = False
    return project


def _tool_body(tool: bundle_search.Tool) -> ToolBody:
    profile = tool.profile
    radius = (profile.radius(profile.cutting_length)
              if type(profile) is v_region.VProfile else profile.radius)
    reach = profile.cutting_length
    return ToolBody(tool.tool_id, (
        ToolBand("cutter", 0, reach, radius),
        ToolBand("shank", reach, reach + 1, radius),
        ToolBand("holder", reach + 1, reach + 3, radius + 0.2),
    ))


def _setup(design: bundle_document.NativeDesign,
           tools: tuple[bundle_search.Tool, ...]) -> OccupancySetup:
    source_project = read_cambam_bytes(
        design.source_bytes, source_name="native stock for bundle search")
    part = source_project.get_part(design.part_identifier)
    if part is None or not part.stock_present:
        raise RuntimeError("bound native Part stock is unavailable")
    x0, y0, top_z = part.stock_drawing_origin
    return OccupancySetup(
        design.target.frame or "program",
        Box("selected-part-stock", (
            x0, y0, top_z - part.stock_thickness,
            x0 + part.stock_width, y0 + part.stock_height, top_z,
        )),
        (), tuple(_tool_body(tool) for tool in tools),
    )


def _ornament_job(source_bytes: bytes):
    design = bundle_document.bind_design(
        source_bytes, "ornamental-region", "ornament-stock",
        cap_depth_mm=0.4, design_angle_degrees=60, frame="ornament-top",
    )
    tools = (
        bundle_search.Tool(
            "T1", v_region.VProfile("flat", 60, 0.15, 2, 2),
            ordered.AxialLimits(0.75, 2), 10000, 100, 120, 600,
        ),
        bundle_search.Tool(
            "T2", replay.ToolProfile("T2", "cylinder", 0.3, 2),
            ordered.AxialLimits(0.75, 2), 10000, 1800, 120, 600,
        ),
    )
    families = (
        bundle_search.Family("raster", "raster", 0.45, 0.5,
                             max_paths=128, max_sites=256),
        bundle_search.Family("feature", "feature", 0.45, 0.5, max_cusp_mm=0.25,
                             max_paths=128, max_sites=256),
    )
    result = bundle_search.search(
        design.target, tools, families,
        constraints=bundle_search.Constraints(1.0, 2.2),
        cost_model=bundle_search.CostModel(3000, 0, 0, 1),
        setup=_setup(design, tools), initial_tip=(-1, -1, 3), safe_z=3,
        max_operations=2, max_evaluations=1, objective="time",
        manual_order=(("T2", "raster"), ("T1", "feature")),
    )
    return design, result


def _component_job(source_bytes: bytes, side: str):
    primitive = f"{side}/insert-region"
    part = f"{side}/insert-{side}"
    design = bundle_document.bind_design(
        source_bytes, primitive, part, cap_depth_mm=0.4,
        design_angle_degrees=60, frame=f"insert-{side}-top",
    )
    tools = (bundle_search.Tool(
        "T2", replay.ToolProfile("T2", "cylinder", 0.3, 2),
        ordered.AxialLimits(0.75, 2), 10000, 800, 100, 600,
    ),)
    families = (bundle_search.Family("raster", "raster", 0.6, 1,
                                     max_paths=64, max_sites=96),)
    result = bundle_search.search(
        design.target, tools, families,
        constraints=bundle_search.Constraints(1.0, 1.2),
        cost_model=bundle_search.CostModel(3000, 0, 0, 1),
        setup=_setup(design, tools), initial_tip=(-1, -1, 3), safe_z=3,
        max_operations=1, max_evaluations=1, objective="time",
    )
    return design, result


def _summary(design, result, fidelity_report: dict[str, str],
             residual_limits: tuple[float, float]) -> dict:
    assessments = []
    for assessment in result.assessments:
        bundle = assessment.bundle
        assessments.append({
            "actions": [list(action) for action in assessment.actions],
            "executed_actions": [list(action) for action in assessment.executed_actions],
            "omitted_actions": [list(action) for action in assessment.omitted_actions],
            "status": assessment.status,
            "reasons": list(assessment.reasons),
            "residual_area_mm2": list(bundle.area_mm2) if bundle else None,
            "residual_volume_mm3": list(bundle.volume_mm3) if bundle else None,
            "estimated_cost_seconds": bundle.cost.estimated_time_s if bundle else None,
            "tool_changes": bundle.cost.tool_changes if bundle else None,
            "setup_boundaries": bundle.cost.setup_boundaries if bundle else None,
        })
    chosen = result.chosen
    return {
        "source_sha256": design.source_sha256,
        "target_fingerprint": design.target.fingerprint,
        "bundle_document": fidelity_report,
        "search": {
            "status": result.status,
            "objective": result.objective,
            "quality": result.quality,
            "residual_limits_mm2_mm3": list(residual_limits),
            "total_orders": result.total_orders,
            "evaluated_orders": result.evaluated_orders,
            "enumeration_complete": result.enumeration_complete,
            "unresolved_orders": result.unresolved_orders,
            "selected_actions": ([list(action) for action in chosen.executed_actions]
                                 if chosen else None),
            "assessments": assessments,
        },
        "execution_certification": "not_evaluated; actual CamBam post audit required",
    }


def _write_document(name: str, design, result, source_bytes: bytes,
                    residual_limits: tuple[float, float],
                    output_directory: Path) -> tuple[bundle_document.NativeDocument, dict]:
    if design.source_bytes != source_bytes:
        raise RuntimeError("bound design was not built from the saved source bytes")
    if result.chosen is None:
        details = [assessment.reasons for assessment in result.assessments]
        raise RuntimeError(f"{name} has no selected search bundle: {details}")
    document = bundle_document.author(design, result.chosen.bundle, name=name)
    candidate_path = output_directory / f"{name}-candidate.cb"
    candidate_path.write_bytes(document.xml_bytes)
    reopened = read_cambam_file(str(candidate_path))
    if reopened is None:
        raise RuntimeError(f"could not reopen {name} candidate")
    cycle_path = output_directory / f"{name}-roundtrip.cb"
    reopened.save(str(cycle_path))
    roundtrip_bytes = cycle_path.read_bytes()
    strict = read_cambam_bytes(roundtrip_bytes, source_name=f"{name} round trip")
    if strict is None:
        raise RuntimeError(f"could not strictly read {name} round trip")
    fidelity_report = document.check_document(
        candidate_bytes=roundtrip_bytes, source_bytes=source_bytes)
    return document, _summary(design, result, fidelity_report, residual_limits)


def build_workflows(output_directory: Path) -> dict[str, bundle_document.NativeDocument]:
    """Write and return the source-bound ornament and composed face documents.

    All generated ``.cb`` artifacts and the JSON summary are confined to the
    explicit output directory. The returned live documents can be used by an
    integration caller to audit actual CamBam posts without rebuilding sources.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = {"workflows": {}}

    ornament = _mixed_region_source()
    ornament_source = output_directory / "ornament-source.cb"
    ornament.save(str(ornament_source))
    ornament_bytes = ornament_source.read_bytes()
    ornament_design, ornament_result = _ornament_job(ornament_bytes)
    ornament_document, ornament_summary = _write_document(
        "ornament", ornament_design, ornament_result, ornament_bytes,
        (1.0, 2.2), output_directory)
    summary["workflows"]["ornament"] = ornament_summary

    component = _component_design()
    stock = SetupStock(14, 8, 3, "Aluminum")
    setups = {
        "front": compose_setup(
            "insert-front-setup", stock,
            (ComponentPlacement(component, "insert-root", "front", ("insert-front",),
                                SetupFrame(x=2, y=1)),),
        ),
        "back": compose_setup(
            "insert-back-setup", stock,
            (ComponentPlacement(component, "insert-root", "back", ("insert-back",),
                                SetupFrame(x=12, y=1, reflect_x=True)),),
        ),
    }
    docs: dict[str, bundle_document.NativeDocument] = {"ornament": ornament_document}
    coordinate_checks = {}
    for side, project in setups.items():
        source_path = output_directory / f"insert-{side}-source.cb"
        project.save(str(source_path))
        source_bytes = source_path.read_bytes()
        design, result = _component_job(source_bytes, side)
        document, workflow_summary = _write_document(
            f"insert-{side}", design, result, source_bytes,
            (1.0, 1.2), output_directory)
        docs[f"insert_{side}"] = document
        summary["workflows"][f"insert_{side}"] = workflow_summary
        source_project = read_cambam_bytes(
            source_bytes, source_name=f"insert-{side} composed source")
        primitive = source_project.get_primitive(f"{side}/insert-region")
        coordinates = primitive.get_absolute_coordinates_xyz()
        region_shell = coordinates["outer_curve"]
        coordinate_checks[side] = [(point[0], point[1]) for point in region_shell]

    front_xy = coordinate_checks["front"]
    back_xy = coordinate_checks["back"]
    if len(front_xy) != len(back_xy):
        raise RuntimeError("front/back composed region point counts differ")
    reflection_errors = [abs(back[0] - (14 - front[0])) + abs(back[1] - front[1])
                         for front, back in zip(front_xy, back_xy)]
    if max(reflection_errors, default=math.inf) > 1e-8:
        raise RuntimeError("back composed Region does not reflect across stock width")
    summary["composed_reflection"] = {
        "stock_width_mm": 14,
        "witness": "back_x = stock_width - front_x; front_y = back_y",
        "maximum_coordinate_error_mm": max(reflection_errors, default=0),
    }
    summary_path = output_directory / "native-search-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    documents = build_workflows(args.output_directory)
    print(f"Wrote {len(documents)} source-bound native workflows to {args.output_directory}")
    print("Search cost is estimated; actual native post audit remains not evaluated.")


if __name__ == "__main__":
    main()
