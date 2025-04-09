"""
main.py

Demonstrates usage of the refactored CamBam framework based on centralized
relationship management in the CamBamProject.
"""

import os
import logging
import sys # For basic logging setup

# Assuming modules are in the same directory orPYTHONPATH is set
from cambam_project import CamBamProject
from cambam_writer import save_cambam_file
from cambam_reader import read_cambam_file # Import reader for testing

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to see detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout # Log to console
)
# Optionally reduce verbosity of specific libraries if needed
# logging.getLogger('some_library').setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def run_demo_1():
    """Demonstrates basic project setup, adding entities, groups, MOPs, and saving."""
    logger.info("--- Starting Demo 1 ---")
    project = CamBamProject("Demo Project 1", default_tool_diameter=5.0)

    # --- Layers ---
    # Add layers using string identifiers. The project manages the Layer objects.
    layer_top = project.add_layer("Top Layer", color="Cyan")
    layer_bottom = project.add_layer("Bottom Layer", color="Red")
    if not layer_top or not layer_bottom:
        logger.error("Failed to create layers. Aborting demo.")
        return

    # --- Parts ---
    part_a = project.add_part("Part_A", stock_thickness=12.4, machining_origin=(0, 0), default_spindle_speed=20000, default_tool_diameter=3.175)
    part_b = project.add_part("Part_B", stock_thickness=9.3, machining_origin=(1220, 0), default_spindle_speed=24000, default_tool_diameter=3.175)
    if not part_a or not part_b:
        logger.error("Failed to create parts. Aborting demo.")
        return

    # --- Primitives ---
    # Add primitives, specifying layer and optionally parent/groups directly in the call.
    # The project handles linking.

    # Top primitives assigned to the Top Layer.
    rect_top = project.add_rect(
        layer="Top Layer", # Identify layer by string
        corner=(10, 10), width=800, height=300,
        identifier="rect_top", groups=["frame", "top_items"],
        description="Top frame main rectangle"
    )
    if not rect_top: logger.error("Failed to add rect_top"); return

    # Add text as a child of the rectangle
    text_top = project.add_text(
        layer=layer_top, # Identify layer by object
        text="Top Board", position=(410, 160), height=20,
        identifier="text_top", groups=["label", "top_items"],
        description="Label for top board",
        parent=rect_top # Identify parent by object
    )
    if not text_top: logger.error("Failed to add text_top"); return

    # Bottom primitives assigned to the Bottom Layer.
    rect_bottom = project.add_rect(
        layer=layer_bottom, # Identify layer by object
        corner=(10, 10), width=800, height=300,
        identifier="rect_bottom", groups=["frame", "bottom_items"],
        description="Bottom frame main rectangle"
    )
    if not rect_bottom: logger.error("Failed to add rect_bottom"); return

    text_bottom = project.add_text(
        layer="Bottom Layer", # Identify layer by string
        text="Bottom Board", position=(410, 160), height=20,
        identifier="text_bottom", groups=["label", "bottom_items"],
        description="Label for bottom board",
        parent="rect_bottom" # Identify parent by string identifier
    )
    if not text_bottom: logger.error("Failed to add text_bottom"); return

    # --- Machine Operations (MOPs) ---
    # Add MOPs, specifying the part and the primitive source (group name or list of identifiers).
    # Target depth negative means cut into stock from Z=0 (StockSurface)

    # Profile MOP targeting the "frame" group on Part A
    profile_mop = project.add_profile_mop(
        part=part_a, # Identify part by object
        pid_source="frame", # Target the group named "frame"
        name="Frame Profile Cut",
        identifier="mop_profile_frame",
        target_depth=-12.9, # Cut slightly through 12.4mm stock
        profile_side="Outside",
        tool_diameter=6.0 # Override part default
    )
    if not profile_mop: logger.error("Failed to add profile_mop"); return

    # Pocket MOP targeting specific primitives (the two text labels) on Part B
    pocket_mop = project.add_pocket_mop(
        part="Part_B", # Identify part by string
        pid_source=["text_top", text_bottom], # Target specific primitives by identifier/object
        name="Label Pocket Engrave",
        identifier="mop_pocket_labels",
        target_depth=-0.5,
        tool_diameter=1.0, # Use a small tool for engraving
        roughing_clearance=0.0 # Follow lines exactly
    )
    if not pocket_mop: logger.error("Failed to add pocket_mop"); return


    # --- Transformations ---
    # Perform transformations using project methods. The project handles propagation.
    logger.info("Translating bottom rectangle (rect_bottom)...")
    success = project.translate_primitive("rect_bottom", dx=900, dy=0) # Translate globally
    if not success: logger.error("Failed to translate rect_bottom")

    # Rotate the top text around its own center
    logger.info("Rotating top text (text_top)...")
    success = project.rotate_primitive_deg(text_top, 45) # Rotate around its geometric center
    if not success: logger.error("Failed to rotate text_top")

    # Bake the transformation for the bottom rectangle (will affect children if any were added)
    logger.info("Baking transformation for bottom rectangle...")
    success = project.bake_primitive_transform("rect_bottom", recursive=True)
    if not success: logger.error("Failed to bake rect_bottom")


    # --- Output ---
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Save CamBam XML file
    cb_file_path = os.path.join(output_dir, "demo_project_1.cb")
    logger.info(f"Saving CamBam file to: {cb_file_path}")
    save_cambam_file(project, cb_file_path, pretty_print=True)

    # Save Project State (Pickle)
    state_file_path = os.path.join(output_dir, "demo_project_1.pkl")
    logger.info(f"Saving project state to: {state_file_path}")
    project.save_state(state_file_path)

    logger.info("--- Demo 1 Finished ---")


def run_demo_2():
    """Demonstrates more complex linking, transformations, and loading."""
    logger.info("--- Starting Demo 2 ---")
    proj = CamBamProject("Project Test Linking", default_tool_diameter=6.0)

    # --- Layers & Parts ---
    layer_frame = proj.add_layer("Frame Layer", color="Blue")
    part_main = proj.add_part("Main Part", stock_thickness=18.0, default_tool_diameter=3.175)
    if not layer_frame or not part_main: logger.error("Setup failed"); return

    # --- Primitives with Hierarchy ---
    # Base frame
    rect_base = proj.add_rect(layer=layer_frame, identifier="frame_base", width=500, height=200, groups=["frame"])
    if not rect_base: logger.error("Failed to add rect_base"); return

    # Add children linked to the base
    circle_1 = proj.add_circle(layer=layer_frame, identifier="hole_1", center=(50, 50), diameter=20, groups=["holes"], parent=rect_base)
    circle_2 = proj.add_circle(layer=layer_frame, identifier="hole_2", center=(450, 50), diameter=20, groups=["holes"], parent=rect_base)
    rect_child = proj.add_rect(layer=layer_frame, identifier="cutout_1", corner=(100, 75), width=300, height=50, groups=["cutouts"], parent=rect_base)
    text_child = proj.add_text(layer=layer_frame, identifier="id_text", text="ID:123", position=(250, 150), height=15, parent=rect_base) # Auto group?
    if not circle_1 or not circle_2 or not rect_child or not text_child: logger.error("Failed adding children"); return

    # --- Transformations (applied to parent) ---
    logger.info("Applying transformations to parent 'frame_base'...")
    # Translate the whole group
    proj.translate_primitive(rect_base, dx=10, dy=10)
    # Rotate the whole group around the base's corner (10,10) after translation
    # Note: get_geometric_center() gives center, use specific coords if needed
    base_coords = rect_base.get_absolute_coordinates() # Get corners
    if base_coords:
        corner_x, corner_y = base_coords[0] # Bottom-left corner's absolute position
        proj.rotate_primitive_deg(rect_base, 15, cx=corner_x, cy=corner_y)
    else:
         logger.warning("Could not get base coordinates for rotation anchor.")
         proj.rotate_primitive_deg(rect_base, 15) # Rotate around center as fallback


    # --- MOPs ---
    proj.add_profile_mop(part=part_main, pid_source="holes", name="Drill Holes (Profile)", profile_side="Inside", target_depth=-18.5)
    proj.add_profile_mop(part=part_main, pid_source="cutouts", name="Cutouts", profile_side="Inside", target_depth=-18.5)
    proj.add_profile_mop(part=part_main, pid_source=["frame_base"], name="Outer Frame Cut", profile_side="Outside", target_depth=-18.5)
    proj.add_engrave_mop(part=part_main, pid_source=[text_child], name="Engrave ID", target_depth=-0.3)

    # --- Output ---
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    cb_file_path = os.path.join(output_dir, "demo_project_2.cb")
    state_file_path = os.path.join(output_dir, "demo_project_2.pkl")

    logger.info(f"Saving CamBam file to: {cb_file_path}")
    save_cambam_file(proj, cb_file_path, pretty_print=True)
    logger.info(f"Saving project state to: {state_file_path}")
    proj.save_state(state_file_path)

    # --- Load Test ---
    logger.info(f"\n--- Loading Project State from: {state_file_path} ---")
    loaded_proj = CamBamProject.load_state(state_file_path)

    if loaded_proj:
        logger.info(f"Successfully loaded project: {loaded_proj.project_name}")
        logger.info(f" Loaded Layers: {len(loaded_proj.list_layers())}")
        logger.info(f" Loaded Parts: {len(loaded_proj.list_parts())}")
        logger.info(f" Loaded Primitives: {len(loaded_proj.list_primitives())}")
        logger.info(f" Loaded MOPs: {len(loaded_proj.list_mops())}")

        # Verify some relationships
        loaded_text = loaded_proj.get_primitive("id_text")
        if loaded_text:
             parent = loaded_proj.get_parent_of_primitive(loaded_text)
             logger.info(f" Parent of 'id_text' is: {parent.user_identifier if parent else 'None'}")
             layer = loaded_proj.get_layer_of_primitive(loaded_text)
             logger.info(f" Layer of 'id_text' is: {layer.user_identifier if layer else 'None'}")

        # Try loading from the saved CB file as well (basic test)
        logger.info(f"\n--- Reading CB file: {cb_file_path} ---")
        read_proj = read_cambam_file(cb_file_path)
        if read_proj:
            logger.info(f"Successfully read project from CB file: {read_proj.project_name}")
            logger.info(f" Read Layers: {len(read_proj.list_layers())}")
            logger.info(f" Read Parts: {len(read_proj.list_parts())}")
            # Note: Primitive/MOP count might differ if reader is incomplete
            logger.info(f" Read Primitives (via Layer query): {sum(len(read_proj.get_primitives_on_layer(l)) for l in read_proj.list_layers())}")
            logger.info(f" Read MOPs (via Part query): {sum(len(read_proj.get_mops_in_part(p)) for p in read_proj.list_parts())}")
        else:
            logger.error("Failed to read project back from CB file.")

    else:
        logger.error("Failed to load project state from pickle file.")

    logger.info("--- Demo 2 Finished ---")


def run_demo_3():
    logger.info("--- Demo 3: Loading from CamBam file ---")
    project = CamBamProject("demo_project_3", default_tool_diameter=5.0)

    # Layers
    layer_frame_top = project.add_layer("Frame Top", color="Cyan")
    layer_frame_bottom = project.add_layer("Frame Bottom", color="Red")

    # Parts
    part_a = project.add_part("12.4 | EM6.0", stock_thickness=12.4, machining_origin=(0,0), default_spindle_speed=20000, default_tool_diameter=3.175)
    part_b = project.add_part("9.3 | EM3.175", stock_thickness=9.3, machining_origin=(1220,0), default_spindle_speed=24000, default_tool_diameter=3.175)

    # Top and Bottom Frame Boards primitives
    # project.set_cursor(20,20) # Set machine origin offset

    rect_frame_top = project.add_rect(layer=layer_frame_top, identifier="frame_top", width=800, height=300, groups=["frame_cutout", "frame"])
    rect_frame_top_edge_left = project.add_rect(layer=layer_frame_top, identifier="frame_top_edge_left", corner=(0,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_edge_right = project.add_rect(layer=layer_frame_top, identifier="frame_top_edge_right", corner=(800-12.4,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_edge_back = project.add_rect(layer=layer_frame_top, identifier="frame_top_edge_back", corner=(0-3.175,300-12.4), width=800+2*3.175, height=12.4, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_groove_1 = project.add_rect(layer=layer_frame_top, identifier="frame_top_groove_1", corner=(200-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_groove_2 = project.add_rect(layer=layer_frame_top, identifier="frame_top_groove_2", corner=(400-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_top)
    text_frame_top = project.add_text(layer=layer_frame_top, identifier="frame_top_id", text="Frame Top Board", position=(800/2,300/2), height=20, groups=["frame_id"], parent=rect_frame_top)

    rect_frame_bottom = project.add_rect(layer=layer_frame_bottom, identifier="frame_bottom", width=800, height=300, groups=["frame_cutout", "frame"])
    rect_frame_bottom_edge_left = project.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_left", corner=(0,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_edge_right = project.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_right", corner=(800-12.4,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_edge_back = project.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_back", corner=(0-3.175,300-12.4), width=800+2*3.175, height=12.4, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_groove_1 = project.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_groove_1", corner=(200-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_groove_2 = project.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_groove_2", corner=(400-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_bottom)
    # text_frame_bottom_id = project.add_text(layer=layer_frame_bottom, identifier="frame_bottom_id", text="Frame Bottom Board", position=(800/2,300/2), height=20, groups=["frame_id"], parent=rect_frame_bottom)

    # Transformations
    # project.translate_primitive(rect_frame_top, 20, 20)
    # project.translate_primitive(rect_frame_bottom, 300, 0)
    # project.translate_primitive(rect_frame_bottom, 20+800+30, 20)
    project.mirror_primitive_y(rect_frame_bottom)
    rect_frame_bottom.bake_geometry()
    text_frame_bottom_id = project.add_text(layer=layer_frame_bottom, identifier="frame_bottom_id", text="Frame Bottom Board", position=rect_frame_bottom.get_geometric_center(), height=20, groups=["frame_id"], parent=rect_frame_bottom)
    # c = rect_frame_bottom.get_absolute_coordinates()[0]
    # project.rotate_primitive_deg(rect_frame_bottom, 90, cx=c[0], cy=c[1])
    # project.align_primitive(rect_frame_bottom, "lower_left", (0,0))


    # MOP
    project.add_pocket_mop(part=part_a, pid_source="frame_edge_groove", name="frame_edge_groove", target_depth=-12.4*(1-0.75), roughing_clearance=-0.1)
    project.add_pocket_mop(part=part_a, pid_source="frame_groove", name="frame_groove", target_depth=-12.4*(1-0.75), roughing_clearance=-0.2)
    project.add_profile_mop(part=part_a, pid_source="frame_cutout", name="frame_cutout", profile_side="Outside", target_depth=-12.9, roughing_clearance=-0.1)
    
    # --- Output ---
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Save CamBam XML file
    cb_file_path = os.path.join(output_dir, project.project_name + ".cb")
    logger.info(f"Saving CamBam file to: {cb_file_path}")
    save_cambam_file(project, cb_file_path, pretty_print=True)

    # Save Project State (Pickle)
    state_file_path = os.path.join(output_dir, project.project_name + ".pkl")
    logger.info(f"Saving project state to: {state_file_path}")
    project.save_state(state_file_path)

    logger.info("--- Demo 3 Finished ---")

    # --- Load Test ---
    # loaded = CamBamProject.load_state(os.path.join(output_dir, project.project_name + ".pkl"))
    # print(f"\nLoaded Project '{loaded.project_name}'")
    # print(f" Bounding Box: {loaded.get_bounding_box()}")
    # print(f" Primitives: {len(loaded.list_primitives())}")


    # Print links with names
    for key in project._primitive_children_link.keys():
        parent = project._primitives.get(key)
        parent_name = parent.user_identifier
        child_uuids = project._primitive_children_link[key]
        child_names = []
        for child_uuid in child_uuids:
            current_child = project._get_entity_by_uuid(child_uuid)
            if current_child is not None:
                child_names.append(current_child.user_identifier)
        print(f"{parent_name} <- {child_names}")

if __name__ == '__main__':
    # run_demo_1()
    # print("\n" + "="*60 + "\n") # Separator
    run_demo_3()

# TODO FIX BAKING COMBINE MATRIX
# Bake children when baking parent. Bake method should be added to project, so it can iterate over links correctly