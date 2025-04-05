import logging
import os
import uuid
import numpy as np

# Module imports
from cambam_project import CamBamProject
from cambam_entities import Layer, Part, Pline, Rect, Text, Circle, ProfileMop, PocketMop
from cad_common import BoundingBox # For type hint if needed
from cad_transformations import translation_matrix, rotation_matrix_deg, identity_matrix

if __name__ == '__main__':
    # --- Setup Logging ---
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format) # Use DEBUG for detailed framework output
    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger('some_library').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info("--- Starting CamBam Framework Example ---")

    # --- Output Directory ---
    output_dir = "./refactored_output"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- Create Project ---
    proj = CamBamProject("Refactored Project Example", default_tool_diameter=6.0)

    # --- Create Layers ---
    layer_top = Layer(user_identifier="Top Layer", color="Cyan")
    layer_bottom = Layer(user_identifier="Bottom Layer", color="Red", visible=True)
    proj.add_layer(layer_top)
    proj.add_layer(layer_bottom)

    # --- Create Parts ---
    part1 = Part(user_identifier="MDF 12mm", stock_thickness=12.0, default_spindle_speed=18000)
    part2 = Part(user_identifier="Plywood 6mm", stock_thickness=6.0, default_tool_diameter=3.175)
    proj.add_part(part1)
    proj.add_part(part2)

    # --- Create Primitives ---
    # Basic rectangle on Top Layer
    rect1 = Rect(layer_id=layer_top.internal_id, user_identifier="BaseRect",
                 relative_corner=(10, 10), width=100, height=50, groups=["Outer", "Profile"])
    proj.add_primitive(rect1)

    # Circle inside rect1, linked as a child
    circle1 = Circle(layer_id=layer_top.internal_id, user_identifier="InnerCircle",
                     relative_center=(60, 35), diameter=30, groups=["Inner", "Pocket"],
                     parent_primitive_id=rect1.internal_id) # Link to rect1
    proj.add_primitive(circle1)

    # Text label for rect1, also a child
    label1 = Text(layer_id=layer_top.internal_id, user_identifier="RectLabel",
                  text_content="Base", relative_position=(60, 35), height=8,
                  groups=["Labels"], parent_primitive_id=rect1.internal_id)
    proj.add_primitive(label1)

    # Another rectangle on Bottom Layer
    rect2 = Rect(layer_id=layer_bottom.internal_id, user_identifier="SecondRect",
                 relative_corner=(150, 10), width=80, height=80, groups=["Outer"])
    proj.add_primitive(rect2)

    # --- Transformations ---
    logger.info("--- Applying Transformations ---")

    # 1. Translate the entire base rectangle structure (rect1, circle1, label1)
    logger.info(f"Rect1 Original Center: {rect1.get_geometric_center()}")
    proj.translate_primitive(rect1.internal_id, dx=20, dy=30, bake=False)
    logger.info(f"Rect1 Translated Center: {rect1.get_geometric_center()}")
    logger.info(f"Circle1 (child) Translated Center: {circle1.get_geometric_center()}") # Should move with parent

    # 2. Rotate the base rectangle structure around its new center
    center = rect1.get_geometric_center()
    proj.rotate_primitive_deg(rect1.internal_id, angle_deg=45, cx=center[0], cy=center[1], bake=False)
    logger.info(f"Rect1 Rotated Center: {rect1.get_geometric_center()}") # Center should remain same
    logger.info(f"Circle1 (child) Rotated Position: {circle1.get_geometric_center()}") # Should rotate around parent center

    # 3. Scale the second rectangle (non-uniform) without baking
    center2 = rect2.get_geometric_center()
    proj.scale_primitive(rect2.internal_id, sx=1.5, sy=0.8, cx=center2[0], cy=center2[1], bake=False)
    logger.info(f"Rect2 Scaled BBox: {rect2.get_bounding_box()}")

    # 4. Align the scaled rect2's lower-left corner to (200, 200)
    proj.align_primitive(rect2.internal_id, align_point='lower_left', target_coord=(200, 200), bake=False)
    logger.info(f"Rect2 Aligned Lower-Left: {rect2.get_bounding_box().min_x}, {rect2.get_bounding_box().min_y}")

    # --- Baking ---
    logger.info("--- Baking Transformations ---")
    # Bake the transformations into rect1 (and its children)
    # Note: Baking the rotated Rect1 will convert it to a Pline
    logger.info(f"Baking Rect1 (ID: {rect1.internal_id}). Expect replacement.")
    success_bake = proj.transform_primitive(rect1.internal_id, identity_matrix(), bake=True) # Bake current state

    # Verify rect1 was replaced
    rect1_after_bake = proj.get_primitive(rect1.internal_id) # Original ID should be gone
    logger.info(f"Original Rect1 object after bake: {rect1_after_bake}")
    assert rect1_after_bake is None, "Original Rect1 should be removed after baking with replacement."

    # Find the replacement (assuming only one for simplicity here)
    # This is a bit fragile, depends on naming convention or searching by geometry/group
    baked_pline = proj.get_primitive("BaseRect_baked_pline") # Find by expected name
    if baked_pline:
         logger.info(f"Found replacement Pline: {baked_pline.user_identifier}")
         logger.info(f"  Baked Pline has effective transform close to identity: {np.allclose(baked_pline.effective_transform, identity_matrix())}")
         # Verify children are now parented to the pline and their transforms reset
         circle1_after_bake = proj.get_primitive(circle1.internal_id)
         label1_after_bake = proj.get_primitive(label1.internal_id)
         if circle1_after_bake:
             logger.info(f"  Child Circle parent ID: {circle1_after_bake.parent_primitive_id} (Expected: {baked_pline.internal_id})")
             logger.info(f"  Child Circle effective transform near identity: {np.allclose(circle1_after_bake.effective_transform, identity_matrix())}")
             assert circle1_after_bake.parent_primitive_id == baked_pline.internal_id
             assert np.allclose(circle1_after_bake.effective_transform, identity_matrix())
         if label1_after_bake:
              logger.info(f"  Child Label parent ID: {label1_after_bake.parent_primitive_id}")
              logger.info(f"  Child Label effective transform near identity: {np.allclose(label1_after_bake.effective_transform, identity_matrix())}")
              assert label1_after_bake.parent_primitive_id == baked_pline.internal_id
              assert np.allclose(label1_after_bake.effective_transform, identity_matrix()) # Text bake resets matrix
    else:
         logger.error("Could not find the replacement Pline after baking Rect1.")


    # --- Copying ---
    logger.info("--- Copying Primitives ---")
    # Copy the scaled & aligned rect2
    copied_rect2 = proj.copy_primitive(rect2.internal_id, new_identifier_suffix="_copy1")
    if copied_rect2:
         logger.info(f"Copied rect2 as '{copied_rect2.user_identifier}' (ID: {copied_rect2.internal_id})")
         # Move the copy
         proj.translate_primitive(copied_rect2.internal_id, dx=100, dy=0, bake=False)
         logger.info(f"  Original rect2 bbox: {rect2.get_bounding_box()}")
         logger.info(f"  Copied rect2 bbox:   {copied_rect2.get_bounding_box()}")
         assert copied_rect2.internal_id != rect2.internal_id
         assert copied_rect2.user_identifier != rect2.user_identifier
    else:
         logger.error("Failed to copy rect2.")

    # --- Create MOPs ---
    logger.info("--- Creating MOPs ---")
    # Pocket the inner circle (use its UUID)
    if circle1_after_bake: # It might have been removed if bake failed? Check needed.
        pocket1 = PocketMop(part_id=part1.internal_id, name="Circle Pocket",
                            pid_source=[circle1_after_bake.internal_id], # Reference by UUID
                            target_depth=-5.0)
        proj.add_mop(pocket1)
    else:
         logger.warning("Skipping pocket MOP creation as inner circle was not found after bake.")

    # Profile the outer shapes (use group name)
    profile1 = ProfileMop(part_id=part1.internal_id, name="Outer Profile",
                          pid_source="Outer", # Reference by group name
                          target_depth=-12.5, # Cut through 12mm stock + little extra
                          profile_side="Outside",
                          tool_diameter=6.0) # Explicitly set for this MOP
    proj.add_mop(profile1)

    # Profile the second rectangle using its specific ID (on part 2)
    profile2 = ProfileMop(part_id=part2.internal_id, name="Rect2 Profile",
                          pid_source=[rect2.internal_id], # Reference original by UUID
                          target_depth=-6.5,
                          profile_side="Outside")
    proj.add_mop(profile2)

    # --- Project Info ---
    logger.info("--- Project Summary ---")
    logger.info(f"Project BBox: {proj.get_bounding_box()}")
    logger.info(f"Primitives: {len(proj.list_primitives())}")
    for group in proj.list_groups():
         count = len(proj.get_primitives_in_group(group))
         logger.info(f"  Group '{group}': {count} primitives")
    logger.info(f"MOPs: {len(proj.list_mops())}")

    # --- Save Project ---
    xml_file_path = os.path.join(output_dir, "refactored_output.cb")
    pickle_file_path = os.path.join(output_dir, "refactored_output.pkl")

    logger.info(f"--- Saving Project to XML: {xml_file_path} ---")
    proj.save_to_xml(xml_file_path, pretty_print=True)

    logger.info(f"--- Saving Project State (Pickle): {pickle_file_path} ---")
    proj.save_state(pickle_file_path)

    # --- Load Project ---
    logger.info(f"--- Loading Project State (Pickle): {pickle_file_path} ---")
    try:
        loaded_proj_pkl = CamBamProject.load_state(pickle_file_path)
        logger.info(f"Successfully loaded '{loaded_proj_pkl.project_name}' from pickle.")
        logger.info(f"  Loaded Project BBox: {loaded_proj_pkl.get_bounding_box()}")
        # Verify a known primitive's transform after load
        loaded_rect2 = loaded_proj_pkl.get_primitive(rect2.internal_id)
        if loaded_rect2:
             logger.info(f"  Loaded Rect2 transform determinant: {np.linalg.det(loaded_rect2.effective_transform)}")
        else:
             logger.warning("Could not find Rect2 in loaded pickle project.")

    except Exception as e:
        logger.error(f"Failed to load project from pickle: {e}", exc_info=True)


    logger.info(f"--- Loading Project from XML: {xml_file_path} ---")
    try:
        loaded_proj_xml = CamBamProject.load_from_xml(xml_file_path)
        logger.info(f"Successfully loaded '{loaded_proj_xml.project_name}' from XML.")
        logger.info(f"  Loaded Project BBox: {loaded_proj_xml.get_bounding_box()}")
        # Verify XML loaded objects exist
        logger.info(f"  Primitives loaded from XML: {len(loaded_proj_xml.list_primitives())}")
        logger.info(f"  MOPs loaded from XML: {len(loaded_proj_xml.list_mops())}")
        # Check a loaded MOP's primitive source (will be list of UUIDs)
        loaded_mop = loaded_proj_xml.get_mop("Outer Profile") # Find by name
        if loaded_mop:
             logger.info(f"  Loaded MOP 'Outer Profile' pid_source (type {type(loaded_mop.pid_source)}): {loaded_mop.pid_source}")
             # We could further check if these UUIDs map back to the expected primitives
        else:
             logger.warning("Could not find MOP 'Outer Profile' in XML loaded project.")

    except Exception as e:
        logger.error(f"Failed to load project from XML: {e}", exc_info=True)


    logger.info("--- CamBam Framework Example Finished ---")