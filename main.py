"""
main.py

Demonstrates usage of the CamBam framework.
Creates a project, adds layers, parts, primitives and MOPs,
assigns groups, descriptions, and tests assignment to different layers and parts.
Then writes the final CamBam file and saves the project state.
"""

import os
import logging
from cambam_project import CamBamProject
from cambam_writer import save_cambam_file

logger = logging.getLogger(__name__)

def test():
    # INFO logger level for this test case
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    project = CamBamProject("Demo Project", default_tool_diameter=5.0)
    
    # Create layers using string identifiers for robust lookup
    top_layer_id = "Top Layer"
    bottom_layer_id = "Bottom Layer"
    project.add_layer(top_layer_id, color="Cyan")
    project.add_layer(bottom_layer_id, color="Red")
    
    # Create parts
    project.add_part("Part_A", stock_thickness=12.4, machining_origin=(0, 0), default_spindle_speed=20000, default_tool_diameter=3.175)
    project.add_part("Part_B", stock_thickness=9.3, machining_origin=(1220, 0), default_spindle_speed=24000, default_tool_diameter=3.175)
    
    # Add primitives with separate assignment properties.
    # Top primitives assigned to the Top Layer.
    rect_top = project.add_rect(top_layer_id, corner=(0, 0), width=800, height=300,
                                identifier="rect_top", groups=["frame"],
                                description="Top frame cutout")
    text_top = project.add_text(top_layer_id, text="Top Board", position=(400, 150), height=20,
                                identifier="text_top", groups=["label"],
                                description="Label for top board", parent=rect_top)
    
    # Bottom primitives assigned to the Bottom Layer.
    rect_bottom = project.add_rect(bottom_layer_id, corner=(0, 0), width=800, height=300,
                                   identifier="rect_bottom", groups=["frame"],
                                   description="Bottom frame cutout")
    text_bottom = project.add_text(bottom_layer_id, text="Bottom Board", position=(400, 150), height=20,
                                   identifier="text_bottom", groups=["label"],
                                   description="Label for bottom board", parent=rect_bottom)
    
    # Add MOPs based on group assignment ("frame")
    project.add_profile_mop("Part_A", pid_source="frame", name="profile_op", target_depth=-10, roughing_clearance=-0.1)
    project.add_pocket_mop("Part_B", pid_source="frame", name="pocket_op", target_depth=-10, roughing_clearance=-0.2)
    
    # Perform a transformation: translate the bottom rectangle
    project.translate_primitive("rect_bottom", dx=300, dy=0, bake=True)

    # Save CamBam file and project state
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "demo_project.cb")
    save_cambam_file(project, file_path, pretty_print=True)
    
    state_file = os.path.join(output_dir, "demo_project.pkl")
    project.save_state(state_file)
    logger.info("Demo project processing complete.")



def test2():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    logger.setLevel(logging.DEBUG) # Show debug messages

    proj = CamBamProject("Project Test with linking and transformations", default_tool_diameter=5.0)

    # Layers
    layer_frame_top = proj.add_layer("Frame Top", color="Cyan")
    layer_frame_bottom = proj.add_layer("Frame Bottom", color="Red")

    # Parts
    part_a = proj.add_part("12.4 | EM6.0", stock_thickness=12.4, machining_origin=(0,0), default_spindle_speed=20000, default_tool_diameter=3.175)
    part_b = proj.add_part("9.3 | EM3.175", stock_thickness=9.3, machining_origin=(1220,0), default_spindle_speed=24000, default_tool_diameter=3.175)

    # Top Frame Board primitives
    rect_frame_top = proj.add_rect(layer=layer_frame_top, identifier="frame_top", width=800, height=300, groups=["frame_cutout", "frame"])
    rect_frame_top_edge_left = proj.add_rect(layer=layer_frame_top, identifier="frame_top_edge_left", corner=(0,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_edge_right = proj.add_rect(layer=layer_frame_top, identifier="frame_top_edge_right", corner=(800-12.4,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_edge_back = proj.add_rect(layer=layer_frame_top, identifier="frame_top_edge_back", corner=(0-3.175,300-12.4), width=800+2*3.175, height=12.4, groups=["frame_edge_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_groove_1 = proj.add_rect(layer=layer_frame_top, identifier="frame_top_groove_1", corner=(200-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_top)
    rect_frame_top_groove_2 = proj.add_rect(layer=layer_frame_top, identifier="frame_top_groove_2", corner=(400-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_top)
    text_frame_top = proj.add_text(layer=layer_frame_top, identifier="frame_top_id", text="Frame Top Board", position=(800/2,300/2), height=20, groups=["frame_id"], parent=rect_frame_top)

    # Bottom Frame Board primitives
    rect_frame_bottom = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom", width=800, height=300, groups=["frame_cutout", "frame"])
    rect_frame_bottom_edge_left = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_left", corner=(0,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_edge_right = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_right", corner=(800-12.4,-3.175), width=12.4, height=300+2*3.175, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_edge_back = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_edge_back", corner=(0-3.175,300-12.4), width=800+2*3.175, height=12.4, groups=["frame_edge_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_groove_1 = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_groove_1", corner=(200-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_bottom)
    rect_frame_bottom_groove_2 = proj.add_rect(layer=layer_frame_bottom, identifier="frame_bottom_groove_2", corner=(400-12.4/2,10), width=12.4, height=300-10+3.175, groups=["frame_groove", "frame"], parent=rect_frame_bottom)
    # text_frame_bottom_id = proj.add_text(layer=layer_frame_bottom, identifier="frame_bottom_id", text="Frame Bottom Board", position=(800/2,300/2), height=20, groups=["frame_id"], parent=rect_frame_bottom)

    # Transformations of bottom board and linked children
    # proj.translate_primitive(rect_frame_top, 20, 20)
    proj.translate_primitive(rect_frame_bottom, 300, 0)
    # proj.translate_primitive(rect_frame_bottom, 20+800+30, 20)
    proj.mirror_primitive_y(rect_frame_bottom, float(rect_frame_bottom.get_absolute_coordinates()[0][0]))
    rect_frame_bottom.bake()
    text_frame_bottom_id = proj.add_text(layer=layer_frame_bottom, identifier="frame_bottom_id", text="Frame Bottom Board", position=rect_frame_bottom.get_geometric_center(), height=20, groups=["frame_id"], parent=rect_frame_bottom)
    c = rect_frame_bottom.get_absolute_coordinates()[0]
    proj.rotate_primitive_deg(rect_frame_bottom, 90, cx=c[0], cy=c[1])
    # proj.align_primitive(rect_frame_bottom, "lower_left", (0,0))


    # MOP
    proj.add_pocket_mop(part=part_a, pid_source="frame_edge_groove", name="frame_edge_groove", target_depth=-12.4*(1-0.75), roughing_clearance=-0.1)
    proj.add_pocket_mop(part=part_a, pid_source="frame_groove", name="frame_groove", target_depth=-12.4*(1-0.75), roughing_clearance=-0.2)
    proj.add_profile_mop(part=part_a, pid_source="frame_cutout", name="frame_id", profile_side="Outside", target_depth=-12.9, roughing_clearance=-0.1)
    
    output_dir = "./output"
    save_cambam_file(proj, os.path.join(output_dir, "v5.cb"))
    proj.save_state(os.path.join(output_dir, "v5.pkl"))

    # --- Load Test ---
    loaded = CamBamProject.load_state(os.path.join(output_dir, "v5.pkl"))
    print(f"\nLoaded Project '{loaded.project_name}'")
    print(f" Bounding Box: {loaded.get_bounding_box()}")
    print(f" Primitives: {len(loaded.list_primitives())}")





if __name__ == '__main__':
    test2()
