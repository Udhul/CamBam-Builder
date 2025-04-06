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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
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
    project.add_pocket_mop("Part_A", pid_source="frame", name="pocket_op", target_depth=-10, roughing_clearance=-0.2)
    
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

if __name__ == '__main__':
    main()
