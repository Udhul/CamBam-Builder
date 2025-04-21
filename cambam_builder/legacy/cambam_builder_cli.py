import os
import sys
from tempfile import gettempdir
import argparse

# Get the path of the application
if getattr(sys, 'frozen', False):  # Check if .exe
    application_path = os.path.dirname(sys.executable)
elif __file__:  # or .py
    application_path = os.path.dirname(__file__)
parent_dir_name = os.path.dirname(application_path)

sys.path.append(parent_dir_name)
# sys.path.append(os.path.join(parent_dir_name, 'Resources'))

# Import project modules
from .legacy_cambam_builder import CamBam

# Define the path for the serialized state file
TMPDIR = gettempdir()
STATE_FILE = os.path.join(TMPDIR, 'cambam_legacy_state.pkl')

def load_cambam_state() -> CamBam:
    if os.path.exists(STATE_FILE):
        return CamBam.load_state(STATE_FILE)
    return None

def save_cambam_state(cb):
    cb.save_state(STATE_FILE)

def main():
    parser = argparse.ArgumentParser(description="CamBam CLI tool")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for creating a new file
    parser_new = subparsers.add_parser('new', help='Create a new CamBam file')
    parser_new.add_argument('name', type=str, help='Name of the new file')
    parser_new.add_argument('--endmill', type=float, default=5, help='Endmill diameter')
    parser_new.add_argument('--stock_thickness', type=float, default=12.5, help='Stock thickness')
    parser_new.add_argument('--stock_width', type=int, default=1220, help='Stock width')
    parser_new.add_argument('--stock_height', type=int, default=2440, help='Stock height')
    parser_new.add_argument('--spindle_speed', type=int, default=24000, help='Spindle speed')

    # Subparser for adding a layer
    parser_layer = subparsers.add_parser('add_layer', help='Add a new layer')
    parser_layer.add_argument('name', type=str, help='Name of the new layer')
    parser_layer.add_argument('--color', type=str, default='Green', help='Layer color')
    parser_layer.add_argument('--alpha', type=float, default=1.0, help='Layer transparency')
    parser_layer.add_argument('--pen', type=float, default=1.0, help='Layer stroke width')
    parser_layer.add_argument('--visible', type=bool, default=True, help='Layer visibility')
    parser_layer.add_argument('--locked', type=bool, default=False, help='Layer lock state')
    parser_layer.add_argument('--target', type=str, default='', help='Target layer to place before/after')
    parser_layer.add_argument('--place_last', type=bool, default=True, help='Place after the target layer')

    # Subparser for adding a part
    parser_part = subparsers.add_parser('add_part', help='Add a new part')
    parser_part.add_argument('name', type=str, help='Name of the new part')
    parser_part.add_argument('--enabled', type=bool, default=True, help='Enable/disable the part')
    parser_part.add_argument('--target', type=str, default='', help='Target part to place before/after')
    parser_part.add_argument('--place_last', type=bool, default=True, help='Place after the target part')

    # Subparser for adding a polyline
    parser_pline = subparsers.add_parser('add_pline', help='Add a new polyline')
    parser_pline.add_argument('points', type=str, help='List of points (x,y,(bulge))')
    parser_pline.add_argument('--closed', type=bool, default=False, help='Whether to close the polyline')
    parser_pline.add_argument('--target', type=str, default='', help='Target layer')
    parser_pline.add_argument('--place_last', type=bool, default=True, help='Place as last object in layer')
    parser_pline.add_argument('--tags', type=str, default='', help='Tags associated with the object')
    parser_pline.add_argument('--groups', type=str, default='', help='Groups to assign the object to')

    # Subparser for adding a circle
    parser_circle = subparsers.add_parser('add_circle', help='Add a new circle')
    parser_circle.add_argument('center_x', type=float, help='Center x coordinate')
    parser_circle.add_argument('center_y', type=float, help='Center y coordinate')
    parser_circle.add_argument('diameter', type=float, help='Circle diameter')
    parser_circle.add_argument('--target', type=str, default='', help='Target layer')
    parser_circle.add_argument('--place_last', type=bool, default=True, help='Place as last object in layer')
    parser_circle.add_argument('--tags', type=str, default='', help='Tags associated with the object')
    parser_circle.add_argument('--groups', type=str, default='', help='Groups to assign the object to')

    # Subparser for saving the file
    parser_save = subparsers.add_parser('save', help='Save the CamBam file')
    parser_save.add_argument('file_path', type=str, help='Full path to the file to save')

    args = parser.parse_args()

    cb = load_cambam_state() if args.command != 'new' else None

    # Execute based on parsed arguments
    if args.command == 'new':
        cb = CamBam(args.name, args.endmill, args.stock_thickness, args.stock_width, args.stock_height, args.spindle_speed)
    elif cb is None:
        print("Error: No CamBam file has been created. Use the 'new' command to create a new file.")
        return
    elif args.command == 'add_layer':
        cb.layer(name=args.name, color=args.color, alpha=args.alpha, pen=args.pen, visible=args.visible,
                 locked=args.locked, target=args.target, place_last=args.place_last)
    elif args.command == 'add_part':
        cb.part(name=args.name, enabled=args.enabled, target=args.target, place_last=args.place_last)
    elif args.command == 'add_pline':
        points = eval(args.points)
        cb.obj_pline(points=points, closed=args.closed, target=args.target, place_last=args.place_last,
                     tags=args.tags, groups=args.groups)
    elif args.command == 'add_circle':
        cb.obj_circle(center_x=args.center_x, center_y=args.center_y, diameter=args.diameter, target=args.target,
                      place_last=args.place_last, tags=args.tags, groups=args.groups)
    elif args.command == 'save':
        cb.save_file(file_path=args.file_path)
        os.remove(STATE_FILE)  # Remove the state file after saving
        return

    save_cambam_state(cb)

if __name__ == '__main__':
    main()
