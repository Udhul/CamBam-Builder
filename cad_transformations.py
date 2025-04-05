# --- 2D CAD Transformation Helpers (NumPy) ---

import numpy as np
import math
import logging
from typing import Tuple, List, Optional, Union, Sequence

# Logger
logger = logging.getLogger(__name__)

def identity_matrix() -> np.ndarray:
    """
    Returns a 3x3 identity matrix.
    
    Returns:
        A 3x3 identity matrix representing no transformation.
    """
    return np.identity(3, dtype=float)

def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """
    Returns a 3x3 translation matrix.
    
    Args:
        dx: Translation distance along X axis
        dy: Translation distance along Y axis
        
    Returns:
        A 3x3 matrix representing the translation.
    """
    mat = np.identity(3, dtype=float)
    mat[0, 2] = dx
    mat[1, 2] = dy
    return mat

def rotation_matrix_deg(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix around point (cx, cy).
    
    Args:
        angle_deg: Rotation angle in degrees (positive is counterclockwise)
        cx: X coordinate of rotation center (default: 0)
        cy: Y coordinate of rotation center (default: 0)
        
    Returns:
        A 3x3 matrix representing the rotation.
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Rotation around origin matrix
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
    
    # If rotating around a point other than origin, translate to origin, rotate, translate back
    if not np.isclose(cx, 0.0) or not np.isclose(cy, 0.0):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        # Order: Translate to origin, rotate, translate back
        return from_origin @ rot_mat @ to_origin
    
    return rot_mat

def rotation_matrix_rad(angle_rad: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix around point (cx, cy) using radians.
    
    Args:
        angle_rad: Rotation angle in radians (positive is counterclockwise)
        cx: X coordinate of rotation center (default: 0)
        cy: Y coordinate of rotation center (default: 0)
        
    Returns:
        A 3x3 matrix representing the rotation.
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Rotation around origin matrix
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
    
    # If rotating around a point other than origin, translate to origin, rotate, translate back
    if not np.isclose(cx, 0.0) or not np.isclose(cy, 0.0):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        # Order: Translate to origin, rotate, translate back
        return from_origin @ rot_mat @ to_origin
    
    return rot_mat

def scale_matrix(sx: float, sy: Optional[float] = None, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 scaling matrix.
    
    Args:
        sx: Scale factor for X axis
        sy: Scale factor for Y axis (defaults to sx if None for uniform scaling)
        cx: X coordinate of scaling center (default: 0)
        cy: Y coordinate of scaling center (default: 0)
        
    Returns:
        A 3x3 matrix representing the scaling.
    """
    if sy is None:
        sy = sx
        
    # Basic scaling matrix
    scale_mat = np.array([
        [sx,  0,  0],
        [0,  sy,  0],
        [0,   0,  1]
    ], dtype=float)
    
    # If scaling around a point other than origin, translate to origin, scale, translate back
    if not np.isclose(cx, 0.0) or not np.isclose(cy, 0.0):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        # Order: Translate to origin, scale, translate back
        return from_origin @ scale_mat @ to_origin
    
    return scale_mat

def mirror_x_matrix(cy: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 matrix for mirroring across the X axis (or a horizontal line at y=cy).
    
    Args:
        cy: Y coordinate of the horizontal mirror line (default: 0 for X axis)
        
    Returns:
        A 3x3 matrix representing the mirror transformation.
    """
    return scale_matrix(1, -1, 0, cy)

def mirror_y_matrix(cx: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 matrix for mirroring across the Y axis (or a vertical line at x=cx).
    
    Args:
        cx: X coordinate of the vertical mirror line (default: 0 for Y axis)
        
    Returns:
        A 3x3 matrix representing the mirror transformation.
    """
    return scale_matrix(-1, 1, cx, 0)

def skew_matrix(angle_x_deg: float = 0.0, angle_y_deg: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 skew/shear matrix.
    
    Args:
        angle_x_deg: Skew angle in degrees for X axis
        angle_y_deg: Skew angle in degrees for Y axis
        
    Returns:
        A 3x3 matrix representing the skew transformation.
    """
    tan_x = math.tan(math.radians(angle_x_deg))
    tan_y = math.tan(math.radians(angle_y_deg))
    
    return np.array([
        [1,     tan_y, 0],
        [tan_x, 1,     0],
        [0,     0,     1]
    ], dtype=float)

def combine_transformations(*matrices: np.ndarray) -> np.ndarray:
    """
    Combine multiple transformation matrices into a single matrix.
    Transformations are applied in the order they are provided (left to right).
    
    Args:
        *matrices: Variable number of 3x3 transformation matrices
        
    Returns:
        A single 3x3 matrix representing the combined transformation.
    """
    if not matrices:
        return identity_matrix()
    
    result = matrices[0]
    for matrix in matrices[1:]:
        result = result @ matrix
    
    return result


# --- Matrix shape ---
def to_4x4_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a matrix of any size to a 4x4 matrix.
    
    If the input matrix is smaller than 4x4, it will be padded with identity matrix values.
    If the input matrix is larger than 4x4, it will be truncated.
    
    Args:
        matrix: NumPy matrix of any size
        
    Returns:
        A 4x4 NumPy matrix
    """
    result = np.identity(4, dtype=float)
    
    # Get the dimensions to copy (limited to 4x4)
    rows = min(matrix.shape[0], 4)
    cols = min(matrix.shape[1], 4)
    
    # Copy the values from the input matrix
    result[:rows, :cols] = matrix[:rows, :cols]
    
    return result

# --- Apply Transformations ---
def apply_transform(points: Sequence[Union[Tuple[float, float], np.ndarray]], matrix: np.ndarray) -> List[Tuple[float, float]]:
    """Applies a 3x3 transformation matrix to a list of 2D points."""
    if not points: 
        return []
    # Convert points to homogeneous coordinates (nx3 matrix)
    points_h = np.ones((len(points), 3), dtype=float)
    for i, p in enumerate(points):
        points_h[i, 0] = p[0]
        points_h[i, 1] = p[1]

    # Apply transformation (matrix multiplication)
    transformed_points_h = (matrix @ points_h.T).T

    # Convert back to 2D points (list of tuples), handling potential perspective division
    result = []
    for row in transformed_points_h:
        w = row[2]
        if np.isclose(w, 0):
            logger.error("Transformation resulted in point at infinity (w=0). Skipping point.")
            continue
        result.append((row[0] / w, row[1] / w))
    if not np.allclose([row[2] for row in transformed_points_h if not np.isclose(row[2], 0)], 1.0):
         logger.warning("Non-affine transformation detected (perspective division != 1).")
    return result

def get_transformed_point(point: Tuple[float, float], matrix: np.ndarray) -> Tuple[float, float]:
    """Applies a 3x3 transformation matrix to a single 2D point."""
    res = apply_transform([point], matrix)
    if not res:
        raise ValueError(f"Transformation resulted in invalid point for input: {point}")
    return res[0]

# --- CamBam XML Matrix helpers ---
def to_cambam_matrix(matrix_3x3: np.ndarray) -> str:
    """
    Convert a NumPy 3x3 transformation matrix to CamBam's 4x4 matrix string format.
    
    In CamBam's XML format, the 4x4 matrix is represented in column-major order with
    translation values in the last row (positions 12, 13, 14 in the flattened array).
    
    Args:
        matrix_3x3: 3x3 NumPy transformation matrix where translation is in the
                   rightmost column (positions [0,2], [1,2])
        
    Returns:
        String representation for CamBam XML in the format:
        "r11 r21 r31 0 r12 r22 r32 0 r13 r23 r33 0 tx ty tz 1"
    """
    # Extract translation values from the input matrix
    tx = matrix_3x3[0, 2]
    ty = matrix_3x3[1, 2]
    tz = 0.0  # Z translation is 0 in a 2D context
    
    # Create a 4x4 matrix for CamBam format
    cambam_matrix = np.identity(4, dtype=float)
    
    # Copy the rotation part (2x2 upper-left submatrix)
    cambam_matrix[0:2, 0:2] = matrix_3x3[0:2, 0:2]
    
    # Set the translation values in the last row
    cambam_matrix[0, 3] = tx
    cambam_matrix[1, 3] = ty
    cambam_matrix[2, 3] = tz
    
    # Flatten in column-major order
    result = []
    for col in range(4):
        for row in range(4):
            result.append(str(cambam_matrix[row, col]))
    
    return " ".join(result)

def from_cambam_matrix(cambam_matrix_str: str) -> np.ndarray:
    """
    Convert a CamBam 4x4 matrix string to a NumPy 3x3 transformation matrix.
    
    Args:
        cambam_matrix_str: CamBam matrix string in the format:
                          "r11 r21 r31 0 r12 r22 r32 0 r13 r23 r33 0 tx ty tz 1"
        
    Returns:
        A 3x3 NumPy transformation matrix.
    """
    # Parse the string into a list of floats
    values = [float(val) for val in cambam_matrix_str.split()]
    
    # Create a 4x4 matrix from the column-major values
    matrix_4x4 = np.zeros((4, 4), dtype=float)
    for col in range(4):
        for row in range(4):
            matrix_4x4[row, col] = values[col * 4 + row]
    
    # Extract the 3x3 matrix
    matrix_3x3 = np.identity(3, dtype=float)
    matrix_3x3[0:2, 0:2] = matrix_4x4[0:2, 0:2]  # Copy rotation/scale part
    
    # Copy translation values
    matrix_3x3[0, 2] = matrix_4x4[0, 3]  # tx
    matrix_3x3[1, 2] = matrix_4x4[1, 3]  # ty
    
    return matrix_3x3


# --- Test ---

def extract_transformation_components(matrix: np.ndarray) -> dict:
    """
    Extract transformation components (translation, rotation, scale) from a 3x3 matrix.
    
    Args:
        matrix: 3x3 NumPy transformation matrix
        
    Returns:
        Dictionary containing transformation components:
        {
            'translation': (tx, ty),
            'rotation_deg': angle in degrees,
            'scale': (sx, sy)
        }
    """
    # Extract translation
    tx, ty = matrix[0, 2], matrix[1, 2]
    
    # Extract rotation (atan2 of the first column gives rotation angle)
    rotation_rad = math.atan2(matrix[1, 0], matrix[0, 0])
    rotation_deg = math.degrees(rotation_rad)
    
    # Extract scale (length of the first two column vectors)
    scale_x = math.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
    scale_y = math.sqrt(matrix[0, 1]**2 + matrix[1, 1]**2)
    
    # Check for negative scaling (mirroring)
    if matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0] < 0:
        if abs(matrix[0, 0]) > abs(matrix[1, 0]):  # Mirror across Y
            scale_x = -scale_x
        else:  # Mirror across X
            scale_y = -scale_y
    
    return {
        'translation': (tx, ty),
        'rotation_deg': rotation_deg,
        'scale': (scale_x, scale_y)
    }

def print_matrix_info(name: str, matrix: np.ndarray):
    """
    Helper function to print matrix information for debugging.
    
    Args:
        name: Name or description of the matrix
        matrix: 3x3 NumPy transformation matrix
    """
    print(f"\n{name}:")
    print(matrix)
    print(f"CamBam format: {to_cambam_matrix(matrix)}")
    
    components = extract_transformation_components(matrix)
    
    print(f"Translation: ({components['translation'][0]}, {components['translation'][1]})")
    print(f"Rotation: {components['rotation_deg']:.2f}°")
    print(f"Scale: ({components['scale'][0]:.2f}, {components['scale'][1]:.2f})")


# Example:
if __name__ == "__main__":

    mat = identity_matrix()
    mat = mat @ translation_matrix(25, 50)
    mat = mat @ mirror_x_matrix(100/2) # Should use the center point before translation *original geometry
    # mat = mat @ mirror_y_matrix(100/2)

    print_matrix_info("Final", mat)

    mat_4x4 = to_4x4_matrix(mat)
    print("\nmat 4x4: \n", mat_4x4)

    print("\nmat 4x4 transposed: \n", mat_4x4.T)


    # print("=== CAD TRANSFORMATION MATRIX TEST ===")
    
    # # Start with identity matrix
    # mat = identity_matrix()
    # print_matrix_info("Identity Matrix", mat)
    
    # # Apply translation
    # mat_translated = mat @ translation_matrix(25, 50)
    # print_matrix_info("After Translation (25, 50)", mat_translated)
    
    # # Apply rotation
    # mat_rotated = mat_translated @ rotation_matrix_deg(45)
    # print_matrix_info("After Rotation (45°)", mat_rotated)
    
    # # Apply scaling
    # mat_scaled = mat_rotated @ scale_matrix(2, 1.5)
    # print_matrix_info("After Scaling (2x, 1.5y)", mat_scaled)
    
    # # Apply mirroring across X axis
    # mat_mirrored_x = mat_scaled @ mirror_x_matrix()
    # print_matrix_info("After Mirror X", mat_mirrored_x)
    
    # # Apply mirroring across Y axis
    # mat_mirrored_y = mat_scaled @ mirror_y_matrix()
    # print_matrix_info("After Mirror Y", mat_mirrored_y)
    
    # # Apply skew
    # mat_skewed = mat_scaled @ skew_matrix(15, 10)
    # print_matrix_info("After Skew (15° X, 10° Y)", mat_skewed)
    
    # # Complex transformation: rotate around a specific point
    # center_rotation = rotation_matrix_deg(30, 100, 100)
    # mat_center_rotated = mat_translated @ center_rotation
    # print_matrix_info("Rotation around point (100, 100)", mat_center_rotated)
    
    # # Final CamBam matrix string
    # final_matrix = mat_scaled  # Choose which transformation to use as final
    # cb_mat = to_cambam_matrix(final_matrix)
    # print("\nFinal CamBam Matrix String:")
    # print(cb_mat)
    # print("\nXML format:")
    # print(f'<mat m="{cb_mat}" />')