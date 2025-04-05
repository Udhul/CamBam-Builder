# cad_transformations.py
# 2D CAD Transformation Helpers (NumPy)

import numpy as np
import math
import logging
from typing import Tuple, List, Optional, Union, Sequence

# Logger
logger = logging.getLogger(__name__)

# Tolerance for floating point comparisons, especially in matrix checks
MATRIX_TOLERANCE = 1e-9

def identity_matrix() -> np.ndarray:
    """Returns a 3x3 identity matrix."""
    return np.identity(3, dtype=float)

def is_identity(matrix: np.ndarray, tol: float = MATRIX_TOLERANCE) -> bool:
    """Checks if a matrix is close to the identity matrix."""
    return np.allclose(matrix, np.identity(matrix.shape[0]), atol=tol)

def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """Returns a 3x3 translation matrix."""
    mat = np.identity(3, dtype=float)
    mat[0, 2] = dx
    mat[1, 2] = dy
    return mat

def rotation_matrix_deg(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Returns a 3x3 rotation matrix around point (cx, cy) in degrees."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
    if not np.isclose(cx, 0.0, atol=MATRIX_TOLERANCE) or not np.isclose(cy, 0.0, atol=MATRIX_TOLERANCE):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        return from_origin @ rot_mat @ to_origin
    return rot_mat

def scaling_matrix(sx: float, sy: Optional[float] = None, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Returns a 3x3 scaling/mirroring matrix around point (cx, cy)."""
    sy_val = sy if sy is not None else sx
    scale_mat = np.array([
        [sx, 0,    0],
        [0,  sy_val, 0],
        [0,  0,    1]
    ], dtype=float)
    if not np.isclose(cx, 0.0, atol=MATRIX_TOLERANCE) or not np.isclose(cy, 0.0, atol=MATRIX_TOLERANCE):
        to_origin = translation_matrix(-cx, -cy)
        from_origin = translation_matrix(cx, cy)
        return from_origin @ scale_mat @ to_origin
    return scale_mat

def mirror_x_matrix(cy: float = 0.0) -> np.ndarray:
    """Returns a 3x3 matrix for mirroring across a horizontal line at y=cy."""
    return scaling_matrix(sx=1, sy=-1, cx=0, cy=cy)

def mirror_y_matrix(cx: float = 0.0) -> np.ndarray:
    """Returns a 3x3 matrix for mirroring across a vertical line at x=cx."""
    return scaling_matrix(sx=-1, sy=1, cx=cx, cy=0)

def apply_transform(points: Sequence[Union[Tuple[float, float], np.ndarray]], matrix: np.ndarray) -> List[Tuple[float, float]]:
    """Applies a 3x3 transformation matrix to a list of 2D points."""
    if not points:
        return []
    points_arr = np.array(points, dtype=float)
    if points_arr.ndim == 1: # Single point case
        points_arr = points_arr.reshape(1, -1)
    if points_arr.shape[1] != 2:
        raise ValueError("Input points must be 2D (list of (x, y) tuples or Nx2 array).")

    # Convert points to homogeneous coordinates (Nx3 matrix)
    points_h = np.hstack([points_arr, np.ones((points_arr.shape[0], 1), dtype=float)])

    # Apply transformation (matrix multiplication)
    # Matrix is 3x3, points_h.T is 3xN -> result is 3xN
    transformed_points_h_t = matrix @ points_h.T

    # Transpose back to Nx3
    transformed_points_h = transformed_points_h_t.T

    # Convert back to 2D points (list of tuples), handling perspective division
    result = []
    warnings = []
    for row in transformed_points_h:
        w = row[2]
        if np.isclose(w, 0, atol=MATRIX_TOLERANCE):
            warnings.append("Transformation resulted in point at infinity (w=0). Skipping point.")
            continue # Or handle as an error?
        if not np.isclose(w, 1.0, atol=MATRIX_TOLERANCE):
            warnings.append("Non-affine transformation detected (perspective division required).")
        result.append((row[0] / w, row[1] / w))

    if warnings:
        # Log unique warnings
        for warning in sorted(list(set(warnings))):
             logger.warning(warning)

    return result

def get_transformed_point(point: Tuple[float, float], matrix: np.ndarray) -> Tuple[float, float]:
    """Applies a 3x3 transformation matrix to a single 2D point."""
    res = apply_transform([point], matrix)
    if not res:
        # This should only happen if w was close to 0
        raise ValueError(f"Transformation resulted in invalid point (likely at infinity) for input: {point}")
    return res[0]

# --- Matrix Decomposition (for baking checks) ---

def get_translation(matrix: np.ndarray) -> Tuple[float, float]:
    """Extracts the translation component (dx, dy) from a 3x3 matrix."""
    return (matrix[0, 2], matrix[1, 2])

def get_rotation_deg(matrix: np.ndarray) -> float:
    """Extracts the rotation component in degrees from a 3x3 matrix."""
    # atan2(m10, m00) handles quadrants correctly
    return math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))

def get_scale(matrix: np.ndarray) -> Tuple[float, float]:
    """
    Extracts the scale components (sx, sy) from a 3x3 matrix.
    Handles mirroring (negative scale).
    """
    # Scale is the length of the transformed basis vectors
    sx = np.linalg.norm(matrix[:2, 0]) # Length of first column vector
    sy = np.linalg.norm(matrix[:2, 1]) # Length of second column vector

    # Check determinant sign for mirroring
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if det < 0:
        # Simple heuristic: assume mirroring is along the axis with near-zero cross-term
        # A more robust method might involve SVD if complex shears are present
        if abs(matrix[0,1]) < abs(matrix[1,0]): # If closer to diagonal scaling/y-mirror
             sy = -sy
        else: # Closer to x-mirror
             sx = -sx
    return (sx, sy)

def has_rotation(matrix: np.ndarray, tol_deg: float = 0.1) -> bool:
    """Checks if the matrix includes a rotation component beyond tolerance."""
    # Check if the upper-left 2x2 is close to a scaled identity/mirror matrix
    # More robust check: Compare atan2(m10, m00) and atan2(-m01, m11)
    angle1 = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    angle2 = math.degrees(math.atan2(-matrix[0, 1], matrix[1, 1]))

    # Normalize angles to be comparable (e.g., 0 to 360)
    angle1 = angle1 % 360
    angle2 = angle2 % 360

    # Difference should be close to 0 or 180 if no shear/complex transform
    diff = abs(angle1 - angle2)
    is_rot_or_scale_mirror = np.isclose(diff, 0, atol=tol_deg) or np.isclose(diff, 180, atol=tol_deg)

    # Check if angle1 itself is non-zero (actual rotation)
    is_rotated = not np.isclose(angle1 % 360, 0, atol=tol_deg)

    # It has rotation if it's rotated AND it's not just scale/mirror
    return is_rotated and is_rot_or_scale_mirror

def has_shear(matrix: np.ndarray, tol: float = MATRIX_TOLERANCE) -> bool:
     """Checks if the matrix has a shear component."""
     # Check if transformed axes are orthogonal
     dot_product = np.dot(matrix[:2, 0], matrix[:2, 1])
     return not np.isclose(dot_product, 0.0, atol=tol)


# --- CamBam XML Matrix helpers ---
def to_cambam_4x4_matrix(matrix_3x3: np.ndarray) -> np.ndarray:
    """Converts a 3x3 2D transform matrix to CamBam's 4x4 format."""
    m = matrix_3x3
    # CamBam 4x4 structure (seems column-major in XML string but let's build it logically first)
    # [[ R R 0 T ]]
    # [[ R R 0 T ]]
    # [[ 0 0 1 0 ]]  <- CamBam uses Z=1 ? Check example files. Assuming Z=1 for scale
    # [[ 0 0 0 1 ]]  <- CamBam seems to put T in last *column* usually? But XML is flattened...
    # Let's stick to the common 3D graphics convention where T is last column
    # And convert the user's original 3x3->4x4 logic which seems specific to CamBam's flatten order
    return np.array([
        [m[0, 0], m[1, 0], 0, 0],  # Column 1 (X basis)
        [m[0, 1], m[1, 1], 0, 0],  # Column 2 (Y basis)
        [0,       0,       1, 0],  # Column 3 (Z basis)
        [m[0, 2], m[1, 2], 0, 1]   # Column 4 (Translation) - Note Z=0 translation
    ], dtype=float)

def matrix_to_cambam_string(matrix_3x3: np.ndarray) -> str:
    """Converts a 3x3 matrix to the flattened CamBam 4x4 string format."""
    # CamBam format seems to be column-major flattened.
    # Example: "m11 m21 m31 m41 m12 m22 m32 m42 ..."
    # Using the structure from to_cambam_4x4_matrix:
    mat4x4 = to_cambam_4x4_matrix(matrix_3x3)
    # Flatten in Fortran order (column-major)
    return " ".join(map(str, mat4x4.flatten('F')))

def cambam_string_to_matrix(cambam_matrix_str: str) -> np.ndarray:
    """Converts a CamBam 4x4 matrix string back to a 3x3 transformation matrix."""
    values = [float(val) for val in cambam_matrix_str.split()]
    if len(values) != 16:
        raise ValueError("CamBam matrix string must contain 16 values.")

    # Reconstruct the 4x4 matrix assuming column-major order
    matrix_4x4 = np.array(values).reshape((4, 4), order='F')

    # Extract the relevant 3x3 transformation part
    matrix_3x3 = identity_matrix()
    matrix_3x3[0:2, 0:2] = matrix_4x4[0:2, 0:2] # Top-left 2x2 for rotation/scale
    matrix_3x3[0, 2] = matrix_4x4[0, 3]         # Translation x
    matrix_3x3[1, 2] = matrix_4x4[1, 3]         # Translation y

    return matrix_3x3