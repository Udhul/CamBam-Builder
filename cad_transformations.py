"""
cad_transformations.py

2D CAD Transformation Helpers using NumPy.
Provides functions to create transformation matrices (identity, translation, rotation,
scaling, mirroring, skewing) and to apply them to point data. Also includes functions
to convert to/from CamBam’s expected 4x4 matrix string format.
"""

import numpy as np
import math
import logging
from typing import Tuple, List, Optional, Union, Sequence

logger = logging.getLogger(__name__)

# --- Matrix Creation Functions ---

def identity_matrix() -> np.ndarray:
    """Return a 3x3 identity matrix."""
    return np.identity(3, dtype=float)

def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """Return a 3x3 translation matrix for translating by (dx, dy)."""
    mat = np.identity(3, dtype=float)
    mat[0, 2] = dx
    mat[1, 2] = dy
    return mat

def rotation_matrix_deg(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """
    Return a 3x3 rotation matrix for rotating by angle_deg (degrees) about point (cx, cy).
    Positive angles rotate counterclockwise.
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
    # If center is not origin, translate to origin, rotate, translate back
    if not (math.isclose(cx, 0.0) and math.isclose(cy, 0.0)):
        return translation_matrix(cx, cy) @ rot_mat @ translation_matrix(-cx, -cy)
    return rot_mat

def rotation_matrix_rad(angle_rad: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Return a 3x3 rotation matrix using radians, rotating about point (cx, cy)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
    # If center is not origin, translate to origin, rotate, translate back
    if not (math.isclose(cx, 0.0) and math.isclose(cy, 0.0)):
        return translation_matrix(cx, cy) @ rot_mat @ translation_matrix(-cx, -cy)
    return rot_mat

def scale_matrix(sx: float, sy: Optional[float] = None, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """
    Return a 3x3 scaling matrix.
    If sy is None, uniform scaling (sx) is applied.
    Scaling is performed about the point (cx, cy) if provided.
    """
    if sy is None:
        sy = sx
    scale_mat = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=float)
    # If center is not origin, translate to origin, scale, translate back
    if not (math.isclose(cx, 0.0) and math.isclose(cy, 0.0)):
        return translation_matrix(cx, cy) @ scale_mat @ translation_matrix(-cx, -cy)
    return scale_mat

def mirror_x_matrix(cy: float = 0.0) -> np.ndarray:
    """Return a 3x3 matrix to mirror across the horizontal line at y=cy."""
    # Equivalent to scaling by (1, -1) around the point (0, cy)
    return scale_matrix(1, -1, 0, cy)

def mirror_y_matrix(cx: float = 0.0) -> np.ndarray:
    """Return a 3x3 matrix to mirror across the vertical line at x=cx."""
    # Equivalent to scaling by (-1, 1) around the point (cx, 0)
    return scale_matrix(-1, 1, cx, 0)

def skew_matrix(angle_x_deg: float = 0.0, angle_y_deg: float = 0.0) -> np.ndarray:
    """Return a 3x3 skew (shear) matrix with the given angles (in degrees)."""
    tan_x = math.tan(math.radians(angle_x_deg)) # Shear parallel to x-axis based on y
    tan_y = math.tan(math.radians(angle_y_deg)) # Shear parallel to y-axis based on x
    return np.array([
        [1,     tan_y, 0], # y-shear affects x
        [tan_x, 1,     0], # x-shear affects y
        [0,     0,     1]
    ], dtype=float)

def combine_transformations(*matrices: np.ndarray) -> np.ndarray:
    """
    Combine multiple 3x3 transformation matrices.
    Transformations are applied in the order given (left to right multiplication).
    """
    result = identity_matrix()
    for m in matrices:
        result = result @ m
    return result

# --- Point Transformation ---

def apply_transform(points: Sequence[Union[Tuple[float, float], np.ndarray]], matrix: np.ndarray) -> List[Tuple[float, float]]:
    """
    Apply a 3x3 transformation matrix to a sequence of 2D points.
    Points are assumed to be given as (x, y) pairs.
    """
    if not points:
        return []
    pts = np.ones((len(points), 3), dtype=float)
    for i, p in enumerate(points):
        pts[i, 0], pts[i, 1] = p[0], p[1]
    transformed = (matrix @ pts.T).T
    result = []
    for row in transformed:
        w = row[2]
        if math.isclose(w, 0.0):
            logger.error("Transformation produced a point at infinity (w=0); skipping.")
            continue
        result.append((row[0] / w, row[1] / w))
    if not all(math.isclose(row[2], 1.0) for row in (matrix @ pts.T).T):
        logger.warning("Non-affine transformation detected (perspective division != 1).")
    return result

def get_transformed_point(point: Tuple[float, float], matrix: np.ndarray) -> Tuple[float, float]:
    """
    Apply a 3x3 transformation matrix to a single 2D point.
    """
    res = apply_transform([point], matrix)
    if not res:
        raise ValueError(f"Invalid transformation for point: {point}")
    return res[0]

# --- CamBam Matrix Format Conversion ---

# TODO: Revise to both functions follow cb mat forrmat, putting tx, ty, tz in bottom row
# TEST them!
def to_cambam_matrix_str(matrix_3x3: np.ndarray) -> str:
    """
    Convert a 3x3 transformation matrix to a CamBam XML 4x4 matrix string.
    The CamBam format expects a 4x4 matrix in a particular column-major order,
    where the translations are registered in the bottom row [..., [tx, ty, tz, 1]].
    The output is a string for of the flattened matrix, listing one row at a time,
    from the top down, separating the numbers with a space. 
    Example: 'sx 0 0 0 0 sy 0 0 0 0 sz 0 tx ty tz 1'
    """
    tx = matrix_3x3[0, 2]
    ty = matrix_3x3[1, 2]
    tz = 0.0  # Z translation is zero for 2D
    cambam_matrix = np.identity(4, dtype=float)
    cambam_matrix[0:2, 0:2] = matrix_3x3[0:2, 0:2]
    cambam_matrix[0, 3] = tx
    cambam_matrix[1, 3] = ty
    cambam_matrix[2, 3] = tz
    # Flatten in column-major order
    flat = []
    for col in range(4):
        for row in range(4):
            flat.append(str(cambam_matrix[row, col]))
    return " ".join(flat)


def from_cambam_matrix_str(cambam_matrix_str: str) -> np.ndarray:
    """
    Convert a CamBam XML 4x4 matrix string back to a 3x3 transformation matrix,
    from CamBams unique matrix format.
    """
    matrix_3x3 = np.identity(3, dtype=float)

    # If registered as "Identity", just return an identity matrix
    if cambam_matrix_str.lower() == "identity":
        return matrix_3x3
    
    # Else, convert the string to a 4x4 matrix and encode it in a 3x3 matrix to be returned
    values = [float(v) for v in cambam_matrix_str.split()]
    matrix_4x4 = np.zeros((4,4), dtype=float)
    for col in range(4):
        for row in range(4):
            matrix_4x4[row, col] = values[col*4 + row]

    matrix_3x3[0:2, 0:2] = matrix_4x4[0:2, 0:2]
    matrix_3x3[0, 2] = matrix_4x4[0, 3]
    matrix_3x3[1, 2] = matrix_4x4[1, 3]
    return matrix_3x3

