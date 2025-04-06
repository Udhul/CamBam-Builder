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
    if not (math.isclose(cx, 0.0) and math.isclose(cy, 0.0)):
        return translation_matrix(cx, cy) @ rot_mat @ translation_matrix(-cx, -cy)
    return rot_mat

def rotation_matrix_rad(angle_rad: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Return a 3x3 rotation matrix using radians."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=float)
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
    if not (math.isclose(cx, 0.0) and math.isclose(cy, 0.0)):
        return translation_matrix(cx, cy) @ scale_mat @ translation_matrix(-cx, -cy)
    return scale_mat

def mirror_x_matrix(cy: float = 0.0) -> np.ndarray:
    """Return a 3x3 matrix to mirror across the horizontal line at y=cy."""
    return scale_matrix(1, -1, 0, cy)

def mirror_y_matrix(cx: float = 0.0) -> np.ndarray:
    """Return a 3x3 matrix to mirror across the vertical line at x=cx."""
    return scale_matrix(-1, 1, cx, 0)

def skew_matrix(angle_x_deg: float = 0.0, angle_y_deg: float = 0.0) -> np.ndarray:
    """Return a 3x3 skew (shear) matrix with the given angles (in degrees)."""
    tan_x = math.tan(math.radians(angle_x_deg))
    tan_y = math.tan(math.radians(angle_y_deg))
    return np.array([
        [1,     tan_y, 0],
        [tan_x, 1,     0],
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

def to_4x4_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 matrix to a 4x4 matrix.
    The 3x3 is embedded into the upper-left; remaining elements come from the identity.
    """
    result = np.identity(4, dtype=float)
    rows = min(matrix.shape[0], 3)
    cols = min(matrix.shape[1], 3)
    result[:rows, :cols] = matrix[:rows, :cols]
    # Place translation values into the last column (except bottom-right element)
    result[0, 3] = matrix[0, 2]
    result[1, 3] = matrix[1, 2]
    return result

def to_cambam_matrix(matrix_3x3: np.ndarray) -> str:
    """
    Convert a 3x3 transformation matrix to a CamBam XML 4x4 matrix string.
    The CamBam format expects a 4x4 matrix in column-major order.
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

def from_cambam_matrix(cambam_matrix_str: str) -> np.ndarray:
    """
    Convert a CamBam XML 4x4 matrix string back to a 3x3 transformation matrix.
    """
    values = [float(v) for v in cambam_matrix_str.split()]
    matrix_4x4 = np.zeros((4,4), dtype=float)
    for col in range(4):
        for row in range(4):
            matrix_4x4[row, col] = values[col*4 + row]
    matrix_3x3 = np.identity(3, dtype=float)
    matrix_3x3[0:2, 0:2] = matrix_4x4[0:2, 0:2]
    matrix_3x3[0, 2] = matrix_4x4[0, 3]
    matrix_3x3[1, 2] = matrix_4x4[1, 3]
    return matrix_3x3

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
