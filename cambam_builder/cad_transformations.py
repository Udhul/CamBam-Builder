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
    Returns matrices[0] @ matrices[1] @ ... . With column-vector points,
    the rightmost transformation acts first.
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

# --- Single Transformation Component ---

def extract_transform_component(matrix: np.ndarray, component_type: str) -> np.ndarray:
    """
    Extracts a specific component from a transformation matrix.
    
    Args:
        matrix: The 3x3 transformation matrix
        component_type: The type of transformation to extract ('translation', 'rotation', 'scale', 'mirror')
        
    Returns:
        A 3x3 matrix containing only the requested component
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got {matrix.shape}")
    
    result = identity_matrix()
    
    if component_type == 'translation':
        # Extract only the translation component
        result[0, 2] = matrix[0, 2]
        result[1, 2] = matrix[1, 2]
    elif component_type == 'rotation':
        # Extract rotation (ignoring scale)
        # First, calculate the scale factors
        sx = np.linalg.norm(matrix[0:2, 0])
        sy = np.linalg.norm(matrix[0:2, 1])
        
        # Remove scale to get pure rotation
        if not (math.isclose(sx, 0.0) or math.isclose(sy, 0.0)):
            # Normalized rotation component
            result[0:2, 0:2] = matrix[0:2, 0:2].copy()
            result[0, 0] /= sx
            result[0, 1] /= sy
            result[1, 0] /= sx
            result[1, 1] /= sy
    elif component_type == 'scale':
        # Extract only the scale component (diagonal)
        sx = np.linalg.norm(matrix[0:2, 0])
        sy = np.linalg.norm(matrix[0:2, 1])
        
        # Apply scale to identity matrix
        result[0, 0] = sx
        result[1, 1] = sy
    elif component_type == 'mirror_x':
        # Extract only x-mirror (check if determinant is negative and y-scale is positive)
        if np.linalg.det(matrix[0:2, 0:2]) < 0:
            det_sign = np.sign(np.linalg.det(matrix[0:2, 0:2]))
            sy_sign = np.sign(np.linalg.norm(matrix[0:2, 1]))
            
            if det_sign * sy_sign < 0:  # X-mirror
                result[0, 0] = -1.0
    elif component_type == 'mirror_y':
        # Extract only y-mirror (check if determinant is negative and x-scale is positive)
        if np.linalg.det(matrix[0:2, 0:2]) < 0:
            det_sign = np.sign(np.linalg.det(matrix[0:2, 0:2]))
            sx_sign = np.sign(np.linalg.norm(matrix[0:2, 0]))
            
            if det_sign * sx_sign < 0:  # Y-mirror
                result[1, 1] = -1.0
    else:
        raise ValueError(f"Unknown component_type: {component_type}")
    
    return result

def remove_transform_component(matrix: np.ndarray, component_type: str) -> np.ndarray:
    """
    Removes a specific component from a transformation matrix.
    
    Args:
        matrix: The 3x3 transformation matrix
        component_type: The type of transformation to remove ('translation', 'rotation', 'scale', 'mirror')
        
    Returns:
        A 3x3 matrix with the specified component removed
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Matrix must be 3x3, got {matrix.shape}")
        
    result = matrix.copy()
    
    if component_type == 'translation':
        # Remove translation
        result[0, 2] = 0.0
        result[1, 2] = 0.0
    elif component_type == 'rotation':
        # Remove rotation while preserving scale and translation
        # Extract scale
        sx = np.linalg.norm(matrix[0:2, 0])
        sy = np.linalg.norm(matrix[0:2, 1])
        
        # Apply scale without rotation
        result[0:2, 0:2] = np.array([[sx, 0], [0, sy]])
    elif component_type in ('mirror_x', 'mirror_y', 'mirror'):
        # Convert any negative scale to positive (remove mirroring)
        sx = np.linalg.norm(matrix[0:2, 0])
        sy = np.linalg.norm(matrix[0:2, 1])
        
        det = np.linalg.det(matrix[0:2, 0:2])
        if det < 0:  # There's a mirror
            if component_type == 'mirror' or component_type == 'mirror_x':
                # Remove x-mirror
                result[0, 0] = abs(result[0, 0])
                result[1, 0] = abs(result[1, 0])
            if component_type == 'mirror' or component_type == 'mirror_y':
                # Remove y-mirror
                result[0, 1] = abs(result[0, 1])
                result[1, 1] = abs(result[1, 1])
    else:
        raise ValueError(f"Unknown component_type: {component_type}")
    
    return result

# --- CamBam Matrix Format Conversion ---

# CamBam's flat column-major pose stores translation at offsets 12, 13 and 14.
def to_cambam_matrix_str(matrix_3x3: np.ndarray, output_decimals: Optional[int] = None, *, z_offset: float = 0.0) -> str:
    """
    Convert a 3x3 transformation matrix to a CamBam XML 4x4 matrix string.
    The CamBam format expects a 4x4 matrix in a particular column-major order,
    where the translations are registered in the bottom row [..., [tx, ty, tz, 1]].
    The output is a string for of the flattened matrix, listing one row at a time,
    from the top down, separating the numbers with a space. 
    Example: 'sx 0 0 0 0 sy 0 0 0 0 sz 0 tx ty tz 1'
    """
    if np.iscomplexobj(matrix_3x3):
        raise ValueError("Expected a real XY affine matrix")
    matrix_3x3 = np.asarray(matrix_3x3, dtype=float)
    if (matrix_3x3.shape != (3, 3) or not np.isfinite(matrix_3x3).all()
            or not np.array_equal(matrix_3x3[2], [0., 0., 1.])
            or not math.isfinite(z_offset)):
        raise ValueError("Expected a finite XY affine matrix and Z translation")
    tx = matrix_3x3[0, 2]
    ty = matrix_3x3[1, 2]
    tz = z_offset
    cambam_matrix = np.identity(4, dtype=float)
    cambam_matrix[0:2, 0:2] = matrix_3x3[0:2, 0:2]
    cambam_matrix[0, 3] = tx
    cambam_matrix[1, 3] = ty
    cambam_matrix[2, 3] = tz

    # Flatten in column-major order
    flat = []
    for col in range(4):
        for row in range(4):
            if output_decimals is not None:
                flat.append(str(round(cambam_matrix[row, col], output_decimals)))
            else:
                flat.append(str(cambam_matrix[row, col]))
    return " ".join(flat)


def from_cambam_matrix_str(cambam_matrix_str: str, *, return_z: bool = False):
    """
    Decode the supported XY affine pose and optional independent Z translation.

    With return_z=True return (xy_matrix, z_offset). The XY-only form rejects
    nonzero Z translation. Spatial mixing, Z scaling and perspective raise.
    """
    matrix_3x3 = np.identity(3, dtype=float)

    # If registered as "Identity", just return an identity matrix
    if cambam_matrix_str.strip().lower() == "identity":
        return (matrix_3x3, 0.0) if return_z else matrix_3x3
    
    # Else, convert the string to a 4x4 matrix and encode it in a 3x3 matrix to be returned
    values = [float(v) for v in cambam_matrix_str.split()]
    if len(values) != 16 or not np.isfinite(values).all():
        raise ValueError("CamBam matrix must contain 16 finite values")
    matrix_4x4 = np.zeros((4,4), dtype=float)
    for col in range(4):
        for row in range(4):
            matrix_4x4[row, col] = values[col*4 + row]

    if (not np.array_equal(matrix_4x4[3], [0., 0., 0., 1.])
            or not np.array_equal(matrix_4x4[:3, 2], [0., 0., 1.])
            or not np.array_equal(matrix_4x4[2, :2], [0., 0.])):
        raise ValueError("Unsupported matrix: only XY affine transforms and Z translation are supported")
    z_offset = float(matrix_4x4[2, 3])
    if z_offset and not return_z:
        raise ValueError("Z translation requires return_z=True; refusing to discard elevation")

    matrix_3x3[0:2, 0:2] = matrix_4x4[0:2, 0:2]
    matrix_3x3[0, 2] = matrix_4x4[0, 3]
    matrix_3x3[1, 2] = matrix_4x4[1, 3]
    return (matrix_3x3, z_offset) if return_z else matrix_3x3




def to_cambam_matrix_str_v2(matrix_3x3: np.ndarray, *, z_offset: float = 0.0) -> str:
    """Compatibility alias for the canonical CamBam matrix encoder."""
    return to_cambam_matrix_str(matrix_3x3, z_offset=z_offset)


def from_cambam_matrix_str_v2(cambam_matrix_str: str, *, return_z: bool = False):
    """Compatibility alias; unsupported spatial/perspective forms fail explicitly."""
    return from_cambam_matrix_str(cambam_matrix_str, return_z=return_z)


if __name__ == "__main__":
    # Test
    a = np.identity(3)
    a = a @ rotation_matrix_deg(35)
    a = a @ scale_matrix(0.7, 2.1)
    a = a @ translation_matrix(0.2, 1.2)

    a[0,2] = 0.2
    a[1,2] = 1.2
    print(a)

    b1 = to_cambam_matrix_str(a)
    print(b1)

    b2 = to_cambam_matrix_str_v2(a)
    print(b2)

    print(from_cambam_matrix_str(b1))
