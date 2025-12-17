"""Here the rope nearest and below the ankle of person is computed and selected and assigned for each person"""
import os
import json
from typing import List, Optional

import numpy as np

ROPE_DIR = "ropes"


# -------------------------------------------------------------------
# JSON loading utilities
# -------------------------------------------------------------------
def load_ropes_from_config(config, camera_id):
    ropes_section = config.get("ropes", {}) 
    camera_ropes = ropes_section.get(camera_id, [])

    rope_polylines = []
    for rope in camera_ropes:
        arr = np.array(rope, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 2:
            rope_polylines.append(arr)
    return rope_polylines


# -------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------
def interpolate_rope_y(x: float, rope_polyline: np.ndarray) -> float:
    """
    Given an x-coordinate and a rope polyline (Nx2 array of (x, y)),
    return the interpolated y-coordinate of the rope at that x.

    - Polyline points are sorted by x internally.
    - If x lies outside the polyline x-range, the y-value of the nearest
      endpoint is returned (clamped).
    """
    if rope_polyline is None or len(rope_polyline) == 0:
        raise ValueError("rope_polyline is empty")

    pts = np.asarray(rope_polyline, dtype=np.float32)
    # Ensure sorted by x
    xs = pts[:, 0]
    ys = pts[:, 1]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # If all x are identical (degenerate but possible), return that y
    if np.allclose(xs[0], xs[-1]):
        return float(ys[0])

    # Clamp if x is outside the range
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])

    # Find the segment where xs[i-1] <= x <= xs[i]
    idx = np.searchsorted(xs, x)

    x1 = xs[idx - 1]
    y1 = ys[idx - 1]
    x2 = xs[idx]
    y2 = ys[idx]

    if np.isclose(x1, x2):
        # Avoid division by zero if two points have the same x
        return float((y1 + y2) / 2.0)

    # Linear interpolation
    t = (x - x1) / (x2 - x1)
    y = y1 + t * (y2 - y1)
    return float(y)


def interpolate_all_ropes_y(x: float,
                            rope_polylines: List[np.ndarray]) -> List[float]:
    """
    Compute the rope y-value at x for all given rope polylines.

    Returns:
        List of y-values, one per rope. If rope_polylines is empty,
        returns an empty list.
    """
    return [interpolate_rope_y(x, polyline) for polyline in rope_polylines]


def get_nearest_rope_below(point_y: float,
                           rope_y_values: List[float],
                           min_vertical_gap: float = 0.0) -> Optional[int]:
    """
    Given a point's y-coordinate (e.g., ankle_mid_y) and a list of rope y-values
    at the same x-coordinate, find the index of the rope that is:

        - Strictly below the point (rope_y > point_y + min_vertical_gap)
        - Closest in vertical distance (minimum (rope_y - point_y))

    Args:
        point_y:          y-coordinate of the point (e.g., ankle midpoint).
        rope_y_values:    List of rope heights at the relevant x.
        min_vertical_gap: Optional minimum gap (in pixels) to require between
                          the point and the rope to consider it "below".

    Returns:
        Index of the nearest rope below the point, or None if no rope satisfies
        the constraint.
    """
    best_idx: Optional[int] = None
    best_distance: float = float("inf")

    for idx, rope_y in enumerate(rope_y_values):
        vertical_distance = rope_y - point_y  # positive => rope is below point

        # Require rope to be below point (with margin)
        if vertical_distance > min_vertical_gap and vertical_distance < best_distance:
            best_distance = vertical_distance
            best_idx = idx

    return best_idx


# -------------------------------------------------------------------
# Convenience helper tying everything together for assignment
# -------------------------------------------------------------------
def assign_rope_below_point(point_x: float,
                            point_y: float,
                            rope_polylines: List[np.ndarray],
                            min_vertical_gap: float = 0.0) -> Optional[int]:
    """
    Convenience function for typical usage in fall detection:

    1. For a given point (point_x, point_y), compute rope y-values at point_x
       for all ropes.
    2. Select nearest rope below the point.

    This is typically used with:
        - point = ankle midpoint (for rope assignment)

    Args:
        point_x:         x-coordinate of the point (e.g., ankle midpoint x).
        point_y:         y-coordinate of the point (e.g., ankle midpoint y).
        rope_polylines:  List of rope polylines for the camera.
        min_vertical_gap:Minimum distance (pixels) required for a rope
                         to be considered "below" the point.

    Returns:
        Index of assigned rope, or None if no rope below.
    """
    if not rope_polylines:
        return None

    rope_y_values = interpolate_all_ropes_y(point_x, rope_polylines)
    assigned_idx = get_nearest_rope_below(point_y, rope_y_values, min_vertical_gap)
    return assigned_idx
