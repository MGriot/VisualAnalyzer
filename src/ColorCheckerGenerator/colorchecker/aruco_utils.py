"""
aruco_utils.py

Helpers to create/generate ArUco markers and embed them around a target image.
"""

import cv2
import numpy as np
from typing import Tuple
from .config import DEFAULT_MARKER_DICT, DEFAULT_ARUCO_SIZE_PX

def make_aruco_marker(id_: int, marker_size: int = DEFAULT_ARUCO_SIZE_PX, dict_name: str = DEFAULT_MARKER_DICT) -> np.ndarray:
    """
    Create a single ArUco marker as BGR image.
    """
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    }
    aruco_id = dict_map.get(dict_name, cv2.aruco.DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_id)
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, id_ % 50, marker_size, marker, 1)
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

def add_aruco_markers_around(image: np.ndarray, marker_size:int = DEFAULT_ARUCO_SIZE_PX, margin_px:int = 40) -> np.ndarray:
    """
    Place 4 markers (top-left, top-right, bottom-left, bottom-right) outside the rectangle.
    Returns a new canvas with markers and margin.
    """
    h, w = image.shape[:2]
    canvas_h = h + 2*margin_px + marker_size
    canvas_w = w + 2*margin_px + marker_size
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    offset = margin_px + marker_size // 2
    canvas[offset:offset+h, offset:offset+w] = image

    # create markers
    markers = [make_aruco_marker(i, marker_size) for i in range(4)]
    # top-left
    canvas[margin_px:margin_px+marker_size, margin_px:margin_px+marker_size] = markers[0]
    # top-right
    canvas[margin_px:margin_px+marker_size, -margin_px-marker_size:-margin_px] = markers[1]
    # bottom-left
    canvas[-margin_px-marker_size:-margin_px, margin_px:margin_px+marker_size] = markers[2]
    # bottom-right
    canvas[-margin_px-marker_size:-margin_px, -margin_px-marker_size:-margin_px] = markers[3]

    return canvas
