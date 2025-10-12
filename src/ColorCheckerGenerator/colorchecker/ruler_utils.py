"""
ruler_utils.py

Functions to draw centimeter and inch rulers as OpenCV images.
"""

import numpy as np
import cv2
from typing import Literal
from .config import DEFAULT_DPI
from .drawing_utils import put_text_cv2

def generate_ruler_dual(length_px: int, dpi: int = DEFAULT_DPI, thickness: int = 100, mode: Literal["cm","inch","both"]="both"):
    """
    Generate a ruler image (numpy array) of width length_px and height thickness.
    mode selects "cm", "inch", or "both" (centimeter scale on top, inch on bottom).
    """
    # Start with white canvas
    ruler = np.ones((thickness, length_px, 3), dtype=np.uint8) * 255
    color = (0,0,0)

    if mode in ("both", "cm"):
        # cm parameters
        cm_per_px = 2.54 / dpi
        px_per_cm = 1 / cm_per_px
        # top 40% reserved for cm ruler
        top_h = thickness // 2
        for cm in range(0, int((length_px / px_per_cm)) + 1):
            x = int(round(cm * px_per_cm))
            tick = int(top_h * 0.6) if cm % 5 == 0 else int(top_h * 0.35)
            cv2.line(ruler, (x, 0), (x, tick), color, 2) # Increased thickness to 2
            if cm % 5 == 0:
                ruler = put_text_cv2(ruler, f"{cm}", (x+2, tick+12), font_size=16, color=color)
        ruler = put_text_cv2(ruler, "cm", (5, 14), font_size=20, color=color)

    if mode in ("both", "inch"):
        # inch parameters
        px_per_in = dpi
        bottom_h = thickness // 2
        # We'll draw ticks for each 1/8" (for resolution); label full inches
        subdivisions = 8
        for i_sub in range(0, int(length_px * subdivisions / px_per_in) + 1):
            x = int(round(i_sub * (px_per_in / subdivisions)))
            tick = int(bottom_h * 0.6) if (i_sub % subdivisions == 0) else int(bottom_h * 0.35)
            y0 = thickness - 1 - tick
            cv2.line(ruler, (x, y0), (x, thickness-1), color, 2) # Increased thickness to 2
            if i_sub % subdivisions == 0:
                inches = i_sub // subdivisions
                ruler = put_text_cv2(ruler, f"{inches}", (x+2, thickness - 10), font_size=16, color=color)
        ruler = put_text_cv2(ruler, "in", (5, thickness - 20), font_size=20, color=color)

    return ruler
