"""
drawing_utils.py

Low level drawing helpers using OpenCV + Pillow (for nice text).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
from .config import DEFAULT_FONT

def _load_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """
    Load a TTF font for Pillow. Tries default system font, falls back to PIL default.
    """
    try:
        return ImageFont.truetype(DEFAULT_FONT, size=size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            raise RuntimeError("No available fonts for Pillow; install DejaVuSans or similar.")

def put_text_cv2(img: np.ndarray, text: str, xy: Tuple[int,int], font_size: int = 20, color=(0,0,0), anchor="lt"):
    """
    Draw high-quality text on OpenCV image by delegating to Pillow.
    xy is the top-left coordinate of the text baseline area.
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(font_size)
    draw.text(xy, text, font=font, fill=tuple(color))
    return np.array(img_pil)

def draw_color_grid(base_w: int, patch_w: int, patch_h: int, color_array: np.ndarray, labels: Optional[np.ndarray] = None, label_color=(255,255,255)) -> np.ndarray:
    """
    Draw a grid of patches from a numpy array of shape (rows, cols, 3).
    Returns an image (numpy array).
    """
    rows, cols = color_array.shape[0], color_array.shape[1]
    img_h = rows * patch_h
    img_w = cols * patch_w
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    for r in range(rows):
        for c in range(cols):
            color = tuple(int(x) for x in color_array[r,c])
            y0, y1 = r*patch_h, (r+1)*patch_h
            x0, x1 = c*patch_w, (c+1)*patch_w
            canvas[y0:y1, x0:x1] = color
            # optional label
            if labels is not None:
                label = labels[r,c]
                canvas = put_text_cv2(canvas, label, (x0+8, y0+12), font_size=18, color=label_color)
    return canvas

def hex_to_bgr(hex_color: str) -> Tuple[int,int,int]:
    """
    Convert "#RRGGBB" to BGR tuple (int).
    """
    h = hex_color.lstrip('#')
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)
