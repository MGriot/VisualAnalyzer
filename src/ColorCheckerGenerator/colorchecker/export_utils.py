"""
export_utils.py

Functions to save the generated OpenCV image to PNG/JPG or PDF (with physical mm size).
Uses ReportLab for PDF export so printed rulers match physical dimensions.
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader
from io import BytesIO
from typing import Tuple
from .unit_utils import parse_size

def save_image(cv_img: np.ndarray, filename: str):
    """
    Save a numpy OpenCV image to filename. File extension decides the format.
    """
    ext = filename.split('.')[-1].lower()
    if ext in ('png', 'jpg', 'jpeg', 'tiff', 'bmp'):
        cv2.imwrite(filename, cv_img)
    elif ext == 'pdf':
        save_pdf(cv_img, filename, width_px=cv_img.shape[1])
    else:
        # fallback to png
        cv2.imwrite(filename, cv_img)

def save_pdf(cv_img: np.ndarray, filename: str, width_px:int, dpi:int = 300):
    """
    Save cv_img to a single-page PDF sized according to width_px and dpi.
    The PDF page is oriented portrait with width in pixels converted to points.
    """
    # convert BGR (OpenCV) to RGB for Pillow/ReportLab
    rgb = cv_img[..., ::-1]
    # Convert the image into a PIL-compatible stream
    import PIL.Image as PilImage
    pil = PilImage.fromarray(rgb)
    img_buffer = BytesIO()
    pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    # Convert pixel width to inches -> points (1 in = 72 points)
    width_in = width_px / dpi
    height_px = cv_img.shape[0]
    height_in = height_px / dpi
    page_width_pt = width_in * 72
    page_height_pt = height_in * 72

    c = pdfcanvas.Canvas(filename, pagesize=(page_width_pt, page_height_pt))
    # place image at (0,0) with full page size
    img_reader = ImageReader(img_buffer)
    c.drawImage(img_reader, 0, 0, width=page_width_pt, height=page_height_pt)
    c.showPage()
    c.save()
