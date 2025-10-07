"""
This module provides the `DatasetItemProcessor` class, which is responsible for
extracting and calculating HSV color information from dataset images.

It supports extracting colors from the entire image or from specific points/regions
of interest, and can calculate HSV ranges based on these extracted values.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

from src.utils.image_utils import load_image

class DatasetItemProcessor:
    """
    Processes dataset item images to calculate HSV color ranges based on full image
    average or specified points.

    This class provides methods to extract HSV color values from images, either
    from the entire image or from defined regions of interest (ROIs) around points.
    It also includes a method to calculate a basic HSV range from average values.
    """

    def extract_hsv_from_image(
        self, image: np.ndarray, method: str, points: List[Dict] = None, radius: int = 7
    ) -> np.ndarray:
        """
        Extracts HSV color values from an image based on the specified method.

        Args:
            image (np.ndarray): The input image (already loaded, in BGR format).
            method (str): The extraction method ("full_average" or "points").
            points (List[Dict], optional): A list of point dictionaries for the 'points' method.
            radius (int, optional): Default radius for the 'points' method.

        Returns:
            np.ndarray: An array of HSV color values.
        """
        img_bgr, alpha = (image[:, :, :3], image[:, :, 3]) if image.shape[2] == 4 else (image, None)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        if method == "full_average":
            pixels = hsv_img[alpha > 0] if alpha is not None else hsv_img.reshape(-1, 3)
            if len(pixels) == 0:
                raise ValueError("No non-transparent pixels found for full image average.")
            return pixels

        elif method == "points":
            if not points:
                raise ValueError("Method 'points' requires a list of points.")
            
            all_pixels = []
            for point in points:
                x, y = point['x'], point['y']
                current_radius = point.get('radius', radius)
                x_min, y_min = max(0, x - current_radius), max(0, y - current_radius)
                x_max, y_max = min(image.shape[1], x + current_radius + 1), min(image.shape[0], y + current_radius + 1)

                roi_hsv = hsv_img[y_min:y_max, x_min:x_max]
                roi_alpha = alpha[y_min:y_max, x_min:x_max] if alpha is not None else None

                pixels = roi_hsv[roi_alpha > 0] if roi_alpha is not None else roi_hsv.reshape(-1, 3)
                if len(pixels) > 0:
                    all_pixels.append(pixels)

            if not all_pixels:
                raise ValueError("No valid pixels found around any specified points.")
            return np.vstack(all_pixels)

        else:
            raise ValueError(f"Unknown extraction method: {method}")