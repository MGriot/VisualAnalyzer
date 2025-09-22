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

    def calculate_hsv_from_full_image(self, image_path: Path) -> Tuple[float, float, float]:
        """
        Calculates the average HSV color from the entire image, considering only
        non-transparent pixels if an alpha channel is present.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Tuple[float, float, float]: A tuple containing the average Hue, Saturation,
                                       and Value components of the image.

        Raises:
            ValueError: If the image cannot be loaded or if no non-transparent pixels are found.
        """
        img, alpha = load_image(str(image_path), handle_transparency=True)
        if img is None:
            raise ValueError(f"Could not load image {image_path}")

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if alpha is not None:
            non_transparent_pixels = hsv_img[alpha > 0]
        else:
            non_transparent_pixels = hsv_img.reshape(-1, 3)

        if len(non_transparent_pixels) == 0:
            raise ValueError(f"No non-transparent pixels found in {image_path} for full image average.")

        avg_h = np.mean(non_transparent_pixels[:, 0])
        avg_s = np.mean(non_transparent_pixels[:, 1])
        avg_v = np.mean(non_transparent_pixels[:, 2])

        return avg_h, avg_s, avg_v

    def calculate_hsv_from_points(self, image_path: Path, points: List[Dict], radius: int = 7) -> List[Tuple[float, float, float]]:
        """
        Calculates the average HSV color for the Region of Interest (ROI) around each specified point.

        Args:
            image_path (Path): The path to the image file.
            points (List[Dict]): A list of dictionaries, where each dictionary represents a point
                                 with 'x', 'y' coordinates and an optional 'radius'.
            radius (int, optional): The default radius for the ROI around each point if not
                                    specified in the point dictionary. Defaults to 7.

        Returns:
            List[Tuple[float, float, float]]: A list of tuples, where each tuple contains the
                                              average (H, S, V) for the ROI of a corresponding point.

        Raises:
            ValueError: If the image cannot be loaded or if no valid pixels are found around any point.
        """
        img, alpha = load_image(str(image_path), handle_transparency=True)
        if img is None:
            raise ValueError(f"Could not load image {image_path}")

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        point_avg_colors = []

        for point in points:
            x, y = point['x'], point['y']
            current_radius = point.get('radius', radius)

            # Define the region of interest (ROI) around the point
            x_min = max(0, x - current_radius)
            y_min = max(0, y - current_radius)
            x_max = min(img.shape[1], x + current_radius + 1)
            y_max = min(img.shape[0], y + current_radius + 1)

            roi_hsv = hsv_img[y_min:y_max, x_min:x_max]
            roi_alpha = alpha[y_min:y_max, x_min:x_max] if alpha is not None else None

            if roi_alpha is not None:
                non_transparent_pixels = roi_hsv[roi_alpha > 0]
            else:
                non_transparent_pixels = roi_hsv.reshape(-1, 3)

            if len(non_transparent_pixels) > 0:
                avg_h = np.mean(non_transparent_pixels[:, 0])
                avg_s = np.mean(non_transparent_pixels[:, 1])
                avg_v = np.mean(non_transparent_pixels[:, 2])
                point_avg_colors.append((avg_h, avg_s, avg_v))

        if not point_avg_colors:
            raise ValueError(f"No valid pixels found around any of the specified points in {image_path}.")

        return point_avg_colors

    def extract_hsv_from_full_image(self, image_path: Path) -> np.ndarray:
        """
        Extracts all HSV color values from the entire image, considering only
        non-transparent pixels if an alpha channel is present.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            np.ndarray: A NumPy array of shape (N, 3) containing all extracted HSV colors,
                        where N is the number of non-transparent pixels.

        Raises:
            ValueError: If the image cannot be loaded or if no non-transparent pixels are found.
        """
        img, alpha = load_image(str(image_path), handle_transparency=True)
        if img is None:
            raise ValueError(f"Could not load image {image_path}")

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if alpha is not None:
            non_transparent_pixels = hsv_img[alpha > 0]
        else:
            non_transparent_pixels = hsv_img.reshape(-1, 3)

        if len(non_transparent_pixels) == 0:
            raise ValueError(f"No non-transparent pixels found in {image_path} for full image average.")

        return non_transparent_pixels

    def extract_hsv_from_points(self, image_path: Path, points: List[Dict], radius: int = 7) -> np.ndarray:
        """
        Extracts all HSV color values from the Regions of Interest (ROIs) around specified points.

        Args:
            image_path (Path): The path to the image file.
            points (List[Dict]): A list of dictionaries, where each dictionary represents a point
                                 with 'x', 'y' coordinates and an optional 'radius'.
            radius (int, optional): The default radius for the ROI around each point if not
                                    specified in the point dictionary. Defaults to 7.

        Returns:
            np.ndarray: A NumPy array of shape (N, 3) containing all extracted HSV colors
                        from the ROIs, where N is the total number of pixels extracted.

        Raises:
            ValueError: If the image cannot be loaded or if no valid pixels are found around any point.
        """
        img, alpha = load_image(str(image_path), handle_transparency=True)
        if img is None:
            raise ValueError(f"Could not load image {image_path}")

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        all_pixels = []

        for point in points:
            x, y = point['x'], point['y']
            current_radius = point.get('radius', radius)

            # Define the region of interest (ROI) around the point
            x_min = max(0, x - current_radius)
            y_min = max(0, y - current_radius)
            x_max = min(img.shape[1], x + current_radius + 1)
            y_max = min(img.shape[0], y + current_radius + 1)

            roi_hsv = hsv_img[y_min:y_max, x_min:x_max]
            roi_alpha = alpha[y_min:y_max, x_min:x_max] if alpha is not None else None

            if roi_alpha is not None:
                non_transparent_pixels = roi_hsv[roi_alpha > 0]
            else:
                non_transparent_pixels = roi_hsv.reshape(-1, 3)

            if len(non_transparent_pixels) > 0:
                all_pixels.append(non_transparent_pixels)

        if not all_pixels:
            raise ValueError(f"No valid pixels found around specified points in {image_path}.")

        return np.vstack(all_pixels)

    def calculate_hsv_range(self, avg_h: float, avg_s: float, avg_v: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates a simple HSV color range (lower, upper, and center) based on
        given average HSV values and predefined tolerances.

        Args:
            avg_h (float): Average Hue value.
            avg_s (float): Average Saturation value.
            avg_v (float): Average Value (Brightness) value.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - lower_limit (np.ndarray): NumPy array [H, S, V] for the lower bounds.
                - upper_limit (np.ndarray): NumPy array [H, S, V] for the upper bounds.
                - center_color (np.ndarray): NumPy array [H, S, V] for the center color.
        """
        h_tolerance = 10  # Degrees
        s_tolerance = 30  # Percentage
        v_tolerance = 30  # Percentage

        lower_h = max(0, avg_h - h_tolerance)
        upper_h = min(179, avg_h + h_tolerance)  # OpenCV HSV H range is 0-179

        lower_s = max(0, avg_s - s_tolerance)
        upper_s = min(255, avg_s + s_tolerance) # OpenCV HSV S range is 0-255

        lower_v = max(0, avg_v - v_tolerance)
        upper_v = min(255, avg_v + v_tolerance) # OpenCV HSV V range is 0-255

        lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        center_color = np.array([avg_h, avg_s, avg_v], dtype=np.uint8)

        return lower_limit, upper_limit, center_color