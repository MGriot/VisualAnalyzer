import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

from src.utils.image_utils import load_image

class SampleProcessor:
    """
    Processes sample images to calculate HSV color ranges based on full image average or specified points.
    """

    def calculate_hsv_from_full_image(self, image_path: Path, alpha_channel: np.ndarray = None) -> Tuple[float, float, float]:
        """
        Calculates the average HSV color from the entire image.
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

    def calculate_hsv_from_points(self, image_path: Path, points: List[Dict], radius: int = 7) -> Tuple[float, float, float]:
        """
        Calculates the average HSV color from specified points within the image.
        Each point is a dictionary with 'x', 'y', and optional 'radius'.
        """
        img, alpha = load_image(str(image_path), handle_transparency=True)
        if img is None:
            raise ValueError(f"Could not load image {image_path}")

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_values = []
        s_values = []
        v_values = []

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
                h_values.extend(non_transparent_pixels[:, 0])
                s_values.extend(non_transparent_pixels[:, 1])
                v_values.extend(non_transparent_pixels[:, 2])

        if len(h_values) == 0:
            raise ValueError(f"No valid pixels found around specified points in {image_path}.")

        avg_h = np.mean(h_values)
        avg_s = np.mean(s_values)
        avg_v = np.mean(v_values)

        return avg_h, avg_s, avg_v

    def extract_hsv_from_full_image(self, image_path: Path) -> np.ndarray:
        """
        Extracts all HSV color from the entire image.
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
        Extracts all HSV colors from specified points within the image.
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
        Calculates the HSV range (lower, upper, center) from average HSV values.
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
