"""
This module provides the `ColorCorrector` class, which is responsible for
performing color correction on images using color checker detection.

It leverages YOLO for initial patch detection and falls back to OpenCV-based
methods if YOLO detection is insufficient. It calculates a 3x3 color correction
matrix and applies it to images, providing rich debug visualizations.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List, Dict
import os
import math

# Assuming these utilities are in your project structure
from src import config
from src.utils.image_utils import load_image, save_image


def _sort_patches(
    patches: List[np.ndarray], patch_coords: List[Tuple[int, int, int, int]]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Sorts patches based on their y and then x coordinates."""
    if not patches:
        return [], []
    # Combine patches and coordinates for sorting
    combined = sorted(
        zip(patches, patch_coords), key=lambda item: (item[1][1], item[1][0])
    )
    # Unzip after sorting
    sorted_patches, sorted_coords = zip(*combined)
    return list(sorted_patches), list(sorted_coords)


class ColorCorrector:
    def __init__(self):
        self.model = YOLO(str(config.YOLO_MODEL_PATH))

    # --- Step 1: Detection Helpers ---
    def _find_checker_bounds(
        self, image: np.ndarray
    ) -> Tuple[int, int, int, int] | None:
        """Finds the bounding box of the largest contour, assumed to be the color checker."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use a high contrast stretch to make the checker stand out
        gray = cv2.equalizeHist(gray)
        inverted_gray = cv2.bitwise_not(gray)

        # Use a binary threshold instead of Otsu which can be sensitive to lighting
        _, thresh = cv2.threshold(inverted_gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon to get a tighter fit
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        return cv2.boundingRect(approx)

    def detect_color_checker_patches(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        """
        Detects color checker patches using a tiered approach:
        1. YOLO model for fast and accurate detection.
        2. Robust OpenCV grid-line detection if YOLO fails.
        3. Simple grid-based OpenCV detection as a final fallback.
        """
        patches, patch_coords = [], []
        try:
            results = self.model(image, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    patch = image[y1:y2, x1:x2]
                    if patch.size > 0:
                        patches.append(patch)
                        patch_coords.append((x1, y1, x2, y2))
        except Exception as e:
            if debug_mode:
                print(f"[DEBUG] YOLO detection failed with error: {e}")
            patches = []  # Ensure patches is empty on failure

        if debug_mode:
            print(f"[DEBUG] YOLO detected {len(patches)} patches.")

        # If YOLO detection is insufficient, start fallback chain
        if len(patches) < 18:
            if debug_mode:
                print(
                    "[DEBUG] YOLO detection insufficient. Falling back to robust OpenCV grid-line detection."
                )

            opencv_result = self._detect_patches_opencv_robust(
                image, output_dir=output_dir, debug_mode=debug_mode
            )
            if len(opencv_result.get("patches", [])) == 24:  # Expect exactly 24 patches
                if debug_mode:
                    print(
                        f"[DEBUG] Robust OpenCV detected {len(opencv_result['patches'])} patches."
                    )
                return opencv_result

            if debug_mode:
                print(
                    "[DEBUG] Robust OpenCV detection insufficient. Falling back to simple grid method."
                )
            return self._detect_patches_opencv(
                image, output_dir=output_dir, debug_mode=debug_mode
            )

        return {"patches": patches, "patch_coords": patch_coords}

    def _detect_patches_opencv_robust(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        """
        Detects patches by finding the grid lines of the color checker using
        Canny edge detection and Hough Line Transform. This is robust to images
        with no physical gaps between patches.
        """
        bounds = self._find_checker_bounds(image)
        if not bounds:
            return {"patches": [], "patch_coords": []}

        x, y, w, h = bounds
        cropped_checker = image[y : y + h, x : x + w]

        gray = cv2.cvtColor(cropped_checker, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection will find the borders between the color patches
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Use Hough Line Transform to detect lines in the edge map
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=h // 4, maxLineGap=20
        )

        if lines is None:
            return {"patches": [], "patch_coords": []}

        # --- Process detected lines to find grid coordinates ---
        vertical_x = []
        horizontal_y = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:  # Horizontal line
                horizontal_y.append(y1)
                horizontal_y.append(y2)
            elif abs(abs(angle) - 90) < 15:  # Vertical line
                vertical_x.append(x1)
                vertical_x.append(x2)

        def cluster_and_average(coords, threshold=10):
            if not coords:
                return []
            coords.sort()
            clusters = []
            current_cluster = [coords[0]]
            for coord in coords[1:]:
                if coord - current_cluster[-1] < threshold:
                    current_cluster.append(coord)
                else:
                    clusters.append(int(np.mean(current_cluster)))
                    current_cluster = [coord]
            clusters.append(int(np.mean(current_cluster)))
            return clusters

        # Add image boundaries to the grid lines
        vertical_x.extend([0, w])
        horizontal_y.extend([0, h])

        x_coords = cluster_and_average(vertical_x, threshold=w / 12)
        y_coords = cluster_and_average(horizontal_y, threshold=h / 8)

        # We expect 7 vertical and 5 horizontal lines for a 6x4 grid
        if len(x_coords) < 5 or len(y_coords) < 4:
            return {"patches": [], "patch_coords": []}

        if debug_mode and output_dir:
            debug_img = cropped_checker.copy()
            for x_c in x_coords:
                cv2.line(debug_img, (x_c, 0), (x_c, h), (0, 255, 255), 2)
            for y_c in y_coords:
                cv2.line(debug_img, (0, y_c), (w, y_c), (255, 0, 255), 2)
            path = os.path.join(output_dir, "debug_robust_grid_lines.png")
            save_image(path, debug_img)

        # --- Extract patches using the detected grid ---
        patches, patch_coords = [], []
        inset = 5  # Inset to avoid edges
        for r_idx in range(len(y_coords) - 1):
            for c_idx in range(len(x_coords) - 1):
                y1, y2 = y_coords[r_idx] + inset, y_coords[r_idx + 1] - inset
                x1, x2 = x_coords[c_idx] + inset, x_coords[c_idx + 1] - inset

                if x1 >= x2 or y1 >= y2:
                    continue

                patch = cropped_checker[y1:y2, x1:x2]
                if patch.size > 0:
                    patches.append(patch)
                    # Adjust coordinates to be relative to the original image
                    patch_coords.append((x + x1, y + y1, x + x2, y + y2))

        return {"patches": patches, "patch_coords": patch_coords}

    def _detect_patches_opencv(
        self, image: np.ndarray, output_dir: str | None = None, debug_mode: bool = False
    ) -> Dict:
        """
        Original simple fallback: Detects patches by dividing the color checker's
        bounding box into a fixed grid. Less robust to rotation or distortion.
        """
        bounds = self._find_checker_bounds(image)
        if not bounds:
            return {"patches": [], "patch_coords": []}
        x, y, w, h = bounds
        cropped_color_checker = image[y : y + h, x : x + w]
        rows, cols = 4, 6
        patch_height = h // rows
        patch_width = w // cols
        patches, patch_coords = [], []
        # Add a small inset to avoid picking up patch borders
        inset = int(min(patch_height, patch_width) * 0.15)
        for r_idx in range(rows):
            for c_idx in range(cols):
                y1 = r_idx * patch_height + inset
                y2 = (r_idx + 1) * patch_height - inset
                x1 = c_idx * patch_width + inset
                x2 = (c_idx + 1) * patch_width - inset
                patch = cropped_color_checker[y1:y2, x1:x2]
                if patch.size > 0:
                    patches.append(patch)
                    patch_coords.append((x + x1, y + y1, x + x2, y + y2))
        return {"patches": patches, "patch_coords": patch_coords}

    # --- Step 2: Color Calculation ---
    def calculate_average_colors(
        self, patches: List[np.ndarray], color_space="bgr"
    ) -> List[np.ndarray]:
        # ... (existing implementation remains the same)
        average_colors = []
        for patch in patches:
            img = patch[:, :, :3]
            if color_space == "hsv":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            avg_color = np.mean(img, axis=(0, 1))
            average_colors.append(avg_color.astype(np.uint8))
        return average_colors

    # --- Step 3: Model Calculation (Dispatcher) ---
    def calculate_correction_model(
        self, source, target, method: str = "linear"
    ) -> Dict:
        # ... (existing implementation remains the same)
        if method == "histogram":
            return {"luts": self._calculate_histogram_luts(source, target)}
        if method == "linear":
            return {"matrix": self._calculate_linear_matrix(source, target)}
        elif method == "polynomial":
            return {"matrix": self._calculate_root_polynomial_matrix(source, target)}
        elif method == "hsv":
            return {"luts": self._calculate_hsv_luts(source, target)}
        else:
            raise ValueError(f"Unknown color correction method: {method}")

    def _calculate_histogram_luts(
        self, source_roi: np.ndarray, target_roi: np.ndarray
    ) -> List[np.ndarray]:
        # ... (existing implementation remains the same)
        luts = []
        for i in range(3):  # For B, G, R channels
            src_hist = cv2.calcHist([source_roi], [i], None, [256], [0, 256])
            tgt_hist = cv2.calcHist([target_roi], [i], None, [256], [0, 256])

            src_cdf = src_hist.cumsum()
            tgt_cdf = tgt_hist.cumsum()
            src_cdf_norm = (src_cdf * tgt_cdf.max()) / src_cdf.max()

            lut = np.zeros(256, dtype=np.uint8)
            j = 0
            for val in range(256):
                while j < 255 and src_cdf_norm[val] > tgt_cdf[j]:
                    j += 1
                lut[val] = j
            luts.append(lut)
        return luts

    def _calculate_linear_matrix(self, source_colors, target_colors):
        # ... (existing implementation remains the same)
        source_matrix = np.array(source_colors, dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)
        M, _, _, _ = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)
        return M

    def _calculate_root_polynomial_matrix(self, source_colors, target_colors):
        # ... (existing implementation remains the same)
        source_matrix = np.array([np.sqrt(c) for c in source_colors], dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)
        M, _, _, _ = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)
        return M

    def _calculate_hsv_luts(self, source_colors_hsv, target_colors_hsv):
        # ... (existing implementation remains the same)
        luts = []
        source_channels = cv2.split(
            np.array(source_colors_hsv, dtype=np.uint8).reshape(-1, 1, 3)
        )
        target_channels = cv2.split(
            np.array(target_colors_hsv, dtype=np.uint8).reshape(-1, 1, 3)
        )
        for i in range(3):
            src_hist, _ = np.histogram(
                source_channels[i].flatten(), bins=256, range=[0, 256]
            )
            tgt_hist, _ = np.histogram(
                target_channels[i].flatten(), bins=256, range=[0, 256]
            )
            src_cdf = src_hist.cumsum()
            tgt_cdf = tgt_hist.cumsum()
            src_cdf_norm = src_cdf * (tgt_cdf.max() / src_cdf.max())
            lut = np.zeros(256, dtype=np.uint8)
            j = 0
            for val in range(256):
                while j < 255 and src_cdf_norm[val] > tgt_cdf[j]:
                    j += 1
                lut[val] = j
            luts.append(lut)
        return luts

    # --- Step 4: Model Application (Dispatcher) ---
    def apply_correction_model(
        self, image: np.ndarray, model: Dict, method: str = "linear"
    ) -> np.ndarray:
        # ... (existing implementation remains the same)
        if method == "histogram":
            return self._apply_histogram_luts(image, model["luts"])
        elif method == "linear":
            return self._apply_linear_matrix(image, model["matrix"])
        elif method == "polynomial":
            return self._apply_root_polynomial_matrix(image, model["matrix"])
        elif method == "hsv":
            return self._apply_hsv_luts(image, model["luts"])
        else:
            raise ValueError(f"Unknown color correction method: {method}")

    def _apply_histogram_luts(
        self, image: np.ndarray, luts: List[np.ndarray]
    ) -> np.ndarray:
        # ... (existing implementation remains the same)
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        b, g, r = cv2.split(bgr_image)
        b_corr = cv2.LUT(b, luts[0])
        g_corr = cv2.LUT(g, luts[1])
        r_corr = cv2.LUT(r, luts[2])
        corrected_bgr = cv2.merge([b_corr, g_corr, r_corr])
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def _apply_linear_matrix(self, image, matrix):
        # ... (existing implementation remains the same)
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        corrected_pixels = np.dot(pixels, matrix)
        corrected_bgr = (
            np.clip(corrected_pixels, 0, 255).astype(np.uint8).reshape(bgr_image.shape)
        )
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def _apply_root_polynomial_matrix(self, image, matrix):
        # ... (existing implementation remains the same)
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        corrected_pixels = np.dot(np.sqrt(pixels), matrix)
        corrected_bgr = (
            np.clip(corrected_pixels, 0, 255).astype(np.uint8).reshape(bgr_image.shape)
        )
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    def _apply_hsv_luts(self, image, luts):
        # ... (existing implementation remains the same)
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        h_corr = cv2.LUT(h, luts[0])
        s_corr = cv2.LUT(s, luts[1])
        v_corr = cv2.LUT(v, luts[2])
        corrected_hsv = cv2.merge([h_corr, s_corr, v_corr])
        corrected_bgr = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
        return (
            cv2.merge([corrected_bgr, image[:, :, 3]])
            if is_4_channel
            else corrected_bgr
        )

    # --- Step 5: Main Workflow ---
    def correct_image_colors(
        self,
        source_image_path: str,
        reference_image_path: str,
        output_dir: str | None = None,
        debug_mode: bool = False,
        method: str = "linear",
    ) -> Dict:
        # ... (existing implementation remains the same, but with a stricter check for matched patches)
        all_debug_paths = {}
        source_image, _ = load_image(source_image_path)
        reference_image, _ = load_image(reference_image_path)

        if method == "histogram":
            src_bounds = self._find_checker_bounds(source_image)
            ref_bounds = self._find_checker_bounds(reference_image)
            if not src_bounds or not ref_bounds:
                raise ValueError(
                    "Could not find color checker bounds for histogram alignment."
                )

            src_roi = source_image[
                src_bounds[1] : src_bounds[1] + src_bounds[3],
                src_bounds[0] : src_bounds[0] + src_bounds[2],
            ]
            ref_roi = reference_image[
                ref_bounds[1] : ref_bounds[1] + ref_bounds[3],
                ref_bounds[0] : ref_bounds[0] + ref_bounds[2],
            ]

            correction_model = self.calculate_correction_model(src_roi, ref_roi, method)
        else:
            source_detection = self.detect_color_checker_patches(
                source_image, output_dir, debug_mode
            )
            ref_detection = self.detect_color_checker_patches(
                reference_image, output_dir, debug_mode
            )
            sorted_src_patches, sorted_src_coords = _sort_patches(
                source_detection["patches"], source_detection["patch_coords"]
            )
            sorted_ref_patches, sorted_ref_coords = _sort_patches(
                ref_detection["patches"], ref_detection["patch_coords"]
            )

            if debug_mode and output_dir:
                src_patch_debug_img = source_image.copy()
                for i, (x1, y1, x2, y2) in enumerate(sorted_src_coords):
                    cv2.rectangle(
                        src_patch_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        src_patch_debug_img,
                        str(i),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                src_patch_debug_path = os.path.join(
                    output_dir, f"01_debug_source_sorted_patches_{method}.png"
                )
                save_image(src_patch_debug_path, src_patch_debug_img)
                all_debug_paths[f"source_sorted_patches"] = src_patch_debug_path

                ref_patch_debug_img = reference_image.copy()
                for i, (x1, y1, x2, y2) in enumerate(sorted_ref_coords):
                    cv2.rectangle(
                        ref_patch_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        ref_patch_debug_img,
                        str(i),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                ref_patch_debug_path = os.path.join(
                    output_dir, f"02_debug_reference_sorted_patches_{method}.png"
                )
                save_image(ref_patch_debug_path, ref_patch_debug_img)
                all_debug_paths[f"reference_sorted_patches"] = ref_patch_debug_path

            num_matched = min(len(sorted_src_patches), len(sorted_ref_patches))
            if num_matched < 24:  # Stricter check
                raise ValueError(
                    f"Not enough matched patches found ({num_matched}). A full set of 24 is required."
                )

            color_space = "hsv" if method == "hsv" else "bgr"
            source_colors = self.calculate_average_colors(
                sorted_src_patches[:num_matched], color_space
            )
            ref_colors = self.calculate_average_colors(
                sorted_ref_patches[:num_matched], color_space
            )

            correction_model = self.calculate_correction_model(
                source_colors, ref_colors, method
            )

        corrected_source_image = self.apply_correction_model(
            source_image, correction_model, method
        )

        if debug_mode and output_dir:
            path = os.path.join(output_dir, f"04_corrected_image_{method}.png")
            save_image(path, corrected_source_image)
            all_debug_paths["corrected_source_image"] = path

        return {
            "corrected_image": corrected_source_image,
            "correction_model": correction_model,
            "debug_paths": all_debug_paths,
        }
