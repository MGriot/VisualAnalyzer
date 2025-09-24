"""
This module provides the `ColorCorrector` class, which is responsible for
performing color correction on images using color checker detection.

It leverages YOLO for initial patch detection and falls back to OpenCV-based
methods if YOLO detection is insufficient. It calculates a 3x3 color correction
matrix and applies it to images.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List

from src import config
from src.utils.image_utils import load_image

class ColorCorrector:
    """
    Handles color correction and alignment using a color checker.

    This class provides methods to detect color checker patches in an image,
    calculate average colors from these patches, compute a 3x3 color correction
    matrix, and apply this matrix to correct the colors of an image.
    """

    def __init__(self):
        """
        Initializes the ColorCorrector by loading the YOLO model for color checker detection.
        """
        self.model = YOLO(str(config.YOLO_MODEL_PATH))

    def detect_color_checker_patches(self, image: np.ndarray, debug_mode: bool = False) -> List[np.ndarray]:
        """
        Detects color checker patches in an image using a YOLO model, with an OpenCV-based
        fallback for robustness.

        Args:
            image (np.ndarray): The input image (BGR format) in which to detect patches.
            debug_mode (bool): If True, prints debug information to the console.

        Returns:
            List[np.ndarray]: A list of detected color patch images (cropped NumPy arrays).
                              Returns an empty list if no patches are detected.
        """
        # Try YOLO detection first
        yolo_patches = []
        results = self.model(image, verbose=False)  # Run inference
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                patch = image[y1:y2, x1:x2]
                if patch.size > 0:
                    yolo_patches.append(patch)

        if debug_mode: print(f"[DEBUG] YOLO detected {len(yolo_patches)} patches.")

        if len(yolo_patches) >= 3:  # If YOLO finds enough patches, use them
            return yolo_patches
        else:
            # Fallback to OpenCV-based detection
            if debug_mode: print("[DEBUG] YOLO did not detect enough patches. Falling back to OpenCV-based detection.")
            return self._detect_patches_opencv(image, debug_mode=debug_mode)

    def _detect_patches_opencv(self, image: np.ndarray, debug_mode: bool = False) -> List[np.ndarray]:
        """
        Detects color checker patches in an image using traditional OpenCV image processing
        techniques (thresholding, contour detection, and grid assumption).

        This method is used as a fallback when YOLO detection is insufficient.
        It assumes a standard 6x4 ColorChecker Classic layout.

        Args:
            image (np.ndarray): The input image (BGR format).
            debug_mode (bool): If True, prints debug information to the console.

        Returns:
            List[np.ndarray]: A list of detected color patch images (cropped NumPy arrays),
                              sorted by their assumed grid position. Returns an empty list
                              if no significant contours are found or cropping fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert the image to make borders white and patches dark
        inverted_gray = cv2.bitwise_not(gray)
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(inverted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Optional: Morphological operations to clean up the image
        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, assuming it's the color checker itself
        if not contours:
            if debug_mode: print("[DEBUG] No contours found for color checker.")
            return []

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image to the bounding box of the largest contour
        cropped_color_checker = image[y:y+h, x:x+w]

        if cropped_color_checker.size == 0:
            if debug_mode: print("[DEBUG] Cropped color checker is empty.")
            return []

        # Assume a 6x4 grid for ColorChecker Classic
        rows, cols = 4, 6
        patch_width = cropped_color_checker.shape[1] // cols
        patch_height = cropped_color_checker.shape[0] // rows

        patches = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * patch_width
                y1 = r * patch_height
                x2 = (c + 1) * patch_width
                y2 = (r + 1) * patch_height
                patch = cropped_color_checker[y1:y2, x1:x2]
                if patch.size > 0:
                    patches.append(patch)

        if debug_mode: print(f"[DEBUG] OpenCV detected {len(patches)} patches.")
        return patches

    def calculate_average_color(self, patches: List[np.ndarray], debug_mode: bool = False) -> List[np.ndarray]:
        """
        Calculates the average BGR color for each provided image patch.

        If a patch has an alpha channel, only non-transparent pixels are considered
        for the average calculation.

        Args:
            patches (List[np.ndarray]): A list of image patches (NumPy arrays).
            debug_mode (bool): If True, prints debug information including the calculated
                               average colors.

        Returns:
            List[np.ndarray]: A list of NumPy arrays, where each array represents the
                              [B, G, R] average color of a corresponding patch.
        """
        average_colors = []
        for patch in patches:
            # Calculate average BGR color, ignoring transparent pixels if any
            if patch.shape[2] == 4: # RGBA
                b, g, r, a = cv2.split(patch)
                # Only consider pixels where alpha is not zero
                non_transparent_pixels = patch[a > 0]
                if len(non_transparent_pixels) > 0:
                    avg_bgr = np.mean(non_transparent_pixels[:, :3], axis=0)
                else:
                    avg_bgr = np.array([0, 0, 0]) # Default to black if no solid pixels
            else:
                avg_bgr = np.mean(patch, axis=(0, 1))
            average_colors.append(avg_bgr.astype(np.uint8))
        
        if debug_mode: print(f"[DEBUG] Calculated average colors for {len(average_colors)} patches.")
        if debug_mode: print(f"[DEBUG] Average Colors (BGR): {average_colors}")
        return average_colors

    def calculate_color_correction_matrix(self, source_colors: List[np.ndarray], target_colors: List[np.ndarray], debug_mode: bool = False) -> np.ndarray:
        """
        Calculates a 3x3 color correction matrix that transforms `source_colors` to `target_colors`
        using a least squares approach.

        Args:
            source_colors (List[np.ndarray]): A list of average BGR colors from the source image.
            target_colors (List[np.ndarray]): A list of average BGR colors from the target (reference) image.
            debug_mode (bool): If True, prints the calculated correction matrix.

        Returns:
            np.ndarray: A 3x3 NumPy array representing the color correction matrix.
                        Returns an identity matrix if the calculation fails.

        Raises:
            ValueError: If there are fewer than 3 color pairs, or if the number of
                        source and target colors do not match.
        """
        if len(source_colors) < 3 or len(target_colors) < 3:
            if debug_mode: print("[DEBUG] Not enough color pairs to calculate a 3x3 matrix.")
            raise ValueError("Need at least 3 color pairs to calculate a 3x3 matrix.")
        if len(source_colors) != len(target_colors):
            if debug_mode: print("[DEBUG] Source and target color lists have different number of elements.")
            raise ValueError("Source and target color lists must have the same number of elements.")

        # Convert lists of arrays to 2D numpy arrays
        source_matrix = np.array(source_colors, dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)

        # Solve for the 3x3 transformation matrix
        try:
            M = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)[0]
            if debug_mode: print(f"[DEBUG] Calculated Correction Matrix:\n{M}")
        except np.linalg.LinAlgError as e:
            if debug_mode: print(f"[DEBUG] Linear algebra error during matrix calculation: {e}")
            return np.eye(3, dtype=np.float32)

        return M

    def apply_color_correction(self, image: np.ndarray, correction_matrix: np.ndarray, output_dir: str | None = None) -> dict:
        """
        Applies a given 3x3 color correction matrix to an input image, handling BGRA images
        by preserving the alpha channel. Optionally saves the output.

        Args:
            image (np.ndarray): The input image (BGR or BGRA) to correct.
            correction_matrix (np.ndarray): The 3x3 color correction matrix.
            output_dir (str | None, optional): If provided, the directory where the corrected image is saved.
                                           Defaults to None.

        Returns:
            dict: A dictionary containing:
                - 'image' (np.ndarray): The color-corrected image.
                - 'debug_path' (str | None): The path to the saved image, or None.
        """
        # Handle 4-channel images by separating alpha, correcting BGR, and merging back.
        is_4_channel = len(image.shape) == 3 and image.shape[2] == 4
        
        if is_4_channel:
            alpha = image[:, :, 3]
            bgr_image = image[:, :, :3]
            original_shape = bgr_image.shape
            pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        else:
            original_shape = image.shape
            pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply the transformation
        corrected_pixels = np.dot(pixels, correction_matrix)

        # Clip values to valid BGR range [0, 255]
        corrected_pixels = np.clip(corrected_pixels, 0, 255).astype(np.uint8)

        # Reshape back to original image dimensions
        corrected_bgr = corrected_pixels.reshape(original_shape)

        if is_4_channel:
            corrected_image = cv2.merge([corrected_bgr, alpha])
        else:
            corrected_image = corrected_bgr

        debug_path = None
        if output_dir:
            import os
            from src.utils.image_utils import save_image
            debug_path = os.path.join(output_dir, "color_corrected.png")
            save_image(debug_path, corrected_image)

        return {
            'image': corrected_image,
            'debug_path': debug_path
        }

    def correct_image_colors(self, source_image_path: str, reference_image_path: str, debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a complete color correction workflow from a source image (e.g., a photo
        of a color checker taken under specific lighting) to a reference image (e.g.,
        an ideal representation of the same color checker).

        This involves detecting patches in both images, calculating their average colors,
        and then computing a color correction matrix to transform colors from the source
        lighting conditions to the reference conditions.

        Args:
            source_image_path (str): Path to the source image containing a color checker.
            reference_image_path (str): Path to the reference image of the same color checker.
            debug_mode (bool): If True, enables verbose output during the process.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - source_image (np.ndarray): The loaded source image.
                - correction_matrix (np.ndarray): The calculated 3x3 color correction matrix.

        Raises:
            ValueError: If either image cannot be loaded, if color checker patches cannot
                        be detected in one or both images, or if an insufficient number
                        of matching patches are found for matrix calculation.
        """
        source_image, _ = load_image(source_image_path)
        reference_image, _ = load_image(reference_image_path)

        if source_image is None or reference_image is None:
            raise ValueError("Could not load source or reference image for color correction.")

        source_patches = self.detect_color_checker_patches(source_image, debug_mode=debug_mode)
        reference_patches = self.detect_color_checker_patches(reference_image, debug_mode=debug_mode)

        if not source_patches or not reference_patches:
            raise ValueError("Could not detect color checker patches in one or both images.")

        source_colors = self.calculate_average_color(source_patches, debug_mode=debug_mode)
        reference_colors = self.calculate_average_color(reference_patches, debug_mode=debug_mode)

        # Ensure the number of detected patches is consistent
        if len(source_colors) != len(reference_colors):
            if debug_mode: print(f"[DEBUG] Warning: Mismatch in number of detected patches. Source: {len(source_colors)}, Reference: {len(reference_colors)}")
            # Attempt to match based on common number of patches, or raise error
            min_patches = min(len(source_colors), len(reference_colors))
            source_colors = source_colors[:min_patches]
            reference_colors = reference_colors[:min_patches]
            if min_patches < 3: # Need at least 3 for a 3x3 matrix
                raise ValueError("Not enough matching color patches detected for correction.")

        correction_matrix = self.calculate_color_correction_matrix(source_colors, reference_colors, debug_mode=debug_mode)

        return source_image, correction_matrix