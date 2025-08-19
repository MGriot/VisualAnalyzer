import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List

from src import config
from src.utils.image_utils import load_image

class ColorCorrector:
    """
    Handles color correction and alignment using a color checker.
    """

    def __init__(self):
        """
        Initializes the ColorCorrector by loading the YOLO model.
        """
        self.model = YOLO(str(config.YOLO_MODEL_PATH))

    def detect_color_checker_patches(self, image: np.ndarray, debug_mode: bool = False) -> List[np.ndarray]:
        """
        Detects color checker patches in an image using the YOLO model.
        If YOLO doesn't find enough patches, it falls back to an OpenCV-based method.

        Args:
            image (np.ndarray): The input image (BGR format).
            debug_mode (bool): If True, prints debug information.

        Returns:
            List[np.ndarray]: A list of detected color patch images.
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
        Detects color checker patches in an image using OpenCV image processing techniques.

        Args:
            image (np.ndarray): The input image (BGR format).
            debug_mode (bool): If True, prints debug information.

        Returns:
            List[np.ndarray]: A list of detected color patch images, sorted by position.
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
        Calculates the average color of each patch.

        Args:
            patches (List[np.ndarray]): A list of color patch images.
            debug_mode (bool): If True, prints debug information.

        Returns:
            List[np.ndarray]: A list of average BGR colors for each patch.
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
        Calculates a 3x3 color correction matrix using least squares.

        Args:
            source_colors (List[np.ndarray]): List of average BGR colors from the source image.
            target_colors (List[np.ndarray]): List of average BGR colors from the target image.
            debug_mode (bool): If True, prints debug information.

        Returns:
            np.ndarray: A 3x3 color correction matrix.
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

    def apply_color_correction(self, image: np.ndarray, correction_matrix: np.ndarray) -> np.ndarray:
        """
        Applies the color correction matrix to an image.

        Args:
            image (np.ndarray): The input image (BGR format).
            correction_matrix (np.ndarray): The 3x3 color correction matrix.

        Returns:
            np.ndarray: The color-corrected image.
        """
        # Reshape image to a 2D array of pixels (height*width, 3)
        original_shape = image.shape
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply the transformation
        corrected_pixels = np.dot(pixels, correction_matrix)

        # Clip values to valid BGR range [0, 255]
        corrected_pixels = np.clip(corrected_pixels, 0, 255).astype(np.uint8)

        # Reshape back to original image dimensions
        corrected_image = corrected_pixels.reshape(original_shape)

        return corrected_image

    def correct_image_colors(self, source_image_path: str, reference_image_path: str, debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs full color correction from a source image to a reference image.

        Args:
            source_image_path (str): Path to the source image (e.g., user's color checker).
            reference_image_path (str): Path to the reference color checker image.
            debug_mode (bool): If True, prints debug information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The original source image and the color correction matrix.
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