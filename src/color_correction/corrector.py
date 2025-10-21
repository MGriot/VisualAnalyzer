import cv2
import numpy as np
from typing import List, Dict


class ColorCorrector:
    """
    A collection of core mathematical models for color correction.

    This class provides methods to calculate a correction model from source and
    target colors, and to apply that model to an image. It does not handle
    image processing tasks like patch detection or alignment.
    """

    def __init__(self):
        """Initializes the ColorCorrector."""
        pass

    def calculate_correction_model(
        self, source: List[np.ndarray], target: List[np.ndarray], method: str = "linear"
    ) -> Dict:
        """
        Calculates a color correction model based on the specified method.

        Args:
            source: A list of source colors (e.g., from a photographed color checker).
            target: A list of target colors (e.g., from a reference color checker).
            method: The correction method to use ('linear', 'polynomial', 'hsv', 'histogram').

        Returns:
            A dictionary representing the calculated model.
        """
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
        """Calculates Look-Up Tables (LUTs) based on histogram matching."""
        luts = []
        for i in range(3):
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

    def _calculate_linear_matrix(self, source_colors: List[np.ndarray], target_colors: List[np.ndarray]) -> np.ndarray:
        """Calculates a linear transformation matrix using least squares."""
        source_matrix = np.array(source_colors, dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)
        M, _, _, _ = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)
        return M

    def _calculate_root_polynomial_matrix(self, source_colors: List[np.ndarray], target_colors: List[np.ndarray]) -> np.ndarray:
        """Calculates a root-polynomial transformation matrix."""
        source_matrix = np.array([np.sqrt(c) for c in source_colors], dtype=np.float32)
        target_matrix = np.array(target_colors, dtype=np.float32)
        M, _, _, _ = np.linalg.lstsq(source_matrix, target_matrix, rcond=None)
        return M

    def _calculate_hsv_luts(self, source_colors_hsv: List[np.ndarray], target_colors_hsv: List[np.ndarray]) -> List[np.ndarray]:
        """Calculates Look-Up Tables (LUTs) for each HSV channel."""
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

    def apply_correction_model(
        self, image: np.ndarray, model: Dict, method: str = "linear"
    ) -> np.ndarray:
        """
        Applies a pre-calculated color correction model to an image.

        Args:
            image: The BGR image to correct.
            model: The model dictionary, containing either a 'matrix' or 'luts'.
            method: The correction method corresponding to the model.

        Returns:
            The color-corrected BGR image.
        """
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
        """Applies Look-Up Tables to the B, G, and R channels."""
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        b, g, r = cv2.split(bgr_image)
        b_corr = cv2.LUT(b, luts[0])
        g_corr = cv2.LUT(g, luts[1])
        r_corr = cv2.LUT(r, luts[2])
        corrected_bgr = cv2.merge([b_corr, g_corr, r_corr])
        if is_4_channel:
            return cv2.merge([corrected_bgr, image[:, :, 3]])
        return corrected_bgr

    def _apply_linear_matrix(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Applies a linear transformation matrix to the image pixels."""
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        corrected_pixels = np.dot(pixels, matrix)
        corrected_bgr = (
            np.clip(corrected_pixels, 0, 255).astype(np.uint8).reshape(bgr_image.shape)
        )
        if is_4_channel:
            return cv2.merge([corrected_bgr, image[:, :, 3]])
        return corrected_bgr

    def _apply_root_polynomial_matrix(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Applies a root-polynomial transformation matrix."""
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        pixels = bgr_image.reshape(-1, 3).astype(np.float32)
        corrected_pixels = np.dot(np.sqrt(pixels), matrix)
        corrected_bgr = (
            np.clip(corrected_pixels, 0, 255).astype(np.uint8).reshape(bgr_image.shape)
        )
        if is_4_channel:
            return cv2.merge([corrected_bgr, image[:, :, 3]])
        return corrected_bgr

    def _apply_hsv_luts(self, image: np.ndarray, luts: List[np.ndarray]) -> np.ndarray:
        """Applies Look-Up Tables to the H, S, and V channels."""
        is_4_channel = image.shape[2] == 4
        bgr_image = image[:, :, :3] if is_4_channel else image
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        h_corr = cv2.LUT(h, luts[0])
        s_corr = cv2.LUT(s, luts[1])
        v_corr = cv2.LUT(v, luts[2])
        corrected_hsv = cv2.merge([h_corr, s_corr, v_corr])
        corrected_bgr = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
        if is_4_channel:
            return cv2.merge([corrected_bgr, image[:, :, 3]])
        return corrected_bgr
