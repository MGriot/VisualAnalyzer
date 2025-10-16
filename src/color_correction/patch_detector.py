import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PatchInfo:
    """Information about a detected patch"""
    center: Tuple[int, int]
    color_rgb: Tuple[int, int, int]
    color_lab: Tuple[float, float, float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    index: int


class PatchExtractor:
    """Extracts color patches from a pre-aligned image of a color checker."""

    def __init__(self, aligned_image: np.ndarray):
        self.image = aligned_image

    def detect_patches(
        self,
        grid_size: Optional[Tuple[int, int]] = None,
        margin_ratio: float = 0.05,
        adaptive: bool = True,
    ) -> List[PatchInfo]:
        """
        Detect patches in the aligned image.

        Args:
            grid_size: Expected (rows, cols) of patches. If None, auto-detect.
            margin_ratio: Ratio of margin to exclude from edges.
            adaptive: If True, automatically detect grid configuration.

        Returns:
            List of PatchInfo objects ordered from top-left to bottom-right.
        """
        if self.image is None:
            raise ValueError("PatchExtractor requires an aligned image upon initialization.")

        h, w = self.image.shape[:2]

        # Calculate working area (exclude margins)
        margin_h = int(h * margin_ratio)
        margin_w = int(w * margin_ratio)
        working_area = self.image[
            margin_h : h - margin_h, margin_w : w - margin_w
        ]

        if adaptive or grid_size is None:
            grid_size = self._detect_grid_size(working_area)

        if not grid_size or grid_size[0] == 0 or grid_size[1] == 0:
            return []

        patches = self._extract_patches_grid(
            working_area, grid_size, margin_w, margin_h
        )

        return patches

    def _detect_grid_size(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Automatically detect the grid size of patches using robust methods.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Method 1: Edge-based detection
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=int(min(w, h) * 0.15), maxLineGap=20
        )

        # Method 2: Contour-based detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Method 3: Color variance analysis
        rows_estimate, cols_estimate = self._estimate_grid_from_variance(image)

        rows_lines, cols_lines = self._extract_grid_from_lines(lines, h, w) if lines is not None and len(lines) > 4 else (0, 0)
        rows_contours, cols_contours = self._estimate_grid_from_contours(contours, h, w)

        # Vote between methods for robustness
        all_rows = [r for r in [rows_lines, rows_contours, rows_estimate] if r > 0]
        all_cols = [c for c in [cols_lines, cols_contours, cols_estimate] if c > 0]

        if all_rows and all_cols:
            rows = int(np.median(all_rows))
            cols = int(np.median(all_cols))
        elif rows_estimate > 0 and cols_estimate > 0:
            rows, cols = rows_estimate, cols_estimate
        else:
            rows, cols = 4, 6  # Default to standard size

        return (max(min(rows, 10), 2), max(min(cols, 12), 2))

    def _extract_grid_from_lines(self, lines: np.ndarray, h: int, w: int) -> Tuple[int, int]:
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < min(h, w) * 0.1: continue
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165: h_lines.append((y1 + y2) / 2)
            elif 75 < angle < 105: v_lines.append((x1 + x2) / 2)

        h_clusters = self._cluster_lines(h_lines, threshold=h * 0.05) if h_lines else []
        v_clusters = self._cluster_lines(v_lines, threshold=w * 0.05) if v_lines else []
        return (max(len(h_clusters) - 1, 0), max(len(v_clusters) - 1, 0))

    def _estimate_grid_from_contours(self, contours: list, h: int, w: int) -> Tuple[int, int]:
        if not contours: return (0, 0)
        valid_contours = []
        min_area, max_area = (h * w) * 0.005, (h * w) * 0.2
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if 0.5 < (cw / ch if ch > 0 else 0) < 2.0:
                    valid_contours.append((x, y, cw, ch))
        if len(valid_contours) < 4: return (0, 0)
        y_pos = sorted([y + ch / 2 for x, y, cw, ch in valid_contours])
        x_pos = sorted([x + cw / 2 for x, y, cw, ch in valid_contours])
        return (len(self._cluster_lines(y_pos, h * 0.08)), len(self._cluster_lines(x_pos, w * 0.08)))

    def _estimate_grid_from_variance(self, image: np.ndarray) -> Tuple[int, int]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        row_var, col_var = np.var(gray, axis=1), np.var(gray, axis=0)
        try:
            from scipy import signal
            row_var_smooth = signal.savgol_filter(row_var, min(51, len(row_var) // 2 * 2 + 1), 3)
            col_var_smooth = signal.savgol_filter(col_var, min(51, len(col_var) // 2 * 2 + 1), 3)
        except ImportError:
            kernel = np.ones(21) / 21
            row_var_smooth, col_var_smooth = np.convolve(row_var, kernel, mode="same"), np.convolve(col_var, kernel, mode="same")
        
        row_peaks = self._find_peaks_simple(row_var_smooth, np.percentile(row_var_smooth, 70), h * 0.05)
        col_peaks = self._find_peaks_simple(col_var_smooth, np.percentile(col_var_smooth, 70), w * 0.05)
        return (len(row_peaks) - 1 if len(row_peaks) > 1 else 0, len(col_peaks) - 1 if len(col_peaks) > 1 else 0)

    def _find_peaks_simple(self, signal: np.ndarray, threshold: float, min_distance: float) -> list:
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                if not peaks or abs(i - peaks[-1]) > min_distance:
                    peaks.append(i)
        return peaks

    def _cluster_lines(self, positions: List[float], threshold: float = 20) -> List[float]:
        if not positions: return []
        positions.sort()
        clusters = [[positions[0]]]
        for pos in positions[1:]:
            if pos - clusters[-1][-1] < threshold: clusters[-1].append(pos)
            else: clusters.append([pos])
        return [np.mean(cluster) for cluster in clusters]

    def _extract_patches_grid(self, image: np.ndarray, grid_size: Tuple[int, int], offset_x: int, offset_y: int) -> List[PatchInfo]:
        rows, cols = grid_size
        h, w = image.shape[:2]
        patch_h, patch_w = h // rows, w // cols
        patches = []
        for r in range(rows):
            for c in range(cols):
                margin = 5
                y1, y2 = r * patch_h + margin, (r + 1) * patch_h - margin
                x1, x2 = c * patch_w + margin, (c + 1) * patch_w - margin
                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)
                patch = image[y1:y2, x1:x2]
                if patch.size == 0: continue

                avg_color_bgr = cv2.mean(patch)[:3]
                patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
                avg_color_lab_raw = cv2.mean(patch_lab)[:3]

                patches.append(PatchInfo(
                    center=((x1 + x2) // 2 + offset_x, (y1 + y2) // 2 + offset_y),
                    color_rgb=(int(avg_color_bgr[2]), int(avg_color_bgr[1]), int(avg_color_bgr[0])),
                    color_lab=tuple(float(x) for x in avg_color_lab_raw),
                    bounding_box=(x1 + offset_x, y1 + offset_y, x2 - x1, y2 - y1),
                    index=len(patches)
                ))
        return patches

    def visualize_patches(self, patches: List[PatchInfo], show_numbers: bool = True) -> np.ndarray:
        vis = self.image.copy()
        for patch in patches:
            x, y, w, h = patch.bounding_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, patch.center, 5, (0, 0, 255), -1)
            if show_numbers:
                text = str(patch.index)
                cv2.putText(vis, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return vis