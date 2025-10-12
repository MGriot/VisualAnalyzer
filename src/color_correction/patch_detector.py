import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os
from src.ColorCheckerGenerator.colorchecker.generator import ColorCheckerGenerator
from src.geometric_alignment.geometric_aligner import ArucoAligner
from src import config


def get_or_generate_reference_checker(filename: str = "generated_cc_with_aruco.png") -> np.ndarray:
    """
    Generates a standard ColorChecker with ArUco markers if it doesn't exist, then loads it.
    """
    filepath = config.REFERENCE_COLOR_CHECKERS_DIR / filename
    
    if not filepath.exists():
        print(f"Reference color checker not found. Generating at {filepath}...")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        gen = ColorCheckerGenerator(
            size="20cm", 
            dpi=300,
            checker_type="classic",
            include_aruco=True,
            logo_text="Reference"
        )
        canvas = gen.build()
        gen.save(str(filepath))
        print("Reference generated.")
        return canvas
    
    return cv2.imread(str(filepath))


@dataclass
class PatchInfo:
    """Information about a detected patch"""

    center: Tuple[int, int]
    color_rgb: Tuple[int, int, int]
    color_lab: Tuple[float, float, float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    index: int


class ColorCheckerAligner:
    """Core logic for aligning and extracting patches from a color checker"""

    def __init__(self, image: np.ndarray):
        """
        Initialize with an image

        Args:
            image: BGR image from OpenCV
        """
        self.original_image = image.copy()
        self.aligned_image = None
        self.transform_matrix = None

    def align_with_aruco(self, reference_image: np.ndarray, target_size: Tuple[int, int] = (1000, 700)) -> Optional[np.ndarray]:
        """
        Align the image using ArUco markers by comparing to a reference image.

        Args:
            reference_image: The reference image containing ArUco markers.
            target_size: Desired size (width, height) of the aligned rectangle.

        Returns:
            Aligned image if successful, otherwise None.
        """
        aligner = ArucoAligner(debug_mode=False) # Or pass a debug flag
        aligned_img, homography, _ = aligner.align_image_to_reference(self.original_image, reference_image)

        if aligned_img is not None:
            # The ArucoAligner already warps the image to match the reference's dimensions.
            # We need to resize it to the expected target_size for consistency with the manual method.
            self.aligned_image = cv2.resize(aligned_img, target_size, interpolation=cv2.INTER_AREA)
            # We don't have a transform_matrix in the same way, but we could store the homography if needed.
            self.transform_matrix = homography
            return self.aligned_image
        
        return None

    def align_rectangle(
        self, points: List[Tuple[int, int]], target_size: Tuple[int, int] = (1000, 700)
    ) -> np.ndarray:
        """
        Align a quadrilateral defined by points to a rectangle

        Args:
            points: List of 4 points defining the quadrilateral (top-left, top-right, bottom-right, bottom-left)
            target_size: Desired size (width, height) of aligned rectangle

        Returns:
            Aligned image
        """
        if len(points) != 4:
            raise ValueError("Exactly 4 points are required for alignment")

        # Order points: top-left, top-right, bottom-right, bottom-left
        points = self._order_points(np.array(points, dtype=np.float32))

        # Define destination points
        dst_points = np.array(
            [
                [0, 0],
                [target_size[0] - 1, 0],
                [target_size[0] - 1, target_size[1] - 1],
                [0, target_size[1] - 1],
            ],
            dtype=np.float32,
        )

        # Compute perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(points, dst_points)

        # Apply perspective transform
        self.aligned_image = cv2.warpPerspective(
            self.original_image, self.transform_matrix, target_size
        )

        return self.aligned_image

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left

        Args:
            pts: Array of 4 points

        Returns:
            Ordered points array
        """
        # Sort by y-coordinate
        sorted_pts = pts[np.argsort(pts[:, 1])]

        # Top two points
        top_pts = sorted_pts[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]  # Sort by x

        # Bottom two points
        bottom_pts = sorted_pts[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]  # Sort by x

        # Return in order: top-left, top-right, bottom-right, bottom-left
        return np.array(
            [top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32
        )

    def detect_patches(
        self,
        grid_size: Optional[Tuple[int, int]] = None,
        margin_ratio: float = 0.05,
        adaptive: bool = True,
    ) -> List[PatchInfo]:
        """
        Detect patches in the aligned image

        Args:
            grid_size: Expected (rows, cols) of patches. If None, auto-detect
            margin_ratio: Ratio of margin to exclude from edges
            adaptive: If True, automatically detect grid configuration

        Returns:
            List of PatchInfo objects ordered from top-left to bottom-right
        """
        if self.aligned_image is None:
            raise ValueError("Image must be aligned first using align_rectangle()")

        h, w = self.aligned_image.shape[:2]

        # Calculate working area (exclude margins)
        margin_h = int(h * margin_ratio)
        margin_w = int(w * margin_ratio)
        working_area = self.aligned_image[
            margin_h : h - margin_h, margin_w : w - margin_w
        ]

        if adaptive or grid_size is None:
            # Auto-detect grid configuration
            grid_size = self._detect_grid_size(working_area)

        patches = self._extract_patches_grid(
            working_area, grid_size, margin_w, margin_h
        )

        return patches

    def _detect_grid_size(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Automatically detect the grid size of patches using robust methods

        Args:
            image: Working area image

        Returns:
            (rows, cols) tuple
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multiple detection methods for robustness

        # Method 1: Edge-based detection with improved parameters
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)

        # Dilate edges slightly to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Detect lines with more lenient parameters
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=int(min(w, h) * 0.15),
            maxLineGap=20,
        )

        # Method 2: Morphological operations to find grid structure
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours for patch detection
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Method 3: Analyze color variance to find patch boundaries
        rows_estimate, cols_estimate = self._estimate_grid_from_variance(image)

        # Combine methods
        if lines is not None and len(lines) > 4:
            rows_lines, cols_lines = self._extract_grid_from_lines(lines, h, w)
        else:
            rows_lines, cols_lines = (0, 0)

        # Use contours as backup
        rows_contours, cols_contours = self._estimate_grid_from_contours(contours, h, w)

        # Vote between methods
        all_rows = [r for r in [rows_lines, rows_contours, rows_estimate] if r > 0]
        all_cols = [c for c in [cols_lines, cols_contours, cols_estimate] if c > 0]

        if all_rows and all_cols:
            rows = int(np.median(all_rows))
            cols = int(np.median(all_cols))
        elif rows_estimate > 0 and cols_estimate > 0:
            rows, cols = rows_estimate, cols_estimate
        else:
            # Default to standard color checker size
            rows, cols = 4, 6

        # Ensure reasonable values
        rows = max(min(rows, 10), 2)
        cols = max(min(cols, 12), 2)

        return (rows, cols)

    def _extract_grid_from_lines(
        self, lines: np.ndarray, h: int, w: int
    ) -> Tuple[int, int]:
        """Extract grid dimensions from detected lines"""
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Skip very short lines
            if length < min(h, w) * 0.1:
                continue

            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Horizontal lines
            if angle < 15 or angle > 165:
                h_lines.append((y1 + y2) / 2)
            # Vertical lines
            elif 75 < angle < 105:
                v_lines.append((x1 + x2) / 2)

        # Cluster lines
        h_clusters = self._cluster_lines(h_lines, threshold=h * 0.05) if h_lines else []
        v_clusters = self._cluster_lines(v_lines, threshold=w * 0.05) if v_lines else []

        rows = max(len(h_clusters) - 1, 0)
        cols = max(len(v_clusters) - 1, 0)

        return (rows, cols)

    def _estimate_grid_from_contours(
        self, contours: list, h: int, w: int
    ) -> Tuple[int, int]:
        """Estimate grid size from contours"""
        if not contours:
            return (0, 0)

        # Filter contours by area
        min_area = (h * w) * 0.005  # At least 0.5% of image
        max_area = (h * w) * 0.2  # At most 20% of image

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Check aspect ratio is reasonable
                aspect = cw / ch if ch > 0 else 0
                if 0.5 < aspect < 2.0:
                    valid_contours.append((x, y, cw, ch))

        if len(valid_contours) < 4:
            return (0, 0)

        # Cluster by position to find grid
        y_positions = sorted([y + ch / 2 for x, y, cw, ch in valid_contours])
        x_positions = sorted([x + cw / 2 for x, y, cw, ch in valid_contours])

        y_clusters = self._cluster_lines(y_positions, threshold=h * 0.08)
        x_clusters = self._cluster_lines(x_positions, threshold=w * 0.08)

        return (len(y_clusters), len(x_clusters))

    def _estimate_grid_from_variance(self, image: np.ndarray) -> Tuple[int, int]:
        """Estimate grid by analyzing color variance across rows and columns"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate variance along each axis
        row_var = np.var(gray, axis=1)
        col_var = np.var(gray, axis=0)

        # Smooth the variance
        from scipy import signal

        try:
            row_var_smooth = signal.savgol_filter(
                row_var, min(51, len(row_var) // 2 * 2 + 1), 3
            )
            col_var_smooth = signal.savgol_filter(
                col_var, min(51, len(col_var) // 2 * 2 + 1), 3
            )
        except:
            # Fallback to simple smoothing if scipy not available
            kernel_size = 21
            kernel = np.ones(kernel_size) / kernel_size
            row_var_smooth = np.convolve(row_var, kernel, mode="same")
            col_var_smooth = np.convolve(col_var, kernel, mode="same")

        # Find peaks (grid lines) in variance
        row_threshold = np.percentile(row_var_smooth, 70)
        col_threshold = np.percentile(col_var_smooth, 70)

        # Find transitions (peaks in variance indicate grid lines)
        row_peaks = self._find_peaks_simple(row_var_smooth, row_threshold, h * 0.05)
        col_peaks = self._find_peaks_simple(col_var_smooth, col_threshold, w * 0.05)

        rows = len(row_peaks) - 1 if len(row_peaks) > 1 else 0
        cols = len(col_peaks) - 1 if len(col_peaks) > 1 else 0

        return (rows, cols)

    def _find_peaks_simple(
        self, signal: np.ndarray, threshold: float, min_distance: float
    ) -> list:
        """Simple peak finding without scipy"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if (
                signal[i] > threshold
                and signal[i] > signal[i - 1]
                and signal[i] > signal[i + 1]
            ):
                # Check minimum distance from previous peak
                if not peaks or abs(i - peaks[-1]) > min_distance:
                    peaks.append(i)
        return peaks

    def _cluster_lines(
        self, positions: List[float], threshold: float = 20
    ) -> List[float]:
        """
        Cluster line positions that are close together

        Args:
            positions: List of line positions
            threshold: Distance threshold for clustering

        Returns:
            List of cluster centers
        """
        if not positions:
            return []

        positions = sorted(positions)
        clusters = [[positions[0]]]

        for pos in positions[1:]:
            if pos - clusters[-1][-1] < threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])

        return [np.mean(cluster) for cluster in clusters]

    def _extract_patches_grid(
        self,
        image: np.ndarray,
        grid_size: Tuple[int, int],
        offset_x: int,
        offset_y: int,
    ) -> List[PatchInfo]:
        """
        Extract patches using grid-based approach

        Args:
            image: Working area image
            grid_size: (rows, cols) tuple
            offset_x: X offset from margin
            offset_y: Y offset from margin

        Returns:
            List of PatchInfo objects
        """
        rows, cols = grid_size
        h, w = image.shape[:2]

        patch_h = h // rows
        patch_w = w // cols

        patches = []
        idx = 0

        for r in range(rows):
            for c in range(cols):
                # Calculate patch boundaries with small margin
                margin = 5
                y1 = r * patch_h + margin
                y2 = (r + 1) * patch_h - margin
                x1 = c * patch_w + margin
                x2 = (c + 1) * patch_w - margin

                # Ensure boundaries are valid
                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)

                # Extract patch
                patch = image[y1:y2, x1:x2]

                if patch.size == 0:
                    continue

                # Calculate center
                center_x = (x1 + x2) // 2 + offset_x
                center_y = (y1 + y2) // 2 + offset_y

                # Calculate average color
                avg_color_bgr = cv2.mean(patch)[:3]
                avg_color_rgb = (
                    int(avg_color_bgr[2]),
                    int(avg_color_bgr[1]),
                    int(avg_color_bgr[0]),
                )

                # Convert to LAB for better color representation
                patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
                avg_color_lab_raw = cv2.mean(patch_lab)[:3]
                avg_color_lab = tuple(float(x) for x in avg_color_lab_raw)

                patch_info = PatchInfo(
                    center=(center_x, center_y),
                    color_rgb=avg_color_rgb,
                    color_lab=avg_color_lab,
                    bounding_box=(x1 + offset_x, y1 + offset_y, x2 - x1, y2 - y1),
                    index=idx,
                )

                patches.append(patch_info)
                idx += 1

        return patches

    def visualize_patches(
        self, patches: List[PatchInfo], show_numbers: bool = True
    ) -> np.ndarray:
        """
        Create visualization of detected patches

        Args:
            patches: List of PatchInfo objects
            show_numbers: Whether to show patch numbers

        Returns:
            Visualization image
        """
        if self.aligned_image is None:
            raise ValueError("No aligned image available")

        vis = self.aligned_image.copy()

        for patch in patches:
            x, y, w, h = patch.bounding_box

            # Draw rectangle
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(vis, patch.center, 5, (0, 0, 255), -1)

            # Draw patch number
            if show_numbers:
                text = str(patch.index)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(vis, text, (x + 5, y + 20), font, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, text, (x + 5, y + 20), font, 0.5, (0, 0, 0), 1)

        return vis
