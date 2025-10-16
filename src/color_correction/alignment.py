import cv2
import numpy as np
from typing import List, Tuple, Optional
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


class Aligner:
    """Handles alignment of a color checker in an image, using automatic or manual methods."""

    def __init__(self, image: np.ndarray):
        """
        Initialize with an image containing a color checker.

        Args:
            image: BGR image from OpenCV.
        """
        self.original_image = image.copy()
        self.aligned_image = None
        self.transform_matrix = None

    def align_with_aruco(self, reference_image: np.ndarray, debug_mode: bool = False, output_dir: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Align the image using ArUco markers by comparing to a reference image.

        Args:
            reference_image: The reference image containing ArUco markers.
            debug_mode: If True, generate and return debug images.
            output_dir: Directory to save debug images.

        Returns:
            Aligned image if successful, otherwise None.
        """
        aligner = ArucoAligner(debug_mode=debug_mode, output_dir=output_dir)
        aligned_img, homography, _ = aligner.align_image_to_reference(self.original_image, reference_image)

        if aligned_img is not None:
            self.aligned_image = aligned_img
            self.transform_matrix = homography
            return self.aligned_image
        
        return None

    def align_with_manual_points(self, points: List[Tuple[int, int]], target_size: Tuple[int, int] = (1000, 700)) -> Optional[np.ndarray]:
        """
        Align a quadrilateral defined by user-provided points to a rectangle.

        Args:
            points: List of 4 points defining the quadrilateral (top-left, top-right, bottom-right, bottom-left).
            target_size: Desired size (width, height) of the aligned rectangle.

        Returns:
            Aligned image.
        """
        if not points or len(points) != 4:
            print("[ERROR] Manual alignment requires exactly 4 points.")
            return None

        # Order points: top-left, top-right, bottom-right, bottom-left
        ordered_points = self._order_points(np.array(points, dtype=np.float32))

        # Define destination points for the perspective transform
        dst_points = np.array(
            [
                [0, 0],
                [target_size[0] - 1, 0],
                [target_size[0] - 1, target_size[1] - 1],
                [0, target_size[1] - 1],
            ],
            dtype=np.float32,
        )

        # Compute and apply the perspective transform
        self.transform_matrix = cv2.getPerspectiveTransform(ordered_points, dst_points)
        self.aligned_image = cv2.warpPerspective(
            self.original_image, self.transform_matrix, target_size
        )

        return self.aligned_image

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 points in clockwise order starting from top-left.

        Args:
            pts: Array of 4 points.

        Returns:
            Ordered points array.
        """
        # Sort by y-coordinate to find top and bottom pairs
        sorted_pts = pts[np.argsort(pts[:, 1])]

        # Top two points, sorted by x-coordinate
        top_pts = sorted_pts[:2]
        top_pts = top_pts[np.argsort(top_pts[:, 0])]

        # Bottom two points, sorted by x-coordinate
        bottom_pts = sorted_pts[2:]
        bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]

        # Return in [top-left, top-right, bottom-right, bottom-left] order
        return np.array(
            [top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32
        )
