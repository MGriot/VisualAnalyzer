<<<<<<< Updated upstream
version https://git-lfs.github.com/spec/v1
oid sha256:d89ff875b4634c50c3db57408c707027b63e780422541cb7ce98c3372f5818c6
size 19138
=======
"""
This module provides the `AdvancedAligner` class for robust image alignment
using various computer vision techniques.

It supports feature-based alignment (ORB/SIFT), ECC maximization,
contour centroid alignment, and a new default method based on circumscribed
geometric shapes, prioritizing pentagons and falling back to quadrilaterals.
"""

import cv2
import numpy as np
import os
from src.utils.image_utils import save_image

# This class is well-formed and requires no corrections.
# Ensure it is defined or imported before calling the align_image function.


class AdvancedAligner:
    """
    A comprehensive image aligner class offering various alignment methods.

    The default method, 'geometric_shape', attempts to align using a 5-point
    pentagon circumscribed around the object, falling back to a 4-point
    quadrilateral if a pentagon is not detected.
    """

    def __init__(
        self,
        max_features=2000,
        good_match_percent=0.15,
        motion_model="homography",
        ecc_iters=5000,
        ecc_eps=1e-8,
        min_contour_area=100,
        poly_epsilon_ratio=0.02,  # Crucial for polygon approximation
        edge_method="canny",
        corner_method="shi-tomasi",
        debug_mode=False,
        output_dir=None,
        default_align_method="geometric_shape",
    ):
        """
        Initializes the AdvancedAligner.

        Args:
            max_features (int): Max keypoints for ORB/SIFT.
            good_match_percent (float): Fraction of good feature matches.
            motion_model (str): Transform model ('affine', 'homography'). Defaults to "homography".
            min_contour_area (int): Minimum contour area for detection.
            poly_epsilon_ratio (float): Approximation precision for simplifying contours
                                        into polygons. A key parameter for pentagon detection.
            debug_mode (bool): Enables saving of intermediate debug images.
            output_dir (str): Directory to save debug images.
            default_align_method (str): Default alignment method. Defaults to "geometric_shape".
        """
        self.max_features = max_features
        self.good_match_percent = good_match_percent
        self.motion_model = motion_model
        self.ecc_iters = ecc_iters
        self.ecc_eps = ecc_eps
        self.min_contour_area = min_contour_area
        self.poly_epsilon_ratio = poly_epsilon_ratio
        self.edge_method = edge_method
        self.corner_method = corner_method
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        self.default_align_method = default_align_method

        self.orb = (
            cv2.ORB_create(self.max_features) if hasattr(cv2, "ORB_create") else None
        )
        self.sift = (
            cv2.SIFT_create(self.max_features) if hasattr(cv2, "SIFT_create") else None
        )

    def _find_largest_contour(self, img):
        """Helper to find the largest contour in an image."""
        gray = self.preprocess_gray(img)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Add a dilation step to close small gaps in the object
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = [
            c for c in contours if cv2.contourArea(c) > self.min_contour_area
        ]
        if not valid_contours:
            return None
        return max(valid_contours, key=cv2.contourArea)

    def _order_points_quad(self, pts):
        """
        Sorts 4 points for a quadrilateral in a consistent order:
        top-left, top-right, bottom-right, bottom-left.
        """
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _order_polygon_points(self, pts):
        """
        Sorts vertices of a convex polygon into clockwise order using their
        angle with respect to the centroid.
        """
        pts = pts.reshape(-1, 2)
        centroid = np.mean(pts, axis=0)
        # Sort by angle around the centroid
        sorted_pts = sorted(
            pts, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return np.array(sorted_pts, dtype="float32")

    def preprocess_gray(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def align_by_feature(self, img_to_align, ref_img, use_sift=False):
        """Aligns images using feature detection (ORB or SIFT)."""
        # (This method remains unchanged)
        detector = self.sift if use_sift and self.sift else self.orb
        if detector is None:
            raise RuntimeError("Selected feature detector not available.")
        norm = cv2.NORM_L2 if use_sift and self.sift else cv2.NORM_HAMMING

        gray_to_align, gray_ref = self.preprocess_gray(
            img_to_align
        ), self.preprocess_gray(ref_img)
        kp1, des1 = detector.detectAndCompute(gray_to_align, None)
        kp2, des2 = detector.detectAndCompute(gray_ref, None)

        if des1 is None or des2 is None:
            return None, {}

        bf = cv2.BFMatcher(norm, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [
            m
            for m, n in matches
            if len(matches[0]) == 2 and m.distance < 0.75 * n.distance
        ]

        debug_paths = {}
        if self.debug_mode and self.output_dir and good_matches:
            img_matches = cv2.drawMatches(
                img_to_align, kp1, ref_img, kp2, good_matches, None
            )
            debug_paths["feature_matches"] = os.path.join(
                self.output_dir, "debug_feature_matches.png"
            )
            save_image(debug_paths["feature_matches"], img_matches)

        if len(good_matches) < 10:
            return None, debug_paths

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        M, _ = (
            cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if self.motion_model == "homography"
            else cv2.estimateAffine2D(src_pts, dst_pts)
        )
        if M is None:
            return None, debug_paths

        h, w = ref_img.shape[:2]
        warp_func = (
            cv2.warpPerspective if self.motion_model == "homography" else cv2.warpAffine
        )
        return warp_func(img_to_align, M, (w, h)), debug_paths

    def align_by_geometric_shape(self, src, ref):
        """
        Aligns images by circumscribing the object with a polygon.
        It prioritizes a 5-vertex pentagon and falls back to a 4-vertex
        quadrilateral if pentagons are not found in both images.
        """
        debug_paths = {}
        src_contour = self._find_largest_contour(src)
        ref_contour = self._find_largest_contour(ref)
        if src_contour is None or ref_contour is None:
            raise RuntimeError(
                "Could not find a dominant contour in one or both images."
            )

        # Attempt to approximate contours to polygons
        src_epsilon = self.poly_epsilon_ratio * cv2.arcLength(src_contour, True)
        ref_epsilon = self.poly_epsilon_ratio * cv2.arcLength(ref_contour, True)
        src_poly = cv2.approxPolyDP(src_contour, src_epsilon, True)
        ref_poly = cv2.approxPolyDP(ref_contour, ref_epsilon, True)

        if self.debug_mode:
            print(f"[DEBUG] Source polygon approximation has {len(src_poly)} vertices.")
            print(
                f"[DEBUG] Reference polygon approximation has {len(ref_poly)} vertices."
            )

        # **CORE LOGIC: Prioritize Pentagon (5 points)**
        if len(src_poly) == 5 and len(ref_poly) == 5:
            print(
                "[INFO] Pentagon detected for both images. Using pentagon-based alignment."
            )
            # Use the general polygon sorter for the 5 points
            src_pts = self._order_polygon_points(src_poly)
            ref_pts = self._order_polygon_points(ref_poly)
            shape_used = "pentagon"
        else:
            # **FALLBACK: Use Quadrilateral (4 points) from minAreaRect**
            if len(src_poly) != 5 or len(ref_poly) != 5:
                print(
                    "[INFO] Pentagon not found or vertex counts mismatch. Falling back to quadrilateral alignment."
                )

            src_rect = cv2.minAreaRect(src_contour)
            ref_rect = cv2.minAreaRect(ref_contour)
            # Use the specialized quadrilateral sorter for the 4 points
            src_pts = self._order_points_quad(cv2.boxPoints(src_rect))
            ref_pts = self._order_points_quad(cv2.boxPoints(ref_rect))
            shape_used = "quadrilateral"

        # --- DEBUGGING STEPS ---
        if self.debug_mode and self.output_dir:
            print(f"[DEBUG] Aligning with {shape_used} shape.")
            src_debug, ref_debug = src.copy(), ref.copy()
            # Draw contours and ordered vertices on debug images
            cv2.polylines(src_debug, [src_pts.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(ref_debug, [ref_pts.astype(np.int32)], True, (0, 255, 0), 2)
            for i, p in enumerate(src_pts):
                cv2.putText(
                    src_debug,
                    str(i),
                    tuple(p.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )
            for i, p in enumerate(ref_pts):
                cv2.putText(
                    ref_debug,
                    str(i),
                    tuple(p.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

            debug_paths[f"geom_src_{shape_used}"] = os.path.join(
                self.output_dir, f"debug_geom_src_{shape_used}.png"
            )
            debug_paths[f"geom_ref_{shape_used}"] = os.path.join(
                self.output_dir, f"debug_geom_ref_{shape_used}.png"
            )
            save_image(debug_paths[f"geom_src_{shape_used}"], src_debug)
            save_image(debug_paths[f"geom_ref_{shape_used}"], ref_debug)
        # --- END DEBUGGING ---

        M, _ = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
        if M is None:
            raise RuntimeError("Homography computation failed with geometric points.")

        h, w = ref.shape[:2]
        aligned_img = cv2.warpPerspective(src, M, (w, h))
        return aligned_img, debug_paths

    def align_by_contour_centroid(self, src, ref):
        """Aligns by matching the centroids of their largest contours."""
        # (This method remains unchanged)
        src_contour = self._find_largest_contour(src)
        ref_contour = self._find_largest_contour(ref)
        if src_contour is None or ref_contour is None:
            raise RuntimeError("No valid contours found for centroid alignment")

        M_src = cv2.moments(src_contour)
        M_ref = cv2.moments(ref_contour)
        if M_src["m00"] == 0 or M_ref["m00"] == 0:
            raise RuntimeError("Cannot compute centroid due to zero-area contour.")

        cx_src, cy_src = int(M_src["m10"] / M_src["m00"]), int(
            M_src["m01"] / M_src["m00"]
        )
        cx_ref, cy_ref = int(M_ref["m10"] / M_ref["m00"]), int(
            M_ref["m01"] / M_ref["m00"]
        )

        dx, dy = cx_ref - cx_src, cy_ref - cy_src
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(src, M, (ref.shape[1], ref.shape[0]))
        return aligned, {}

    def align(self, src, ref, method=None):
        """
        Main alignment interface that dispatches to different alignment methods.
        """
        method = method or self.default_align_method
        aligned_image, debug_paths = None, {}

        try:
            if method == "geometric_shape":
                aligned_image, debug_paths = self.align_by_geometric_shape(src, ref)
            elif method in ["feature_orb", "feature_sift"]:
                aligned_image, debug_paths = self.align_by_feature(
                    src, ref, use_sift=(method == "feature_sift")
                )
            elif method == "contour_centroid":
                aligned_image, debug_paths = self.align_by_contour_centroid(src, ref)
            else:
                raise ValueError(f"Unknown alignment method: {method}")
        except Exception as e:
            print(f"[Alignment Error] Method '{method}' failed: {e}")
            return {"image": None, "debug_paths": debug_paths}

        if aligned_image is not None and self.output_dir:
            final_path = os.path.join(self.output_dir, f"final_aligned_{method}.png")
            save_image(final_path, aligned_image)
            debug_paths["final_aligned_image"] = final_path

        return {"image": aligned_image, "debug_paths": debug_paths}
>>>>>>> Stashed changes
