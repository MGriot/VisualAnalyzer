"""
This module provides the AdvancedAligner class for robust image alignment
using various computer vision techniques.

It supports feature-based alignment (ORB/SIFT), a robust method based on
contour pose estimation, and a new default geometric method that prioritizes
a 5-point pentagon and falls back to a 4-point quadrilateral.
"""

import cv2
import numpy as np
import os
import math


def save_image(path, image):
    """
    Saves an image to a specified path, creating directories if they don't exist.

    Args:
        path (str): The full path where the image will be saved.
        image (np.ndarray): The image data to save.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")


class AdvancedAligner:
    """
    A comprehensive image aligner that offers various alignment methods.

    The new default method, 'geometric_shape', attempts to align using a 5-point
    pentagon circumscribed around the object, falling back to a 4-point
    quadrilateral if a pentagon is not detected.

    Args:
        max_features (int): Maximum features to detect for ORB/SIFT.
        min_contour_area (int): The minimum area for a contour to be considered valid.
        poly_epsilon_ratio (float): Ratio for approximating polygons. Crucial for pentagon detection.
        debug_mode (bool): If True, saves intermediate images for debugging.
        output_dir (str): Directory to save debug images. Required if `debug_mode` is True.
        default_align_method (str): The default alignment method to use.
        shadow_removal_method (str): The default shadow removal technique.
    """

    def __init__(
        self,
        max_features=2000,
        min_contour_area=100,
        poly_epsilon_ratio=0.02,
        debug_mode=False,
        output_dir=None,
        default_align_method="geometric_shape",  # Changed default to the new method
        shadow_removal_method="clahe",
    ):
        self.max_features = max_features
        self.min_contour_area = min_contour_area
        self.poly_epsilon_ratio = poly_epsilon_ratio
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        self.default_align_method = default_align_method
        self.shadow_removal_method = shadow_removal_method

        if self.debug_mode and not self.output_dir:
            raise ValueError("output_dir must be provided when debug_mode is True.")

        self.orb = (
            cv2.ORB_create(self.max_features) if hasattr(cv2, "ORB_create") else None
        )
        self.sift = (
            cv2.SIFT_create(self.max_features) if hasattr(cv2, "SIFT_create") else None
        )

    def _save_debug_image(self, name, image, debug_paths):
        """Saves a debug image if debug_mode is enabled."""
        if self.debug_mode and self.output_dir:
            path = os.path.join(self.output_dir, f"debug_{name}.png")
            save_image(path, image)
            debug_paths[name] = path

    def _apply_clahe_contrast(self, img):
        """Enhances contrast using CLAHE in the L*a*b* color space."""
        if len(img.shape) != 3:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_simple_gamma(self, img, gamma=1.5):
        """Applies a simple gamma correction to brighten the image."""
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(img, table)

    def _find_largest_contour(self, img):
        """
        Helper to find the largest contour in an image after preprocessing.
        This version adds a fallback to the second-largest contour if the largest
        one is likely the image frame itself.
        """
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )
        # Use OTSU's thresholding for robust binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use morphological closing to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by minimum area and sort them from largest to smallest
        valid_contours = sorted(
            [c for c in contours if cv2.contourArea(c) > self.min_contour_area],
            key=cv2.contourArea,
            reverse=True
        )

        if not valid_contours:
            return None

        # Check if the largest contour is suspiciously large (e.g., >95% of image area)
        image_area = img.shape[0] * img.shape[1]
        largest_contour_area = cv2.contourArea(valid_contours[0])
        
        # If the largest contour is almost the size of the whole image and a second contour exists,
        # it's likely the frame. In that case, choose the second largest.
        if len(valid_contours) > 1 and (largest_contour_area / image_area) > 0.95:
            if self.debug_mode:
                print(f"[DEBUG] Largest contour area ({largest_contour_area}) is >95% of image area ({image_area}). Falling back to second-largest contour.")
            return valid_contours[1] # Return the second largest
        
        # Otherwise, return the largest contour as intended
        return valid_contours[0]

    # --- NEW HELPER METHODS for POINT ORDERING ---
    def _order_points_quad(self, pts):
        """
        Sorts 4 points for a quadrilateral in a consistent order:
        top-left, top-right, bottom-right, bottom-left.
        This ensures correct orientation mapping.
        """
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
        return rect

    def _order_polygon_points(self, pts):
        """
        Sorts vertices of a convex polygon into clockwise order using their
        angle with respect to the centroid. This works for any N-sided polygon.
        """
        pts = pts.reshape(-1, 2)
        centroid = np.mean(pts, axis=0)
        # Sort by angle around the centroid
        sorted_pts = sorted(
            pts, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return np.array(sorted_pts, dtype="float32")

    # --- NEW ALIGNMENT METHOD ---
    def align_by_geometric_shape(self, src_processed, ref_processed):
        """
        Aligns images by circumscribing the object with a polygon.
        It prioritizes a 5-vertex pentagon and falls back to a 4-vertex
        quadrilateral if pentagons are not found in both images.
        """
        debug_paths = {}
        src_contour = self._find_largest_contour(src_processed)
        ref_contour = self._find_largest_contour(ref_processed)
        if src_contour is None or ref_contour is None:
            raise RuntimeError(
                "Could not find a dominant contour in one or both images."
            )

        # Attempt to approximate contours to polygons
        src_epsilon = self.poly_epsilon_ratio * cv2.arcLength(src_contour, True)
        ref_epsilon = self.poly_epsilon_ratio * cv2.arcLength(ref_contour, True)
        src_poly = cv2.approxPolyDP(src_contour, src_epsilon, True)
        ref_poly = cv2.approxPolyDP(ref_contour, ref_epsilon, True)

        print(f"[INFO] Source polygon approximation has {len(src_poly)} vertices.")
        print(f"[INFO] Reference polygon approximation has {len(ref_poly)} vertices.")

        # **CORE LOGIC: Prioritize Pentagon (5 points)**
        if len(src_poly) == 5 and len(ref_poly) == 5:
            print(
                "[INFO] Pentagon detected in both images. Using pentagon-based alignment."
            )
            src_pts = self._order_polygon_points(src_poly)
            ref_pts = self._order_polygon_points(ref_poly)
            shape_used = "pentagon"
        else:
            # **FALLBACK: Use Quadrilateral (4 points) from minAreaRect**
            print(
                "[INFO] Pentagon not found or vertex counts mismatch. Falling back to quadrilateral alignment."
            )
            src_rect = cv2.minAreaRect(src_contour)
            ref_rect = cv2.minAreaRect(ref_contour)
            src_pts = self._order_points_quad(cv2.boxPoints(src_rect))
            ref_pts = self._order_points_quad(cv2.boxPoints(ref_rect))
            shape_used = "quadrilateral"

        # --- Generate Debug Images ---
        src_debug, ref_debug = src_processed.copy(), ref_processed.copy()
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

        self._save_debug_image(f"03_geom_src_{shape_used}", src_debug, debug_paths)
        self._save_debug_image(f"03_geom_ref_{shape_used}", ref_debug, debug_paths)
        # --- End Debugging ---

        M, _ = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
        if M is None:
            raise RuntimeError("Homography computation failed with geometric points.")

        return M, debug_paths

    # --- EXISTING METHODS (Unchanged) ---
    def align_by_feature(self, src_processed, ref_processed, use_sift=False):
        """Aligns images based on feature matching (ORB or SIFT)."""
        debug_paths = {}
        detector = self.sift if use_sift and self.sift else self.orb
        if detector is None:
            raise RuntimeError(
                f"Feature detector {'SIFT' if use_sift else 'ORB'} not available."
            )
        gray_src = cv2.cvtColor(src_processed, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_processed, cv2.COLOR_BGR2GRAY)
        kp1, des1 = detector.detectAndCompute(gray_src, None)
        kp2, des2 = detector.detectAndCompute(gray_ref, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, {}

        norm = cv2.NORM_L2 if use_sift and self.sift else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        img_matches = cv2.drawMatches(
            src_processed, kp1, ref_processed, kp2, good_matches, None
        )
        self._save_debug_image("02_feature_matches", img_matches, debug_paths)

        if len(good_matches) < 10:
            return None, debug_paths

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M, debug_paths

    def align_by_contour_pose(self, src_processed, ref_processed):
        """Aligns images by estimating the pose of their largest contours."""
        debug_paths = {}
        # This now uses the old _find_largest_contour logic. To make it use the new one, we would need to adapt it.
        # For now, let's keep its original logic separate.
        gray_src = (
            cv2.cvtColor(src_processed, cv2.COLOR_BGR2GRAY)
            if len(src_processed.shape) == 3
            else src_processed.copy()
        )
        binary_src = cv2.adaptiveThreshold(
            gray_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours_src, _ = cv2.findContours(
            binary_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        src_contour = max(
            [c for c in contours_src if cv2.contourArea(c) > self.min_contour_area],
            key=cv2.contourArea,
        )
        M_src = cv2.moments(src_contour)
        src_centroid = (
            int(M_src["m10"] / M_src["m00"]),
            int(M_src["m01"] / M_src["m00"]),
        )

        gray_ref = (
            cv2.cvtColor(ref_processed, cv2.COLOR_BGR2GRAY)
            if len(ref_processed.shape) == 3
            else ref_processed.copy()
        )
        binary_ref = cv2.adaptiveThreshold(
            gray_ref, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours_ref, _ = cv2.findContours(
            binary_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ref_contour = max(
            [c for c in contours_ref if cv2.contourArea(c) > self.min_contour_area],
            key=cv2.contourArea,
        )
        M_ref = cv2.moments(ref_contour)
        ref_centroid = (
            int(M_ref["m10"] / M_ref["m00"]),
            int(M_ref["m01"] / M_ref["m00"]),
        )

        src_rect = cv2.minAreaRect(src_contour)
        ref_rect = cv2.minAreaRect(ref_contour)
        src_angle = src_rect[2] + (90 if src_rect[1][0] < src_rect[1][1] else 0)
        ref_angle = ref_rect[2] + (90 if ref_rect[1][0] < ref_rect[1][1] else 0)
        angle_diff = ref_angle - src_angle
        src_area = max(1, src_rect[1][0] * src_rect[1][1])
        ref_area = ref_rect[1][0] * ref_rect[1][1]
        scale = math.sqrt(ref_area / src_area)

        M = cv2.getRotationMatrix2D(src_centroid, angle_diff, scale)
        rotated_src_centroid = M @ np.array([src_centroid[0], src_centroid[1], 1])
        M[0, 2] += ref_centroid[0] - rotated_src_centroid[0]
        M[1, 2] += ref_centroid[1] - rotated_src_centroid[1]

        return M, debug_paths

    # --- MAIN ALIGNMENT DISPATCHER (Updated) ---
    def align(self, src, ref, method=None, shadow_removal=None):
        """
        Main alignment interface that dispatches to different alignment methods.
        """
        method = method or self.default_align_method
        shadow_method = shadow_removal or self.shadow_removal_method
        debug_paths = {}

        self._save_debug_image("00_input_source", src, debug_paths)
        self._save_debug_image("00_input_reference", ref, debug_paths)

        try:
            src_processed, ref_processed = src.copy(), ref.copy()
            if shadow_method == "clahe":
                src_processed = self._apply_clahe_contrast(src_processed)
                ref_processed = self._apply_clahe_contrast(ref_processed)
                self._save_debug_image(
                    "01_src_clahe_enhanced", src_processed, debug_paths
                )
                self._save_debug_image(
                    "01_ref_clahe_enhanced", ref_processed, debug_paths
                )
            elif shadow_method == "gamma":
                src_processed = self._apply_simple_gamma(src_processed)
                ref_processed = self._apply_simple_gamma(ref_processed)
                self._save_debug_image(
                    "01_src_gamma_corrected", src_processed, debug_paths
                )
                self._save_debug_image(
                    "01_ref_gamma_corrected", ref_processed, debug_paths
                )
            elif shadow_method not in ["none", None]:
                print(
                    f"[Warning] Unknown shadow removal method '{shadow_method}'. Skipping."
                )

            M, dbg = None, {}
            print(f"[INFO] Attempting alignment with method: '{method}'")

            # --- Updated Method Dispatcher ---
            if method == "geometric_shape":
                M, dbg = self.align_by_geometric_shape(src_processed, ref_processed)
            elif method == "contour_pose":
                M, dbg = self.align_by_contour_pose(src_processed, ref_processed)
            elif method == "feature_sift":
                M, dbg = self.align_by_feature(
                    src_processed, ref_processed, use_sift=True
                )
            elif method == "feature_orb":
                M, dbg = self.align_by_feature(
                    src_processed, ref_processed, use_sift=False
                )
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            debug_paths.update(dbg)

        except Exception as e:
            print(f"[Alignment Error] Method '{method}' failed: {e}")
            return {"image": None, "debug_paths": debug_paths, "transform_matrix": None}

        if M is None:
            print(
                f"[Alignment Warning] Method '{method}' could not compute a transformation matrix."
            )
            return {"image": None, "debug_paths": debug_paths, "transform_matrix": None}

        h, w = ref.shape[:2]
        warp_func = cv2.warpAffine if M.shape[0] == 2 else cv2.warpPerspective
        aligned_image = warp_func(src, M, (w, h))
        self._save_debug_image(f"04_final_aligned_{method}", aligned_image, debug_paths)

        return {
            "image": aligned_image,
            "debug_paths": debug_paths,
            "transform_matrix": M,
        }
