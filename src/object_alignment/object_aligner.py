"""
This module provides the `AdvancedAligner` class for robust image alignment
using various computer vision techniques.

It supports feature-based alignment (ORB/SIFT), ECC (Enhanced Correlation Coefficient)
maximization, contour centroid alignment, and polygon-based alignment.
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

    This class provides tools for image preprocessing, feature detection,
    and different alignment strategies including feature matching (ORB/SIFT),
    ECC, contour centroid, and polygon-based alignment.
    """
    def __init__(
        self,
        max_features=2000,
        good_match_percent=0.15,
        motion_model="affine",
        ecc_iters=5000,
        ecc_eps=1e-8,
        min_contour_area=100,
        poly_epsilon_ratio=0.02,
        edge_method="canny",
        corner_method="shi-tomasi",
        debug_mode=False,
        output_dir=None,
    ):
        """
        Initializes the AdvancedAligner with various configuration parameters.

        Args:
            max_features (int, optional): Maximum number of keypoints for feature detectors (ORB/SIFT).
                                          Defaults to 2000.
            good_match_percent (float, optional): Fraction of good feature matches to keep for alignment.
                                                  Defaults to 0.15.
            motion_model (str, optional): Type of geometric transform to use ('affine', 'euclidean', 'homography').
                                          Defaults to "affine".
            ecc_iters (int, optional): Number of iterations for the ECC algorithm.
                                       Defaults to 5000.
            ecc_eps (float, optional): Epsilon for ECC algorithm convergence.
                                       Defaults to 1e-8.
            min_contour_area (int, optional): Minimum contour area to consider for contour-based methods.
                                              Defaults to 100.
            poly_epsilon_ratio (float, optional): Approximation precision for polygon simplification.
                                                  Defaults to 0.02.
            edge_method (str, optional): Edge detection method ('canny', 'sobel', 'laplacian').
                                         Defaults to "canny".
            corner_method (str, optional): Corner detection technique ('shi-tomasi', 'harris').
                                           Defaults to "shi-tomasi".
            debug_mode (bool, optional): Enables saving of intermediate debug images. Defaults to False.
            output_dir (str, optional): Directory to save debug images. Defaults to None.
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

        # Feature detectors
        try:
            self.orb = cv2.ORB_create(self.max_features)
        except:
            self.orb = None
        try:
            self.sift = cv2.SIFT_create(self.max_features)
        except:
            self.sift = None

    def preprocess_gray(self, img):
        """
        Converts an input image to grayscale if it's a color image, otherwise returns a copy.

        Args:
            img (np.ndarray): The input image (BGR, BGRA, or grayscale).

        Returns:
            np.ndarray: The grayscale version of the image.
        """
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()
    def detect_edges(self, gray):
        """
        Detects edges in a grayscale image using the configured edge detection method.

        Args:
            gray (np.ndarray): The input grayscale image.

        Returns:
            np.ndarray: A binary image with detected edges.
        """
        if self.edge_method == "canny":
            edges = cv2.Canny(gray, 50, 150)
        elif self.edge_method == "sobel":
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            grad = cv2.convertScaleAbs(np.sqrt(grad_x**2 + grad_y**2))
            _, edges = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)
        else:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges = cv2.convertScaleAbs(laplacian)
            _, edges = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
        return edges

    def find_polygons(self, edges):
        """
        Finds contours in an edge image and approximates them as polygons.

        Args:
            edges (np.ndarray): A binary image containing edges.

        Returns:
            List[np.ndarray]: A list of approximated polygonal contours.
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polys = []
        for c in contours:
            if cv2.contourArea(c) > self.min_contour_area:
                epsilon = self.poly_epsilon_ratio * cv2.arcLength(c, True)
                approx_poly = cv2.approxPolyDP(c, epsilon, True)
                polys.append(approx_poly)
        return polys

    def extract_corners(self, gray):
        """
        Extracts corners from a grayscale image using the configured corner detection method.

        Args:
            gray (np.ndarray): The input grayscale image.

        Returns:
            np.ndarray: An array of detected corner coordinates (x, y).
        """
        if self.corner_method == "shi-tomasi":
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=50, qualityLevel=0.01, minDistance=10
            )
            if corners is not None:
                return np.int0(corners).reshape(-1, 2)
        elif self.corner_method == "harris":
            dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
            dst = cv2.dilate(dst, None)
            corners = np.argwhere(dst > 0.01 * dst.max())
            return corners[:, ::-1]  # swap to (x,y)
        return np.array([])

    def minimum_area_rectangle(self, contour):
        """
        Calculates the minimum area bounding rectangle for a given contour.

        Args:
            contour (np.ndarray): The input contour.

        Returns:
            Tuple[np.ndarray, tuple]: A tuple containing the 4 corner points of the rectangle
                                     and the `RotatedRect` object.
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int0(box), rect

    def convex_hull(self, contour):
        """
        Computes the convex hull of a point set or object contour.

        Args:
            contour (np.ndarray): The input contour.

        Returns:
            np.ndarray: The convex hull of the contour.
        """
        return cv2.convexHull(contour)

    def align_by_feature(self, img_to_align, ref_img, use_sift=False):
        """
        Aligns images using feature detection (ORB or SIFT).
        """
        if use_sift:
            if self.sift is None:
                print("[WARNING] SIFT detector not available. Please install opencv-contrib-python. Falling back to ORB.")
                detector = self.orb
                norm = cv2.NORM_HAMMING
            else:
                detector = self.sift
                norm = cv2.NORM_L2
        else:
            if self.orb is None:
                raise RuntimeError("ORB detector not available.")
            detector = self.orb
            norm = cv2.NORM_HAMMING

        # Preprocess images to grayscale for feature detection
        gray_to_align = self.preprocess_gray(img_to_align)
        gray_ref = self.preprocess_gray(ref_img)

        kp1, des1 = detector.detectAndCompute(gray_to_align, None)
        kp2, des2 = detector.detectAndCompute(gray_ref, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            if self.debug_mode:
                print("[DEBUG] Alignment failed: No features/descriptors found in one of the images.")
            return None

        # Use BFMatcher and k-NN matching
        bf = cv2.BFMatcher(norm, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to find good matches
        good_matches = []
        # Ensure matches is not empty and contains lists of 2
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        debug_paths = {}
        if self.debug_mode and self.output_dir:
            # Draw matches for debugging
            img_matches = cv2.drawMatches(img_to_align, kp1, ref_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            debug_image_path = os.path.join(self.output_dir, "feature_matches.png")
            save_image(debug_image_path, img_matches)
            debug_paths['feature_matches_image'] = debug_image_path
            print(f"[DEBUG] Saved feature matching visualization to {debug_image_path}")
        
        debug_paths['good_feature_matches'] = len(good_matches)

        MIN_MATCHES = 10
        if len(good_matches) < MIN_MATCHES:
            if self.debug_mode:
                print(f"[DEBUG] Alignment failed: Not enough good matches found ({len(good_matches)}/{MIN_MATCHES}).")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Determine motion model from constructor
        if self.motion_model == "homography":
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:  # Default to affine
            M, mask = cv2.estimateAffine2D(src_pts, dst_pts)

        if M is None:
            if self.debug_mode:
                print("[DEBUG] Alignment failed: Could not compute the transformation matrix (homography/affine).")
            return None

        h, w = ref_img.shape[:2]
        if self.motion_model == "homography":
            aligned_img = cv2.warpPerspective(img_to_align, M, (w, h))
        else:
            aligned_img = cv2.warpAffine(img_to_align, M, (w, h))

        return aligned_img, debug_paths

    def align_by_ecc(self, src, ref):
        """
        Aligns a source image to a reference image using the Enhanced Correlation Coefficient (ECC) maximization algorithm.

        This method first performs a rough alignment using feature matching, then refines it using ECC.

        Args:
            src (np.ndarray): The source image to be aligned.
            ref (np.ndarray): The reference image.

        Returns:
            np.ndarray: The aligned image.

        Raises:
            RuntimeError: If initial feature-based alignment fails or ECC refinement encounters an error.
        """
        # First, get a rough alignment using features (ORB by default)
        aligned_src = self.align_by_feature(src, ref, use_sift=False)
        if aligned_src is None:
            raise RuntimeError(
                "Initial feature-based alignment failed, so ECC cannot proceed."
            )

        src_gray = self.preprocess_gray(aligned_src)
        ref_gray = self.preprocess_gray(ref)

        warp_mode = {
            "homography": cv2.MOTION_HOMOGRAPHY,
            "affine": cv2.MOTION_AFFINE,
            "euclidean": cv2.MOTION_EUCLIDEAN,
        }.get(self.motion_model, cv2.MOTION_AFFINE)

        # Start with an identity matrix for refinement
        warp_matrix = (
            np.eye(3, dtype=np.float32)
            if warp_mode == cv2.MOTION_HOMOGRAPHY
            else np.eye(2, 3, dtype=np.float32)
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iters,
            self.ecc_eps,
        )

        try:
            # findTransformECC finds the warp that aligns src_gray with ref_gray
            cc, warp_matrix = cv2.findTransformECC(
                ref_gray, src_gray, warp_matrix, warp_mode, criteria
            )
        except cv2.error as e:
            raise RuntimeError(f"ECC refinement failed: {e}")

        # Warp the already aligned image with the refined matrix
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(
                aligned_src,
                warp_matrix,
                (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            aligned = cv2.warpAffine(
                aligned_src,
                warp_matrix,
                (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return aligned

    def align_by_contour_centroid(self, src, ref):
        """
        Aligns a source image to a reference image by matching the centroids of their largest contours.

        Args:
            src (np.ndarray): The source image to be aligned.
            ref (np.ndarray): The reference image.

        Returns:
            np.ndarray: The aligned image.

        Raises:
            RuntimeError: If no valid contours are found in either image.
        """
        def largest_contour(img):
            gray = self.preprocess_gray(img)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filtered = [
                c for c in contours if cv2.contourArea(c) > self.min_contour_area
            ]
            return max(filtered, key=cv2.contourArea) if filtered else None

        src_contour = largest_contour(src)
        ref_contour = largest_contour(ref)
        if src_contour is None or ref_contour is None:
            raise RuntimeError("No valid contours found for centroid alignment")
        M_src = cv2.moments(src_contour)
        M_ref = cv2.moments(ref_contour)
        cx_src = int(M_src["m10"] / M_src["m00"])
        cy_src = int(M_src["m01"] / M_src["m00"])
        cx_ref = int(M_ref["m10"] / M_ref["m00"])
        cy_ref = int(M_ref["m01"] / M_ref["m00"])
        dx = cx_ref - cx_src
        dy = cy_ref - cy_src
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(
            src, M, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def align_by_polygon(self, src, ref):
        """
        Aligns a source image to a reference image by approximating contours to polygons
        and computing an affine transformation based on key vertices.

        Args:
            src (np.ndarray): The source image to be aligned.
            ref (np.ndarray): The reference image.

        Returns:
            np.ndarray: The aligned image.

        Raises:
            RuntimeError: If polygons cannot be found or if there are not enough points
                          for affine transformation.
        """

        def largest_polygon(img):
            gray = self.preprocess_gray(img)
            edges = self.detect_edges(gray)
            polygons = self.find_polygons(edges)
            return max(polygons, key=cv2.contourArea) if polygons else None

        # Find largest polygon for src and ref
        src_poly = largest_polygon(src)
        ref_poly = largest_polygon(ref)
        if src_poly is None or ref_poly is None:
            raise RuntimeError("Could not find polygons for alignment")
        # At least 3 points needed for affine
        if len(src_poly) < 3 or len(ref_poly) < 3:
            raise RuntimeError("Not enough polygon points for affine transform")
        # Select 3 points (e.g., first 3 vertices)
        src_pts = np.float32(src_poly[:3].reshape(-1, 2))
        ref_pts = np.float32(ref_poly[:3].reshape(-1, 2))
        M = cv2.getAffineTransform(src_pts, ref_pts)
        aligned = cv2.warpAffine(
            src, M, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def align(self, src, ref, method="feature_orb"):
        """
        Main alignment interface that dispatches to different alignment methods.

        Args:
            src (np.ndarray): The source image to be aligned.
            ref (np.ndarray): The reference image.
            method (str, optional): The alignment method to use. Options include:
                                    'feature_orb', 'feature_sift', 'ecc',
                                    'contour_centroid', 'polygon'.
                                    Defaults to "feature_orb".

        Returns:
            dict: A dictionary containing:
                - 'image' (np.ndarray | None): The aligned image, or None if alignment fails.
                - 'debug_paths' (dict): A dictionary of paths to any generated debug images.
        """
        aligned_image = None
        debug_paths = {}
        try:
            if method in ["feature_orb", "feature_sift"]:
                aligned_image, feature_debug_paths = self.align_by_feature(src, ref, use_sift=(method=="feature_sift"))
                debug_paths.update(feature_debug_paths)
            elif method == "ecc":
                # Note: align_by_ecc calls align_by_feature internally, but we don't get its debug paths here.
                # This could be improved in a future refactoring.
                aligned_image = self.align_by_ecc(src, ref)
            elif method == "contour_centroid":
                aligned_image = self.align_by_contour_centroid(src, ref)
            elif method == "polygon":
                aligned_image = self.align_by_polygon(src, ref)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"[Alignment error with method {method}]: {e}")
            aligned_image = None

        if aligned_image is not None and self.output_dir:
            final_path = os.path.join(self.output_dir, "object_aligned.png")
            save_image(final_path, aligned_image)
            debug_paths['final_aligned'] = final_path
        
        return {
            'image': aligned_image,
            'debug_paths': debug_paths
        }
