import cv2
import numpy as np

# This class is well-formed and requires no corrections.
# Ensure it is defined or imported before calling the align_image function.


class AdvancedAligner:
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
    ):
        """
        max_features: max keypoints for ORB/SIFT
        good_match_percent: fraction of good feature matches to keep
        motion_model: 'affine', 'euclidean', 'homography' for geometric transforms
        ecc_iters, ecc_eps: ECC algorithm convergence parameters
        min_contour_area: filter small contours
        poly_epsilon_ratio: approximation precision for polygon simplification
        edge_method: edge detection method ('canny', 'sobel', 'laplacian')
        corner_method: corner detection technique ('shi-tomasi', 'harris')
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
        return (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )

    def detect_edges(self, gray):
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
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int0(box), rect

    def convex_hull(self, contour):
        return cv2.convexHull(contour)

    def align_by_feature(self, src, ref, use_sift=False):
        src_gray = self.preprocess_gray(src)
        ref_gray = self.preprocess_gray(ref)
        detector = self.sift if use_sift and self.sift is not None else self.orb
        if detector is None:
            raise RuntimeError("Feature detector not available in OpenCV installation")
        kp1, des1 = detector.detectAndCompute(src_gray, None)
        kp2, des2 = detector.detectAndCompute(ref_gray, None)
        if des1 is None or des2 is None:
            raise RuntimeError("No descriptors found for feature matching")
        bf = cv2.BFMatcher(
            cv2.NORM_L2 if use_sift else cv2.NORM_HAMMING, crossCheck=True
        )
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[: int(len(matches) * self.good_match_percent)]
        if len(good_matches) < 4:
            raise RuntimeError(
                f"Not enough good matches for alignment: {len(good_matches)} found"
            )
        pts_src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        pts_ref = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        if self.motion_model == "homography":
            H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC)
            aligned = cv2.warpPerspective(
                src, H, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE
            )
        else:
            M, _ = cv2.estimateAffinePartial2D(pts_src, pts_ref)
            aligned = cv2.warpAffine(
                src, M, (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REPLICATE
            )
        return aligned

    def align_by_ecc(self, src, ref):
        # First, get a rough alignment using features
        aligned_src = self.align_by_feature(src, ref)
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
        """Approximate contours to polygons, extract key vertices, compute affine."""

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
        """Interface to select alignment method."""
        try:
            if method == "feature_orb":
                return self.align_by_feature(src, ref, use_sift=False)
            elif method == "feature_sift":
                return self.align_by_feature(src, ref, use_sift=True)
            elif method == "ecc":
                return self.align_by_ecc(src, ref)
            elif method == "contour_centroid":
                return self.align_by_contour_centroid(src, ref)
            elif method == "polygon":
                return self.align_by_polygon(src, ref)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"[Alignment error with method {method}]: {e}")
            return None
