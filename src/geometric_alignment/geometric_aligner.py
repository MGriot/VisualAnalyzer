"""
This module provides classes for robust image alignment using ArUco markers.

The code includes a compatibility layer to work across different OpenCV versions.
The core logic is designed to be flexible, using all commonly detected markers
between an image and a reference (either a map or another image) to compute a
highly accurate and stable perspective transformation.
"""

import cv2
import numpy as np
import os


def generate_aruco_marker_map(
    output_size_wh: tuple, marker_locations: dict, marker_size_px: int
) -> dict:
    """
    Generates a map of ideal corner coordinates for a flexible set of ArUco markers.
    """
    w, h = output_size_wh
    m_size = marker_size_px
    marker_map = {}
    for marker_id, location in marker_locations.items():
        if isinstance(location, str):
            if location == "top-left":
                x, y = 0, 0
            elif location == "top-right":
                x, y = w - m_size, 0
            elif location == "bottom-right":
                x, y = w - m_size, h - m_size
            elif location == "bottom-left":
                x, y = 0, h - m_size
            else:
                raise ValueError(f"Unknown string location: {location}")
        elif isinstance(location, tuple) and len(location) == 2:
            x, y = location
        else:
            raise TypeError("Location must be a string or a (x, y) tuple.")
        corners = [[x, y], [x + m_size, y], [x + m_size, y + m_size], [x, y + m_size]]
        marker_map[marker_id] = np.array(corners, dtype=np.float32)
    return marker_map


class ArucoAligner:
    """Handles robust image alignment by using all available ArUco markers."""

    def __init__(
        self,
        aruco_dict=cv2.aruco.DICT_4X4_50,
        debug_mode: bool = False,
        output_dir: str = "output_aruco",
    ):
        """
        Initializes the ArucoAligner with a compatibility layer for different
        OpenCV versions.
        """
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        if self.debug_mode and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        else:
            self.aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dict)

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.aruco_parameters = cv2.aruco.DetectorParameters()
        else:
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.aruco_parameters.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_SUBPIX
            )

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dictionary, self.aruco_parameters
            )
        else:
            self.detector = self

    def detectMarkers(self, gray_image):
        """Compatibility method for older OpenCV versions."""
        return cv2.aruco.detectMarkers(
            gray_image, self.aruco_dictionary, parameters=self.aruco_parameters
        )

    def _save_debug_image(self, name, image, debug_paths):
        if self.debug_mode and self.output_dir:
            path = os.path.join(self.output_dir, f"debug_{name}.png")
            cv2.imwrite(path, image)
            debug_paths[name] = path

    def align_image_by_markers(
        self, image: np.ndarray, marker_map: dict, output_size_wh: tuple
    ):
        # (This method is complete and correct from the previous version)
        # ... (implementation omitted for brevity)
        pass

    def align_image_to_reference(self, image: np.ndarray, reference_image: np.ndarray):
        """
        Aligns an image to a reference image using all common ArUco markers,
        with full debug output.
        """
        debug_paths = {}
        # Pre-process the source image (image) for more robust marker detection under varied lighting
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_image_enhanced = clahe.apply(gray_image)
        
        src_corners, src_ids, _ = self.detector.detectMarkers(gray_image_enhanced)

        # Reference image is pristine, no enhancement needed
        gray_ref_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        dst_corners, dst_ids, _ = self.detector.detectMarkers(gray_ref_image)

        if self.debug_mode:
            src_detected_img = image.copy()
            if src_ids is not None:
                cv2.aruco.drawDetectedMarkers(src_detected_img, src_corners, src_ids)
            self._save_debug_image(
                "aruco_ref_src_detected", src_detected_img, debug_paths
            )

            ref_detected_img = reference_image.copy()
            if dst_ids is not None:
                cv2.aruco.drawDetectedMarkers(ref_detected_img, dst_corners, dst_ids)
            self._save_debug_image(
                "aruco_ref_target_detected", ref_detected_img, debug_paths
            )

        if src_ids is None or dst_ids is None:
            print("[DEBUG] No markers found in either source or reference image.")
            return None, None, debug_paths

        common_ids = np.intersect1d(src_ids.flatten(), dst_ids.flatten())
        if len(common_ids) < 3:
            print(
                f"[DEBUG] Not enough common markers. Need at least 3, found {len(common_ids)}."
            )
            return None, None, debug_paths

        src_points, dst_points = [], []
        for marker_id in common_ids:
            src_idx = np.where(src_ids.flatten() == marker_id)[0][0]
            dst_idx = np.where(dst_ids.flatten() == marker_id)[0][0]
            src_points.extend(src_corners[src_idx][0])
            dst_points.extend(dst_corners[dst_idx][0])

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        if self.debug_mode:
            debug_points_img = image.copy()
            for i, point in enumerate(src_points):
                x, y = int(point[0]), int(point[1])
                cv2.circle(debug_points_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    debug_points_img,
                    str(i),
                    (x + 7, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
            self._save_debug_image(
                "aruco_ref_homography_points", debug_points_img, debug_paths
            )

        h, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        if h is None:
            print("[DEBUG] Homography calculation failed.")
            return None, None, debug_paths

        output_size = (reference_image.shape[1], reference_image.shape[0])
        aligned_image = cv2.warpPerspective(image, h, output_size)

        return aligned_image, h, debug_paths


class Aligner:
    """Simplified interface for image alignment using the robust ArucoAligner."""

    def __init__(self, debug_mode: bool = False, output_dir: str = "output"):
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        aruco_output_dir = (
            os.path.join(self.output_dir, "aruco_debug") if self.output_dir else None
        )
        self.aruco_aligner = ArucoAligner(
            debug_mode=debug_mode, output_dir=aruco_output_dir
        )

    def align_image(
        self,
        image: np.ndarray,
        aruco_reference_path: str = None,
        marker_map: dict = None,
        output_size_wh: tuple = None,
    ):
        aligned_image = None
        alignment_data = None
        debug_paths = {}

        if aruco_reference_path:
            # *** FIX: Re-implemented this logic branch ***
            reference_image = cv2.imread(aruco_reference_path)
            if reference_image is None:
                raise FileNotFoundError(
                    f"Could not read ArUco reference image at: {aruco_reference_path}"
                )

            aligned_image, homography, aruco_debug_paths = (
                self.aruco_aligner.align_image_to_reference(image, reference_image)
            )
            debug_paths.update(aruco_debug_paths)

            if aligned_image is not None:
                alignment_data = {"homography_matrix": homography.tolist()}

        elif marker_map is not None and output_size_wh:
            img, h, corners, ids, used_map, src_pts, dest_pts, aruco_debug_paths = (
                self.aruco_aligner.align_image_by_markers(
                    image, marker_map, output_size_wh
                )
            )
            aligned_image = img
            debug_paths.update(aruco_debug_paths)

            if img is not None:
                alignment_data = {
                    "homography_matrix": h.tolist(),
                    "detected_corners": (
                        [c.tolist() for c in corners] if corners is not None else []
                    ),
                    "detected_ids": ids.flatten().tolist() if ids is not None else [],
                    "used_marker_map": {
                        int(k): v.tolist() for k, v in used_map.items()
                    },
                    "source_points": src_pts.tolist(),
                    "dest_points": dest_pts.tolist(),
                }
        else:
            raise ValueError(
                "Either 'aruco_reference_path' or both 'marker_map' and 'output_size_wh' must be provided."
            )

        if aligned_image is not None and self.output_dir:
            final_path = os.path.join(self.output_dir, "geometrically_aligned.png")
            cv2.imwrite(final_path, aligned_image)
            debug_paths["final_aligned"] = final_path

        return {
            "image": aligned_image,
            "alignment_data": alignment_data,
            "debug_paths": debug_paths,
        }
