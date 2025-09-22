"""
This module provides classes for image alignment using ArUco markers.

It includes functionalities for generating ArUco marker maps, detecting markers in images,
and performing perspective correction to align images based on marker locations.
"""

import cv2
import numpy as np
import os

# Import the generator function from your other module
# from src.utils.aruco import create_printable_aruco_sheet # Assuming this exists

# ==============================================================================
# HELPER & ALIGNER CLASSES (generate_aruco_marker_map remains the same)
# ==============================================================================


def generate_aruco_marker_map(
    output_size_wh: tuple,
    marker_ids: list,
    marker_size_px: int,
    margin_px: int = 0,
) -> dict:
    """
    Generates a map of ideal corner coordinates for ArUco markers.

    This function is used to define the target positions for ArUco markers
    when performing perspective correction. It assumes a 4-marker setup
    for the four corners of a rectangular region.

    Args:
        output_size_wh (tuple): A tuple (width, height) representing the desired
                                output size of the aligned image in pixels.
        marker_ids (list): A list of exactly 4 integer IDs for the ArUco markers,
                           ordered as [top-left, top-right, bottom-right, bottom-left].
        marker_size_px (int): The size of the square ArUco marker in pixels.
        margin_px (int, optional): The margin from the image border to the marker
                                   in pixels. Defaults to 0.

    Returns:
        dict: A dictionary where keys are marker IDs and values are numpy arrays
              representing the ideal corner coordinates (x, y) for each marker.

    Raises:
        ValueError: If the `marker_ids` list does not contain exactly 4 IDs.
    """
    if len(marker_ids) != 4:
        raise ValueError(
            "This function requires exactly 4 marker IDs for the four corners."
        )
    w, h = output_size_wh
    m_size = marker_size_px
    margin = margin_px

    # Define ideal corner positions (TL, TR, BR, BL) for each marker
    tl_corners = [
        [margin, margin],
        [margin + m_size, margin],
        [margin + m_size, margin + m_size],
        [margin, margin + m_size],
    ]
    tr_corners = [
        [w - margin - m_size, margin],
        [w - margin, margin],
        [w - margin, margin + m_size],
        [w - margin - m_size, margin + m_size],
    ]
    br_corners = [
        [w - margin - m_size, h - margin - m_size],
        [w - margin, h - margin - m_size],
        [w - margin, h - margin],
        [w - margin - m_size, h - margin],
    ]
    bl_corners = [
        [margin, h - margin - m_size],
        [margin + m_size, h - margin - m_size],
        [margin + m_size, h - margin],
        [margin, h - margin],
    ]

    marker_map = {
        marker_ids[0]: np.array(tl_corners, dtype=np.float32),
        marker_ids[1]: np.array(tr_corners, dtype=np.float32),
        marker_ids[2]: np.array(br_corners, dtype=np.float32),
        marker_ids[3]: np.array(bl_corners, dtype=np.float32),
    }
    return marker_map


class ArucoAligner:
    """Handles image alignment by correcting perspective using ArUco markers."""

    def __init__(
        self,
        aruco_dict=cv2.aruco.DICT_4X4_50,
        debug_mode: bool = False,
        output_dir: str = "output_aruco",
    ):
        """
        Initializes the ArucoAligner with a specified ArUco dictionary and debug settings.

        Args:
            aruco_dict (int): The OpenCV predefined ArUco dictionary to use (e.g., cv2.aruco.DICT_4X4_50).
            debug_mode (bool, optional): If True, enables debug output and saves intermediate images.
                                         Defaults to False.
            output_dir (str, optional): Directory to save debug images if `debug_mode` is True.
                                        Defaults to "output_aruco".
        """
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        
        # Aggressive tuning for difficult conditions (wrinkled paper, glare)
        self.aruco_parameters.adaptiveThreshWinSizeMin = 3
        self.aruco_parameters.adaptiveThreshWinSizeMax = 55 # Wider search range for thresholding window
        self.aruco_parameters.adaptiveThreshWinSizeStep = 6
        self.aruco_parameters.adaptiveThreshConstant = 10 # Helps with shadows and non-uniform lighting
        self.aruco_parameters.polygonalApproxAccuracyRate = 0.08 # More lenient with marker shape
        self.aruco_parameters.minMarkerPerimeterRate = 0.015 # Adjust for potentially smaller markers in frame
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR # Better for distorted contours
        self.aruco_parameters.errorCorrectionRate = 0.8 # Higher error correction capability

        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dictionary, self.aruco_parameters
        )
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        if self.debug_mode and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def align_image_by_markers(
        self, image: np.ndarray, marker_map: dict, output_size_wh: tuple
    ):
        """
        Aligns an input image by correcting its perspective based on detected ArUco markers
        and their ideal positions defined in a marker map.

        This method detects ArUco markers in the input `image`, matches them against
        the provided `marker_map`, and calculates a homography matrix to warp the image
        to a corrected perspective.

        Args:
            image (np.ndarray): The input image (BGR format) to be aligned.
            marker_map (dict): A dictionary defining the ideal corner coordinates for
                               each ArUco marker ID. Keys are marker IDs (int), values
                               are numpy arrays of shape (4, 2) representing the
                               top-left, top-right, bottom-right, and bottom-left
                               corners of the marker in the ideal output space.
            output_size_wh (tuple): A tuple (width, height) specifying the desired
                                    dimensions of the aligned output image.

        Returns:
            tuple: A tuple containing:
                - aligned_image (np.ndarray or None): The perspective-corrected image,
                                                      or None if alignment fails.
                - homography (np.ndarray or None): The 3x3 homography matrix used for
                                                  alignment, or None.
                - corners (list or None): List of detected marker corners.
                - ids (np.ndarray or None): Array of detected marker IDs.
                - used_marker_map (dict or None): The subset of `marker_map` corresponding
                                                  to successfully detected markers.
                - source_points (np.ndarray or None): The 4 source points used for homography
                                                      calculation.
                - dest_points (np.ndarray or None): The 4 destination points used for homography
                                                    calculation.

        Raises:
            ValueError: If the input `image` is None.
        """

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray_image)

        if ids is None or len(ids) < 4:
            if self.debug_mode:
                print(
                    f"[DEBUG] Not enough markers found. Needed 4, but found {len(ids) if ids is not None else 0}."
                )
            return None, None, None, None, None, None, None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for c in corners:
            cv2.cornerSubPix(gray_image, c, (5, 5), (-1, -1), criteria)

        if self.debug_mode:
            img_with_markers = image.copy()
            cv2.aruco.drawDetectedMarkers(img_with_markers, corners, ids)
            cv2.imwrite(
                os.path.join(self.output_dir, "aruco_markers_detected.png"),
                img_with_markers,
            )
            print(f"[DEBUG] Detected marker IDs: {ids.flatten()}")

        # --- START OF MODIFIED LOGIC ---

        # Create a dictionary of detected markers for easy lookup
        detected_markers = {
            marker_id[0]: corner for marker_id, corner in zip(ids, corners)
        }

        # Convert marker_map keys from string (from JSON) to int for comparison
        try:
            marker_map_int_keys = {int(k): v for k, v in marker_map.items()}
        except (ValueError, TypeError):
            if self.debug_mode:
                print(f"[DEBUG] Error: Keys in 'aruco_marker_map' must be integers. Found: {list(marker_map.keys())}")
            return None, None, None, None, None

        # We rely on the dictionary insertion order from the JSON file (Python 3.7+).
        ordered_ids = list(marker_map_int_keys.keys())
        
        if len(ordered_ids) < 4:
            if self.debug_mode:
                print(f"[DEBUG] Error: 'aruco_marker_map' must contain at least 4 markers. Found: {len(ordered_ids)}")
            return None, None, None, None, None

        tl_id, tr_id, br_id, bl_id = (
            ordered_ids[0],
            ordered_ids[1],
            ordered_ids[2],
            ordered_ids[3],
        )

        # Check if all required markers are detected
        if not all(
            marker_id in detected_markers for marker_id in [tl_id, tr_id, br_id, bl_id]
        ):
            if self.debug_mode:
                print(f"[DEBUG] Not all four corner markers were detected. Required: {[tl_id, tr_id, br_id, bl_id]}, Found: {list(detected_markers.keys())}")
            return None, None, None, None, None

        # Assemble the 4 source points from the detected markers' corners
        # ArUco corners are ordered: 0:TL, 1:TR, 2:BR, 3:BL
        source_points = np.array(
            [
                detected_markers[tl_id][0][0],  # Top-left corner of TL marker
                detected_markers[tr_id][0][1],  # Top-right corner of TR marker
                detected_markers[br_id][0][2],  # Bottom-right corner of BR marker
                detected_markers[bl_id][0][3],  # Bottom-left corner of BL marker
            ],
            dtype=np.float32,
        )

        # Assemble the 4 destination points from the ideal marker_map
        dest_points = np.array(
            [
                marker_map_int_keys[tl_id][0],  # Top-left corner of ideal TL marker
                marker_map_int_keys[tr_id][1],  # Top-right corner of ideal TR marker
                marker_map_int_keys[br_id][2],  # Bottom-right corner of ideal BR marker
                marker_map_int_keys[bl_id][3],  # Bottom-left corner of ideal BL marker
            ],
            dtype=np.float32,
        )

        # --- END OF MODIFIED LOGIC ---

        h, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC, 5.0)

        if h is None:
            if self.debug_mode:
                print("[DEBUG] Homography calculation failed.")
            return None, None, None, None, None, None, None

        aligned_image = cv2.warpPerspective(image, h, output_size_wh)

        if self.debug_mode:
            print("[DEBUG] Alignment successful.")
            cv2.imwrite(
                os.path.join(self.output_dir, "aruco_aligned_image.png"), aligned_image
            )

        # For compatibility with the wrapper, create a 'used_marker_map'
        used_marker_map = {
            mid: marker_map_int_keys[mid] for mid in ordered_ids if mid in detected_markers
        }

        return aligned_image, h, corners, ids, used_marker_map, source_points, dest_points

    def align_image_to_reference(self, image: np.ndarray, reference_image: np.ndarray):
        """
        Aligns an image to a reference image using ArUco markers.
        """
        if image is None:
            raise ValueError("Input image is None.")
        if reference_image is None:
            raise ValueError("Reference image is None.")

        # Detect markers in both images
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        src_corners, src_ids, _ = self.detector.detectMarkers(gray_image)

        gray_ref_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        dst_corners, dst_ids, _ = self.detector.detectMarkers(gray_ref_image)

        if src_ids is None or dst_ids is None:
            if self.debug_mode:
                print("[DEBUG] Not enough markers found in either the source or reference image.")
            return None, None

        # Find common markers
        common_ids = np.intersect1d(src_ids, dst_ids)
        if len(common_ids) < 4:
            if self.debug_mode:
                print(f"[DEBUG] Not enough common markers found. Needed at least 4, but found {len(common_ids)}.")
            return None, None

        # Get corners for common markers
        src_points = []
        dst_points = []
        for marker_id in common_ids:
            src_idx = np.where(src_ids == marker_id)[0][0]
            dst_idx = np.where(dst_ids == marker_id)[0][0]
            src_points.extend(src_corners[src_idx][0])
            dst_points.extend(dst_corners[dst_idx][0])

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Calculate homography
        h, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        if h is None:
            if self.debug_mode:
                print("[DEBUG] Homography calculation failed.")
            return None, None

        # Warp image
        output_size = (reference_image.shape[1], reference_image.shape[0])
        aligned_image = cv2.warpPerspective(image, h, output_size)

        if self.debug_mode:
            print("[DEBUG] Alignment to reference successful.")
            cv2.imwrite(
                os.path.join(self.output_dir, "aruco_aligned_to_reference.png"), aligned_image
            )

        return aligned_image, h


# The Aligner class and the __main__ block can remain exactly as you had them.
class Aligner:
    """
    A wrapper class that provides a simplified interface for image alignment
    using ArUco markers. It utilizes the `ArucoAligner` internally.
    """

    def __init__(self, debug_mode: bool = False, output_dir: str = "output"):
        """
        Initializes the Aligner.

        Args:
            debug_mode (bool, optional): If True, enables debug output and saves intermediate images.
                                         Defaults to False.
            output_dir (str, optional): Base directory for output, including ArUco debug images.
                                        Defaults to "output".
        """
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        aruco_output_dir = (
            os.path.join(self.output_dir, "aruco_debug") if self.output_dir else None
        )
        self.aruco_aligner = ArucoAligner(
            debug_mode=debug_mode, output_dir=aruco_output_dir
        )

    def align_image(self, image: np.ndarray, aruco_reference_path: str = None, marker_map: dict = None, output_size_wh: tuple = None):
        """
        Aligns an image using either a reference ArUco image or a predefined marker map.

        Args:
            image (np.ndarray): The input image (BGR format) to be aligned.
            aruco_reference_path (str, optional): Path to an image containing the ideal
                                                  ArUco markers for alignment. If provided,
                                                  `marker_map` and `output_size_wh` are ignored.
            marker_map (dict, optional): A dictionary defining the ideal corner coordinates
                                         for each ArUco marker ID. Required if
                                         `aruco_reference_path` is not provided.
            output_size_wh (tuple, optional): A tuple (width, height) specifying the desired
                                              dimensions of the aligned output image.
                                              Required if `marker_map` is provided.

        Returns:
            tuple: A tuple containing:
                - aligned_image (np.ndarray or None): The aligned image, or None if alignment fails.
                - alignment_data (dict or None): A dictionary containing details about the alignment,
                                                 such as the homography matrix, detected corners, etc.,
                                                 or None if alignment fails.

        Raises:
            ValueError: If neither `aruco_reference_path` nor both `marker_map` and
                        `output_size_wh` are provided.
            FileNotFoundError: If `aruco_reference_path` is provided but the file cannot be read.
        """
        if aruco_reference_path:
            reference_image = cv2.imread(aruco_reference_path)
            if reference_image is None:
                raise FileNotFoundError(f"Could not read ArUco reference image at: {aruco_reference_path}")
            
            aligned_image, homography = self.aruco_aligner.align_image_to_reference(image, reference_image)
            
            alignment_data = None
            if aligned_image is not None:
                alignment_data = {
                    "homography_matrix": homography.tolist(),
                }
            return aligned_image, alignment_data

        elif marker_map and output_size_wh:
            aligned_image, homography, corners, ids, used_map, source_points, dest_points = (
                self.aruco_aligner.align_image_by_markers(image, marker_map, output_size_wh)
            )
            alignment_data = None
            if aligned_image is not None:
                alignment_data = {
                    "homography_matrix": homography.tolist(),
                    "detected_corners": [c.tolist() for c in corners],
                    "detected_ids": ids.flatten().tolist(),
                    "used_marker_map": used_map,  # Values are already lists from JSON
                    "source_points": source_points.tolist(),
                    "dest_points": dest_points.tolist(),
                }
            return aligned_image, alignment_data
        
        else:
            raise ValueError("Either 'aruco_reference_path' or both 'marker_map' and 'output_size_wh' must be provided for alignment.")
