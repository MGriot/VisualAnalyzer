# src/alignment/aligner.py

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
    (This function is unchanged)
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
    """A class to handle image alignment by correcting perspective using ArUco markers."""

    def __init__(
        self,
        aruco_dict=cv2.aruco.DICT_5X5_250,
        debug_mode: bool = False,
        output_dir: str = "output_aruco",
    ):
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
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
        # (This method's code is CHANGED)
        if image is None:
            raise ValueError("Input image is None.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray_image)

        if ids is None or len(ids) < 4:
            if self.debug_mode:
                print(
                    f"[DEBUG] Not enough markers found. Needed 4, but found {len(ids) if ids is not None else 0}."
                )
            return None, None, None, None, None

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

        # The marker_map is created from a list of IDs in order: [TL, TR, BR, BL]
        # We can extract this order to correctly assemble our points.
        ordered_ids = list(marker_map.keys())
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
                print("[DEBUG] Not all four corner markers were detected.")
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
                marker_map[tl_id][0],  # Top-left corner of ideal TL marker
                marker_map[tr_id][1],  # Top-right corner of ideal TR marker
                marker_map[br_id][2],  # Bottom-right corner of ideal BR marker
                marker_map[bl_id][3],  # Bottom-left corner of ideal BL marker
            ],
            dtype=np.float32,
        )

        # --- END OF MODIFIED LOGIC ---

        h, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC, 5.0)

        if h is None:
            if self.debug_mode:
                print("[DEBUG] Homography calculation failed.")
            return None, None, None, None, None

        aligned_image = cv2.warpPerspective(image, h, output_size_wh)

        if self.debug_mode:
            print("[DEBUG] Alignment successful.")
            cv2.imwrite(
                os.path.join(self.output_dir, "aruco_aligned_image.png"), aligned_image
            )

        # For compatibility with the wrapper, create a 'used_marker_map'
        used_marker_map = {
            mid: marker_map[mid] for mid in ordered_ids if mid in detected_markers
        }

        return aligned_image, h, corners, ids, used_marker_map


# The Aligner class and the __main__ block can remain exactly as you had them.
class Aligner:
    """Main aligner class."""

    def __init__(self, debug_mode: bool = False, output_dir: str = "output"):
        # (This class is unchanged)
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        aruco_output_dir = (
            os.path.join(self.output_dir, "aruco_debug") if self.output_dir else None
        )
        self.aruco_aligner = ArucoAligner(
            debug_mode=debug_mode, output_dir=aruco_output_dir
        )

    def align_image(self, image: np.ndarray, marker_map: dict, output_size_wh: tuple):
        # (This method's code is unchanged)
        aligned_image, homography, corners, ids, used_map = (
            self.aruco_aligner.align_image_by_markers(image, marker_map, output_size_wh)
        )
        alignment_data = None
        if aligned_image is not None:
            alignment_data = {
                "homography_matrix": homography.tolist(),
                "detected_corners": [c.tolist() for c in corners],
                "detected_ids": ids.flatten().tolist(),
                "used_marker_map": {k: v.tolist() for k, v in used_map.items()},
            }
        return aligned_image, alignment_data
