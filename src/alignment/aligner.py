import cv2
import numpy as np
import os

# ==============================================================================
# HELPER FUNCTION TO PREVENT CORNER ORDER ERRORS
# ==============================================================================


def generate_aruco_marker_map(
    output_size_wh: tuple, marker_ids: list, marker_size_px: int
) -> dict:
    """
    Generates a marker map with markers in the four corners of the target image.
    This function ensures the corner points are in the correct order required by OpenCV.

    Args:
        output_size_wh (tuple): The (width, height) of the target output image.
        marker_ids (list): A list of 4 integer IDs for the markers. The IDs will be assigned
                           to Top-Left, Top-Right, Bottom-Right, and Bottom-Left corners in order.
        marker_size_px (int): The size of the marker in pixels in the target image.

    Returns:
        dict: A correctly formatted marker_map for use with the ArucoAligner.
    """
    if len(marker_ids) != 4:
        raise ValueError(
            "This function requires exactly 4 marker IDs for the four corners."
        )

    w, h = output_size_wh
    m_size = marker_size_px

    # Define the ideal corner positions with the GUARANTEED CORRECT ORDER:
    # 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left

    # Top-Left Marker (ID: marker_ids[0])
    tl_corners = [[0, 0], [m_size, 0], [m_size, m_size], [0, m_size]]

    # Top-Right Marker (ID: marker_ids[1])
    tr_corners = [[w - m_size, 0], [w, 0], [w, m_size], [w - m_size, m_size]]

    # Bottom-Right Marker (ID: marker_ids[2])
    br_corners = [[w - m_size, h - m_size], [w, h - m_size], [w, h], [w - m_size, h]]

    # Bottom-Left Marker (ID: marker_ids[3])
    bl_corners = [[0, h - m_size], [m_size, h - m_size], [m_size, h], [0, h]]

    marker_map = {
        marker_ids[0]: tl_corners,
        marker_ids[1]: tr_corners,
        marker_ids[2]: br_corners,
        marker_ids[3]: bl_corners,
    }

    return marker_map


# ==============================================================================
# THE ROBUST ARUCO ALIGNER CLASS (Unchanged Logic, Added Comments)
# ==============================================================================
class ArucoAligner:
    """
    A class to handle image alignment by correcting perspective using ArUco markers.
    """

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
        if image is None:
            raise ValueError("Input image is None.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray_image)

        if ids is None or len(ids) < 1:
            if self.debug_mode:
                print("[DEBUG] No ArUco markers found.")
            return None, None, None, None, None

        # Refine corner locations to sub-pixel accuracy for better results.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Note: corners is a tuple of arrays. We need to iterate and refine.
        for i in range(len(corners)):
            cv2.cornerSubPix(gray_image, corners[i], (5, 5), (-1, -1), criteria)

        if self.debug_mode:
            img_with_markers = image.copy()
            cv2.aruco.drawDetectedMarkers(img_with_markers, corners, ids)
            cv2.imwrite(
                os.path.join(self.output_dir, "aruco_markers_detected.png"),
                img_with_markers,
            )
            print(f"[DEBUG] Detected marker IDs: {ids.flatten()}")

        source_points, dest_points, used_marker_map = [], [], {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_map:
                # The order of corners from detectMarkers is TL, TR, BR, BL.
                # We add all 4 detected corners to our source list.
                for corner_point in corners[i][0]:
                    source_points.append(corner_point)

                # We add all 4 ideal corners from our map to the destination list.
                # This is where the order MUST match.
                for dest_point in marker_map[marker_id]:
                    dest_points.append(dest_point)

                used_marker_map[marker_id] = marker_map[marker_id]

        if len(source_points) < 4:
            if self.debug_mode:
                print("[DEBUG] Not enough valid markers found to compute homography.")
            return None, None, None, None, None

        # Use RANSAC to calculate a robust homography, ignoring potential outlier points.
        h, mask = cv2.findHomography(
            np.array(source_points), np.array(dest_points), cv2.RANSAC, 5.0
        )

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

        return aligned_image, h, corners, ids, used_marker_map


# ==============================================================================
# THE MAIN ALIGNER CLASS (Unchanged)
# ==============================================================================
class Aligner:
    def __init__(self, debug_mode: bool = False, output_dir: str = "output"):
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        aruco_output_dir = (
            os.path.join(self.output_dir, "aruco_debug") if self.output_dir else None
        )
        self.aruco_aligner = ArucoAligner(
            debug_mode=debug_mode, output_dir=aruco_output_dir
        )

    def align_image(self, image: np.ndarray, marker_map: dict, output_size_wh: tuple):
        if self.debug_mode:
            print("[DEBUG] Starting alignment using ArUcoAligner.")

        aligned_image, homography, corners, ids, used_map = (
            self.aruco_aligner.align_image_by_markers(image, marker_map, output_size_wh)
        )

        alignment_data = None
        if aligned_image is not None:
            alignment_data = {
                "homography_matrix": homography.tolist(),
                "detected_corners": [c.tolist() for c in corners],
                "detected_ids": ids.flatten().tolist(),
                "used_marker_map": used_map,
            }
        return aligned_image, alignment_data


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Define Parameters ---
    OUTPUT_WIDTH, OUTPUT_HEIGHT = 800, 600
    MARKER_IDS = [10, 20, 30, 40]  # TL, TR, BR, BL
    MARKER_SIZE = 100  # Size in pixels for the ideal output image

    # --- 2. Generate the Marker Map Safely ---
    # Use the helper function to avoid corner order errors.
    print("Generating a safe marker map...")
    safe_marker_map = generate_aruco_marker_map(
        output_size_wh=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
        marker_ids=MARKER_IDS,
        marker_size_px=MARKER_SIZE,
    )
    print("Marker map generated successfully.")

    # --- 3. Create a Synthetic Distorted Image for Testing ---
    print("Creating a synthetic test image...")
    # a) Create the ideal layout first
    ideal_canvas = np.ones((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8) * 240
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    for marker_id, corners in safe_marker_map.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE)
        marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        tl_corner = corners[0]  # Top-left corner of the marker
        ideal_canvas[
            tl_corner[1] : tl_corner[1] + MARKER_SIZE,
            tl_corner[0] : tl_corner[0] + MARKER_SIZE,
        ] = marker_img_bgr

    # b) Apply a perspective warp to simulate a camera view
    src_pts = np.float32(
        [[0, 0], [OUTPUT_WIDTH, 0], [OUTPUT_WIDTH, OUTPUT_HEIGHT], [0, OUTPUT_HEIGHT]]
    )
    dst_pts = np.float32(
        [
            [50, 80],
            [OUTPUT_WIDTH - 20, 50],
            [OUTPUT_WIDTH - 80, OUTPUT_HEIGHT - 30],
            [20, OUTPUT_HEIGHT - 100],
        ]
    )
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted_image = cv2.warpPerspective(
        ideal_canvas, perspective_matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )
    cv2.imwrite("test_distorted_input.png", distorted_image)
    print("Test image 'test_distorted_input.png' saved.")

    # --- 4. Initialize and Run the Aligner ---
    print("\nInitializing Aligner in debug mode...")
    aligner = Aligner(debug_mode=True, output_dir="output_folder")

    print("Attempting to align the image...")
    aligned_image, alignment_data = aligner.align_image(
        image=distorted_image,
        marker_map=safe_marker_map,
        output_size_wh=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )

    # --- 5. Check and Save Results ---
    if aligned_image is not None:
        print("\n✅✅✅ ALIGNMENT SUCCEEDED! ✅✅✅")
        cv2.imwrite("test_aligned_output.png", aligned_image)
        print("Final aligned image saved as 'test_aligned_output.png'")
        # print("Alignment data:", alignment_data)
    else:
        print("\n❌❌❌ ALIGNMENT FAILED. ❌❌❌")
