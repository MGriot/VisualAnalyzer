# src/utils/aruco.py

import cv2
import numpy as np
import os

# ==============================================================================
# Introduction to ArUco Markers
# ==============================================================================
"""
What are ArUco Markers?

An ArUco (Augmented Reality University of Cordoba) marker is a specific type of fiducial marker used for camera pose estimation and object detection in computer vision. [1] It is essentially a synthetic square marker composed of a wide black border and an inner binary matrix that determines its unique identifier (ID).

How Do They Work?

The detection process involves several steps:
1.  **Image Thresholding**: The input image is converted to a binary (black and white) format.
2.  **Contour Detection**: The algorithm searches for square-shaped contours in the binary image.
3.  **Canonical Form Extraction**: For each square found, the perspective distortion is removed to obtain a frontal, canonical view of the marker.
4.  **ID Identification**: The inner grid of the marker is analyzed to decode its binary pattern, which is then compared against a predefined dictionary of valid markers to determine its ID. The wide black border facilitates easier and more robust detection. [2]
5.  **Corner Refinement**: The precise locations of the four corners are refined to sub-pixel accuracy, which is crucial for high-precision applications.

Key Advantages and Potential:

ArUco markers offer several significant advantages that make them a popular choice for many applications:

*   **High Detection Speed and Robustness**: The use of a simple black-and-white pattern and a dictionary-based approach allows for very fast and reliable detection, even under varying lighting conditions or partial occlusion.
*   **Unique Identification**: Each marker has a unique ID, allowing a camera to distinguish between multiple markers in its field of view. This is essential for tracking multiple objects or creating complex augmented reality scenes.
*   **Pose Estimation**: Because the real-world size and shape of the marker are known, detecting its four corners in an image allows the system to calculate the camera's 3D position and orientation ("pose") relative to the marker. This is the foundation for augmented reality overlays and robotic navigation. [3]
*   **Simplicity and Low Cost**: ArUco markers can be generated with a simple script (like this one) and printed on standard paper, making them an extremely low-cost solution for high-precision tracking.

Common Applications:

*   **Robotics**: For robot localization, navigation, and grasping tasks. A robot can determine its position in a room by observing markers placed at known locations.
*   **Augmented Reality (AR)**: To accurately overlay virtual objects onto the real world. The marker acts as an anchor for the virtual content.
*   **Camera Calibration**: To determine the intrinsic and extrinsic parameters of a camera with high accuracy.
*   **Industrial Automation**: For tracking parts on a conveyor belt or guiding automated assembly processes.
*   **Gesture and Object Tracking**: For creating interactive systems where the movement of a marked object is tracked in real-time.

This script provides a utility to generate printable sheets of ArUco markers for use in such applications.

Bibliography:

[1] S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. "Automatic generation and detection of highly reliable fiducial markers under occlusion." Pattern Recognition, 2014.
[2] R. Muñoz-Salinas, S. Garrido-Jurado, M.J. Marín-Jiménez, E.J. Palomo-Amador. "ArUco-Net: A Deep Learning based Fiducial Marker Detection." ArXiv, 2021.
[3] OpenCV Documentation on ArUco Markers: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
"""

# ==============================================================================
# ArUco Sheet Generation Function
# ==============================================================================


def create_printable_aruco_sheet(
    marker_ids: list,
    output_path: str = "A4_ArUco_Sheet.png",
    aruco_dict_name: int = cv2.aruco.DICT_5X5_250,
    placement: str = "grid",
    orientation: str = "portrait",  # <<< NEW PARAMETER
    markers_per_row: int = 4,
    marker_size_cm: float = 4.0,
    page_margin_cm: float = 1.5,
    dpi: int = 300,
    add_text: bool = True,
):
    """
    Generates a high-resolution, printable A4 sheet with customizable ArUco markers.

    Args:
        marker_ids (list): A list of integer IDs for the ArUco markers to generate.
        output_path (str): The file path to save the generated PNG image.
        aruco_dict_name (int): The predefined OpenCV ArUco dictionary to use.
        placement (str): The marker placement strategy. Can be 'grid' or 'corners'.
        orientation (str): Page orientation. Can be 'portrait' or 'landscape'.
        markers_per_row (int): Number of markers per row (only for 'grid' placement).
        marker_size_cm (float): Desired size of each marker in centimeters.
        page_margin_cm (float): Margin for the page in cm.
        dpi (int): Resolution in Dots Per Inch for the output image.
        add_text (bool): If True, adds ID numbers and a header to the sheet.
    """
    # --- 1. Define Page and Marker Dimensions in Pixels ---
    A4_IN_PORTRAIT = (8.27, 11.69)
    A4_IN_LANDSCAPE = (11.69, 8.27)

    # --- NEW: Set page dimensions based on orientation ---
    if orientation.lower() == "portrait":
        page_width_in, page_height_in = A4_IN_PORTRAIT
    elif orientation.lower() == "landscape":
        page_width_in, page_height_in = A4_IN_LANDSCAPE
    else:
        raise ValueError(
            f"Invalid orientation '{orientation}'. Choose 'portrait' or 'landscape'."
        )

    page_width_px = int(page_width_in * dpi)
    page_height_px = int(page_height_in * dpi)

    def cm_to_pixels(cm):
        return int(cm * dpi / 2.54)

    marker_size_px = cm_to_pixels(marker_size_cm)
    margin_px = cm_to_pixels(page_margin_cm)

    # --- 2. Set up ArUco Dictionary and Create Canvas ---
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    canvas = np.ones((page_height_px, page_width_px, 3), dtype=np.uint8) * 255

    print(f"Generating A4 '{orientation}' sheet with '{placement}' placement...")

    # --- 3. Helper function to draw a marker and its ID ---
    def draw_marker_with_text(x, y, marker_id):
        # Ensure marker fits within canvas bounds before drawing
        if x + marker_size_px > page_width_px or y + marker_size_px > page_height_px:
            print(
                f"Warning: Marker ID {marker_id} at ({x}, {y}) is out of bounds. Skipping."
            )
            return

        marker_img = cv2.aruco.generateImageMarker(
            aruco_dictionary, marker_id, marker_size_px
        )
        marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[y : y + marker_size_px, x : x + marker_size_px] = marker_img_bgr

        if add_text:
            text = f"ID: {marker_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 1.2, 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (marker_size_px - tw) // 2
            text_y = y + marker_size_px + th + 15
            cv2.putText(
                canvas, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness
            )

    # --- 4. Place Markers based on the chosen strategy ---
    if placement == "corners":
        if len(marker_ids) != 4:
            raise ValueError("'corners' placement requires exactly 4 marker IDs.")

        # Define positions (TL, TR, BR, BL)
        positions = [
            (margin_px, margin_px),  # Top-Left
            (page_width_px - margin_px - marker_size_px, margin_px),  # Top-Right
            (
                page_width_px - margin_px - marker_size_px,
                page_height_px - margin_px - marker_size_px,
            ),  # Bottom-Right
            (margin_px, page_height_px - margin_px - marker_size_px),  # Bottom-Left
        ]
        # Assign markers to corners in the specified order
        for i, marker_id in enumerate(marker_ids):
            draw_marker_with_text(positions[i][0], positions[i][1], marker_id)

    elif placement == "grid":
        drawable_width = page_width_px - (2 * margin_px)
        h_spacing = (
            (drawable_width - (markers_per_row * marker_size_px))
            // (markers_per_row - 1)
            if markers_per_row > 1
            else 0
        )

        current_x, current_y = margin_px, margin_px
        for i, marker_id in enumerate(marker_ids):
            if i > 0 and i % markers_per_row == 0:
                current_x = margin_px
                current_y += marker_size_px + int(
                    marker_size_px * 0.75  # Use relative vertical spacing
                )

            if current_y + marker_size_px > page_height_px - margin_px:
                print(
                    f"Warning: Not all markers could fit. Stopping after marker ID {marker_ids[i-1]}."
                )
                break

            draw_marker_with_text(current_x, current_y, marker_id)
            current_x += marker_size_px + h_spacing
    else:
        raise ValueError(
            f"Invalid placement strategy '{placement}'. Choose 'grid' or 'corners'."
        )

    # --- 5. Add Header Text and Save ---
    if add_text:
        header_text = (
            f"ArUco Sheet | Dict: {aruco_dict_name} | Size: {marker_size_cm}cm"
        )
        cv2.putText(
            canvas,
            header_text,
            (margin_px, margin_px - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (150, 150, 150),
            2,
        )

    cv2.imwrite(output_path, canvas)
    print(f"\n✅ Successfully generated ArUco sheet: {os.path.abspath(output_path)}")
    return canvas


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Generating Portrait Alignment Sheet ('corners' placement) ---")
    create_printable_aruco_sheet(
        marker_ids=[10, 20, 30, 40],  # Order: TL, TR, BR, BL
        output_path="Alignment_Sheet_Corners_Portrait.png",
        orientation="portrait",
        placement="corners",
        marker_size_cm=2.5,
        page_margin_cm=1.5,
    )

    print("\n--- Generating General Purpose Grid Sheet ('grid' placement) ---")
    create_printable_aruco_sheet(
        marker_ids=list(range(50, 62)),
        output_path="General_Sheet_Grid.png",
        placement="grid",
        markers_per_row=4,
        marker_size_cm=3.0,
    )

    # --- NEW EXAMPLE ---
    print("\n--- Generating Landscape Alignment Sheet ('corners' placement) ---")
    create_printable_aruco_sheet(
        marker_ids=[10, 20, 30, 40],  # Order: TL, TR, BR, BL
        output_path="Alignment_Sheet_Corners_Landscape.png",
        orientation="landscape",  # Use the new parameter
        placement="corners",
        marker_size_cm=4.0,
        page_margin_cm=2.0,
    )
