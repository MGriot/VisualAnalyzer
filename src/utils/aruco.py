"""
This module provides utility functions for generating printable ArUco marker sheets.

ArUco markers are fiducial markers used in computer vision for tasks like camera
pose estimation and object detection. This module allows for the creation of
high-resolution sheets with customizable marker layouts and properties.
"""

import cv2
import numpy as np
import os

# Removed extensive ArUco introduction text to keep the module concise.

# ==============================================================================
# ArUco Sheet Generation Function
# ==============================================================================


def create_printable_aruco_sheet(
    marker_ids: list,
    output_path: str = "A4_ArUco_Sheet.png",
    aruco_dict_name: int = cv2.aruco.DICT_5X5_250,
    placement: str = "grid",
    orientation: str = "portrait",
    markers_per_row: int = 4,
    marker_size_cm: float = 4.0,
    page_margin_cm: float = 1.5,
    dpi: int = 300,
    add_text: bool = True,
):
    """
    Generates a high-resolution, printable A4 sheet with customizable ArUco markers.

    This function creates an image file (PNG) containing ArUco markers arranged
    either in a grid or at the corners of the page, suitable for printing.

    Args:
        marker_ids (list): A list of integer IDs for the ArUco markers to generate.
        output_path (str): The file path to save the generated PNG image.
        aruco_dict_name (int): The predefined OpenCV ArUco dictionary to use
                               (e.g., `cv2.aruco.DICT_4X4_50`).
        placement (str): The marker placement strategy. Can be 'grid' or 'corners'.
                         If 'corners', `marker_ids` must contain exactly 4 IDs.
        orientation (str): Page orientation. Can be 'portrait' or 'landscape'.
        markers_per_row (int): Number of markers per row (only applicable for 'grid' placement).
        marker_size_cm (float): Desired size of each marker in centimeters.
        page_margin_cm (float): Margin from the page edges to the markers in centimeters.
        dpi (int): Resolution in Dots Per Inch (DPI) for the output image.
        add_text (bool): If True, adds ID numbers below each marker and a header to the sheet.

    Raises:
        ValueError: If an invalid `orientation` or `placement` strategy is provided,
                    or if 'corners' placement is selected without exactly 4 marker IDs.
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
            (
                drawable_width - (markers_per_row * marker_size_px)
            )
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
