import cv2
import numpy as np
import os


def create_printable_aruco_sheet(
    marker_ids: list,
    output_path: str = "A4_ArUco_Sheet.png",
    aruco_dict_name: int = cv2.aruco.DICT_5X5_250,
    placement: str = "grid",
    markers_per_row: int = 4,
    marker_size_cm: float = 4.0,
    page_margin_cm: float = 1.5,
    dpi: int = 300,
):
    """
    Generates a high-resolution, printable A4 sheet with customizable ArUco markers.

    Args:
        marker_ids (list): A list of integer IDs for the ArUco markers to generate.
        output_path (str): The file path to save the generated PNG image.
        aruco_dict_name (int): The predefined OpenCV ArUco dictionary to use.
        placement (str): The marker placement strategy. Can be 'grid' (default) or 'corners'.
        markers_per_row (int): The number of markers per row (only used for 'grid' placement).
        marker_size_cm (float): The desired size (width and height) of each marker in centimeters.
        page_margin_cm (float): The margin for the top, bottom, left, and right of the page in cm.
        dpi (int): The resolution in Dots Per Inch for the output image (300 is good for printing).
    """
    # --- 1. Define Page and Marker Dimensions in Pixels ---
    A4_WIDTH_IN, A4_HEIGHT_IN = 8.27, 11.69
    page_width_px, page_height_px = int(A4_WIDTH_IN * dpi), int(A4_HEIGHT_IN * dpi)

    def cm_to_pixels(cm):
        return int(cm * dpi / 2.54)

    marker_size_px = cm_to_pixels(marker_size_cm)
    margin_px = cm_to_pixels(page_margin_cm)

    # --- 2. Set up the ArUco Dictionary and Create Canvas ---
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    canvas = np.ones((page_height_px, page_width_px, 3), dtype=np.uint8) * 255

    print(f"Generating sheet with '{placement}' placement strategy...")

    # --- 3. Place Markers based on the chosen strategy ---

    # Helper function to draw a marker and its ID
    def draw_marker(x, y, marker_id):
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dictionary, marker_id, marker_size_px
        )
        marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[y : y + marker_size_px, x : x + marker_size_px] = marker_img_bgr

        text = f"ID: {marker_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness = 1.2, 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x + (marker_size_px - text_size[0]) // 2
        text_y = y + marker_size_px + int(text_size[1] * 1.5)
        cv2.putText(
            canvas, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness
        )

    if placement == "corners":
        if len(marker_ids) > 4:
            print(
                f"Warning: 'corners' placement selected with {len(marker_ids)} markers. Only the first 4 will be used."
            )
            ids_to_place = marker_ids[:4]
        else:
            ids_to_place = marker_ids

        # Define the (x, y) coordinates for the top-left of each corner position
        positions = [
            (margin_px, margin_px),  # Top-Left
            (page_width_px - margin_px - marker_size_px, margin_px),  # Top-Right
            (margin_px, page_height_px - margin_px - marker_size_px),  # Bottom-Left
            (
                page_width_px - margin_px - marker_size_px,
                page_height_px - margin_px - marker_size_px,
            ),  # Bottom-Right
        ]

        # If user provides only 2 markers, place them in diagonal corners for max distance
        if len(ids_to_place) == 2:
            draw_marker(positions[0][0], positions[0][1], ids_to_place[0])  # Top-Left
            draw_marker(
                positions[3][0], positions[3][1], ids_to_place[1]
            )  # Bottom-Right
        else:  # For 1, 3, or 4 markers, place them in order
            for i, marker_id in enumerate(ids_to_place):
                draw_marker(positions[i][0], positions[i][1], marker_id)

    elif placement == "grid":
        drawable_width = page_width_px - (2 * margin_px)
        if markers_per_row > 1:
            h_spacing = (drawable_width - (markers_per_row * marker_size_px)) // (
                markers_per_row - 1
            )
        else:
            h_spacing = 0

        current_x, current_y = margin_px, margin_px
        for i, marker_id in enumerate(marker_ids):
            if i > 0 and i % markers_per_row == 0:
                current_x = margin_px
                current_y += marker_size_px + h_spacing

            if current_y + marker_size_px > page_height_px - margin_px:
                print(
                    f"Warning: Not all markers could fit. Stopping after marker ID {marker_ids[i-1]}."
                )
                break

            draw_marker(current_x, current_y, marker_id)
            current_x += marker_size_px + h_spacing
    else:
        raise ValueError(
            f"Invalid placement strategy '{placement}'. Choose 'grid' or 'corners'."
        )

    # --- 4. Add Header Text and Save ---
    header_text = f"ArUco Sheet | Dict: {aruco_dict_name} | Size: {marker_size_cm}cm | Placement: {placement}"
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


# --- Example 1: New 'corners' placement for an alignment sheet ---
# We want 4 markers, one in each corner, and they should be large.
print("--- Generating Alignment Sheet ('corners' placement) ---")
create_printable_aruco_sheet(
    marker_ids=[10, 20, 30, 40],  # The IDs we will use for our aligner
    output_path="Alignment_Sheet_Corners.png",
    aruco_dict_name=cv2.aruco.DICT_5X5_250,
    placement="corners",  # Use the new strategy
    marker_size_cm=2.0,  # Make markers large
    page_margin_cm=1.5,
)

# --- Example 2: Using 'corners' placement with only 2 markers ---
# They will be placed diagonally for maximum distance.
print("\n--- Generating Alignment Sheet with 2 diagonal markers ---")
create_printable_aruco_sheet(
    marker_ids=[100, 200],
    output_path="Alignment_Sheet_Diagonal.png",
    placement="corners",
    marker_size_cm=2.0,
)

# --- Example 3: Original 'grid' placement for a general purpose sheet ---
print("\n--- Generating General Purpose Sheet ('grid' placement) ---")
create_printable_aruco_sheet(
    marker_ids=list(range(50, 62)),  # A range of IDs
    output_path="General_Sheet_Grid.png",
    placement="grid",  # Explicitly use the grid strategy
    markers_per_row=2,
    marker_size_cm=2.0,
)
