import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt # Import matplotlib
import argparse # Import argparse

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alignment.aligner import Aligner
from src.utils.image_utils import load_image # Import load_image utility

def create_distorted_aruco_image(output_path: str, marker_ids: list, marker_size_px: int, image_size_wh: tuple, perspective_transform_params: dict):
    """
    Generates an image with ArUco markers and applies a perspective transform.
    """
    img = np.zeros((image_size_wh[1], image_size_wh[0], 3), dtype=np.uint8)
    img.fill(255) # White background

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    
    # Define marker positions (simple grid for demonstration)
    marker_positions = {
        marker_ids[0]: (50, 50),
        marker_ids[1]: (image_size_wh[0] - marker_size_px - 50, 50),
        marker_ids[2]: (50, image_size_wh[1] - marker_size_px - 50),
        marker_ids[3]: (image_size_wh[0] - marker_size_px - 50, image_size_wh[1] - marker_size_px - 50),
    }

    for marker_id, pos in marker_positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
        img[pos[1]:pos[1]+marker_size_px, pos[0]:pos[0]+marker_size_px] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # Apply perspective transform
    pts1 = np.float32([[0, 0], [image_size_wh[0], 0], [0, image_size_wh[1]], [image_size_wh[0], image_size_wh[1]]])
    pts2 = np.float32(perspective_transform_params["pts2"])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    distorted_img = cv2.warpPerspective(img, matrix, image_size_wh)

    cv2.imwrite(output_path, distorted_img)
    return distorted_img

def run_aligner_test(image_path: str = None):
    print("--- Running Aligner Test ---")
    print() # Add a newline

    # Define output directory
    test_output_dir = "output/test_aligner_results"
    os.makedirs(test_output_dir, exist_ok=True)

    if image_path:
        print(f"Loading image from: {image_path}")
        distorted_image, _ = load_image(image_path)
        if distorted_image is None:
            print(f"Error: Could not load image from {image_path}. Exiting.")
            return
        # For external images, we might not know the ideal output size or marker layout.
        # For this test, we'll assume the external image has markers that map to the
        # same ideal_marker_layout and output_size_wh as the generated image.
        # In a real scenario, these would also be configurable or detected.
        output_width, output_height = distorted_image.shape[1], distorted_image.shape[0] # Use image's own size
        print(f"Using image dimensions: {output_width}x{output_height}")
    else:
        # 1. Generate a distorted image with ArUco markers
        marker_ids = [10, 20, 30, 40]
        marker_size_px = 80
        image_size_wh = (800, 600)
        perspective_transform_params = {
            "pts2": [[100, 50], [image_size_wh[0]-50, 0], [50, image_size_wh[1]-100], [image_size_wh[0]-100, image_size_wh[1]-50]]
        }
        distorted_image_path = os.path.join(test_output_dir, "distorted_aruco_test_image.png")
        distorted_image = create_distorted_aruco_image(distorted_image_path, marker_ids, marker_size_px, image_size_wh, perspective_transform_params)
        print(f"Generated distorted ArUco test image: {distorted_image_path}")
        output_width, output_height = image_size_wh # Use generated image's size

    # 2. Define marker_map and output_size_wh for alignment
    margin = 50
    # marker_size_px needs to be consistent with the markers in the image
    # For external images, this might need to be adjusted or detected.
    # For now, we'll use a fixed marker_size_px for the ideal layout.
    # If the external image has different sized markers, this will cause issues.
    fixed_marker_size_for_layout = 80 

    ideal_marker_layout = {
        10: [[margin, margin], [margin + fixed_marker_size_for_layout, margin], [margin + fixed_marker_size_for_layout, margin + fixed_marker_size_for_layout], [margin, margin + fixed_marker_size_for_layout]],
        20: [[output_width - margin - fixed_marker_size_for_layout, margin], [output_width - margin, margin], [output_width - margin, margin + fixed_marker_size_for_layout], [output_width - margin - fixed_marker_size_for_layout, margin + fixed_marker_size_for_layout]],
        30: [[output_width - margin - fixed_marker_size_for_layout, output_height - margin - fixed_marker_size_for_layout], [output_width - margin, output_height - margin - fixed_marker_size_for_layout], [output_width - margin, output_height - margin], [output_width - margin - fixed_marker_size_for_layout, output_height - margin]],
        40: [[margin, output_height - margin - fixed_marker_size_for_layout], [margin + fixed_marker_size_for_layout, output_height - margin - fixed_marker_size_for_layout], [margin + fixed_marker_size_for_layout, output_height - margin], [margin, output_height - fixed_marker_size_for_layout]]
    }
    # Convert ideal_marker_layout values to numpy arrays of float32
    for key in ideal_marker_layout:
        ideal_marker_layout[key] = np.array(ideal_marker_layout[key], dtype=np.float32)

    # 3. Instantiate Aligner
    aligner = Aligner(debug_mode=True, output_dir=test_output_dir)

    # 4. Call aligner.align_image
    print("Attempting alignment...")
    aligned_image, alignment_data = aligner.align_image(
        image=distorted_image,
        marker_map=ideal_marker_layout,
        output_size_wh=(output_width, output_height)
    )

    # 5. Verify the output
    if aligned_image is not None:
        final_aligned_image_path = os.path.join(test_output_dir, "final_aligned_aruco_test_result.png")
        cv2.imwrite(final_aligned_image_path, aligned_image)
        print(f"✅ Alignment successful! Final aligned image saved to: {final_aligned_image_path}")
        print(f"Alignment Data: {alignment_data}")
        print() # Add a newline

        # --- Visualization ---
        print("Generating visualization of point connections...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # 1 row, 2 columns

        # Convert images to RGB for matplotlib
        distorted_image_rgb = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB)
        aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

        # Subplot 1: Distorted Image with detected corners
        axes[0].imshow(distorted_image_rgb)
        axes[0].set_title("Distorted Image (Detected Corners)")
        axes[0].axis('off')

        # Subplot 2: Aligned Image with ideal corners
        axes[1].imshow(aligned_image_rgb)
        axes[1].set_title("Aligned Image (Ideal Corners)")
        axes[1].axis('off')

        # Extract data for plotting
        detected_corners = alignment_data['detected_corners']
        detected_ids = alignment_data['detected_ids']
        used_marker_map = alignment_data['used_marker_map']

        # Draw detected corners and ideal corners, and connecting lines
        colors = plt.cm.get_cmap('hsv', len(np.array(detected_ids).flatten())) # Unique color for each marker

        for i, marker_id in enumerate(np.array(detected_ids).flatten()): # Fix: Ensure detected_ids is numpy array
            if marker_id in used_marker_map:
                # Detected corners (source points)
                corners_src = np.array(detected_corners[i][0]) # Fix: Explicitly convert to numpy array
                axes[0].scatter(corners_src[:, 0], corners_src[:, 1], s=100, marker='o', color=colors(i), edgecolor='k', linewidth=1, label=f'ID {marker_id}')
                for j, p in enumerate(corners_src):
                    axes[0].text(p[0]+5, p[1]+5, f'{marker_id}-{j}', color=colors(i), fontsize=8)

                # Ideal corners (destination points)
                corners_dst = np.array(used_marker_map[marker_id]) # Fix: Explicitly convert to numpy array
                axes[1].scatter(corners_dst[:, 0], corners_dst[:, 1], s=100, marker='x', color=colors(i), edgecolor='k', linewidth=1, label=f'ID {marker_id}')
                for j, p in enumerate(corners_dst):
                    axes[1].text(p[0]+5, p[1]+5, f'{marker_id}-{j}', color=colors(i), fontsize=8)

                # Draw connecting lines (conceptual, not actual transformation lines)
                # For each marker, connect its detected corners to its ideal corners
                # This is a simplified representation, not a true homography visualization
                for j in range(4): # Assuming 4 corners per marker
                    p_src = corners_src[j]
                    p_dst = corners_dst[j]
                    
                    # Draw line from detected point to its corresponding ideal point
                    # This requires transforming the ideal point back to the source image space
                    # or transforming the source point to the ideal image space.
                    # For visual clarity, we'll just draw the points and rely on color coding.
                    # A more complex visualization would involve projecting points.
                    pass # Skip drawing lines between subplots for simplicity, rely on color coding

        plt.tight_layout()
        plot_path = os.path.join(test_output_dir, "aruco_alignment_visualization.png")
        plt.savefig(plot_path)
        plt.close(fig) # Close the figure to free memory
        print(f"Visualization saved to: {plot_path}")

    else:
        print("❌ Alignment failed.")

import unittest

class TestAligner(unittest.TestCase):

    def test_align_image_by_markers(self):
        # 1. Generate a distorted image with ArUco markers
        marker_ids = [10, 20, 30, 40]
        marker_size_px = 80
        image_size_wh = (800, 600)
        perspective_transform_params = {
            "pts2": [[100, 50], [image_size_wh[0]-50, 0], [50, image_size_wh[1]-100], [image_size_wh[0]-100, image_size_wh[1]-50]]
        }
        distorted_image = create_distorted_aruco_image("distorted_aruco_test_image.png", marker_ids, marker_size_px, image_size_wh, perspective_transform_params)

        # 2. Define marker_map and output_size_wh for alignment
        margin = 50
        fixed_marker_size_for_layout = 80
        output_width, output_height = image_size_wh

        ideal_marker_layout = {
            10: [[margin, margin], [margin + fixed_marker_size_for_layout, margin], [margin + fixed_marker_size_for_layout, margin + fixed_marker_size_for_layout], [margin, margin + fixed_marker_size_for_layout]],
            20: [[output_width - margin - fixed_marker_size_for_layout, margin], [output_width - margin, margin], [output_width - margin, margin + fixed_marker_size_for_layout], [output_width - margin - fixed_marker_size_for_layout, margin + fixed_marker_size_for_layout]],
            30: [[output_width - margin - fixed_marker_size_for_layout, output_height - margin - fixed_marker_size_for_layout], [output_width - margin, output_height - margin - fixed_marker_size_for_layout], [output_width - margin, output_height - margin], [output_width - margin - fixed_marker_size_for_layout, output_height - margin]],
            40: [[margin, output_height - margin - fixed_marker_size_for_layout], [margin + fixed_marker_size_for_layout, output_height - margin - fixed_marker_size_for_layout], [margin + fixed_marker_size_for_layout, output_height - margin], [margin, output_height - fixed_marker_size_for_layout]]
        }
        for key in ideal_marker_layout:
            ideal_marker_layout[key] = np.array(ideal_marker_layout[key], dtype=np.float32)

        # 3. Instantiate Aligner
        from src.alignment.aligner import ArucoAligner
        aligner = ArucoAligner(debug_mode=True, output_dir="test_output_dir")

        # 4. Call aligner.align_image
        aligned_image, _, _, _, _ = aligner.align_image_by_markers(
            image=distorted_image,
            marker_map=ideal_marker_layout,
            output_size_wh=(output_width, output_height)
        )

        # 5. Verify the output
        self.assertIsNotNone(aligned_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ArUco Aligner with generated or external image.")
    parser.add_argument("--image_path", type=str, help="Optional: Path to an external image to test alignment.")
    args = parser.parse_args()

    if args.image_path:
        run_aligner_test(image_path=args.image_path)
    else:
        unittest.main()
