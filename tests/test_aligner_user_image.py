import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alignment.aligner import Aligner, generate_aruco_marker_map
from src.utils.image_utils import load_image

def run_final_alignment_test(image_path: str):
    """
    Runs the final, improved alignment process on a single specified image.
    """
    print(f"--- Running Final Alignment Test on: {os.path.basename(image_path)} ---")

    # 1. Setup output directory
    test_output_dir = "output/final_alignment_test"
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(test_output_dir)}")

    # 2. Load the image
    try:
        input_image, _ = load_image(image_path)
        if input_image is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"\n*** ERROR: Could not load image from '{image_path}'. Please check the path. ***")
        return

    # 3. Define alignment parameters
    # Using the layout confirmed by the user and the improved logic with a margin.
    marker_ids = [10, 20, 30, 40]  # Order: TL, TR, BR, BL
    output_height, output_width, _ = input_image.shape
    output_size_wh = (output_width, output_height)
    marker_size_in_output_px = 80
    margin_px = 50 # Use a 50 pixel margin for the alignment target

    print(f"\nImage Size (WxH): {output_size_wh}")
    print(f"Using Marker Layout (TL-TR-BR-BL): {marker_ids}")
    print(f"Using Margin: {margin_px}px")

    # Generate the target map using the improved function
    marker_map = generate_aruco_marker_map(
        output_size_wh=output_size_wh,
        marker_ids=marker_ids,
        marker_size_px=marker_size_in_output_px,
        margin_px=margin_px
    )
    print("Generated target coordinate map with margin.")

    # 4. Initialize and run the Aligner
    aligner = Aligner(debug_mode=True, output_dir=test_output_dir)
    
    print("\n--- Starting alignment process... ---")
    aligned_image, alignment_data = aligner.align_image(
        image=input_image,
        marker_map=marker_map,
        output_size_wh=output_size_wh,
    )

    # 5. Save and report results
    if aligned_image is not None and alignment_data is not None:
        print("\n*** ALIGNMENT SUCCEEDED! ***")
        # Save aligned image
        aligned_filename = "aligned_final_result.png"
        cv2.imwrite(os.path.join(test_output_dir, aligned_filename), aligned_image)

        # Save visualization
        vis_filename = "visualization_final_result.png"
        generate_visualization(input_image, aligned_image, alignment_data, os.path.join(test_output_dir, vis_filename))
        print(f"  Results saved to: {os.path.abspath(test_output_dir)}")
    else:
        print("\n*** ALIGNMENT FAILED. ***")

def generate_visualization(input_image, aligned_image, alignment_data, output_path):
    """Generates and saves a visualization of the alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    aligned_image_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)

    axes[0].imshow(input_image_rgb)
    axes[0].set_title("Original Image (Detected Corners)")
    axes[0].axis('off')

    axes[1].imshow(aligned_image_rgb)
    axes[1].set_title("Aligned Image (Ideal Corners with Margin)")
    axes[1].axis('off')

    detected_corners = alignment_data['detected_corners']
    detected_ids = alignment_data['detected_ids']
    used_marker_map = alignment_data['used_marker_map']

    colors = ['red', 'lime', 'blue', 'yellow', 'cyan', 'magenta']
    flattened_ids = np.array(detected_ids).flatten()
    
    unique_ids = sorted(list(used_marker_map.keys()))
    id_to_color = {marker_id: colors[i % len(colors)] for i, marker_id in enumerate(unique_ids)}

    for i, marker_id in enumerate(flattened_ids):
        if marker_id in used_marker_map:
            color = id_to_color.get(marker_id, 'white')
            corners_src = np.array(detected_corners[i][0])
            axes[0].scatter(corners_src[:, 0], corners_src[:, 1], s=120, marker='o', color=color, edgecolor='black', linewidth=2, label=f'ID {marker_id}')
            for j, p in enumerate(corners_src):
                axes[0].text(p[0] + 8, p[1] + 8, f'{j}', color='white', fontsize=10, weight='bold')

            corners_dst = np.array(used_marker_map[marker_id])
            axes[1].scatter(corners_dst[:, 0], corners_dst[:, 1], s=120, marker='x', color=color, linewidth=3, label=f'ID {marker_id}')
            for j, p in enumerate(corners_dst):
                axes[1].text(p[0] + 8, p[1] + 8, f'{j}', color='black', fontsize=10, weight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc='best')
    
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    axes[1].legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the final, improved ArUco alignment test.")
    parser.add_argument(
        "-i", "--image",
        dest="image_path",
        default="data/projects/alignment_test/image.png",
        help="Path to the image file to be aligned."
    )
    args = parser.parse_args()
    
    run_final_alignment_test(args.image_path)