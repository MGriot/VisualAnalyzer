import cv2
import numpy as np
import os
from pathlib import Path

# Add project root to path to allow src imports
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.alignment.aligner import Aligner, generate_aruco_marker_map

# --- Test Script for the Aligner Class ---

def run_aligner_test():
    print("--- Running Aligner Class Test ---")
    
    # 1. Define paths and parameters
    project_root_str = "C:/Users/Admin/Documents/Coding/VisualAnalyzer"
    image_path = os.path.join(project_root_str, "data/projects/benagol/samples/test/sample/image.png")
    output_dir = os.path.join(project_root_str, "output/aligner_class_test")
    
    # 2. Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"!!! ERROR: Could not load image at {image_path}")
        return
    print(f"Successfully loaded image: {image_path}")

    # 3. Manually create the marker map and output size (replicating project config)
    output_size_wh = (1000, 1000)
    # The user's image has IDs 10 (TL), 20 (TR), 40 (BL), 30 (BR).
    # The required order for generate_aruco_marker_map is TL, TR, BR, BL.
    marker_ids = [10, 20, 30, 40]
    
    # generate_aruco_marker_map returns a map with int keys. For the test, we need string keys to simulate JSON loading.
    marker_map_int_keys = generate_aruco_marker_map(
        output_size_wh=output_size_wh,
        marker_ids=marker_ids,
        marker_size_px=100, # A reasonable guess
        margin_px=50        # A reasonable guess
    )
    marker_map_str_keys = {str(k): v.tolist() for k, v in marker_map_int_keys.items()}
    print("Successfully generated a test marker map.")

    # 4. Instantiate Aligner (same as pipeline)
    aligner = Aligner(debug_mode=True, output_dir=output_dir)
    print("Aligner class instantiated with debug mode ON.")

    # 5. Call align_image (same as pipeline)
    print("Calling aligner.align_image...")
    aligned_image, alignment_data = aligner.align_image(
        image=image,
        marker_map=marker_map_str_keys, # Use the map with string keys
        output_size_wh=output_size_wh
    )

    # 6. Report results
    if aligned_image is not None and alignment_data is not None:
        print("--- SUCCESS: Aligner returned an image! ---")
        print(f"Detected IDs: {alignment_data.get('detected_ids')}")
        output_path = os.path.join(output_dir, "aligner_test_SUCCESS.png")
        print(f"Saving aligned image to {output_path}")
        cv2.imwrite(output_path, aligned_image)
    else:
        print("--- FAILURE: Aligner returned None. ---")
        print("Check the debug output in the console and the contents of the output/aligner_class_test/aruco_debug directory.")

if __name__ == "__main__":
    run_aligner_test()
