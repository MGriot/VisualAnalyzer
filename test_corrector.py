
import os
import cv2
from src.color_correction.corrector import ColorCorrector
from src.utils.image_utils import save_image
import numpy as np

def run_test():
    # 1. Setup
    corrector = ColorCorrector()
    ref_image_path = "test/colorchecker.png"
    source_image_path = "test/colorchecker_altered.png"
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reference image: {ref_image_path}")
    print(f"Source image: {source_image_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 40)

    methods = ['linear', 'polynomial', 'hsv', 'histogram']

    # 2. Loop through methods and test
    for method in methods:
        print(f"--- Testing method: {method} ---")
        try:
            method_output_dir = os.path.join(output_dir, method)
            os.makedirs(method_output_dir, exist_ok=True)

            result = corrector.correct_image_colors(
                source_image_path=source_image_path,
                reference_image_path=ref_image_path,
                output_dir=method_output_dir,
                debug_mode=True,
                method=method
            )

            # 3. Save results and show info
            corrected_image = result.get("corrected_image")
            correction_model = result.get("correction_model")
            debug_paths = result.get("debug_paths")

            if corrected_image is not None:
                final_image_path = os.path.join(method_output_dir, f"final_corrected_{method}.png")
                save_image(final_image_path, corrected_image)
                print(f"Successfully corrected image. Saved to: {final_image_path}")

            if correction_model:
                print("Calculated Correction Model:")
                for key, value in correction_model.items():
                    if isinstance(value, np.ndarray):
                        # Pretty print numpy arrays
                        print(f"  {key}: \n{np.round(value, 4)}")
                    elif isinstance(value, list) and all(isinstance(i, np.ndarray) for i in value):
                        print(f"  {key}: LUTs with {len(value)} channels")
                    else:
                        print(f"  {key}: {value}")
            
            if debug_paths:
                print("Debug images created:")
                for name, path in debug_paths.items():
                    print(f"  - {name}: {path}")

        except Exception as e:
            print(f"Error testing method '{method}': {e}")
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    run_test()
