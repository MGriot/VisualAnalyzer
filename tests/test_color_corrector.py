"""
Test suite for the ColorCorrector module.
"""

import pytest
import cv2
import numpy as np
import os
from pathlib import Path

# Add project root to sys.path to allow imports from src
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.color_correction.corrector import ColorCorrector
from src.utils.color_check_generator import generate_16_color_check
from src.utils.image_utils import load_image

@pytest.fixture(scope="module")
def temp_test_dir(tmpdir_factory):
    """
    Pytest fixture to create a temporary directory for storing test assets.
    This fixture has a 'module' scope, so the directory is created once per test module.
    """
    tmp_dir = tmpdir_factory.mktemp("color_correction_test")
    return Path(str(tmp_dir))

def test_color_correction_pipeline(temp_test_dir):
    """
    Tests the full color correction pipeline.

    This test verifies the core functionality of the ColorCorrector class by performing
    the following steps:
    1.  Generates a clean, ideal reference color checker image.
    2.  Creates a distorted version of the reference image to simulate a real-world
        photograph with color inaccuracies (e.g., a blue tint and reduced brightness).
    3.  Instantiates the `ColorCorrector` and uses it to calculate a color correction
        matrix by comparing the distorted image to the reference image.
    4.  Applies the calculated correction matrix to the distorted image to produce a
        corrected image.
    5.  Compares the average colors of the patches in the corrected image against the
        patches in the original reference image.
    6.  Asserts that the color difference after correction is significantly smaller
        than the initial difference, and that the final color error is below an
        acceptable threshold.
    """
    # --- 1. Setup: Generate test images ---
    reference_path = temp_test_dir / "reference_colorchecker.png"
    source_path = temp_test_dir / "source_colorchecker.png"

    # Generate the ideal reference image using the utility function
    generate_16_color_check(width=800, height=600, filename=str(reference_path))

    # Load the reference image and create a distorted source image
    # This simulates a photo taken under different lighting conditions
    reference_img_bgr, _ = load_image(str(reference_path), handle_transparency=False)
    
    # Apply a distortion: add a blue tint and reduce overall brightness
    distorted_img = reference_img_bgr.astype(np.float32)
    distorted_img[:, :, 0] += 40  # Increase blue channel
    distorted_img[:, :, 1] -= 10  # Decrease green channel
    distorted_img *= 0.8          # Reduce brightness
    distorted_img = np.clip(distorted_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(source_path), distorted_img)

    # Ensure that the test assets were created successfully
    assert reference_path.exists(), "Reference color checker image was not created."
    assert source_path.exists(), "Distorted source image was not created."

    # --- 2. Execute the Color Correction ---
    color_corrector = ColorCorrector()

    # Calculate the correction matrix by comparing the source to the reference
    _, correction_matrix = color_corrector.correct_image_colors(
        source_image_path=str(source_path),
        reference_image_path=str(reference_path),
        debug_mode=True  # Enable debug output for better diagnostics on failure
    )

    # Apply the calculated matrix to the distorted image
    corrected_image = color_corrector.apply_color_correction(distorted_img, correction_matrix)

    # --- 3. Verification ---
    # The most reliable way to verify the correction is to compare the average colors
    # of the detected patches in the corrected image to the reference image.

    # Detect patches in the reference, distorted, and corrected images
    reference_patches = color_corrector.detect_color_checker_patches(reference_img_bgr, debug_mode=True)
    corrected_patches = color_corrector.detect_color_checker_patches(corrected_image, debug_mode=True)
    distorted_patches = color_corrector.detect_color_checker_patches(distorted_img, debug_mode=True)

    # Basic sanity checks for patch detection
    assert len(reference_patches) > 0, "Could not detect any patches in the reference image."
    assert len(corrected_patches) > 0, "Could not detect any patches in the corrected image."
    assert len(reference_patches) == len(corrected_patches), \
        f"Mismatch in detected patch count: Ref={len(reference_patches)}, Corrected={len(corrected_patches)}"

    # Calculate the average colors for each set of patches
    reference_colors = color_corrector.calculate_average_color(reference_patches)
    corrected_colors = color_corrector.calculate_average_color(corrected_patches)
    source_colors_distorted = color_corrector.calculate_average_color(distorted_patches)

    # Calculate the mean absolute difference before and after correction
    initial_diff = np.mean(np.abs(np.array(source_colors_distorted, dtype=np.float32) - np.array(reference_colors, dtype=np.float32)))
    final_diff = np.mean(np.abs(np.array(corrected_colors, dtype=np.float32) - np.array(reference_colors, dtype=np.float32)))

    print(f"Initial average color difference (Distorted vs. Ref): {initial_diff:.2f}")
    print(f"Final average color difference (Corrected vs. Ref): {final_diff:.2f}")

    # Assert that the correction significantly reduced the color error
    assert final_diff < initial_diff / 2, "Color correction did not improve color accuracy by at least 50%."
    
    # Assert that the final color error is below a reasonable threshold (e.g., 15 on a 0-255 scale)
    assert final_diff < 15, f"Corrected colors are still too far from reference. Average diff: {final_diff:.2f}"
