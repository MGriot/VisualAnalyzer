import cv2
import os
import unittest
import numpy as np
from pathlib import Path

# Add project root to path to allow src imports
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.object_alignment.object_aligner import AdvancedAligner
from src.utils.image_utils import load_image

class TestObjectAligner(unittest.TestCase):

    def setUp(self):
        """Set up test images and output directory."""
        self.project_root_str = str(project_root)
        self.ref_image_path = os.path.join(self.project_root_str, "data", "projects", "benagol", "dataset", "aruco", "default_aruco_reference.png")
        self.input_image_path = os.path.join(self.project_root_str, "data", "projects", "benagol", "dataset", "training", "image.png")

        self.output_dir = os.path.join(self.project_root_str, "output", "object_aligner_test")
        os.makedirs(self.output_dir, exist_ok=True)

        self.ref_img, _ = load_image(self.ref_image_path)
        self.input_img, _ = load_image(self.input_image_path)

        self.assertIsNotNone(self.ref_img, "Reference image could not be loaded.")
        self.assertIsNotNone(self.input_img, "Input image could not be loaded.")

        self.aligner = AdvancedAligner()

    def test_alignment_methods(self):
        """Test all alignment methods of the AdvancedAligner."""
        # Note: SIFT may not be available in all OpenCV builds.
        methods = ['feature_orb', 'ecc', 'contour_centroid', 'polygon']
        if self.aligner.sift is not None:
            methods.append('feature_sift')
        else:
            print("Skipping 'feature_sift' test as SIFT is not available in this OpenCV build.")

        for method in methods:
            with self.subTest(method=method):
                aligned_image = self.aligner.align(self.input_img, self.ref_img, method=method)

                self.assertIsNotNone(aligned_image, f"Alignment failed for method: {method}")

                # Save the aligned image for visual inspection
                output_path = os.path.join(self.output_dir, f"aligned_{method}.png")
                cv2.imwrite(output_path, aligned_image)
                self.assertTrue(os.path.exists(output_path), f"Failed to save aligned image for method: {method}")

if __name__ == '__main__':
    unittest.main()
