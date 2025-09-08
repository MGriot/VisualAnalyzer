import unittest
import os
import shutil
from pathlib import Path
from src.pipeline import run_analysis
from src import config
import pydantic

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.project_name = "test_project"
        self.project_path = config.PROJECTS_DIR / self.project_name
        self.output_path = config.OUTPUT_DIR / self.project_name
        self.image_path = self.project_path / "test_image_PN123_T1.png"

        # Create a dummy project
        os.makedirs(self.project_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        # Create a dummy image
        import numpy as np
        import cv2
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(self.image_path), dummy_image)

        # Create dummy config files
        project_config = {
            "reference_color_checker_filename": "ref_cc.png",
            "colorchecker_reference_for_project": ["ref_cc.png"],
        }
        with open(self.project_path / "project_config.json", "w") as f:
            import json
            json.dump(project_config, f)

        sample_processing_config = {
            "image_configs": [
                {
                    "filename": "test_image.png",
                    "method": "full_average"
                }
            ]
        }
        with open(self.project_path / "sample_processing_config.json", "w") as f:
            import json
            json.dump(sample_processing_config, f)

        # Create a dummy reference color checker
        ref_cc_path = self.project_path / "ref_cc.png"
        cv2.imwrite(str(ref_cc_path), dummy_image)

    def tearDown(self):
        shutil.rmtree(self.project_path)
        shutil.rmtree(self.output_path)

    def test_run_analysis(self):
        class Args:
            pass
        args = Args()
        args.project = self.project_name
        args.image = str(self.image_path)
        args.video = None
        args.camera = False
        args.debug = False
        args.aggregate = False
        args.blur = False
        args.alignment = False
        args.drawing = None
        args.color_alignment = False
        args.symmetry = False

        run_analysis(args)

        # Check if the report was generated
        report_path = self.output_path / "PN123.pdf"
        self.assertTrue(os.path.exists(report_path))

if __name__ == '__main__':
    unittest.main()
