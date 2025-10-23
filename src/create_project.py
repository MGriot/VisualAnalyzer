import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src import config
from src.ColorCheckerGenerator.colorchecker.generator import ColorCheckerGenerator

def generate_aruco_reference(output_path: Path):
    """
    Generates a default A4 landscape ArUco reference image with 4 markers at the corners.
    """
    width_px, height_px = 3508, 2480
    margin_px, marker_size_px = 150, 300
    image = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_ids = [0, 1, 2, 3]
    positions = [
        (margin_px, margin_px),
        (width_px - margin_px - marker_size_px, margin_px),
        (width_px - margin_px - marker_size_px, height_px - margin_px - marker_size_px),
        (margin_px, height_px - margin_px - marker_size_px),
    ]
    for i, marker_id in enumerate(marker_ids):
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
        marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        x, y = positions[i]
        image[y:y+marker_size_px, x:x+marker_size_px] = marker_image
    cv2.imwrite(str(output_path), image)
    return f"Generated ArUco reference image: {output_path}"

def generate_colorchecker_with_aruco(output_path: Path):
    """
    Generates a standard ColorChecker with ArUco markers.
    """
    gen = ColorCheckerGenerator(
        size="20cm",
        dpi=300,
        checker_type="classic",
        include_aruco=True,
        logo_text="Reference"
    )
    gen.build()
    gen.save(str(output_path))
    return f"Generated ColorChecker with ArUco markers: {output_path}"

def create_project(project_name: str):
    """
    Creates a new project directory with a predefined structure and default configuration files.
    """
    messages = []
    project_path = config.PROJECTS_DIR / project_name
    if project_path.exists():
        return [f"Error: Project '{project_name}' already exists at {project_path}"]

    try:
        dataset_path = project_path / "dataset"
        dirs_to_create = [
            dataset_path / "training_images",
            dataset_path / "drawing_layers",
            project_path / "samples",
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
            messages.append(f"Created directory: {d}")

        readme_path = project_path / "samples" / "README.md"
        with open(readme_path, 'w') as f:
            f.write("# Samples Directory\n\nPlace the images you want to analyze in this directory.")
        messages.append(f"Created README: {readme_path}")

        aruco_ref_filename = "default_geometric_align_reference.png"
        messages.append(generate_aruco_reference(dataset_path / aruco_ref_filename))

        cc_aruco_filename = "default_color_checker_reference.png"
        messages.append(generate_colorchecker_with_aruco(dataset_path / cc_aruco_filename))

        project_config_path = project_path / "project_config.json"
        default_project_config = {
            "training_path": "dataset/training_images",
            "object_reference_path": "dataset/object_reference.png",

            "color_correction": {
                "reference_color_checker_path": f"dataset/{cc_aruco_filename}",
                "project_specific_color_checker_path": "dataset/project_color_checker.png"
            },
            "geometrical_alignment": {
                "reference_path": f"dataset/{aruco_ref_filename}",
                "marker_map": {},
                "output_size": [1000, 1000]
            },
            "masking": {
                "drawing_layers": {
                    "1": "dataset/drawing_layers/layer1.png",
                    "2": "dataset/drawing_layers/layer2.png",
                    "3": "dataset/drawing_layers/layer3.png"
                }
            }
        }
        with open(project_config_path, 'w') as f:
            json.dump(default_project_config, f, indent=4)
        messages.append(f"Created config file: {project_config_path}")

        dataset_config_path = project_path / "dataset_item_processing_config.json"
        default_dataset_config = {"image_configs": []}
        with open(dataset_config_path, 'w') as f:
            json.dump(default_dataset_config, f, indent=4)
        messages.append(f"Created config file: {dataset_config_path}")

        messages.append(f"\nProject '{project_name}' created successfully.")
        messages.append("Reference files for alignment and color correction have been automatically generated in the 'dataset' folder.")

    except Exception as e:
        messages.append(f"An error occurred during project creation: {e}")
    
    return messages

def main():
    """
    Main function for the `create_project.py` script.

    Parses command-line arguments to get the project name and calls the
    `create_project` function to set up the new project.
    """

if __name__ == "__main__":
    main()
