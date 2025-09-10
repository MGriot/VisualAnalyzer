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

def generate_aruco_reference(output_path: Path):
    """
    Generates a default A4 landscape ArUco reference image with 4 markers at the corners.
    """
    # A4 dimensions at 300 DPI (landscape)
    width_px = 3508
    height_px = 2480
    margin_px = 150  # Margin from the edge
    marker_size_px = 300  # Size of the ArUco marker

    # Create a white A4 landscape image
    image = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

    # Load ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Marker IDs
    marker_ids = [0, 1, 2, 3]

    # Positions for the markers (top-left, top-right, bottom-right, bottom-left)
    positions = [
        (margin_px, margin_px),
        (width_px - margin_px - marker_size_px, margin_px),
        (width_px - margin_px - marker_size_px, height_px - margin_px - marker_size_px),
        (margin_px, height_px - margin_px - marker_size_px),
    ]

    # Generate and draw markers
    for i, marker_id in enumerate(marker_ids):
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
        marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        x, y = positions[i]
        image[y:y+marker_size_px, x:x+marker_size_px] = marker_image

    # Save the image
    cv2.imwrite(str(output_path), image)
    print(f"Generated ArUco reference image: {output_path}")

def create_project(project_name: str):
    """
    Creates a new project with the required directory structure and default configuration files.
    """
    project_path = config.PROJECTS_DIR / project_name
    if project_path.exists():
        print(f"Error: Project '{project_name}' already exists at {project_path}")
        return

    try:
        # Create directories
        dataset_path = project_path / "dataset"
        colorchecker_path = dataset_path / "colorchecker"
        aruco_path = dataset_path / "aruco"
        training_path = dataset_path / "training"
        samples_path = project_path / "samples"
        mock_sample_path = samples_path / "test"
        mock_sample_colorchecker_path = mock_sample_path / "colorchecker"
        mock_sample_sample_path = mock_sample_path / "sample"

        colorchecker_path.mkdir(parents=True, exist_ok=True)
        aruco_path.mkdir(exist_ok=True)
        training_path.mkdir(exist_ok=True)
        mock_sample_colorchecker_path.mkdir(parents=True, exist_ok=True)
        mock_sample_sample_path.mkdir(exist_ok=True)

        print(f"Created directory: {colorchecker_path}")
        print(f"Created directory: {aruco_path}")
        print(f"Created directory: {training_path}")
        print(f"Created directory: {mock_sample_colorchecker_path}")
        print(f"Created directory: {mock_sample_sample_path}")

        # Generate default ArUco reference
        aruco_ref_filename = "default_aruco_reference.png"
        generate_aruco_reference(aruco_path / aruco_ref_filename)

        # Create project_config.json
        project_config_path = project_path / "project_config.json"
        default_project_config = {
            "reference_color_checker_path": "dataset/colorchecker",
            "training_path": "dataset/training",
            "technical_drawing_path": None,
            "aruco_reference_path": "dataset/aruco",
            "aruco_marker_map": {},
            "aruco_output_size": [1000, 1000]
        }
        with open(project_config_path, 'w') as f:
            json.dump(default_project_config, f, indent=4)
        print(f"Created config file: {project_config_path}")

        # Create dataset_item_processing_config.json
        dataset_config_path = project_path / "dataset_item_processing_config.json"
        default_dataset_config = {"image_configs": []}
        with open(dataset_config_path, 'w') as f:
            json.dump(default_dataset_config, f, indent=4)
        print(f"Created config file: {dataset_config_path}")

        print(f"\nProject '{project_name}' created successfully.")
        print("Please add your reference color checker image to the 'dataset/colorchecker' directory.")

    except Exception as e:
        print(f"An error occurred during project creation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create a new project for Visual Analyzer.")
    parser.add_argument("--name", required=True, type=str, help="The name of the new project.")
    args = parser.parse_args()
    create_project(args.name)

if __name__ == "__main__":
    main()