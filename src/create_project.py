"""
This module provides functionality to create and initialize new project directories
for the Visual Analyzer application.

It sets up the required folder structure, generates default configuration files,
and creates a default ArUco reference image.
"""

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

    The generated image contains four ArUco markers (IDs 0, 1, 2, 3) placed at
    the corners of an A4 landscape-sized white canvas. This image can be printed
    and used as a reference for geometrical alignment.

    Args:
        output_path (Path): The full path, including filename and extension,
                            where the generated ArUco reference image will be saved.

    Returns:
        str: A message indicating the successful generation and the path of the image.
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

def create_project(project_name: str):
    """
    Creates a new project directory with a predefined structure and default configuration files.

    The project directory will be created under `data/projects/`.
    It includes subdirectories for `dataset` (colorchecker, aruco, training, drawing, object)
    and `samples`, along with `project_config.json` and `dataset_item_processing_config.json`.
    A default ArUco reference image is also generated.

    Args:
        project_name (str): The name of the new project to be created.

    Returns:
        List[str]: A list of status messages indicating the success or failure of each step.
                   If the project already exists, an error message is returned.
    """
    messages = []
    project_path = config.PROJECTS_DIR / project_name
    if project_path.exists():
        return [f"Error: Project '{project_name}' already exists at {project_path}"]

    try:
        # Create directories
        dirs_to_create = [
            project_path / "dataset" / "colorchecker",
            project_path / "dataset" / "aruco",
            project_path / "dataset" / "training",
            project_path / "dataset" / "drawing",
            project_path / "dataset" / "object",
            project_path / "samples" / "test" / "colorchecker",
            project_path / "samples" / "test" / "sample",
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
            messages.append(f"Created directory: {d}")

        # Generate default ArUco reference
        aruco_ref_filename = "default_aruco_reference.png"
        aruco_path = project_path / "dataset" / "aruco"
        messages.append(generate_aruco_reference(aruco_path / aruco_ref_filename))

        # Create project_config.json
        project_config_path = project_path / "project_config.json"
        default_project_config = {
            "reference_color_checker_path": "dataset/colorchecker/colorchecker.png",
            "training_path": "dataset/training",
            "colorchecker_reference_for_project": [],
            "object_reference_path": None,
            "technical_drawing_path_layer_1": None,
            "technical_drawing_path_layer_2": None,
            "technical_drawing_path_layer_3": None,
            "aruco_reference_path": f"dataset/aruco/{aruco_ref_filename}",
            "aruco_marker_map": {},
            "aruco_output_size": [1000, 1000]
        }
        with open(project_config_path, 'w') as f:
            json.dump(default_project_config, f, indent=4)
        messages.append(f"Created config file: {project_config_path}")

        # Create dataset_item_processing_config.json
        dataset_config_path = project_path / "dataset_item_processing_config.json"
        default_dataset_config = {"image_configs": []}
        with open(dataset_config_path, 'w') as f:
            json.dump(default_dataset_config, f, indent=4)
        messages.append(f"Created config file: {dataset_config_path}")

        messages.append(f"\nProject '{project_name}' created successfully.")
        messages.append("Please add your reference color checker image to the 'dataset/colorchecker' directory.")

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
