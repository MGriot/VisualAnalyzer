import argparse
import os
import cv2
import numpy as np
import json
from pathlib import Path
import tkinter as tk

from src.color_analysis.project_manager import ProjectManager
from src.sample_manager.gui import PointSelectorGUI

def main():
    parser = argparse.ArgumentParser(description="Sample Manager for Visual Analyzer.")
    parser.add_argument("--project", type=str, help="Name of the project to manage samples for.")

    args = parser.parse_args()

    project_manager = ProjectManager()
    available_projects = project_manager.list_projects()

    if not available_projects:
        print("No projects found. Please create a project in the 'data/projects' directory.")
        return

    selected_project = args.project
    if not selected_project:
        print("Available projects:")
        for i, project in enumerate(available_projects):
            print(f"{i+1}. {project}")
        while True:
            try:
                choice = input("Select a project by number or name: ")
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_projects):
                        selected_project = available_projects[choice_idx]
                        break
                    else:
                        print("Invalid project number.")
                elif choice in available_projects:
                    selected_project = choice
                    break
                else:
                    print("Invalid project name or number.")
            except EOFError:
                print("\nExiting due to no further input.")
                return

    try:
        project_path = project_manager.projects_root / selected_project
        project_config_file = project_path / "project_config.json"
        sample_processing_config_file = project_path / "sample_processing_config.json"

        # Load existing sample processing config or create new one
        sample_processing_config = {"image_configs": []}
        if sample_processing_config_file.exists():
            with open(sample_processing_config_file, 'r') as f:
                sample_processing_config = json.load(f)

        # Get all sample images for the project
        # This logic is similar to ProjectManager.get_project_file_paths but focuses on samples
        ref_color_checker_filename = None
        colorchecker_ref_for_project_relative = []
        if project_config_file.exists():
            with open(project_config_file, 'r') as f:
                proj_config_data = json.load(f)
                ref_color_checker_filename = proj_config_data.get("reference_color_checker_filename")
                colorchecker_ref_for_project_relative = proj_config_data.get("colorchecker_reference_for_project", [])

        excluded_files = set()
        if ref_color_checker_filename: excluded_files.add(ref_color_checker_filename)
        excluded_files.update({Path(p).name for p in colorchecker_ref_for_project_relative})
        excluded_files.add(project_config_file.name) # Exclude project_config.json itself
        excluded_files.add(sample_processing_config_file.name) # Exclude sample_processing_config.json itself

        sample_images_in_project = []
        for item in project_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg'] and item.name not in excluded_files:
                sample_images_in_project.append(item)
            elif item.is_dir() and item.name == "samples":
                for sample_item in item.iterdir():
                    if sample_item.is_file() and sample_item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        sample_images_in_project.append(sample_item)

        if not sample_images_in_project:
            print(f"No sample images found in project '{selected_project}'.")
            return

        # Iterate through sample images and launch GUI if needed
        for sample_image_path in sample_images_in_project:
            current_image_config = next((item for item in sample_processing_config["image_configs"] if item["filename"] == sample_image_path.name), None)

            if not current_image_config or current_image_config.get("method") == "points":
                print(f"Managing points for: {sample_image_path.name}")
                root = tk.Tk()
                app = PointSelectorGUI(root, sample_image_path, sample_processing_config_file, current_image_config.get("points") if current_image_config else None)
                root.mainloop()
            else:
                print(f"Using full average for: {sample_image_path.name}")

        # After managing samples, trigger cache invalidation for the project
        # by touching the project_config.json or sample_processing_config.json
        # This will force ProjectManager to recalculate on next run of main.py
        if sample_processing_config_file.exists():
            os.utime(sample_processing_config_file, None) # Update modification time
            print(f"Updated timestamp of {sample_processing_config_file.name} to invalidate cache.")
        else:
            # If sample_processing_config_file was just created, ensure it's saved
            with open(sample_processing_config_file, 'w') as f:
                json.dump(sample_processing_config, f, indent=4)
            print(f"Created {sample_processing_config_file.name} to invalidate cache.")

        print("Sample management complete. Run main.py to use updated color space.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
