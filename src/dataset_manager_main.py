import argparse
import tkinter as tk
from pathlib import Path
import json

from src.color_analysis.project_manager import ProjectManager
from src.sample_manager.dataset_gui import DatasetManagerGUI

def main():
    parser = argparse.ArgumentParser(description="Dataset Manager for Visual Analyzer.")
    parser.add_argument("--project", required=True, type=str, help="Name of the project to manage the dataset for.")
    args = parser.parse_args()

    project_manager = ProjectManager()
    
    try:
        # Get all dataset images for the project
        project_files = project_manager.get_project_file_paths(args.project)
        dataset_image_configs = project_files.get("dataset_image_configs", [])
        dataset_image_paths = [cfg['path'] for cfg in dataset_image_configs]

        if not dataset_image_paths:
            print(f"No dataset images found in project '{args.project}'.")
            return

        config_file_path = project_manager.projects_root / args.project / "dataset_item_processing_config.json"

        # Launch the GUI
        root = tk.Tk()
        app = DatasetManagerGUI(root, dataset_image_paths, config_file_path)
        root.mainloop()

        print("Dataset management complete. Run main.py to use updated color space.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()