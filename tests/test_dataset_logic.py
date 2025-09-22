import os
from pathlib import Path
import sys
import json

# Add project root to path to allow src imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.color_analysis.project_manager import ProjectManager

def run_dataset_logic_test():
    """
    Tests the core logic of finding training images, bypassing the GUI.
    """
    project_name = "benagol"
    print(f"--- Running Dataset Logic Test for project: '{project_name}' ---")

    # 1. Instantiate ProjectManager
    try:
        project_manager = ProjectManager()
        print("ProjectManager instantiated successfully.")
    except Exception as e:
        print(f"!!! ERROR: Failed to instantiate ProjectManager: {e}")
        return

    # 2. Call the core logic function
    project_files = None
    try:
        print(f"\nCalling get_project_file_paths for '{project_name}'...")
        project_files = project_manager.get_project_file_paths(project_name, debug_mode=True)
        print("get_project_file_paths executed.")
    except Exception as e:
        print(f"!!! ERROR: get_project_file_paths raised an exception: {e}")
        # Also, let's check the debug log file that the method should have created
        try:
            with open("debug_log.txt", "r") as f:
                print("\n--- Contents of debug_log.txt ---")
                print(f.read())
                print("---------------------------------")
        except FileNotFoundError:
            print("debug_log.txt was not created.")
        return

    # 3. Analyze the results
    print("\n--- Analyzing Results ---")
    training_image_configs = project_files.get("training_image_configs", [])
    
    print(f"Found {len(training_image_configs)} training image(s).")
    
    # 4. Read the debug log file created by the function
    try:
        with open("debug_log.txt", "r") as f:
            print("\n--- Contents of debug_log.txt ---")
            print(f.read())
            print("---------------------------------")
    except FileNotFoundError:
        print("\n--- WARNING: debug_log.txt was not created. ---")

    if not training_image_configs:
        print("\n!!! FAILURE: No training images were found by the ProjectManager.")
        print("This confirms the issue is in the file-finding logic.")
    else:
        print("\n--- SUCCESS: Training images found! ---")
        print("The core logic is working correctly.")
        print("List of found image configurations:")
        print(json.dumps(training_image_configs, indent=2))
        

if __name__ == "__main__":
    run_dataset_logic_test()
