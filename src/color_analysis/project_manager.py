import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import json
import time

from src import config
from src.utils.image_utils import load_image
from src.color_correction.corrector import ColorCorrector # Import ColorCorrector
from src.sample_manager.processor import SampleProcessor # Import SampleProcessor

class ProjectManager:
    """
    Manages projects, including listing available projects, providing paths to
    reference color checkers and sample images, and calculating average HSV colors.
    Also handles caching of calculated color correction matrices and HSV ranges.
    """

    def __init__(self):
        """
        Initializes the ProjectManager.
        """
        self.projects_root = config.PROJECTS_DIR
        self.color_corrector = ColorCorrector() # Initialize ColorCorrector
        self.sample_processor = SampleProcessor() # Initialize SampleProcessor
        self.cache_dir = config.OUTPUT_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file_path(self, project_name: str) -> Path:
        """
        Returns the path to the cache file for a given project.
        """
        return self.cache_dir / f"{project_name}_cache.json"

    def list_projects(self) -> List[str]:
        """
        Lists all available project names.

        Returns:
            List[str]: A list of project names.
        """
        if not self.projects_root.exists():
            return []
        return [d.name for d in self.projects_root.iterdir() if d.is_dir()]

    def _get_project_config(self, project_name: str) -> Dict:
        """
        Reads the project's configuration file.
        """
        project_path = self.projects_root / project_name
        config_file_path = project_path / "project_config.json"
        if not config_file_path.is_file():
            raise FileNotFoundError(f"Configuration file 'project_config.json' not found for project '{project_name}'.")

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        return config_data

    def _get_sample_processing_config(self, project_name: str) -> Dict:
        """
        Reads the sample processing configuration file for a project.
        """
        project_path = self.projects_root / project_name
        config_file_path = project_path / "sample_processing_config.json"
        if not config_file_path.is_file():
            return {"image_configs": []} # Return empty config if not found

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        return config_data

    def get_project_file_paths(self, project_name: str, debug_mode: bool = False) -> Dict[str, Path | List[Path] | List[Dict]]:
        """
        Gets the file paths for a given project based on its configuration.

        Args:
            project_name (str): The name of the project.
            debug_mode (bool): If True, prints debug information.

        Returns:
            Dict[str, Path | List[Path] | List[Dict]]: A dictionary containing paths to the reference color checker,
                                          color checker images for project calibration, and sample image configurations.

        Raises:
            ValueError: If the project or its configuration file does not exist.
            FileNotFoundError: If specified files within the project are not found.
        """
        project_path = self.projects_root / project_name
        if not project_path.is_dir():
            raise ValueError(f"Project '{project_name}' not found.")

        config_data = self._get_project_config(project_name)
        sample_processing_config = self._get_sample_processing_config(project_name)

        ref_color_checker_filename = config_data.get("reference_color_checker_filename")
        colorchecker_ref_for_project_relative = config_data.get("colorchecker_reference_for_project", [])
        technical_drawing_filename = config_data.get("technical_drawing_filename")
        
        # New: ArUco alignment configuration
        aruco_marker_map = config_data.get("aruco_marker_map")
        aruco_output_size = config_data.get("aruco_output_size")

        if not ref_color_checker_filename:
            raise ValueError(f"'reference_color_checker_filename' not specified in project_config.json for project '{project_name}'.")

        ref_color_checker_path = project_path / ref_color_checker_filename
        if not ref_color_checker_path.is_file():
            raise FileNotFoundError(f"Reference color checker '{ref_color_checker_filename}' not found for project '{project_name}'.")

        technical_drawing_path = None
        if technical_drawing_filename:
            technical_drawing_path = project_path / technical_drawing_filename
            if not technical_drawing_path.is_file():
                if debug_mode: print(f"[DEBUG] Warning: Technical drawing '{technical_drawing_filename}' not found for project '{project_name}'. Skipping.")
                technical_drawing_path = None

        colorchecker_ref_for_project_paths = []
        for rel_path in colorchecker_ref_for_project_relative:
            full_path = project_path / rel_path
            if full_path.is_file():
                colorchecker_ref_for_project_paths.append(full_path)
            else:
                if debug_mode: print(f"[DEBUG] Warning: Color checker reference image '{rel_path}' not found for project '{project_name}'. Skipping.")

        # Dynamically discover sample images (all other files in folder, excluding specified color checkers)
        sample_image_configs = []
        excluded_files = {ref_color_checker_path.name} | {Path(p).name for p in colorchecker_ref_for_project_relative}
        if technical_drawing_filename:
            excluded_files.add(technical_drawing_filename)

        if debug_mode: print(f"[DEBUG] Excluded files for sample discovery: {excluded_files}")

        for item in project_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg'] and item.name not in excluded_files:
                # Check if there's a specific config for this image
                img_config = next((cfg for cfg in sample_processing_config["image_configs"] if cfg["filename"] == item.name), None)
                if img_config:
                    sample_image_configs.append({"path": item, "method": img_config["method"], "points": img_config.get("points")})
                else:
                    sample_image_configs.append({"path": item, "method": "full_average"}) # Default to full average
            elif item.is_dir() and item.name == "samples": # Include files from the 'samples' subdirectory
                for sample_item in item.iterdir():
                    if sample_item.is_file() and sample_item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        img_config = next((cfg for cfg in sample_processing_config["image_configs"] if cfg["filename"] == sample_item.name), None)
                        if img_config:
                            sample_image_configs.append({"path": sample_item, "method": img_config["method"], "points": img_config.get("points")})
                        else:
                            sample_image_configs.append({"path": sample_item, "method": "full_average"}) # Default to full average

        if not sample_image_configs:
            if debug_mode: print(f"[DEBUG] Warning: No valid sample images found for project '{project_name}'.")

        if debug_mode:
            print(f"[DEBUG] Project '{project_name}' paths:")
            print(f"[DEBUG]   Reference Color Checker: {ref_color_checker_path}")
            print(f"[DEBUG]   Color Checker Refs for Project: {colorchecker_ref_for_project_paths}")
            print(f"[DEBUG]   Sample Image Configurations: {sample_image_configs}")
            if technical_drawing_path:
                print(f"[DEBUG]   Technical Drawing: {technical_drawing_path}")

        return {
            "reference_color_checker": ref_color_checker_path,
            "colorchecker_reference_for_project": colorchecker_ref_for_project_paths,
            "sample_image_configs": sample_image_configs,
            "technical_drawing": technical_drawing_path,
            "aruco_marker_map": aruco_marker_map, # New
            "aruco_output_size": aruco_output_size # New
        }

    def get_hsv_colors_from_samples(self, sample_image_configs: List[Dict], debug_mode: bool = False) -> np.ndarray:
        """
        Extracts all HSV colors from a list of sample image configurations.

        Args:
            sample_image_configs (List[Dict]): A list of dictionaries, each containing 'path', 'method', and 'points' (if applicable).
            debug_mode (bool): If True, prints debug information.

        Returns:
            np.ndarray: A numpy array of all extracted HSV colors.
        """
        all_hsv_colors = []

        if not sample_image_configs:
            raise ValueError("No sample image configurations provided to extract HSV colors.")

        if debug_mode: print(f"[DEBUG] Extracting HSV colors from {len(sample_image_configs)} sample image configurations.")

        for img_config in sample_image_configs:
            sample_file_path = img_config['path']
            method = img_config['method']
            points = img_config.get('points')

            try:
                if method == "full_average":
                    hsv_colors = self.sample_processor.extract_hsv_from_full_image(sample_file_path)
                    if debug_mode: print(f"[DEBUG]   Processed {sample_file_path.name} using full_average.")
                elif method == "points":
                    if not points: raise ValueError(f"Points not specified for {sample_file_path.name} with 'points' method.")
                    hsv_colors = self.sample_processor.extract_hsv_from_points(sample_file_path, points)
                    if debug_mode: print(f"[DEBUG]   Processed {sample_file_path.name} using points method with {len(points)} points.")
                else:
                    if debug_mode: print(f"[DEBUG]   Unknown method '{method}' for {sample_file_path.name}. Skipping.")
                    continue

                all_hsv_colors.append(hsv_colors)

            except Exception as e:
                if debug_mode: print(f"[DEBUG] Warning: Error processing sample image {sample_file_path.name}: {e}. Skipping.")
                continue

        if not all_hsv_colors:
            raise ValueError(f"No valid sample images processed or no non-transparent pixels in provided paths.")

        return np.vstack(all_hsv_colors)

    def calculate_average_hsv_from_samples(self, sample_image_configs: List[Dict], debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the average HSV color (lower, upper, center) from a list of sample image configurations.

        Args:
            sample_image_configs (List[Dict]): A list of dictionaries, each containing 'path', 'method', and 'points' (if applicable).
            debug_mode (bool): If True, prints debug information.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the lower HSV limit,
                                                      upper HSV limit, and the center HSV color.
        """
        h_values = []
        s_values = []
        v_values = []

        if not sample_image_configs:
            raise ValueError("No sample image configurations provided to calculate average HSV.")

        if debug_mode: print(f"[DEBUG] Calculating average HSV from {len(sample_image_configs)} sample image configurations.")

        for img_config in sample_image_configs:
            sample_file_path = img_config['path']
            method = img_config['method']
            points = img_config.get('points')

            try:
                if method == "full_average":
                    avg_h, avg_s, avg_v = self.sample_processor.calculate_hsv_from_full_image(sample_file_path)
                    if debug_mode: print(f"[DEBUG]   Processed {sample_file_path.name} using full_average.")
                elif method == "points":
                    if not points: raise ValueError(f"Points not specified for {sample_file_path.name} with 'points' method.")
                    avg_h, avg_s, avg_v = self.sample_processor.calculate_hsv_from_points(sample_file_path, points)
                    if debug_mode: print(f"[DEBUG]   Processed {sample_file_path.name} using points method with {len(points)} points.")
                else:
                    if debug_mode: print(f"[DEBUG]   Unknown method '{method}' for {sample_file_path.name}. Skipping.")
                    continue

                h_values.append(avg_h)
                s_values.append(avg_s)
                v_values.append(avg_v)

            except Exception as e:
                if debug_mode: print(f"[DEBUG] Warning: Error processing sample image {sample_file_path.name}: {e}. Skipping.")
                continue

        if not h_values:
            raise ValueError(f"No valid sample images processed or no non-transparent pixels in provided paths.")

        avg_h = np.mean(h_values)
        avg_s = np.mean(s_values)
        avg_v = np.mean(v_values)

        # Define a tolerance for the color range
        h_tolerance = 10  # Degrees
        s_tolerance = 30  # Percentage
        v_tolerance = 30  # Percentage

        lower_h = max(0, avg_h - h_tolerance)
        upper_h = min(179, avg_h + h_tolerance)  # OpenCV HSV H range is 0-179

        lower_s = max(0, avg_s - s_tolerance)
        upper_s = min(255, avg_s + s_tolerance) # OpenCV HSV S range is 0-255

        lower_v = max(0, avg_v - v_tolerance)
        upper_v = min(255, avg_v + v_tolerance) # OpenCV HSV V range is 0-255

        lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        center_color = np.array([avg_h, avg_s, avg_v], dtype=np.uint8)

        if debug_mode:
            print(f"[DEBUG] Calculated HSV Range: Lower={lower_limit}, Upper={upper_limit}, Center={center_color}")

        return lower_limit, upper_limit, center_color

    def get_project_data(self, project_name: str, debug_mode: bool = False) -> Dict[str, any]:
        """
        Calculates and caches the color correction matrix and HSV range for a project.
        Recalculates if source files have changed.

        Returns:
            Dict[str, any]: Contains 'correction_matrix', 'lower_hsv', 'upper_hsv', 'center_hsv'.
        """
        cache_file_path = self._get_cache_file_path(project_name)
        cached_data = None
        source_file_timestamps = {}

        # Get current file paths for comparison
        current_file_paths_dict = self.get_project_file_paths(project_name, debug_mode=debug_mode)
        current_source_files = set()
        current_source_files.add(str(current_file_paths_dict['reference_color_checker']))
        for p in current_file_paths_dict['colorchecker_reference_for_project']:
            current_source_files.add(str(p))
        for img_config in current_file_paths_dict['sample_image_configs']:
            current_source_files.add(str(img_config['path']))
            # Also add the sample_processing_config.json itself to the watched files
        current_source_files.add(str(self.projects_root / project_name / "project_config.json"))
        current_source_files.add(str(self.projects_root / project_name / "sample_processing_config.json"))

        if cache_file_path.exists():
            try:
                with open(cache_file_path, 'r') as f:
                    loaded_cache = json.load(f)
                
                # Deserialize NumPy arrays
                correction_matrix_list = loaded_cache['data']['correction_matrix']
                loaded_cache['data']['correction_matrix'] = np.array(correction_matrix_list, dtype=np.float32)
                loaded_cache['data']['lower_hsv'] = np.array(loaded_cache['data']['lower_hsv'], dtype=np.uint8)
                loaded_cache['data']['upper_hsv'] = np.array(loaded_cache['data']['upper_hsv'], dtype=np.uint8)
                loaded_cache['data']['center_hsv'] = np.array(loaded_cache['data']['center_hsv'], dtype=np.uint8)

                cached_data = loaded_cache['data']
                source_file_timestamps = loaded_cache['source_file_timestamps']

                # Verify cache validity
                is_cache_valid = True

                # 1. Check if the set of source files has changed
                cached_source_files = set(source_file_timestamps.keys())
                if current_source_files != cached_source_files:
                    is_cache_valid = False
                    if debug_mode: print(f"[DEBUG] Cache invalidated: Source file list changed.")
                else:
                    # 2. Check if any source files have been modified
                    for file_path_str, timestamp in source_file_timestamps.items():
                        current_mtime = Path(file_path_str).stat().st_mtime
                        if debug_mode: print(f"[DEBUG]   File: {file_path_str}, Cached mtime: {timestamp}, Current mtime: {current_mtime}")
                        if current_mtime > timestamp:
                            is_cache_valid = False
                            if debug_mode: print(f"[DEBUG]   Cache invalidated for {file_path_str} (modified).")
                            break
                
                if is_cache_valid:
                    if debug_mode: print(f"[DEBUG] Using cached data for project '{project_name}'.")
                    return cached_data
                else:
                    if debug_mode: print(f"[DEBUG] Cache for project '{project_name}' is outdated. Recalculating...")

            except (json.JSONDecodeError, KeyError, FileNotFoundError, ValueError) as e:
                if debug_mode: print(f"[DEBUG] Error loading or validating cache for project '{project_name}': {e}. Recalculating...")
                cached_data = None # Force recalculation

        if debug_mode: print(f"[DEBUG] Calculating data for project '{project_name}'...")
        file_paths = self.get_project_file_paths(project_name, debug_mode=debug_mode)

        # Calculate correction matrix
        correction_matrix = np.eye(3, dtype=np.float32) # Default to identity
        if file_paths['colorchecker_reference_for_project']:
            # For simplicity, we'll use the first image in the list to calculate the matrix
            # A more robust solution might average matrices from multiple images or use a more complex algorithm
            source_image_path = file_paths['colorchecker_reference_for_project'][0]
            try:
                _, correction_matrix = self.color_corrector.correct_image_colors(
                    source_image_path=str(source_image_path),
                    reference_image_path=str(file_paths['reference_color_checker']),
                    debug_mode=debug_mode
                )
                if debug_mode: print("[DEBUG] Project color alignment matrix calculated.")
            except Exception as e:
                if debug_mode: print(f"[DEBUG] Warning: Could not calculate project color alignment matrix: {e}. Using identity matrix.")

        # Calculate HSV range
        lower_hsv, upper_hsv, center_hsv = self.calculate_average_hsv_from_samples(file_paths['sample_image_configs'], debug_mode=debug_mode)
        if debug_mode: print("[DEBUG] Project HSV range calculated.")

        # Store in cache
        source_file_timestamps = {}
        for p in file_paths['colorchecker_reference_for_project']:
            source_file_timestamps[str(p)] = p.stat().st_mtime
        source_file_timestamps[str(file_paths['reference_color_checker'])] = file_paths['reference_color_checker'].stat().st_mtime
        for img_config in file_paths['sample_image_configs']:
            source_file_timestamps[str(img_config['path'])] = img_config['path'].stat().st_mtime
        source_file_timestamps[str(self.projects_root / project_name / "project_config.json")] = (self.projects_root / project_name / "project_config.json").stat().st_mtime
        source_file_timestamps[str(self.projects_root / project_name / "sample_processing_config.json")] = (self.projects_root / project_name / "sample_processing_config.json").stat().st_mtime

        cached_data_to_save = {
            'correction_matrix': correction_matrix.tolist(), # Convert NumPy array to list for JSON serialization
            'lower_hsv': lower_hsv.tolist(),
            'upper_hsv': upper_hsv.tolist(),
            'center_hsv': center_hsv.tolist(),
        }
        
        full_cache_entry = {
            'data': cached_data_to_save,
            'source_file_timestamps': source_file_timestamps,
        }
        
        with open(cache_file_path, 'w') as f:
            json.dump(full_cache_entry, f, indent=4)
        if debug_mode: print(f"[DEBUG] Cached data saved to {cache_file_path}")

        return {
            'correction_matrix': correction_matrix,
            'lower_hsv': lower_hsv,
            'upper_hsv': upper_hsv,
            'center_hsv': center_hsv,
        }
