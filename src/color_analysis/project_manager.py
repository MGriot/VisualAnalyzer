import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import json
import time
from pydantic import ValidationError

from src import config
from src.utils.image_utils import load_image
from src.color_correction.corrector import ColorCorrector
from src.sample_manager.dataset_item_processor import DatasetItemProcessor
from src.config_models import ProjectConfig, DatasetItemProcessingConfig


class ProjectManager:
    """
    Manages projects, including listing available projects, providing paths to
    reference color checkers and dataset images, and calculating average HSV colors.
    Also handles caching of calculated color correction matrices and HSV ranges.
    """

    def __init__(self):
        """
        Initializes the ProjectManager.
        """
        self.projects_root = config.PROJECTS_DIR
        self.color_corrector = ColorCorrector()
        self.dataset_item_processor = DatasetItemProcessor()
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

    def _get_project_config(self, project_name: str) -> ProjectConfig:
        """
        Reads and validates the project's configuration file.
        """
        project_path = self.projects_root / project_name
        config_file_path = project_path / "project_config.json"
        if not config_file_path.is_file():
            raise FileNotFoundError(f"Configuration file 'project_config.json' not found for project '{project_name}'.")

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        
        try:
            return ProjectConfig(**config_data)
        except ValidationError as e:
            raise ValueError(f"Invalid project configuration for '{project_name}':\n{e}")

    def _get_dataset_item_processing_config(self, project_name: str) -> DatasetItemProcessingConfig:
        """
        Reads and validates the dataset item processing configuration file for a project.
        """
        project_path = self.projects_root / project_name
        config_file_path = project_path / "dataset_item_processing_config.json"
        if not config_file_path.is_file():
            return DatasetItemProcessingConfig(image_configs=[])

        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
            
        try:
            return DatasetItemProcessingConfig(**config_data)
        except ValidationError as e:
            raise ValueError(f"Invalid dataset item processing configuration for '{project_name}':\n{e}")


    def get_project_file_paths(self, project_name: str, debug_mode: bool = False) -> Dict[str, Path | List[Path] | List[Dict]]:
        """
        Gets the file paths for a given project based on its configuration.

        Args:
            project_name (str): The name of the project.
            debug_mode (bool): If True, prints debug information.

        Returns:
            Dict[str, Path | List[Path] | List[Dict]]: A dictionary containing paths to the reference color checker,
                                          color checker images for project calibration, and dataset image configurations.

        Raises:
            ValueError: If the project or its configuration file does not exist.
            FileNotFoundError: If specified files within the project are not found.
        """
        project_path = self.projects_root / project_name
        if not project_path.is_dir():
            raise ValueError(f"Project '{project_name}' not found.")

        config_data = self._get_project_config(project_name)
        dataset_item_processing_config = self._get_dataset_item_processing_config(project_name)

        ref_color_checker_rel_path = config_data.reference_color_checker_path
        colorchecker_ref_for_project_relative = config_data.colorchecker_reference_for_project
        technical_drawing_rel_path = config_data.technical_drawing_path
        aruco_ref_rel_path = config_data.aruco_reference_path
        training_rel_path = config_data.training_path
        
        # New: ArUco alignment configuration
        aruco_marker_map = config_data.aruco_marker_map
        aruco_output_size = config_data.aruco_output_size

        if not ref_color_checker_rel_path:
            raise ValueError(f"'reference_color_checker_path' not specified in project_config.json for project '{project_name}'.")

        ref_color_checker_dir = project_path / ref_color_checker_rel_path
        if not ref_color_checker_dir.is_dir():
            raise FileNotFoundError(f"Reference color checker directory '{ref_color_checker_rel_path}' not found for project '{project_name}'.")

        ref_color_checker_path = None
        for item in ref_color_checker_dir.iterdir():
            if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                ref_color_checker_path = item
                break
        
        if not ref_color_checker_path:
            raise FileNotFoundError(f"No reference color checker image found in '{ref_color_checker_dir}'.")

        technical_drawing_path = None
        if technical_drawing_rel_path:
            technical_drawing_path = project_path / technical_drawing_rel_path
            if not technical_drawing_path.is_file():
                if debug_mode: print(f"[DEBUG] Warning: Technical drawing '{technical_drawing_rel_path}' not found for project '{project_name}'. Skipping.")
                technical_drawing_path = None

        colorchecker_ref_for_project_paths = []
        if colorchecker_ref_for_project_relative:
            for rel_path in colorchecker_ref_for_project_relative:
                full_path = project_path / rel_path
                if full_path.is_file():
                    colorchecker_ref_for_project_paths.append(full_path)
                else:
                    if debug_mode: print(f"[DEBUG] Warning: Color checker reference image '{rel_path}' not found for project '{project_name}'. Skipping.")

        aruco_ref_path = None
        if aruco_ref_rel_path:
            aruco_ref_dir = project_path / aruco_ref_rel_path
            if aruco_ref_dir.is_dir():
                for item in aruco_ref_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        aruco_ref_path = item
                        break
            if not aruco_ref_path and debug_mode:
                print(f"[DEBUG] Warning: No ArUco reference image found in '{aruco_ref_dir}'. Skipping.")

        training_image_configs = []
        if training_rel_path:
            training_path = project_path / training_rel_path
            if training_path.is_dir():
                for item in training_path.iterdir():
                    if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        training_image_configs.append({"path": item, "method": "full_average"})

        # Dynamically discover dataset images from the 'dataset' folder
        dataset_image_configs = []
        dataset_path = project_path / "dataset"

        if debug_mode: print(f"[DEBUG] Discovering samples in {dataset_path}")

        if dataset_path.is_dir():
            for item in dataset_path.iterdir():
                if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Check if there's a specific config for this image
                    img_config = next((cfg for cfg in dataset_item_processing_config.image_configs if cfg.filename == item.name), None)
                    if img_config:
                        method = img_config.method
                        points = img_config.points
                        if method == "points" and not points:
                            method = "full_average"
                        points_as_dicts = [p.model_dump() for p in points] if points else None
                        dataset_image_configs.append({"path": item, "method": method, "points": points_as_dicts})
                    else:
                        dataset_image_configs.append({"path": item, "method": "full_average"})

        if not dataset_image_configs:
            if debug_mode: print(f"[DEBUG] Warning: No valid dataset images found for project '{project_name}'.")

        if debug_mode:
            print(f"[DEBUG] Project '{project_name}' paths:")
            print(f"[DEBUG]   Reference Color Checker: {ref_color_checker_path}")
            print(f"[DEBUG]   Color Checker Refs for Project: {colorchecker_ref_for_project_paths}")
            print(f"[DEBUG]   Dataset Image Configurations: {dataset_image_configs}")
            if technical_drawing_path:
                print(f"[DEBUG]   Technical Drawing: {technical_drawing_path}")
            if aruco_ref_path:
                print(f"[DEBUG]   ArUco Reference: {aruco_ref_path}")

        return {
            "reference_color_checker": ref_color_checker_path,
            "colorchecker_reference_for_project": colorchecker_ref_for_project_paths,
            "dataset_image_configs": dataset_image_configs,
            "training_image_configs": training_image_configs,
            "technical_drawing": technical_drawing_path,
            "aruco_reference": aruco_ref_path,
            "aruco_marker_map": aruco_marker_map, # New
            "aruco_output_size": aruco_output_size # New
        }

    def get_hsv_colors_from_dataset(self, dataset_image_configs: List[Dict], debug_mode: bool = False) -> np.ndarray:
        """
        Extracts all HSV colors from a list of dataset image configurations.

        Args:
            dataset_image_configs (List[Dict]): A list of dictionaries, each containing 'path', 'method', and 'points' (if applicable).
            debug_mode (bool): If True, prints debug information.

        Returns:
            np.ndarray: A numpy array of all extracted HSV colors.
        """
        all_hsv_colors = []

        if not dataset_image_configs:
            raise ValueError("No dataset image configurations provided to extract HSV colors.")

        if debug_mode: print(f"[DEBUG] Extracting HSV colors from {len(dataset_image_configs)} dataset image configurations.")

        for img_config in dataset_image_configs:
            dataset_item_file_path = img_config['path']
            method = img_config['method']
            points = img_config.get('points')

            try:
                if method == "full_average":
                    hsv_colors = self.dataset_item_processor.extract_hsv_from_full_image(dataset_item_file_path)
                    if debug_mode: print(f"[DEBUG]   Processed {dataset_item_file_path.name} using full_average.")
                elif method == "points":
                    if not points: raise ValueError(f"Points not specified for {dataset_item_file_path.name} with 'points' method.")
                    hsv_colors = self.dataset_item_processor.extract_hsv_from_points(dataset_item_file_path, points)
                    if debug_mode: print(f"[DEBUG]   Processed {dataset_item_file_path.name} using points method with {len(points)} points.")
                else:
                    if debug_mode: print(f"[DEBUG]   Unknown method '{method}' for {dataset_item_file_path.name}. Skipping.")
                    continue

                all_hsv_colors.append(hsv_colors)

            except Exception as e:
                if debug_mode: print(f"[DEBUG] Warning: Error processing dataset image {dataset_item_file_path.name}: {e}. Skipping.")
                continue

        if not all_hsv_colors:
            raise ValueError(f"No valid dataset images processed or no non-transparent pixels in provided paths.")

        return np.vstack(all_hsv_colors)

    def calculate_hsv_range_from_dataset(self, dataset_image_configs: List[Dict], debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Calculates the HSV color range (lower, upper, center) from a list of dataset image configurations.
        This method computes a bounding box around the average colors of the dataset images or points.

        Args:
            dataset_image_configs (List[Dict]): A list of dictionaries, each containing 'path', 'method', and 'points' (if applicable).
            debug_mode (bool): If True, prints debug information.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]: A tuple containing the lower HSV limit,
                                                                  upper HSV limit, the center HSV color,
                                                                  and a list of dictionaries with debug info for each sample.
        """
        all_hsv_colors = []
        dataset_debug_info = []

        if not dataset_image_configs:
            raise ValueError("No dataset image configurations provided to calculate HSV range.")

        if debug_mode: print(f"[DEBUG] Calculating HSV range from {len(dataset_image_configs)} sample image configurations.")

        for img_config in dataset_image_configs:
            dataset_item_file_path = img_config['path']
            method = img_config['method']
            points = img_config.get('points')
            
            try:
                if method == "full_average":
                    avg_h, avg_s, avg_v = self.dataset_item_processor.calculate_hsv_from_full_image(dataset_item_file_path)
                    all_hsv_colors.append((avg_h, avg_s, avg_v))
                    avg_hsv_for_debug = np.array([[[avg_h, avg_s, avg_v]]], dtype=np.uint8)
                    avg_bgr = cv2.cvtColor(avg_hsv_for_debug, cv2.COLOR_HSV2BGR)[0][0]
                    dataset_debug_info.append({
                        'path': str(dataset_item_file_path),
                        'method': method,
                        'points': None,
                        'avg_color_bgr': avg_bgr.tolist(),
                        'hsv_colors': [(avg_h, avg_s, avg_v)]
                    })
                    if debug_mode: print(f"[DEBUG]   Processed {dataset_item_file_path.name} using full_average.")
                
                elif method == "points":
                    if not points: raise ValueError(f"Points not specified for {dataset_item_file_path.name} with 'points' method.")
                    
                    point_colors_hsv = self.dataset_item_processor.calculate_hsv_from_points(dataset_item_file_path, points)
                    all_hsv_colors.extend(point_colors_hsv)
                    
                    # For debug, we'll show the average of the points for that image
                    avg_h = np.mean([c[0] for c in point_colors_hsv])
                    avg_s = np.mean([c[1] for c in point_colors_hsv])
                    avg_v = np.mean([c[2] for c in point_colors_hsv])
                    avg_hsv_for_debug = np.array([[[avg_h, avg_s, avg_v]]], dtype=np.uint8)
                    avg_bgr = cv2.cvtColor(avg_hsv_for_debug, cv2.COLOR_HSV2BGR)[0][0]
                    dataset_debug_info.append({
                        'path': str(dataset_item_file_path),
                        'method': method,
                        'points': points,
                        'avg_color_bgr': avg_bgr.tolist(),
                        'hsv_colors': point_colors_hsv
                    })
                    if debug_mode: print(f"[DEBUG]   Processed {dataset_item_file_path.name} using points method with {len(points)} points.")

            except Exception as e:
                if debug_mode: print(f"[DEBUG] Warning: Error processing dataset image {dataset_item_file_path.name}: {e}. Skipping.")
                continue

        if not all_hsv_colors:
            raise ValueError("Could not extract any HSV colors from the provided sample images.")

        h_values = np.array([c[0] for c in all_hsv_colors])
        s_values = np.array([c[1] for c in all_hsv_colors])
        v_values = np.array([c[2] for c in all_hsv_colors])

        def get_robust_range(values):
            if len(values) == 0:
                return 0, 0

            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
            
            if len(filtered_values) == 0:
                final_lower = np.min(values)
                final_upper = np.max(values)
            else:
                final_lower = np.min(filtered_values)
                final_upper = np.max(filtered_values)

            return int(final_lower), int(final_upper)

        lower_h, upper_h = get_robust_range(h_values)
        lower_s, upper_s = get_robust_range(s_values)
        lower_v, upper_v = get_robust_range(v_values)
        
        center_h, center_s, center_v = np.mean(h_values), np.mean(s_values), np.mean(v_values)

        lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        center_color = np.array([center_h, center_s, center_v], dtype=np.uint8)

        if debug_mode:
            print(f"[DEBUG] Statistically Calculated HSV Range: Lower={lower_limit}, Upper={upper_limit}, Center={center_color}")

        return lower_limit, upper_limit, center_color, dataset_debug_info

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
        for img_config in current_file_paths_dict['dataset_image_configs']:
            current_source_files.add(str(img_config['path']))
            # Also add the sample_processing_config.json itself to the watched files
        current_source_files.add(str(self.projects_root / project_name / "project_config.json"))
        current_source_files.add(str(self.projects_root / project_name / "dataset_item_processing_config.json"))

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
                # dataset_debug_info is already a list of dicts, so no conversion is needed

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
        lower_hsv, upper_hsv, center_hsv, dataset_debug_info = self.calculate_hsv_range_from_dataset(file_paths['training_image_configs'], debug_mode=debug_mode)
        if debug_mode: print("[DEBUG] Project HSV range calculated.")

        # Store in cache
        source_file_timestamps = {}
        for p in file_paths['colorchecker_reference_for_project']:
            source_file_timestamps[str(p)] = p.stat().st_mtime
        source_file_timestamps[str(file_paths['reference_color_checker'])] = file_paths['reference_color_checker'].stat().st_mtime
        for img_config in file_paths['dataset_image_configs']:
            source_file_timestamps[str(img_config['path'])] = img_config['path'].stat().st_mtime
        source_file_timestamps[str(self.projects_root / project_name / "project_config.json")] = (self.projects_root / project_name / "project_config.json").stat().st_mtime
        source_file_timestamps[str(self.projects_root / project_name / "dataset_item_processing_config.json")] = (self.projects_root / project_name / "dataset_item_processing_config.json").stat().st_mtime

        cached_data_to_save = {
            'correction_matrix': correction_matrix.tolist(), # Convert NumPy array to list for JSON serialization
            'lower_hsv': lower_hsv.tolist(),
            'upper_hsv': upper_hsv.tolist(),
            'center_hsv': center_hsv.tolist(),
            'dataset_debug_info': dataset_debug_info,
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
            'dataset_debug_info': dataset_debug_info,
        }
