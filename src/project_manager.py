<<<<<<< Updated upstream
version https://git-lfs.github.com/spec/v1
oid sha256:6dae941c26a662a4d6e6c2111d4f95963c3aa7de7974f36168e6a862147bb320
size 31348
=======
"""
This module defines the `ProjectManager` class, responsible for handling project-specific
configurations, file paths, and cached analysis data within the Visual Analyzer application.

It provides functionalities to list projects, retrieve project and dataset item processing
configurations, and calculate HSV color ranges and color correction matrices, with caching
mechanisms to optimize performance.
"""

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
    Manages Visual Analyzer projects, including listing available projects,
    providing paths to reference color checkers and dataset images, and
    calculating average HSV colors. It also handles caching of calculated
    color correction matrices and HSV ranges to improve performance.
    """

    def __init__(self):
        """
        Initializes the ProjectManager.

        Sets up paths to project roots, initializes `ColorCorrector` and
        `DatasetItemProcessor` instances, and creates a cache directory.
        """
        self.projects_root = config.PROJECTS_DIR
        self.color_corrector = ColorCorrector()
        self.dataset_item_processor = DatasetItemProcessor()
        self.cache_dir = config.OUTPUT_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file_path(self, project_name: str) -> Path:
        """
        Constructs the file path for the cache file associated with a given project.

        Args:
            project_name (str): The name of the project.

        Returns:
            Path: The absolute path to the project's cache file.
        """
        return self.cache_dir / f"{project_name}_cache.json"

    def list_projects(self) -> List[str]:
        """
        Lists the names of all available projects located in the configured projects root directory.

        Returns:
            List[str]: A list of strings, where each string is the name of a project.
                       Returns an empty list if the projects root directory does not exist
                       or contains no subdirectories.
        """
        if not self.projects_root.exists():
            return []
        return [d.name for d in self.projects_root.iterdir() if d.is_dir()]

    def _get_project_config(self, project_name: str) -> ProjectConfig:
        project_path = self.projects_root / project_name
        config_file_path = project_path / "project_config.json"
        if not config_file_path.is_file():
            raise FileNotFoundError(
                f"Configuration file 'project_config.json' not found for project '{project_name}'."
            )

        # More robust file reading
        config_data = {}
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    config_data = json.loads(content)
        except (IOError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Could not read or parse project_config.json for '{project_name}': {e}"
            )

        try:
            return ProjectConfig(**config_data)
        except ValidationError as e:
            raise ValueError(
                f"Invalid project configuration for '{project_name}':\n{e}"
            )

    def _get_dataset_item_processing_config(
        self, project_name: str
    ) -> DatasetItemProcessingConfig:
        """
        Reads and validates the `dataset_item_processing_config.json` file for a specified project.

        Args:
            project_name (str): The name of the project.

        Returns:
            DatasetItemProcessingConfig: An instance of `DatasetItemProcessingConfig` containing
                                         the validated configuration data. If the file is not found,
                                         an empty configuration is returned.

        Raises:
            ValueError: If the configuration data fails Pydantic validation.
        """
        project_path = self.projects_root / project_name
        config_file_path = project_path / "dataset_item_processing_config.json"
        if not config_file_path.is_file():
            return DatasetItemProcessingConfig(image_configs=[])

        with open(config_file_path, "r") as f:
            config_data = json.load(f)

        try:
            obj = DatasetItemProcessingConfig(**config_data)
            print(f"--- Debugging DatasetItemProcessingConfig ---")
            print(f"Type of returned object: {type(obj)}")
            print(f"Attributes of returned object: {dir(obj)}")
            print(f"-------------------------------------------")
            return obj
        except ValidationError as e:
            raise ValueError(
                f"Invalid dataset item processing configuration for '{project_name}':\n{e}"
            )

    def get_project_file_paths(
        self, project_name: str, debug_mode: bool = False
    ) -> Dict[str, Path | List[Path] | List[Dict]]:
        project_path = self.projects_root / project_name
        if not project_path.is_dir():
            raise ValueError(f"Project '{project_name}' not found.")

        config_data = self._get_project_config(project_name)
        dataset_item_processing_config = self._get_dataset_item_processing_config(
            project_name
        )

        ref_color_checker_rel_path = getattr(
            config_data, "reference_color_checker_path", None
        )
        colorchecker_ref_for_project_relative = getattr(
            config_data, "colorchecker_reference_for_project", []
        )
        technical_drawing_rel_path_layer_1 = getattr(
            config_data, "technical_drawing_path_layer_1", None
        )
        technical_drawing_rel_path_layer_2 = getattr(
            config_data, "technical_drawing_path_layer_2", None
        )
        technical_drawing_rel_path_layer_3 = getattr(
            config_data, "technical_drawing_path_layer_3", None
        )
        aruco_ref_rel_path = getattr(config_data, "aruco_reference_path", None)
        training_rel_path = getattr(config_data, "training_path", None)
        object_reference_rel_path = getattr(config_data, "object_reference_path", None)

        aruco_marker_map = getattr(config_data, "aruco_marker_map", {})
        aruco_output_size = getattr(config_data, "aruco_output_size", [1000, 1000])

        if not ref_color_checker_rel_path:
            if debug_mode:
                print(
                    f"[DEBUG] 'reference_color_checker_path' not specified in project_config.json for project '{project_name}'."
                )
            ref_color_checker_path = None
        else:
            ref_color_checker_path = project_path / ref_color_checker_rel_path
            if not ref_color_checker_path.is_file():
                if debug_mode:
                    print(
                        f"[DEBUG] Reference color checker file not found at '{ref_color_checker_path}'."
                    )
                ref_color_checker_path = None

        technical_drawing_path_layer_1 = (
            project_path / technical_drawing_rel_path_layer_1
            if technical_drawing_rel_path_layer_1
            else None
        )
        if (
            technical_drawing_path_layer_1
            and not technical_drawing_path_layer_1.is_file()
        ):
            if debug_mode:
                print(
                    f"[DEBUG] Warning: Technical drawing layer 1 file not found at '{technical_drawing_path_layer_1}'."
                )
            technical_drawing_path_layer_1 = None

        technical_drawing_path_layer_2 = (
            project_path / technical_drawing_rel_path_layer_2
            if technical_drawing_rel_path_layer_2
            else None
        )
        if (
            technical_drawing_path_layer_2
            and not technical_drawing_path_layer_2.is_file()
        ):
            if debug_mode:
                print(
                    f"[DEBUG] Warning: Technical drawing layer 2 file not found at '{technical_drawing_path_layer_2}'."
                )
            technical_drawing_path_layer_2 = None

        technical_drawing_path_layer_3 = (
            project_path / technical_drawing_rel_path_layer_3
            if technical_drawing_rel_path_layer_3
            else None
        )
        if (
            technical_drawing_path_layer_3
            and not technical_drawing_path_layer_3.is_file()
        ):
            if debug_mode:
                print(
                    f"[DEBUG] Warning: Technical drawing layer 3 file not found at '{technical_drawing_path_layer_3}'."
                )
            technical_drawing_path_layer_3 = None

        colorchecker_ref_for_project_paths = []
        if colorchecker_ref_for_project_relative:
            for rel_path in colorchecker_ref_for_project_relative:
                full_path = project_path / rel_path
                if full_path.is_file():
                    colorchecker_ref_for_project_paths.append(full_path)
                else:
                    if debug_mode:
                        print(
                            f"[DEBUG] Warning: Color checker reference image '{rel_path}' not found for project '{project_name}'. Skipping."
                        )

        aruco_ref_path = (
            project_path / aruco_ref_rel_path if aruco_ref_rel_path else None
        )
        if aruco_ref_path and not aruco_ref_path.is_file():
            if debug_mode:
                print(
                    f"[DEBUG] Warning: ArUco reference file not found at '{aruco_ref_path}'."
                )
            aruco_ref_path = None

        object_ref_path = (
            project_path / object_reference_rel_path
            if object_reference_rel_path
            else None
        )
        if object_ref_path and not object_ref_path.is_file():
            if debug_mode:
                print(
                    f"[DEBUG] Warning: Object reference file not found at '{object_ref_path}'."
                )
            object_ref_path = None

        training_image_configs = []
        if training_rel_path:
            training_path = project_path / training_rel_path
            if not training_path.is_dir():
                if debug_mode:
                    print(
                        f"[ERROR] The specified training path is not a valid directory: {training_path}"
                    )
            else:
                for item in training_path.iterdir():
                    if item.is_file() and item.suffix.lower() in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ]:
                        img_config = next(
                            (
                                cfg
                                for cfg in dataset_item_processing_config.image_configs
                                if cfg.filename == item.name
                            ),
                            None,
                        )
                        if img_config:
                            method = img_config.method
                            points = img_config.points
                            if method == "points" and not points:
                                method = "full_average"
                            points_as_dicts = (
                                [p.model_dump() for p in points] if points else None
                            )
                            training_image_configs.append(
                                {
                                    "filename": item.name,
                                    "path": item,
                                    "method": method,
                                    "points": points_as_dicts,
                                }
                            )
                        else:
                            training_image_configs.append(
                                {
                                    "filename": item.name,
                                    "path": item,
                                    "method": "full_average",
                                }
                            )

        dataset_image_configs = []
        dataset_path = project_path / "dataset"

        if debug_mode:
            print(f"[DEBUG] Discovering samples in {dataset_path}")

        if dataset_path.is_dir():
            for item in dataset_path.iterdir():
                if item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    img_config = next(
                        (
                            cfg
                            for cfg in dataset_item_processing_config.image_configs
                            if cfg.filename == item.name
                        ),
                        None,
                    )
                    if img_config:
                        method = img_config.method
                        points = img_config.points
                        if method == "points" and not points:
                            method = "full_average"
                        points_as_dicts = (
                            [p.model_dump() for p in points] if points else None
                        )
                        dataset_image_configs.append(
                            {
                                "filename": item.name,
                                "path": item,
                                "method": method,
                                "points": points_as_dicts,
                            }
                        )
                    else:
                        dataset_image_configs.append(
                            {
                                "filename": item.name,
                                "path": item,
                                "method": "full_average",
                            }
                        )

        if not dataset_image_configs:
            if debug_mode:
                print(
                    f"[DEBUG] Warning: No valid dataset images found for project '{project_name}'."
                )

        if debug_mode:
            print(f"[DEBUG] Project '{project_name}' paths:")
            print(f"[DEBUG]   Reference Color Checker: {ref_color_checker_path}")
            print(
                f"[DEBUG]   Color Checker Refs for Project: {colorchecker_ref_for_project_paths}"
            )
            print(f"[DEBUG]   Dataset Image Configurations: {dataset_image_configs}")
            if technical_drawing_path_layer_1:
                print(
                    f"[DEBUG]   Technical Drawing Layer 1: {technical_drawing_path_layer_1}"
                )
            if technical_drawing_path_layer_2:
                print(
                    f"[DEBUG]   Technical Drawing Layer 2: {technical_drawing_path_layer_2}"
                )
            if technical_drawing_path_layer_3:
                print(
                    f"[DEBUG]   Technical Drawing Layer 3: {technical_drawing_path_layer_3}"
                )
            if aruco_ref_path:
                print(f"[DEBUG]   ArUco Reference: {aruco_ref_path}")
            if object_ref_path:
                print(f"[DEBUG]   Object Reference: {object_ref_path}")

        return {
            "reference_color_checker": ref_color_checker_path,
            "colorchecker_reference_for_project": colorchecker_ref_for_project_paths,
            "dataset_image_configs": dataset_image_configs,
            "training_image_configs": training_image_configs,
            "technical_drawing_layer_1": technical_drawing_path_layer_1,
            "technical_drawing_layer_2": technical_drawing_path_layer_2,
            "technical_drawing_layer_3": technical_drawing_path_layer_3,
            "aruco_reference": aruco_ref_path,
            "object_reference_path": object_ref_path,
            "aruco_marker_map": aruco_marker_map,
            "aruco_output_size": aruco_output_size,
        }

    def get_hsv_colors_from_dataset(
        self, dataset_image_configs: List[Dict], debug_mode: bool = False
    ) -> np.ndarray:
        """
        Extracts all HSV color values from a list of dataset image configurations.

        This method iterates through the provided image configurations, applying either
        full image average extraction or point-based extraction as specified, and
        collects all extracted HSV colors into a single NumPy array.

        Args:
            dataset_image_configs (List[Dict]): A list of dictionaries, each containing
                                               'path' (Path to image), 'method' (e.g.,
                                               "full_average", "points"), and optionally
                                               'points' (List of point dictionaries).
            debug_mode (bool): If True, prints debug information.

        Returns:
            np.ndarray: A NumPy array of shape (N, 3) containing all extracted HSV colors,
                        where N is the total number of pixels/points extracted.

        Raises:
            ValueError: If no dataset image configurations are provided, or if no valid
                        HSV colors could be extracted from the configured images.
        """
        all_hsv_colors = []

        if not dataset_image_configs:
            raise ValueError(
                "No dataset image configurations provided to extract HSV colors."
            )

        if debug_mode:
            print(
                f"[DEBUG] Extracting HSV colors from {len(dataset_image_configs)} dataset image configurations."
            )

        for img_config in dataset_image_configs:
            dataset_item_file_path = img_config["path"]
            method = img_config["method"]
            points = img_config.get("points")

            try:
                if method == "full_average":
                    hsv_colors = (
                        self.dataset_item_processor.extract_hsv_from_full_image(
                            dataset_item_file_path
                        )
                    )
                    if debug_mode:
                        print(
                            f"[DEBUG]   Processed {dataset_item_file_path.name} using full_average."
                        )
                elif method == "points":
                    if not points:
                        raise ValueError(
                            f"Points not specified for {dataset_item_file_path.name} with 'points' method."
                        )
                    hsv_colors = self.dataset_item_processor.extract_hsv_from_points(
                        dataset_item_file_path, points
                    )
                    if debug_mode:
                        print(
                            f"[DEBUG]   Processed {dataset_item_file_path.name} using points method with {len(points)} points."
                        )
                else:
                    if debug_mode:
                        print(
                            f"[DEBUG]   Unknown method '{method}' for {dataset_item_file_path.name}. Skipping."
                        )
                    continue

                all_hsv_colors.append(hsv_colors)

            except Exception as e:
                if debug_mode:
                    print(
                        f"[DEBUG] Warning: Error processing dataset image {dataset_item_file_path.name}: {e}. Skipping."
                    )
                continue

        if not all_hsv_colors:
            raise ValueError(
                "No valid dataset images processed or no non-transparent pixels in provided paths."
            )

        return np.vstack(all_hsv_colors)

    def calculate_hsv_range_from_dataset(
        self, dataset_image_configs: List[Dict], debug_mode: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Calculates a robust HSV color range (lower bound, upper bound, and center color)
        from a collection of dataset image configurations.

        This method processes each image configuration to extract HSV colors, then uses
        statistical methods (interquartile range) to determine a robust bounding box
        for the HSV values, mitigating the effect of outliers.

        Args:
            dataset_image_configs (List[Dict]): A list of dictionaries, each containing
                                               'path' (Path to image), 'method' (e.g.,
                                               "full_average", "points"), and optionally
                                               'points' (List of point dictionaries).
            debug_mode (bool): If True, prints debug information and collects detailed
                               debug info for each processed sample.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]: A tuple containing:
                - lower_limit (np.ndarray): A NumPy array [H, S, V] representing the
                                            lower bounds of the calculated HSV range.
                - upper_limit (np.ndarray): A NumPy array [H, S, V] representing the
                                            upper bounds of the calculated HSV range.
                - center_color (np.ndarray): A NumPy array [H, S, V] representing the
                                             mean HSV color of the dataset.
                - dataset_debug_info (List[Dict]): A list of dictionaries, each containing
                                                   detailed information about how each
                                                   sample image was processed and its
                                                   extracted colors (useful for reporting).

        Raises:
            ValueError: If no dataset image configurations are provided, or if no valid
                        HSV colors could be extracted from the configured images.
        """
        all_hsv_colors = []
        dataset_debug_info = []

        if not dataset_image_configs:
            raise ValueError(
                "No dataset image configurations provided to calculate HSV range."
            )

        if debug_mode:
            print(
                f"[DEBUG] Calculating HSV range from {len(dataset_image_configs)} sample image configurations."
            )

        for img_config in dataset_image_configs:
            dataset_item_file_path = img_config["path"]
            method = img_config["method"]
            points = img_config.get("points")

            try:
                hsv_colors_for_sample = []
                bgr_colors_for_sample = []

                if method == "full_average":
                    avg_h, avg_s, avg_v = (
                        self.dataset_item_processor.calculate_hsv_from_full_image(
                            dataset_item_file_path
                        )
                    )
                    hsv_colors_for_sample.append((avg_h, avg_s, avg_v))
                    all_hsv_colors.append((avg_h, avg_s, avg_v))
                    if debug_mode:
                        print(
                            f"[DEBUG]   Processed {dataset_item_file_path.name} using full_average."
                        )

                elif method == "points":
                    if not points:
                        raise ValueError(
                            f"Points not specified for {dataset_item_file_path.name} with 'points' method."
                        )
                    point_colors_hsv = (
                        self.dataset_item_processor.calculate_hsv_from_points(
                            dataset_item_file_path, points
                        )
                    )
                    hsv_colors_for_sample.extend(point_colors_hsv)
                    all_hsv_colors.extend(point_colors_hsv)
                    if debug_mode:
                        print(
                            f"[DEBUG]   Processed {dataset_item_file_path.name} using points method with {len(points)} points."
                        )

                # Common logic for debug info
                for hsv_color in hsv_colors_for_sample:
                    hsv_for_debug = np.array([[list(hsv_color)]], dtype=np.uint8)
                    bgr_color = cv2.cvtColor(hsv_for_debug, cv2.COLOR_HSV2BGR)[0][0]
                    bgr_colors_for_sample.append(bgr_color.tolist())

                dataset_debug_info.append(
                    {
                        "path": str(dataset_item_file_path),
                        "method": method,
                        "points": points,
                        "bgr_colors": bgr_colors_for_sample,
                        "hsv_colors": hsv_colors_for_sample,
                    }
                )

            except Exception as e:
                if debug_mode:
                    print(
                        f"[DEBUG] Warning: Error processing dataset image {dataset_item_file_path.name}: {e}. Skipping."
                    )
                continue

        if not all_hsv_colors:
            raise ValueError(
                "Could not extract any HSV colors from the provided sample images."
            )

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

        center_h, center_s, center_v = (
            np.mean(h_values),
            np.mean(s_values),
            np.mean(v_values),
        )

        lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        center_color = np.array([center_h, center_s, center_v], dtype=np.uint8)

        if debug_mode:
            print(
                f"[DEBUG] Statistically Calculated HSV Range: Lower={lower_limit}, Upper={upper_limit}, Center={center_color}"
            )

        return lower_limit, upper_limit, center_color, dataset_debug_info

    def get_project_data(
        self, project_name: str, debug_mode: bool = False
    ) -> Dict[str, any]:
        """
        Retrieves or calculates the essential data for a project, including the color
        correction matrix and the HSV color range for analysis.

        This method implements a caching mechanism: if the data has been previously
        calculated and the source files (project config, dataset images, color checkers)
        have not changed, the cached data is returned. Otherwise, the data is recalculated
        and then cached.

        Args:
            project_name (str): The name of the project.
            debug_mode (bool): If True, enables verbose output for debugging and caching details.

        Returns:
            Dict[str, any]: A dictionary containing:
                - 'correction_matrix' (np.ndarray): A 3x3 color correction matrix.
                - 'lower_hsv' (np.ndarray): The lower bounds of the calculated HSV range.
                - 'upper_hsv' (np.ndarray): The upper bounds of the calculated HSV range.
                - 'center_hsv' (np.ndarray): The center HSV color of the dataset.
                - 'dataset_debug_info' (List[Dict]): Debug information about the dataset
                                                      used for HSV range calculation.
        """
        cache_file_path = self._get_cache_file_path(project_name)
        cached_data = None
        source_file_timestamps = {}

        # Get current file paths for comparison
        current_file_paths_dict = self.get_project_file_paths(
            project_name, debug_mode=debug_mode
        )
        current_source_files = set()
        current_source_files.add(
            str(current_file_paths_dict["reference_color_checker"])
        )
        for p in current_file_paths_dict["colorchecker_reference_for_project"]:
            current_source_files.add(str(p))
        for img_config in current_file_paths_dict["dataset_image_configs"]:
            current_source_files.add(str(img_config["path"]))
            # Also add the sample_processing_config.json itself to the watched files
        current_source_files.add(
            str(self.projects_root / project_name / "project_config.json")
        )
        current_source_files.add(
            str(
                self.projects_root
                / project_name
                / "dataset_item_processing_config.json"
            )
        )

        if cache_file_path.exists():
            try:
                with open(cache_file_path, "r") as f:
                    loaded_cache = json.load(f)

                # Deserialize NumPy arrays
                correction_matrix_list = loaded_cache["data"]["correction_matrix"]
                loaded_cache["data"]["correction_matrix"] = np.array(
                    correction_matrix_list, dtype=np.float32
                )
                loaded_cache["data"]["lower_hsv"] = np.array(
                    loaded_cache["data"]["lower_hsv"], dtype=np.uint8
                )
                loaded_cache["data"]["upper_hsv"] = np.array(
                    loaded_cache["data"]["upper_hsv"], dtype=np.uint8
                )
                loaded_cache["data"]["center_hsv"] = np.array(
                    loaded_cache["data"]["center_hsv"], dtype=np.uint8
                )
                # dataset_debug_info is already a list of dicts, so no conversion is needed

                cached_data = loaded_cache["data"]
                source_file_timestamps = loaded_cache["source_file_timestamps"]

                # Verify cache validity
                is_cache_valid = True

                # 1. Check if the set of source files has changed
                cached_source_files = set(source_file_timestamps.keys())
                if current_source_files != cached_source_files:
                    is_cache_valid = False
                    if debug_mode:
                        print(f"[DEBUG] Cache invalidated: Source file list changed.")
                else:
                    # 2. Check if any source files have been modified
                    for file_path_str, timestamp in source_file_timestamps.items():
                        current_mtime = Path(file_path_str).stat().st_mtime
                        if debug_mode:
                            print(
                                f"[DEBUG]   File: {file_path_str}, Cached mtime: {timestamp}, Current mtime: {current_mtime}"
                            )
                        if current_mtime > timestamp:
                            is_cache_valid = False
                            if debug_mode:
                                print(
                                    f"[DEBUG]   Cache invalidated for {file_path_str} (modified)."
                                )
                            break

                if is_cache_valid:
                    if debug_mode:
                        print(
                            f"[DEBUG] Using cached data for project '{project_name}'."
                        )
                    return cached_data
                else:
                    if debug_mode:
                        print(
                            f"[DEBUG] Cache for project '{project_name}' is outdated. Recalculating..."
                        )

            except (json.JSONDecodeError, KeyError, FileNotFoundError, ValueError) as e:
                if debug_mode:
                    print(
                        f"[DEBUG] Error loading or validating cache for project '{project_name}': {e}. Recalculating..."
                    )
                cached_data = None  # Force recalculation

        if debug_mode:
            print(f"[DEBUG] Calculating data for project '{project_name}'...")
        file_paths = self.get_project_file_paths(project_name, debug_mode=debug_mode)

        # Calculate correction matrix
        correction_matrix = np.eye(3, dtype=np.float32)  # Default to identity
        if file_paths["colorchecker_reference_for_project"]:
            # For simplicity, we'll use the first image in the list to calculate the matrix
            # A more robust solution might average matrices from multiple images or use a more complex algorithm
            source_image_path = file_paths["colorchecker_reference_for_project"][0]
            try:
                _, correction_matrix = self.color_corrector.correct_image_colors(
                    source_image_path=str(source_image_path),
                    reference_image_path=str(file_paths["reference_color_checker"]),
                    debug_mode=debug_mode,
                )
                if debug_mode:
                    print("[DEBUG] Project color alignment matrix calculated.")
            except Exception as e:
                if debug_mode:
                    print(
                        f"[DEBUG] Warning: Could not calculate project color alignment matrix: {e}. Using identity matrix."
                    )

        # Calculate HSV range
        lower_hsv, upper_hsv, center_hsv, dataset_debug_info = (
            self.calculate_hsv_range_from_dataset(
                file_paths["training_image_configs"], debug_mode=debug_mode
            )
        )
        if debug_mode:
            print("[DEBUG] Project HSV range calculated.")

        # Store in cache
        source_file_timestamps = {}
        for p in file_paths["colorchecker_reference_for_project"]:
            source_file_timestamps[str(p)] = p.stat().st_mtime
        source_file_timestamps[str(file_paths["reference_color_checker"])] = (
            file_paths["reference_color_checker"].stat().st_mtime
        )
        for img_config in file_paths["dataset_image_configs"]:
            source_file_timestamps[str(img_config["path"])] = (
                img_config["path"].stat().st_mtime
            )
        source_file_timestamps[
            str(self.projects_root / project_name / "project_config.json")
        ] = (
            (self.projects_root / project_name / "project_config.json").stat().st_mtime
        )
        source_file_timestamps[
            str(
                self.projects_root
                / project_name
                / "dataset_item_processing_config.json"
            )
        ] = (
            (self.projects_root / project_name / "dataset_item_processing_config.json")
            .stat()
            .st_mtime
        )

        cached_data_to_save = {
            "correction_matrix": correction_matrix.tolist(),  # Convert NumPy array to list for JSON serialization
            "lower_hsv": lower_hsv.tolist(),
            "upper_hsv": upper_hsv.tolist(),
            "center_hsv": center_hsv.tolist(),
            "dataset_debug_info": dataset_debug_info,
        }

        full_cache_entry = {
            "data": cached_data_to_save,
            "source_file_timestamps": source_file_timestamps,
        }

        with open(cache_file_path, "w") as f:
            json.dump(full_cache_entry, f, indent=4)
        if debug_mode:
            print(f"[DEBUG] Cached data saved to {cache_file_path}")

        return {
            "correction_matrix": correction_matrix,
            "lower_hsv": lower_hsv,
            "upper_hsv": upper_hsv,
            "center_hsv": center_hsv,
            "dataset_debug_info": dataset_debug_info,
        }
>>>>>>> Stashed changes
