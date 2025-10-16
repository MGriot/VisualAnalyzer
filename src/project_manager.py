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
        """
        self.projects_root = config.PROJECTS_DIR
        self.color_corrector = ColorCorrector()
        self.dataset_item_processor = DatasetItemProcessor()
        self.cache_dir = config.OUTPUT_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file_path(self, project_name: str) -> Path:
        """
        Constructs the file path for the cache file associated with a given project.
        """
        return self.cache_dir / f"{project_name}_cache.json"

    def list_projects(self) -> List[str]:
        """
        Lists the names of all available projects.
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
        self,
        project_name: str
    ) -> DatasetItemProcessingConfig:
        project_path = self.projects_root / project_name
        config_file_path = project_path / "dataset_item_processing_config.json"
        if not config_file_path.is_file():
            return DatasetItemProcessingConfig(image_configs=[])
        with open(config_file_path, "r") as f:
            config_data = json.load(f)
        try:
            obj = DatasetItemProcessingConfig(**config_data)
            return obj
        except ValidationError as e:
            raise ValueError(
                f"Invalid dataset item processing configuration for '{project_name}':\n{e}"
            )

    def get_project_file_paths(
        self,
        project_name: str,
        debug_mode: bool = False
    ) -> Dict[str, Path | List[Path] | List[Dict]]:
        project_path = self.projects_root / project_name
        if not project_path.is_dir():
            raise ValueError(f"Project '{project_name}' not found.")

        config_data = self._get_project_config(project_name)
        dataset_item_processing_config = self._get_dataset_item_processing_config(
            project_name
        )
        color_config = config_data.color_correction
        geo_config = config_data.geometrical_alignment
        mask_config = config_data.masking

        ref_color_checker_path = project_path / color_config.reference_color_checker_path
        proj_spec_checker_path = (
            project_path / color_config.project_specific_color_checker_path
            if color_config.project_specific_color_checker_path
            else None
        )
        object_ref_path = (
            project_path / config_data.object_reference_path
            if config_data.object_reference_path
            else None
        )
        aruco_ref_path = (
            project_path / geo_config.reference_path
            if geo_config.reference_path
            else None
        )
        drawing_paths = {
            key: project_path / path
            for key, path in mask_config.drawing_layers.items()
            if (project_path / path).is_file()
        }

        training_image_configs = []
        if config_data.training_path:
            training_path = project_path / config_data.training_path
            if training_path.is_dir():
                for item in training_path.iterdir():
                    if item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        img_config = next((
                            cfg for cfg in dataset_item_processing_config.image_configs if cfg.filename == item.name
                        ), None)
                        
                        method = img_config.method if img_config else "full_average"
                        points = img_config.points if img_config else None
                        points_as_dicts = [p.model_dump() for p in points] if points else None

                        training_image_configs.append({
                            "filename": item.name,
                            "path": item,
                            "method": method,
                            "points": points_as_dicts,
                        })

        return {
            "reference_color_checker": ref_color_checker_path,
            "project_specific_color_checker": proj_spec_checker_path,
            "training_image_configs": training_image_configs,
            "technical_drawing_paths": drawing_paths,
            "geometrical_alignment_config": geo_config,
            "geometrical_alignment_reference_path": aruco_ref_path,
            "object_reference_path": object_ref_path,
        }

    def calculate_hsv_range_from_dataset(
        self,
        dataset_image_configs: List[Dict],
        correction_matrix: np.ndarray = None,
        debug_mode: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        all_hsv_colors = []
        dataset_debug_info = []

        if not dataset_image_configs:
            raise ValueError("No dataset image configurations provided to calculate HSV range.")

        if debug_mode:
            print(f"[DEBUG] Calculating HSV range from {len(dataset_image_configs)} sample configs.")

        for img_config in dataset_image_configs:
            try:
                image_path = img_config["path"]
                image, _ = load_image(str(image_path))
                if image is None:
                    if debug_mode: print(f"[DEBUG] Skipping {image_path.name}, could not load.")
                    continue

                corrected_image = image
                if correction_matrix is not None:
                    corrected_image = self.color_corrector.apply_correction_model(
                        image, correction_matrix, method='linear'
                    )

                hsv_colors_for_sample = self.dataset_item_processor.extract_hsv_from_image(
                    corrected_image, img_config["method"], img_config.get("points")
                )
                all_hsv_colors.extend(hsv_colors_for_sample)

                if debug_mode:
                    avg_hsv_colors_for_report = self.dataset_item_processor.extract_average_hsv_from_image(
                        corrected_image, img_config["method"], img_config.get("points")
                    )
                    if avg_hsv_colors_for_report.size > 0:
                        avg_bgr_colors_for_report = [
                            cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0] for hsv in avg_hsv_colors_for_report
                        ]
                        dataset_debug_info.append({
                            "path": str(image_path),
                            "method": img_config["method"],
                            "points": img_config.get("points"),
                            "hsv_colors": [c.tolist() for c in avg_hsv_colors_for_report],
                            "bgr_colors": [c.tolist() for c in avg_bgr_colors_for_report],
                        })
            except Exception as e:
                if debug_mode:
                    print(f"[DEBUG] Warning: Error processing dataset image {img_config['path'].name}: {e}. Skipping.")
                continue

        if not all_hsv_colors:
            raise ValueError(
                "Could not extract any HSV colors from the provided sample images."
            )

        h_values = np.array([c[0] for c in all_hsv_colors])
        s_values = np.array([c[1] for c in all_hsv_colors])
        v_values = np.array([c[2] for c in all_hsv_colors])

        def get_robust_range(values):
            if len(values) == 0: return 0, 0
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
            return (int(np.min(filtered_values)), int(np.max(filtered_values))) if len(filtered_values) > 0 else (int(np.min(values)), int(np.max(values)))

        lower_h, upper_h = get_robust_range(h_values)
        lower_s, upper_s = get_robust_range(s_values)
        lower_v, upper_v = get_robust_range(v_values)
        center_h, center_s, center_v = (np.mean(h_values), np.mean(s_values), np.mean(v_values))
        lower_limit = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_limit = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
        center_color = np.array([center_h, center_s, center_v], dtype=np.uint8)

        return lower_limit, upper_limit, center_color, dataset_debug_info

    def get_project_data(
        self,
        project_name: str,
        debug_mode: bool = False
    ) -> Dict[str, any]:
        cache_file_path = self._get_cache_file_path(project_name)
        cached_data = None
        source_file_timestamps = {}

        current_file_paths_dict = self.get_project_file_paths(
            project_name, debug_mode=False # Turn off debug for this internal call
        )
        current_source_files = set()
        if current_file_paths_dict.get("reference_color_checker"):
            current_source_files.add(str(current_file_paths_dict["reference_color_checker"]))
        if current_file_paths_dict.get("project_specific_color_checker"):
            current_source_files.add(str(current_file_paths_dict["project_specific_color_checker"]))
        for img_config in current_file_paths_dict["training_image_configs"]:
            current_source_files.add(str(img_config["path"]))
        current_source_files.add(str(self.projects_root / project_name / "project_config.json"))
        current_source_files.add(str(self.projects_root / project_name / "dataset_item_processing_config.json"))

        if cache_file_path.exists():
            try:
                with open(cache_file_path, "r") as f:
                    loaded_cache = json.load(f)
                correction_matrix_list = loaded_cache["data"]["correction_matrix"]
                loaded_cache["data"]["correction_matrix"] = {'matrix': np.array(correction_matrix_list, dtype=np.float32)}
                loaded_cache["data"]["lower_hsv"] = np.array(loaded_cache["data"]["lower_hsv"], dtype=np.uint8)
                loaded_cache["data"]["upper_hsv"] = np.array(loaded_cache["data"]["upper_hsv"], dtype=np.uint8)
                loaded_cache["data"]["center_hsv"] = np.array(loaded_cache["data"]["center_hsv"], dtype=np.uint8)
                cached_data = loaded_cache["data"]
                source_file_timestamps = loaded_cache["source_file_timestamps"]

                is_cache_valid = True
                cached_source_files = set(source_file_timestamps.keys())
                if current_source_files != cached_source_files:
                    is_cache_valid = False
                    if debug_mode: print("[DEBUG] Cache invalidated: Source file list changed.")
                else:
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
            except Exception as e:
                if debug_mode: print(f"[DEBUG] Error loading or validating cache for project '{project_name}': {e}. Recalculating...")
                cached_data = None

        if debug_mode:
            print(f"[DEBUG] Calculating data for project '{project_name}'...")
        file_paths = self.get_project_file_paths(project_name, debug_mode=debug_mode)

        correction_model = {'matrix': np.eye(3, dtype=np.float32)}
        if file_paths["project_specific_color_checker"] and file_paths["reference_color_checker"]:
            try:
                result = self.color_corrector.calculate_correction_from_images(
                    source_image_path=str(file_paths["project_specific_color_checker"]),
                    reference_image_path=str(file_paths["reference_color_checker"]),
                    debug_mode=debug_mode,
                )
                correction_model = result["correction_model"]
                if debug_mode: print("[DEBUG] Project color alignment matrix calculated.")
            except Exception as e:
                if debug_mode: print(f"[DEBUG] Warning: Could not calculate project color alignment matrix: {e}. Using identity matrix.")

        lower_hsv, upper_hsv, center_hsv, dataset_debug_info = (
            self.calculate_hsv_range_from_dataset(
                file_paths["training_image_configs"], 
                correction_matrix=correction_model, 
                debug_mode=debug_mode
            )
        )
        if debug_mode: print("[DEBUG] Project HSV range calculated.")

        source_file_timestamps = {}
        for p_str in current_source_files:
            p = Path(p_str)
            if p.exists():
                source_file_timestamps[str(p)] = p.stat().st_mtime

        cached_data_to_save = {
            "correction_matrix": correction_model['matrix'].tolist(),
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
        if debug_mode: print(f"[DEBUG] Cached data saved to {cache_file_path}")

        return {
            "correction_matrix": correction_model,
            "lower_hsv": lower_hsv,
            "upper_hsv": upper_hsv,
            "center_hsv": center_hsv,
            "dataset_debug_info": dataset_debug_info,
        }