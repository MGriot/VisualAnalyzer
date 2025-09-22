"""
This module defines Pydantic models for validating and structuring configuration data
used throughout the Visual Analyzer application.

These models ensure that configuration files (like `project_config.json` and
`dataset_item_processing_config.json`) adhere to a defined schema, providing
robustness and type safety.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional

class Point(BaseModel):
    """
    Represents a point with x, y coordinates and an optional radius.
    Used for defining regions of interest in images.
    """
    x: int
    y: int
    radius: int = 7

class ImageConfig(BaseModel):
    """
    Defines how a specific image in a dataset should be processed.

    Attributes:
        filename (str): The name of the image file.
        method (str): The processing method to apply (e.g., "full_average", "points").
        points (Optional[List[Point]]): A list of `Point` objects if the method requires specific points.
    """
    filename: str
    method: str
    points: Optional[List[Point]] = None

class DatasetItemProcessingConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    """
    Represents the configuration for processing multiple images within a dataset.

    Attributes:
        image_configs (List[ImageConfig]): A list of `ImageConfig` objects, each defining
                                           how a particular image should be processed.
    """
    image_configs: List[ImageConfig]

class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    """
    Defines the overall configuration for a Visual Analyzer project.

    Attributes:
        reference_color_checker_path (str): Path to the ideal color checker image.
        training_path: str
        colorchecker_reference_for_project: Optional[List[str]] = None
        object_reference_path: Optional[str] = None
        technical_drawing_path_layer_1: Optional[str] = None
        technical_drawing_path_layer_2: Optional[str] = None
        technical_drawing_path_layer_3: Optional[str] = None
        aruco_reference_path: Optional[str] = None
        aruco_marker_map: Optional[Dict[str, List[List[int]]]] = Field(default_factory=dict)
        aruco_output_size: Optional[List[int]] = Field(default_factory=lambda: [1000, 1000])
    """
    reference_color_checker_path: str
    training_path: str
    colorchecker_reference_for_project: Optional[List[str]] = None
    object_reference_path: Optional[str] = None
    technical_drawing_path_layer_1: Optional[str] = None
    technical_drawing_path_layer_2: Optional[str] = None
    technical_drawing_path_layer_3: Optional[str] = None
    aruco_reference_path: Optional[str] = None
    aruco_marker_map: Optional[Dict[str, List[List[int]]]] = Field(default_factory=dict)
    aruco_output_size: Optional[List[int]] = Field(default_factory=lambda: [1000, 1000])