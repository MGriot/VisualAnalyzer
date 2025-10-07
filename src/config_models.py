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

class ColorCorrectionConfig(BaseModel):
    """Configuration for color correction."""
    reference_color_checker_path: str
    project_specific_color_checker_path: Optional[str] = None

class GeometricalAlignmentConfig(BaseModel):
    """Configuration for ArUco-based geometrical alignment."""
    reference_path: Optional[str] = None
    marker_map: Optional[Dict[str, List[List[int]]]] = Field(default_factory=dict)
    output_size: Optional[List[int]] = Field(default_factory=lambda: [1000, 1000])

class MaskingConfig(BaseModel):
    """Configuration for masking."""
    drawing_layers: Dict[str, str] = Field(default_factory=dict)

class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    """
    Defines the overall configuration for a Visual Analyzer project.
    """
    training_path: str
    object_reference_path: Optional[str] = None
    
    color_correction: ColorCorrectionConfig
    geometrical_alignment: GeometricalAlignmentConfig = Field(default_factory=GeometricalAlignmentConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)