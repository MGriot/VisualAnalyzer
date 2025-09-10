from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Point(BaseModel):
    x: int
    y: int
    radius: int = 7

class ImageConfig(BaseModel):
    filename: str
    method: str
    points: Optional[List[Point]] = None

class DatasetItemProcessingConfig(BaseModel):
    image_configs: List[ImageConfig]

class ProjectConfig(BaseModel):
    reference_color_checker_path: str
    training_path: str
    colorchecker_reference_for_project: Optional[List[str]] = None
    technical_drawing_path: Optional[str] = None
    aruco_reference_path: Optional[str] = None
    aruco_marker_map: Optional[Dict[str, List[List[int]]]] = None
    aruco_output_size: Optional[List[int]] = None
