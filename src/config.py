<<<<<<< Updated upstream
version https://git-lfs.github.com/spec/v1
oid sha256:1d916fed382b4c2860a7af12c49fed09abe393845aaaf3d077aebaebeb1b3328
size 989
=======
"""
This module defines global configuration variables and paths for the Visual Analyzer application.

It centralizes important directory paths, model locations, and report metadata,
making it easier to manage and update project-wide settings.
"""

import os
from pathlib import Path

# --- Project Root ---
ROOT_DIR = Path(__file__).parent.parent

# --- Data Directories ---
DATA_DIR = ROOT_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
REFERENCE_COLOR_CHECKERS_DIR = DATA_DIR / "reference_color_checkers"

# --- Source Directories ---
SRC_DIR = ROOT_DIR / "src"
TEMPLATES_DIR = SRC_DIR / "templates"

# --- Output Directory ---
OUTPUT_DIR = ROOT_DIR / "output"

# --- Report Configuration ---
REPORT_ASSETS_DIR = OUTPUT_DIR / "assets"
LOGO_PATH = TEMPLATES_DIR / "logo.png"

# --- YOLO Model ---
YOLO_MODEL_PATH = ROOT_DIR / "models" / "ColourChecker" / "ColourChecker.pt"

# --- Report Metadata ---
AUTHOR = "Griot Matteo"
DEPARTMENT = "Department"
REPORT_TITLE = "Test image type"
>>>>>>> Stashed changes
