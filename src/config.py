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
LOGO_PATH = REPORT_ASSETS_DIR / "logo.png"

# --- YOLO Model ---
YOLO_MODEL_PATH = ROOT_DIR / "models" / "ColourChecker" / "ColourChecker.pt"

# --- Report Metadata ---
AUTHOR = "Griot Matteo"
DEPARTMENT = "Global Quality"
REPORT_TITLE = "Under Layer Report"
