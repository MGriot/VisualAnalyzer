# Visual Analyzer

## Project Description

Visual Analyzer is a powerful Python-based tool for advanced image and video analysis. It provides a comprehensive suite of features for color correction, color zone analysis, image alignment, and symmetry analysis. The tool is designed to be project-based, allowing users to manage different analysis tasks with specific configurations and reference materials. The application is accessible through a Tkinter-based Graphical User Interface (GUI) and a Command-Line Interface (CLI).

## Core Concepts & Theory

For a deeper, university-level explanation of the core algorithms used in this project, please see the documents below:

-   **[Theory: Color Correction](./docs/THEORY_ColorCorrection.md)**: A detailed guide on the tiered patch detection system and the robust, cost-matrix-based patch matching algorithm.
-   **[Theory: Image Alignment](./docs/THEORY_Alignment.md)**: An in-depth look at the two-stage alignment process, covering both the projective geometry of ArUco markers and the contour-based methods for object alignment.
-   **[Theory: Symmetry Analysis](./docs/THEORY_Symmetry.md)**: An explanation of the geometric principles behind the reflection, rotation, translation, and glide-reflection symmetry analyses.

## Features

*   **Project-Based Management:** Organize your work into projects, each with its own configuration. A helper script scaffolds new projects, automatically generating a standard reference color checker with ArUco markers.
*   **Advanced Color Correction:** Corrects color using various algorithms (Linear, Polynomial, HSV, Histogram). This process is highly robust:
    1.  It uses a tiered fallback system to detect color checker patches: **ArUco Alignment** → **Robust OpenCV (Hough Transform)** → **YOLOv8** → **Simple Grid Slicing**.
    2.  It then uses a sophisticated matching algorithm (Hungarian algorithm with CIELAB color difference) to intelligently pair detected patches with reference patches, making the system resilient to missing patches, false detections, or incorrect ordering.
*   **Two-Stage Alignment:**
    *   **Geometrical Alignment:** Corrects perspective distortion using ArUco markers.
    *   **Object Alignment:** Aligns the primary object in an image to a reference template using a new geometric fitting algorithm. It prioritizes matching a 5-point pentagon circumscribed around the object and falls back to a 4-point bounding box for robust alignment. It also includes shadow removal techniques to improve contour detection.
*   **Multi-Layer Background Removal:** Programmatically remove the background from an image by using up to three layers of technical drawings as masks.
*   **Symmetry Analysis:** Analyze the symmetry of an object's binary mask, providing quantitative scores for vertical/horizontal reflection, 90/180-degree rotation, and more.
*   **Flexible Input:** Analyze single images, directories of images, video files, or live camera streams.
*   **Interactive Project Setup:** A Tkinter-based GUI is provided to interactively:
    *   Define specific points on training images for color range calculation.
    *   Easily place and rename reference files into their correct project folders, with instant validation for the reference color checker.
*   **Tkinter GUI & CLI:** A comprehensive Tkinter GUI allows for easy configuration of all analysis parameters. A full-featured Command-Line Interface (CLI) is also available for scripting and automation.
*   **Archivable PDF Reporting:** Generate detailed PDF reports using ReportLab, summarizing the analysis results with statistics and visualizations. The entire pipeline state can be archived for later regeneration of reports.
*   **Debug Mode:** A debug mode is available for verbose console output and detailed debug reports with intermediate pipeline steps and data.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/VisualAnalyzer.git
    cd VisualAnalyzer
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\.venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Configuration

The Visual Analyzer uses a project-based configuration system. Each project is a directory in the `data/projects` folder. Use the GUI (`python src/gui.py`) to scaffold a new project via the "Create Project" tab.

### `project_config.json`

This file defines the core settings for your project. When a project is created, it is pre-filled with default paths. Paths are relative to the project's root directory.

```json
{
    "training_path": "dataset/training_images",
    "object_reference_path": "dataset/object_reference.png",
    "color_correction": {
        "reference_color_checker_path": "dataset/default_color_checker_reference.png",
        "project_specific_color_checker_path": "dataset/project_color_checker.png"
    },
    "geometrical_alignment": {
        "reference_path": "dataset/default_geometric_align_reference.png",
        "marker_map": {},
        "output_size": [
            1000,
            1000
        ]
    },
    "masking": {
        "drawing_layers": {
            "1": "dataset/drawing_layers/layer1.png",
            "2": "dataset/drawing_layers/layer2.png",
            "3": "dataset/drawing_layers/layer3.png"
        }
    }
}
```

*   `training_path`: Path to the directory of images for calculating the target color range.
*   `object_reference_path`: Path to the reference image for object alignment.
*   `color_correction`: Object containing paths for color correction.
    *   `reference_color_checker_path`: Path to the ideal, canonical color checker image (auto-generated).
    *   `project_specific_color_checker_path`: Path to a photo of the color checker taken in the project's specific lighting. **You must provide this file.**
*   `geometrical_alignment`: Object for ArUco-based alignment settings.
*   `masking`: Object containing paths to drawing layers for masking.

### `dataset_item_processing_config.json`

This file defines how sample images in the `training_path` are processed for color range calculation. You can edit this manually or use the interactive GUI (`python src/gui.py` -> "Manage Dataset" tab).

```json
{
    "image_configs": [
        {
            "filename": "sample1.png",
            "method": "full_average"
        },
        {
            "filename": "sample2.png",
            "method": "points",
            "points": [
                {"x": 100, "y": 150, "radius": 7}
            ]
        }
    ]
}
```

## Usage

### Launching the Application

The application can be launched either via a Graphical User Interface (GUI) or through the Command-Line Interface (CLI).

*   **GUI:** To start the GUI, run `src/gui.py`.
    ```bash
    python src/gui.py
    ```
    The GUI provides an intuitive way to create projects, manage reference files, and configure analysis options.

*   **CLI:** To run the analysis directly from the command line, provide the necessary arguments to `main.py`:
    ```bash
    python src/main.py --project <project_name> --image <path_to_image> [options]
    ```

### Command-Line Options:

*   `--project <project_name>`: **(Required)** The name of the project to use.
*   `--image <path>`: Path to a single image file or a directory of images for analysis.
*   `--video <path>`: Path to a video file for analysis.
*   `--camera`: Use a live camera stream for analysis.
*   `--debug`: Enable debug mode for verbose output and detailed reports.
*   `--color-alignment`: Enable color correction. 
*   `--sample-color-checker <path>`: Path to a color checker image taken with the sample. If provided, it is used for on-the-fly color correction for this specific run.
*   `--color-correction-method <method>`: Specify the algorithm for color correction. Choices: `linear`, `polynomial`, `hsv`, `histogram`. Default is `linear`.
*   `--alignment`: Enable geometrical (ArUco) alignment.
*   `--object-alignment`: Enable object alignment.
*   `--object-alignment-shadow-removal <method>`: Shadow removal method for object alignment. Choices: `clahe` (default), `gamma`, `none`.
*   `--apply-mask`: Enable background removal using technical drawing(s).
*   `--mask-bg-is-white`: When using `--apply-mask`, treat white pixels in the drawing as background.
*   `--masking-order <order>`: Specify the order of masking layers (e.g., '1-2-3', '3-1-2'). Default is '1-2-3'.
*   `--symmetry`: Enable symmetry analysis.
*   `--blur`: Enable blurring of the input image.
*   `--blur-kernel <W> <H>`: Specify a custom kernel size (width height) for blurring. Both values must be odd.
*   `--aggregate`: Enable aggregation of matched pixel areas.
*   `--agg-kernel-size <size>`: Kernel size for the aggregation dilation step (default: 7).
*   `--agg-min-area <ratio>`: Minimum area ratio for keeping a component during aggregation (default: 0.0005).
*   `--agg-density-thresh <threshold>`: Minimum density of original pixels for an aggregated area to be kept (default: 0.5).
*   `--skip-color-analysis`: Skip the color analysis step.
*   `--skip-report-generation`: Skip the final report generation and archiving step.
*   `--save-state-to <path>`: Path to save the entire pipeline state to a `.gri` file for later regeneration.
*   `--load-state-from <path>`: Path to load a previously saved pipeline state from a `.gri` file.
