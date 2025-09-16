# Visual Analyzer

## Project Description

Visual Analyzer is a powerful Python-based tool for advanced image and video analysis. It provides a comprehensive suite of features for color correction, color zone analysis, image alignment, and symmetry analysis. The tool is designed to be project-based, allowing users to manage different analysis tasks with specific configurations and reference materials.

## Features

*   **Project-Based Management:** Organize your work into projects, each with its own configuration, color references, and sample images. New projects can be scaffolded with a helper script, which now creates dedicated folders for drawings and object references.
*   **Advanced Color Correction:** Automatically correct image colors using a color checker. The system detects the color checker, calculates a color correction matrix, and applies it to the input image or video frames.
*   **Color Zone Analysis:** Identify and quantify areas in an image that match a specific HSV color range. The color range is automatically calculated from a set of user-provided sample images.
*   **Two-Step Image Alignment:**
    *   **Geometrical Alignment:** Corrects perspective distortion using ArUco markers.
    *   **Object Alignment:** Aligns the primary object in the image to a reference/template image using feature matching.
*   **Background Removal:** Programmatically remove the background from an image by using a technical drawing as a mask. The masking can be configured to treat white pixels as background in addition to transparent pixels.
*   **Symmetry Analysis:** Analyze the symmetry of an object in an image, including vertical and horizontal reflection, four-quadrant symmetry, and more.
*   **Flexible Input:** Analyze single images, video files, or live camera streams.
*   **Interactive Sample Management:** A GUI is provided to interactively define how sample images are used for color range calculation.
*   **Enhanced GUI:** The main application now features a comprehensive GUI for easy configuration of all analysis parameters, including project selection, file inputs, and advanced options.
*   **Comprehensive Reporting:** Generate detailed PDF reports summarizing the analysis results, including statistics, processed images, and visualizations.
*   **Debug Mode:** A debug mode is available for verbose console output and detailed debug reports with intermediate steps and data.

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

The Visual Analyzer uses a project-based configuration system. Each project is a directory in the `data/projects` folder. Use `python src/create_project.py --name <your_project_name>` to scaffold a new project.

### `project_config.json`

This file defines the core settings for your project. Paths are now direct file paths relative to the project root.

```json
{
    "reference_color_checker_path": "dataset/colorchecker/colorchecker.png",
    "training_path": "dataset/training",
    "object_reference_path": "dataset/object/object.png",
    "technical_drawing_path_layer_1": "dataset/drawing/image.png",
    "technical_drawing_path_layer_2": "dataset/drawing/image1.png",
    "technical_drawing_path_layer_3": "dataset/drawing/image2.png",
    "aruco_reference_path": "dataset/aruco/aruco.png",
    "aruco_marker_map": {},
    "aruco_output_size": [
        1000,
        1000
    ]
}
```

*   `reference_color_checker_path`: Path to the ideal color checker image file.
*   `training_path`: Path to the directory containing images for calculating the color range.
*   `object_reference_path` (Optional): Path to the reference image file for object alignment.
*   `technical_drawing_path_layer_1` (Optional): Path to the first technical drawing file for background removal.
*   `technical_drawing_path_layer_2` (Optional): Path to the second technical drawing file for background removal.
*   `technical_drawing_path_layer_3` (Optional): Path to the third technical drawing file for background removal.
*   `aruco_reference_path` (Optional): Path to the ArUco reference sheet image file.
*   `aruco_marker_map` (Optional): A dictionary mapping ArUco marker IDs to their ideal corner coordinates for perspective correction.
*   `aruco_output_size` (Optional): The output size of the image after ArUco-based alignment.

### `dataset_item_processing_config.json`

This file defines how sample images in the `training_path` are processed for color range calculation. You can edit this manually or use the GUI.

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

*   **GUI:** To start the GUI, run `gui.py`. You can optionally add `--debug` to show all advanced options.
    ```bash
    python src/gui.py
    ```
    or for debug layout:
    ```bash
    python src/gui.py --debug
    ```
    The GUI provides an intuitive way to select projects, input files, and configure analysis options.

*   **CLI:** To run the analysis directly from the command line, provide the necessary arguments to `main.py`:
    ```bash
    python src/main.py --project <project_name> --image <path_to_image> [options]
    ```

### Sample Management

To configure how sample images are processed, use the `dataset_manager_main.py` script:

```bash
python src/dataset_manager_main.py --project <project_name>
```

### Command-Line Options:

*   `--project <project_name>`: The name of the project to use.
*   `--image <path>`: Path to a single image file for analysis.
*   `--video <path>`: Path to a video file for analysis.
*   `--camera`: Use a live camera stream for analysis.
*   `--debug`: Enable debug mode.
*   `--aggregate`: Enable aggregation of matched pixel areas.
*   `--blur`: Enable blurring of the input image.
*   `--color-alignment`: Enable color correction.
*   `--symmetry`: Enable symmetry analysis.
*   `--alignment`: Enable geometrical (ArUco) alignment.
*   `--object-alignment`: Enable object alignment.
*   `--apply-mask`: Enable background removal using a drawing.
*   `--mask-bg-is-white`: When using `--apply-mask`, treat white pixels in the drawing as background.
*   `--masking-order <order>`: Specify the order of masking layers (e.g., '1-2-3', '3-1-2'). Default is '1-2-3'.
*   `--report-type <type>`: Specify the type of PDF report to generate ('html', 'reportlab', 'all'). Default is 'all'.
*   `--agg-kernel-size <size>`: Kernel size for the aggregation dilation step (default: 7).
*   `--agg-min-area <ratio>`: Minimum area ratio for keeping a component during aggregation (default: 0.0005).
*   `--agg-density-thresh <threshold>`: Minimum density of original pixels for an aggregated area to be kept (default: 0.5).
*   `--blur-kernel <W> <H>`: Specify a custom kernel size (width height) for blurring. Both values must be odd.
