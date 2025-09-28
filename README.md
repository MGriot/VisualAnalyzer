<<<<<<< Updated upstream
version https://git-lfs.github.com/spec/v1
oid sha256:a0e1ddf5204c92e2a0583128df57b2f3bb585ddd06547c77e8be16eefbe84de9
size 7713
=======
# Visual Analyzer

## Project Description

Visual Analyzer is a powerful Python-based tool for advanced image and video analysis. It provides a comprehensive suite of features for color correction, color zone analysis, image alignment, and symmetry analysis. The tool is designed to be project-based, allowing users to manage different analysis tasks with specific configurations and reference materials. The application is accessible through a Streamlit-based Graphical User Interface (GUI) and a Command-Line Interface (CLI).

## Features

*   **Project-Based Management:** Organize your work into projects, each with its own configuration, color references, and sample images. New projects can be scaffolded with a helper script.
*   **Advanced Color Correction:** Automatically calculates a color correction matrix using a reference color checker and a sample image of that checker in the target lighting conditions. This matrix can then be applied to any image.
*   **Color Zone Analysis:** Identify and quantify areas in an image that match a specific HSV color range. The color range is automatically and robustly calculated from a set of user-provided sample images using statistical methods.
*   **Multi-Method Image Alignment:**
    *   **Geometrical Alignment:** Corrects perspective distortion using ArUco markers, either from a reference image or a pre-defined marker map.
    *   **Object Alignment:** Aligns the primary object in an image to a reference/template. The default method intelligently fits a pentagon (5-point) or quadrilateral (4-point) to the object's contour for a robust alignment. Feature-based matching (ORB/SIFT) is also available.
*   **Multi-Layer Background Removal:** Programmatically remove the background from an image by using up to three layers of technical drawings as masks. The masking can be configured to treat white pixels as background in addition to transparent pixels.
*   **Symmetry Analysis:** Analyze the symmetry of an object's binary mask, providing quantitative scores for vertical/horizontal reflection, 90/180-degree rotation, and more.
*   **Flexible Input:** Analyze single images, directories of images, video files, or live camera streams.
*   **Interactive Sample Management:** A GUI is provided to interactively define specific points or areas on sample images to be used for color range calculation.
*   **Streamlit GUI & CLI:** A comprehensive Streamlit GUI allows for easy configuration of all analysis parameters. A full-featured Command-Line Interface (CLI) is also available for scripting and automation.
*   **Comprehensive & Archivable Reporting:** Generate detailed PDF and HTML reports summarizing the analysis results with statistics and visualizations. Reports and their assets can be archived for later viewing or regeneration.
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

The Visual Analyzer uses a project-based configuration system. Each project is a directory in the `data/projects` folder. Use `python src/create_project.py --name <your_project_name>` to scaffold a new project.

### `project_config.json`

This file defines the core settings for your project. Paths are relative to the project's root directory.

```json
{
    "reference_color_checker_path": "dataset/colorchecker/colorchecker.png",
    "training_path": "dataset/training",
    "colorchecker_reference_for_project": [
        "samples/test/colorchecker/sample_checker.png"
    ],
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

*   `reference_color_checker_path`: Path to the ideal, canonical color checker image file.
*   `training_path`: Path to the directory containing images for calculating the target color range.
*   `colorchecker_reference_for_project` (Optional): A list of paths to images of the color checker taken in the specific lighting conditions of your sample images. Used to calculate the color correction matrix.
*   `object_reference_path` (Optional): Path to the reference image file for object alignment.
*   `technical_drawing_path_layer_1` (Optional): Path to the first technical drawing file for background removal.
*   `technical_drawing_path_layer_2` (Optional): Path to the second technical drawing file.
*   `technical_drawing_path_layer_3` (Optional): Path to the third technical drawing file.
*   `aruco_reference_path` (Optional): Path to the ArUco reference sheet image file for geometrical alignment.
*   `aruco_marker_map` (Optional): A dictionary mapping ArUco marker IDs to their ideal corner coordinates for an alternative perspective correction method.
*   `aruco_output_size` (Optional): The output size `[width, height]` of the image after ArUco-based alignment.

### `dataset_item_processing_config.json`

This file defines how sample images in the `training_path` are processed for color range calculation. You can edit this manually or use the interactive GUI (`src/sample_manager/dataset_gui.py`).

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

*   **GUI:** To start the GUI, run `streamlit_app.py`.
    ```bash
    streamlit run streamlit_app.py
    ```
    The GUI provides an intuitive way to select projects, input files, and configure analysis options.

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
*   `--color-alignment`: Enable color correction. Requires `colorchecker_reference_for_project` to be set in the project config.
*   `--alignment`: Enable geometrical (ArUco) alignment.
*   `--object-alignment`: Enable object alignment.
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
*   `--report-type <type>`: Specify the type of PDF report to generate (`html`, `reportlab`, `all`). Default is `all`.
*   `--skip-color-analysis`: Skip the color analysis step.
*   `--skip-report-generation`: Skip the final report generation step.
*   `--save-state-to <path>`: Path to save the entire pipeline state to a `.gri` file for later regeneration.
*   `--load-state-from <path>`: Path to load a previously saved pipeline state from a `.gri` file.
>>>>>>> Stashed changes
