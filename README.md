# Visual Analyzer

## Project Description

Visual Analyzer is a powerful Python-based tool for advanced image and video analysis. It provides a comprehensive suite of features for color correction, color zone analysis, image alignment, and symmetry analysis. The tool is designed to be project-based, allowing users to manage different analysis tasks with specific configurations and reference materials.

## Features

*   **Project-Based Management:** Organize your work into projects, each with its own configuration, color references, and sample images.
*   **Advanced Color Correction:** Automatically correct image colors using a color checker. The system detects the color checker using a YOLOv8 model with a fallback to an OpenCV-based method, calculates a color correction matrix, and applies it to the input image or video frames.
*   **Color Zone Analysis:** Identify and quantify areas in an image that match a specific HSV color range. The color range is automatically calculated from a set of user-provided sample images.
*   **Image Alignment:**
    *   **Perspective Correction:** Correct perspective distortion using ArUco markers.
    *   **Object Alignment:** Align objects with a technical drawing for precise analysis.
*   **Symmetry Analysis:** Analyze the symmetry of an object in an image, including vertical and horizontal reflection, four-quadrant symmetry, rotational symmetry, translational symmetry, and glide-reflection symmetry.
*   **Flexible Input:** Analyze single images, video files, or live camera streams.
*   **Interactive Sample Management:** A GUI is provided to interactively define how sample images are used for color range calculation (full image average or specific points).
*   **Comprehensive Reporting:** Generate detailed HTML and PDF reports summarizing the analysis results, including statistics, processed images, and visualizations of the color space.
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

The Visual Analyzer uses a project-based configuration system. Each project is a directory in the `data/projects` folder and should contain the following files:

### `project_config.json`

This file defines the core settings for your project.

```json
{
    "reference_color_checker_filename": "reference_color_checker.png",
    "colorchecker_reference_for_project": [
        "reference_color_checker.png"
    ],
    "technical_drawing_filename": "drawing.png",
    "aruco_marker_map": {
        "10": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "20": [[900, 0], [1000, 0], [1000, 100], [900, 100]],
        "30": [[900, 900], [1000, 900], [1000, 1000], [900, 1000]],
        "40": [[0, 900], [100, 900], [100, 1000], [0, 1000]]
    },
    "aruco_output_size": [1000, 1000]
}
```

*   `reference_color_checker_filename`: The filename of the ideal color checker image.
*   `colorchecker_reference_for_project`: A list of paths to images containing color checkers for calculating the color correction matrix.
*   `technical_drawing_filename` (Optional): The filename of the technical drawing for alignment.
*   `aruco_marker_map` (Optional): A dictionary mapping ArUco marker IDs to their ideal corner coordinates for perspective correction.
*   `aruco_output_size` (Optional): The output size of the image after ArUco-based alignment.

### `sample_processing_config.json`

This file defines how sample images are processed for color range calculation.

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
                {"x": 100, "y": 150, "radius": 7},
                {"x": 200, "y": 250, "radius": 7}
            ]
        }
    ]
}
```

*   `image_configs`: A list of configurations for each sample image.
    *   `filename`: The filename of the sample image.
    *   `method`: The method for extracting color from the image (`full_average` or `points`).
    *   `points` (Required for `points` method): A list of points to sample color from.

## Usage

### Sample Management

To configure how sample images are processed, use the `sample_manager_main.py` script:

```bash
python src/sample_manager_main.py --project <project_name>
```

This will open a GUI to interactively select points on the sample images.

### Running the Analysis

To run the analysis, use the `main.py` script:

```bash
python src/main.py --project <project_name> --image <path_to_image> [options]
```

**Command-Line Options:**

*   `--project <project_name>`: The name of the project to use.
*   `--image <path>`: Path to a single image file for analysis.
*   `--video <path>`: Path to a video file for analysis.
*   `--camera`: Use a live camera stream for analysis.
*   `--debug`: Enable debug mode.
*   `--aggregate`: Enable aggregation of matched pixel areas.
*   `--blur`: Enable blurring of the input image.
*   `--alignment`: Enable geometrical alignment.
*   `--drawing <path>`: Path to a technical drawing for masking.
*   `--color-alignment`: Enable color correction.
*   `--symmetry`: Enable symmetry analysis.

### GUI

A graphical user interface is also available to run the analysis. To start the GUI, run:

```bash
python src/gui_main.py
```

## Future Improvements

*   **Pydantic Configuration:** Use Pydantic for configuration validation and management.
*   **Modular Codebase:** Refactor the `main.py` file to be more modular and easier to maintain.
*   **Improved Error Handling:** Implement more robust error handling and provide more informative error messages.
*   **Enhanced GUI:** Improve the main GUI to be more interactive and provide real-time feedback.
*   **Integrated Symmetry Analysis:** Integrate the symmetry analysis results into the main report.
*   **Batch Processing:** Implement batch processing of images and videos.