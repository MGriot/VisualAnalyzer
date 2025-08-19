# Visual Analyzer

## Project Description

Visual Analyzer is a powerful tool designed for color correction and analysis of images and video streams. It allows users to define projects with specific color checker references and sample images to accurately analyze color zones, apply color corrections, and generate comprehensive reports.

## Features

*   **Project Management:** Organize your analysis with dedicated projects, each defined by a `project_config.json` file that specifies reference color checkers and sample images.
*   **Robust Color Correction:** Automatically corrects colors in input images/frames by comparing them to project-defined reference color checkers. Utilizes a YOLO-based detection with an OpenCV-based fallback for accurate color patch identification.
*   **Image Alignment:** Corrects perspective distortion using a chessboard pattern and aligns the object with a technical drawing for precise analysis.
*   **Persistent Caching:** Automatically caches calculated color alignment matrices and HSV color ranges for each project, significantly speeding up subsequent analyses by avoiding recalculation unless source files change.
*   **Color Zone Analysis:** Analyzes images and video frames to identify and quantify areas matching a user-defined color range in HSV color space. Handles transparent image formats by analyzing only solid pixels.
*   **Granular Sample Processing:** Allows defining how each sample image contributes to the color range calculation (full image average or specific points with a radius), configurable via a dedicated JSON file and an interactive GUI.
*   **Comprehensive Reporting:** Generates detailed HTML and PDF reports summarizing the analysis, including statistics, processed images, and visual representations of color spaces, with an improved layout.
*   **Flexible Input Options:** Supports analysis of single image files, video files, and live camera streams.
*   **Debug Mode:** Provides verbose output in the console and generates enhanced debug reports (HTML and PDF) with intermediate steps and data, including blurred images and pre-aggregation masks.
*   **Image Pre-processing (Blur):** Option to apply Gaussian blur to the input image before color matching, which can help in reducing noise and smoothing color transitions.

## Installation

To set up the Visual Analyzer, follow these steps:

1.  **Navigate to the project directory:**

    ```bash
    cd C:\Users\Admin\Documents\Coding\VisualAnalyzer
    ```

2.  **Create a new Python virtual environment:**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    ```bash
    .\.venv\Scripts\activate
    ```

4.  **Install Python dependencies:**

    ```bash
    pip install --upgrade pip
    pip install numpy opencv-python ultralytics jinja2 weasyprint Pillow
    ```

5.  **Install system-level dependencies for WeasyPrint (Windows):**
    WeasyPrint requires some external libraries (GTK+ components) to generate PDFs. If you encounter errors related to missing libraries (e.g., `libgobject-2.0-0`), you need to install the GTK+ for Windows Runtime Environment. Follow the instructions on the official WeasyPrint documentation:

    [WeasyPrint Installation Guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation)

    Ensure that the installed libraries are accessible to your system's PATH.

## Project Structure

```
VisualAnalyzer/
├── data/
│   ├── projects/
│   │   └── sample_project/
│   │       └── project_config.json
│   │       └── sample_processing_config.json (New: Defines how samples are processed)
│   │       └── reference_color_checker.png
│   │       └── samples/ (Contains sample images, dynamically discovered)
├── src/
│   ├── alignment/
│   │   └── aligner.py
│   ├── color_analysis/
│   │   ├── analyzer.py
│   │   └── project_manager.py
│   ├── color_correction/
│   │   ├── corrector.py
│   ├── reporting/
│   │   ├── generator.py
│   │   └── templates/
│   │       ├── Report_Default.html
│   │       └── Report_Debug.html
│   ├── sample_manager/ (New: Module for sample processing management)
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   └── gui.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── video_utils.py
│   ├── main.py
│   └── sample_manager_main.py (New: Main script for sample management)
├── models/
│   └── ColourChecker/
│       └── ColourChecker.pt
├── output/ (Generated reports and processed images)
│   └── cache/ (Cached project data)
├── .venv/ (Python Virtual Environment)
├── README.md
└── ... (Other project files)
```

## Usage

### 1. Project Setup

Before running the analysis, you need to set up your projects in the `data/projects/` directory. Each project should be a separate folder containing:

*   `project_config.json`: A JSON file defining the project's settings:
    ```json
    {
        "reference_color_checker_filename": "reference_color_checker.png",
        "colorchecker_reference_for_project": [
            "reference_color_checker.png" 
        ],
        "technical_drawing_filename": "drawing.png"
    }
    ```
    *   `reference_color_checker_filename`: The filename of the ideal color checker image within this project's directory, used as the target for color alignment.
    *   `colorchecker_reference_for_project`: A list of paths (relative to the project folder) to images containing color checkers. These images are used to calculate the color alignment matrix for the project. The matrix is calculated once and cached.
    *   `technical_drawing_filename` (Optional): The filename of the technical drawing to be used for alignment. This should be a black image with the object's profile in white.

*   `sample_processing_config.json` (Optional): A JSON file that specifies how individual sample images should be processed for color range calculation. If this file doesn't exist for an image, it defaults to `full_average`. Example structure:
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

*   **Sample Images:** All other image files (e.g., `.png`, `.jpg`, `.jpeg`) directly within the project folder or its `samples/` subdirectory (excluding those specified in `project_config.json` as color checkers) will be automatically discovered and used to calculate the project's color range.

### 2. Sample Management (New)

To configure how sample images are processed (full average vs. point-based), use the `sample_manager_main.py` script:

Activate your virtual environment:

```bash
.\.venv\Scripts\activate
```

Run the sample manager:

```bash
python src/sample_manager_main.py --project <project_name>
```

Replace `<project_name>` with your project name. This will launch a GUI for each sample image that doesn't have a configuration or is set to `points` method. You can click on the image to select points. After selecting points, click "Save Points" to update `sample_processing_config.json` and close the GUI. This will also invalidate the project's cache, forcing a recalculation of the color range on the next `main.py` run.

### 3. Running the Analysis

Activate your virtual environment (if not already active):

```bash
.\.venv\Scripts\activate
```

Then, run the `main.py` script with the desired arguments:

*   **List available projects:**

    ```bash
    python src/main.py
    ```

    You will be prompted to select a project by number or name.

*   **Process a single image:**

    ```bash
    python src/main.py --project <project_name> --image <path_to_image.png>
    ```

    Replace `<project_name>` with the name of your project (e.g., `sample_project`) and `<path_to_image.png>` with the absolute or relative path to your image file. The color alignment (based on your project's configuration) will be automatically applied to this image before analysis.

*   **Process an image with alignment:**

    ```bash
    python src/main.py --project <project_name> --image <path_to_image.png> --alignment
    ```

    This will enable the image alignment feature. Make sure your project is configured with a `technical_drawing_filename`.

*   **Process a video file:**

    ```bash
    python src/main.py --project <project_name> --video <path_to_video.mp4>
    ```

    Replace `<project_name>` and `<path_to_video.mp4>` accordingly. Color alignment will be applied to each frame.

*   **Process live camera stream:**

    ```bash
    python src/main.py --project <project_name> --camera
    ```

    Color alignment will be applied to each frame. Press `q` to quit the camera stream.

### 4. Reports

After processing an image or video, detailed HTML and PDF reports will be generated in the `output/<project_name>/` directory. These reports include analysis statistics, processed images, and visual representations of the color space, with an improved layout.

## Future Improvements

*   **Advanced Color Correction:** Explore more sophisticated color correction algorithms.
*   **Batch Processing:** Add functionality to process multiple images or videos in a single run.
*   **Customizable Grid Detection:** Allow users to define the grid size (e.g., 5x5, 7x7) for color checkers other than the standard 6x4.
*   **Configuration File for Projects:** Implement a more robust way to define project settings (e.g., JSON or YAML files) instead of relying solely on directory structure.
