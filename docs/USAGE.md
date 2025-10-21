# Visual Analyzer - Advanced Usage Guide

This document provides a detailed guide for advanced users of the Visual Analyzer, covering in-depth explanations of command-line arguments, debug reports, and the symmetry analysis feature.

## Dependencies

In addition to the libraries listed in `requirements.txt` (which include `numpy`, `opencv-python`, `ultralytics`, etc.), the tool relies on:

*   `scipy` and `scikit-image`: For robust color patch matching using the Hungarian algorithm and CIELAB color difference calculations.
*   `reportlab`: For generating PDF reports.
*   `colour-science`: For accessing standardized color data for reference generation.

## Command-Line Arguments

The `main.py` script provides a variety of command-line arguments to customize the analysis pipeline:

### Core Arguments

*   `--project <project_name>`: **(Required)** Specifies the project to use for the analysis. The project must be a directory in the `data/projects` folder.
*   `--image <path>`: Path to a single image file or a directory of images for analysis. The filename can be structured to automatically extract metadata for the report; see `WORKFLOW.md` for details.
*   `--video <path>`: Path to a video file for analysis.
*   `--camera`: Use a live camera stream for analysis.
*   `--debug`: Enable debug mode. This will generate a more detailed report with intermediate steps and data, and print verbose output to the console.

### Pipeline Step Arguments

*   `--color-alignment`: Enable color correction. This is a two-stage process:
    1.  **Patch Detection**: The system uses a tiered approach to find color checker patches: **ArUco Alignment** -> **Robust OpenCV (Hough)** -> **YOLOv8** -> **Simple Grid**.
    2.  **Robust Patch Matching**: After detection, it uses the Hungarian algorithm to intelligently match detected patches to their correct reference swatches based on color similarity (CIELAB ΔE*). This makes the process resilient to incorrectly ordered, missing, or falsely detected patches.
*   `--color-correction-method <method>`: Specify the algorithm for color correction. Choices: `linear`, `polynomial`, `hsv`, `histogram`. Default is `linear`.
*   `--sample-color-checker <path>`: (Optional) Path to a color checker image taken with the sample. If not provided, the system assumes the checker is in the main image and will fall back to a manual GUI alignment if automatic detection fails.
*   `--alignment`: Enable geometrical alignment. This feature, handled by the `geometric_alignment` module, uses ArUco markers to correct perspective distortion.
*   `--object-alignment`: Enable object alignment. This second alignment step aligns the object to a template image. The default method (`geometric_shape`) finds the main contour of the object in the source and reference images. It then attempts to fit a 5-sided polygon (a pentagon) to both. If successful, the 5 vertices are used to compute a highly accurate homography. If a pentagon cannot be resolved, it falls back to using the 4 corners of the minimum area bounding box. This geometric approach is robust to changes in scale and rotation. Requires `object_reference_path` to be set.
*   `--object-alignment-shadow-removal <method>`: Pre-processes the image to remove shadows before object alignment, which can significantly improve contour detection. Choices are `clahe` (default, advanced contrast enhancement), `gamma` (simple brightness lift), and `none`.
*   `--apply-mask`: Enable background removal using one or more technical drawing layers.
*   `--blur`: Enable blurring of the input image before color matching. This can help to reduce noise and smooth out color variations.
*   `--symmetry`: Enable symmetry analysis. This will analyze the symmetry of the object in the image and include the results in the debug report.
*   `--skip-color-analysis`: A flag to completely skip the color analysis step.
*   `--skip-report-generation`: A flag to completely skip the final report generation and archiving step.

### Masking Arguments

*   `--mask-bg-is-white`: A modifier for `--apply-mask`. If specified, any pure white pixels in the drawing file will also be treated as background and removed.
*   `--masking-order <order>`: Specify the order of masking layers (e.g., '1-2-3', '3-1-2'). Default is '1-2-3'.

### Color Aggregation Arguments

*   `--aggregate`: Enable aggregation of nearby matched pixel areas. This is useful for noisy images where the matched pixels are scattered.
*   `--agg-kernel-size <int>`: (Optional) Sets the kernel size for the aggregation dilation step. Default is 7.
*   `--agg-min-area <float>`: (Optional) Sets the minimum area ratio for keeping a detected color region during aggregation. Default is 0.0005.
*   `--agg-density-thresh <float>`: (Optional) Sets the minimum density (0.0-1.0) of original pixels for an aggregated area to be kept. Prevents over-aggregation. Default is 0.5.

### Other Optional Arguments

*   `--blur-kernel <W H>`: (Optional) Specify a custom kernel size (width height) for blurring. Both values must be odd integers. Overrides the default adaptive kernel.

*   `--save-state-to <path>`: Path to save the entire pipeline state to a `.gri` file. This allows for later report regeneration.
*   `--load-state-from <path>`: Path to load a previously saved pipeline state from a `.gri` file and re-run the pipeline from that point.

## GUI Usage

To launch the Graphical User Interface (GUI), run `python -m src.gui`.

For a detailed guide on all GUI features, including the interactive manual alignment fallback and dataset management tools, please see **[GUI_USAGE.md](GUI_USAGE.md)**.

## Report Outputs and Archiving

For each analysis run, the tool generates several output files in the `output/<project_name>/<sample_name>/` directory:

1.  **PDF Report:** A detailed PDF report generated with ReportLab (`<sample_name>_reportlab.pdf`).
2.  **State Archive:** A `.gri` (Gemini Report Information) file containing the entire pipeline state, created using Python's `pickle` module. These are stored in the `archives/` subdirectory.

*Note: The specific reports generated depends on the `--report-type` argument and whether the required libraries are installed.*

### Report Regeneration from State

You can regenerate any report from its `.gri` state file using the `regenerate_from_state.py` script. This is useful for recreating a report with different settings (like `--report-type`) without re-running the entire analysis.

**Usage:**
```bash
# Activate your virtual environment first
.venv\Scripts\activate.bat

# Run the regeneration script
python src/tools/regenerate_from_state.py --state-file "output/<project_name>/<sample_name>/archives/<archive_name>.gri"
```

## Debug Reports

When you run the analysis with the `--debug` flag, the Visual Analyzer generates a detailed debug report. The debug report contains the following sections:

*   **Project Information:** Information about the project, including the project name and the HSV color range used for the analysis.
*   **Analysis Settings:** The settings used for the analysis, such as which alignment, masking, and blurring options were enabled.
*   **Analysis Results:** The results of the analysis, including the number of matched pixels, the total number of pixels, and the percentage of matched pixels.
*   **Image Pipeline:** A visual representation of the image processing pipeline, showing the intermediate images at each step of the analysis. This is crucial for debugging the new alignment and masking steps.
*   **Symmetry Analysis Results:** The results of the symmetry analysis, including the symmetry scores for each type of symmetry and visualizations of the symmetry axes.
*   **Dataset Color Space Definition:** A detailed breakdown of how the color space was derived from the training images, including a scatter plot showing the distribution of training colors.
