# Visual Analyzer - Advanced Usage Guide

This document provides a detailed guide for advanced users of the Visual Analyzer, covering in-depth explanations of command-line arguments, debug reports, and the symmetry analysis feature.

## Dependencies

In addition to the libraries listed in `requirements.txt`, this tool now uses `reportlab` for generating an alternative PDF report format.

## Command-Line Arguments

The `main.py` script provides a variety of command-line arguments to customize the analysis pipeline:

*   `--project <project_name>`: **(Required)** Specifies the project to use for the analysis. The project must be a directory in the `data/projects` folder.

*   `--image <path>`: Path to a single image file for analysis.

*   `--video <path>`: Path to a video file for analysis.

*   `--camera`: Use a live camera stream for analysis.

*   `--debug`: Enable debug mode. This will generate a more detailed report with intermediate steps and data, and print verbose output to the console.

*   `--aggregate`: Enable aggregation of nearby matched pixel areas. This is useful for noisy images where the matched pixels are scattered.

*   `--blur`: Enable blurring of the input image before color matching. This can help to reduce noise and smooth out color variations.

*   `--blur-kernel <W H>`: (Optional) Specify a custom kernel size (width height) for blurring. Both values must be odd integers. Overrides the default adaptive kernel.

*   `--color-alignment`: Enable color correction. This feature uses a color checker to correct the colors of the input image.

*   `--symmetry`: Enable symmetry analysis. This will analyze the symmetry of the object in the image and include the results in the debug report.

*   `--report-type <type>`: Specify the type of PDF report to generate. Options are `html` (for the WeasyPrint-based PDF), `reportlab`, or `all` (default) to generate both.

*   `--agg-kernel-size <int>`: (Optional) Sets the kernel size for the aggregation dilation step. Default is 7.

*   `--agg-min-area <float>`: (Optional) Sets the minimum area ratio for keeping a detected color region during aggregation. Default is 0.0005.

*   `--agg-density-thresh <float>`: (Optional) Sets the minimum density (0.0-1.0) of original pixels for an aggregated area to be kept. Prevents over-aggregation. Default is 0.5.

### Alignment and Masking Arguments

*   `--alignment`: Enable geometrical alignment. This feature uses ArUco markers to correct perspective distortion based on the `aruco_reference_path` in the project config.

*   `--object-alignment`: Enable object alignment. This second alignment step aligns the object to a template image using feature matching. It requires the `object_reference_path` to be set in the project config.

*   `--apply-mask`: Enable background removal. This step uses a drawing to create a mask and remove the background from the image. It requires the `technical_drawing_path` to be set in the project config.

*   `--mask-bg-is-white`: A modifier for `--apply-mask`. If specified, any pure white pixels in the drawing file will also be treated as background and removed.

*   `--drawing <path>`: **(Legacy)** Path to a technical drawing for masking. The `--apply-mask` feature, which uses the project's configured `technical_drawing_path`, is now the recommended method for masking.

## Report Outputs and Archiving

For each analysis run, the tool generates several output files in the `output/<project_name>/<sample_name>/` directory:

1.  **HTML Report:** A detailed, interactive report (`<sample_name>.html`).
2.  **Primary PDF Report:** A PDF version of the report generated with WeasyPrint (`<sample_name>.pdf`).
3.  **Alternative PDF Report:** An alternative PDF version generated with ReportLab (`<sample_name>_reportlab.pdf`).
4.  **Report Archive:** A `.zip` file containing all data and assets required to fully regenerate the report. These are stored in the `archives/` subdirectory.

*Note: The specific reports generated depends on the `--report-type` argument and whether the required libraries are installed.*

### Report Regeneration

You can regenerate any report from its archive file using the `regenerate_report.py` script. This is useful for recreating a report if the output files have been deleted or if you want to apply a new report template.

**Usage:**
```bash
# Activate your virtual environment first
.venv\Scripts\activate.bat

# Run the regeneration script
python regenerate_report.py --archive "output/<project_name>/<sample_name>/archives/<archive_name>.zip"
```

## Debug Reports

When you run the analysis with the `--debug` flag, the Visual Analyzer generates a detailed debug report. The debug report contains the following sections:

*   **Project Information:** Information about the project, including the project name and the HSV color range used for the analysis.

*   **Analysis Settings:** The settings used for the analysis, such as which alignment, masking, and blurring options were enabled.

*   **Analysis Results:** The results of the analysis, including the number of matched pixels, the total number of pixels, and the percentage of matched pixels.

*   **Image Pipeline:** A visual representation of the image processing pipeline, showing the intermediate images at each step of the analysis. This is crucial for debugging the new alignment and masking steps.

*   **Symmetry Analysis Results:** The results of the symmetry analysis, including the symmetry scores for each type of symmetry and visualizations of the symmetry axes.

*   **Dataset Color Space Definition:** A detailed breakdown of how the color space was derived from the training images, including a scatter plot showing the distribution of training colors.
