<<<<<<< Updated upstream
version https://git-lfs.github.com/spec/v1
oid sha256:f495c4a57cddfd7785e840a61dfa24c68beb7c7d4a0344006bbce92409fd522c
size 8224
=======
# Visual Analyzer - Advanced Usage Guide

This document provides a detailed guide for advanced users of the Visual Analyzer, covering in-depth explanations of command-line arguments, debug reports, and the symmetry analysis feature.

## Dependencies

In addition to the libraries listed in `requirements.txt`, this tool uses `reportlab` and `weasyprint` for generating PDF reports from different sources.

## Command-Line Arguments

The `main.py` script provides a variety of command-line arguments to customize the analysis pipeline:

### Core Arguments

*   `--project <project_name>`: **(Required)** Specifies the project to use for the analysis. The project must be a directory in the `data/projects` folder.
*   `--image <path>`: Path to a single image file or a directory of images for analysis.
*   `--video <path>`: Path to a video file for analysis.
*   `--camera`: Use a live camera stream for analysis.
*   `--debug`: Enable debug mode. This will generate a more detailed report with intermediate steps and data, and print verbose output to the console.

### Pipeline Step Arguments

*   `--color-alignment`: Enable color correction. This requires `reference_color_checker_path` and at least one path in `colorchecker_reference_for_project` to be set in the project config.
*   `--alignment`: Enable geometrical alignment. This feature uses ArUco markers to correct perspective distortion.
*   `--object-alignment`: Enable object alignment. This second alignment step aligns the object to a template image. The default method uses a robust geometric shape fitting (pentagon/quadrilateral). Requires `object_reference_path` to be set.
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
*   `--report-type <type>`: Specify the type of PDF report to generate. Options are `html` (for the WeasyPrint-based PDF), `reportlab`, or `all` (default) to generate both.
*   `--save-state-to <path>`: Path to save the entire pipeline state to a `.gri` file. This allows for later report regeneration.
*   `--load-state-from <path>`: Path to load a previously saved pipeline state from a `.gri` file and re-run the pipeline from that point.

## GUI Usage

To launch the Graphical User Interface (GUI), run `streamlit_app.py`:

```bash
streamlit run streamlit_app.py
```

The GUI provides an intuitive way to configure and run the analysis. Here's a breakdown of its elements:

*   **Project Section:**
    *   **Project Name:** Select your project from the dropdown list. This corresponds to the `--project` CLI argument.

*   **Files Section:**
    *   **Select Image:** Click this button to browse and select the image file you want to analyze. This corresponds to the `--image` CLI argument.

*   **Analysis Steps & Options Section:** This section contains checkboxes for enabling/disabling various analysis steps, ordered by their typical execution flow in the pipeline. Some steps also have associated input fields for specific parameters.
    *   **Color Alignment:** Enables color correction (`--color-alignment`).
    *   **Geometrical Alignment (ArUco):** Enables ArUco-based alignment (`--alignment`).
    *   **Object Alignment:** Enables object alignment (`--object-alignment`).
    *   **Apply Mask:** Enables background removal (`--apply-mask`).
        *   **Treat White as BG:** Modifier for `--apply-mask` (`--mask-bg-is-white`).
        *   **Masking Order (e.g., 1-2-3):** Specifies the order of technical drawing layers to apply (`--masking-order`).
    *   **Blur Image:** Enables image blurring (`--blur`).
        *   **Blur Kernel (W H, odd):** Custom kernel size for blurring (`--blur-kernel`).
    *   **Aggregate Matched Pixels:** Enables aggregation of color regions (`--aggregate`).
        *   **Agg Kernel Size:** Kernel size for aggregation dilation (`--agg-kernel-size`).
        *   **Agg Min Area:** Minimum area ratio for aggregated components (`--agg-min-area`).
        *   **Agg Density Thresh:** Minimum density for aggregated areas (`--agg-density-thresh`).
    *   **Symmetry Analysis:** Enables symmetry calculation (`--symmetry`).

*   **Report Section:**
    *   **Report Type:** Select the desired report format from the dropdown. This corresponds to the `--report-type` CLI argument.

*   **Run Analysis Button:** Click this button to start the analysis. A dialog will prompt you to choose between running in **Debug Mode** (`--debug`) or **Normal Mode**. The GUI will execute the underlying CLI command and display a success or error message.

## Report Outputs and Archiving

For each analysis run, the tool generates several output files in the `output/<project_name>/<sample_name>/` directory:

1.  **HTML Report:** A detailed, interactive report (`<sample_name>.html`).
2.  **Primary PDF Report:** A PDF version of the report generated with WeasyPrint (`<sample_name>.pdf`).
3.  **Alternative PDF Report:** An alternative PDF version generated with ReportLab (`<sample_name>_reportlab.pdf`).
4.  **State Archive:** A `.gri` (Gemini Report Information) file containing the entire pipeline state, created using Python's `pickle` module. These are stored in the `archives/` subdirectory.

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
>>>>>>> Stashed changes
