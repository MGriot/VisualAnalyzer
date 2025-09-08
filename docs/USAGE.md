# Visual Analyzer - Advanced Usage Guide

This document provides a detailed guide for advanced users of the Visual Analyzer, covering in-depth explanations of command-line arguments, debug reports, and the symmetry analysis feature.

## Command-Line Arguments

The `main.py` script provides a variety of command-line arguments to customize the analysis pipeline:

*   `--project <project_name>`: **(Required)** Specifies the project to use for the analysis. The project must be a directory in the `data/projects` folder.

*   `--image <path>`: Path to a single image file for analysis.

*   `--video <path>`: Path to a video file for analysis.

*   `--camera`: Use a live camera stream for analysis.

*   `--debug`: Enable debug mode. This will generate a more detailed report with intermediate steps and data, and print verbose output to the console.

*   `--aggregate`: Enable aggregation of nearby matched pixel areas. This is useful for noisy images where the matched pixels are scattered.

*   `--blur`: Enable blurring of the input image before color matching. This can help to reduce noise and smooth out color variations.

*   `--alignment`: Enable geometrical alignment. This feature uses ArUco markers to correct perspective distortion.

*   `--drawing <path>`: Path to a technical drawing for masking. The drawing should be a black and white image where the white area represents the object of interest.

*   `--color-alignment`: Enable color correction. This feature uses a color checker to correct the colors of the input image.

*   `--symmetry`: Enable symmetry analysis. This will analyze the symmetry of the object in the image and include the results in the debug report.

## Debug Reports

When you run the analysis with the `--debug` flag, the Visual Analyzer generates a detailed debug report in HTML and PDF format. The debug report contains the following sections:

*   **Project Information:** Information about the project, including the project name and the HSV color range used for the analysis.

*   **Analysis Settings:** The settings used for the analysis, such as whether color alignment, geometrical alignment, and blurring were enabled.

*   **Analysis Results:** The results of the analysis, including the number of matched pixels, the total number of pixels, and the percentage of matched pixels.

*   **Image Pipeline:** A visual representation of the image processing pipeline, showing the intermediate images at each step of the analysis.

*   **Symmetry Analysis Results:** The results of the symmetry analysis, including the symmetry scores for each type of symmetry and visualizations of the symmetry axes.

## Symmetry Analysis

The symmetry analysis feature allows you to analyze the symmetry of an object in an image. To use this feature, you need to enable it with the `--symmetry` flag.

The symmetry analysis is performed by the `SymmetryAnalyzer` class in the `src/symmetry_analysis/symmetry.py` file. This class provides methods for analyzing the following types of symmetry:

*   **Vertical and Horizontal Reflection:** The symmetry of the object across its vertical and horizontal axes.

*   **Four-Quadrant Symmetry:** The symmetry of the object across its four quadrants.

*   **Rotational Symmetry:** The symmetry of the object when it is rotated by a certain angle.

*   **Translational Symmetry:** The symmetry of the object when it is translated by a certain distance.

*   **Glide-Reflection Symmetry:** The symmetry of the object when it is reflected and then translated.

The results of the symmetry analysis are included in the debug report. The report shows the symmetry score for each type of symmetry, as well as visualizations of the symmetry axes.