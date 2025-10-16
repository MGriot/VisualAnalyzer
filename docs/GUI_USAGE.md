# GUI Usage Guide

This document provides a detailed guide to using the Visual Analyzer's Graphical User Interface (GUI), launched by running `python -m src.gui`.

## Main Tabs

The GUI is organized into three main tabs:

1.  **Run Analysis:** The primary interface for executing an analysis pipeline on an image.
2.  **Create Project:** A simple tool to scaffold a new project directory.
3.  **Manage Dataset:** Tools for preparing and configuring your project's data assets.

## Run Analysis Tab

This is the main tab for running an analysis. The workflow is as follows:

1.  **Select Project:** Choose your project from the dropdown menu.
2.  **Select Image:** Click to select the image file you want to analyze.
3.  **Select Color Checker (Optional):** If your analysis image does not contain a color checker, you can provide a separate photo of a checker taken under the same lighting conditions.
4.  **Configure Analysis (Debug Mode):** If you run the GUI with the `--debug` flag (`python -m src.gui --debug`), a detailed options panel becomes visible, allowing you to toggle every step of the pipeline (e.g., Color Alignment, Object Alignment, Masking) and configure their parameters.
5.  **Run Analysis:** Click the "Run Analysis" button to start the pipeline.

### Interactive Fallback: Manual Color Checker Alignment

A key feature of the pipeline is its robust fallback system. If you run a process that requires color correction (e.g., the main analysis pipeline) and the system **fails to automatically detect the color checker** using ArUco markers, a new GUI window will automatically open.

-   **How it works:** The window will display the image of your color checker.
-   **Your action:** Click on the **four corners** of the color checker grid, starting with the **top-left** and proceeding clockwise.
-   Once you have selected four points, you can close the window.
-   The pipeline will automatically use these four points to perform a perspective alignment and continue the analysis.

This ensures that the analysis can always proceed, even if automatic detection is not possible.

## Create Project Tab

This tab provides a simple one-click utility to create a new project.

1.  **Enter Project Name:** Type the desired name for your new project.
2.  **Click Create:** The application will create a new folder in `data/projects/` with the standard sub-directory structure (`dataset`, `samples`, etc.) and generate the default configuration files and reference images (`project_config.json`, `default_color_checker_reference.png`, etc.).

## Manage Dataset Tab

This tab provides powerful tools to help you set up your project's data.

1.  **Launch Point Selector:** This opens a GUI for interactively defining regions of interest on your training images. For each image in your project's `training_images` folder, you can click to place points. The coordinates of these points are saved to `dataset_item_processing_config.json` and are used to define the target color for analysis.

2.  **Setup Project Files:** This opens the **File Placer**, a utility designed to streamline project setup. It reads your `project_config.json` and shows the status of all required files (e.g., Object Reference, Drawing Layers). For each missing file, you can click a button to open a file dialog, select the correct file from anywhere on your computer, and the tool will automatically copy and rename it to the correct location within your project.
