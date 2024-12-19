# Visual Analyzer

## Overview

The Visual Analyzer is a tool designed to analyze images and generate detailed HTML reports. It uses color analysis to identify specific colors in images and provides visualizations and statistics about the color distribution.

## Features

- Browse and select a dataset directory.
- Browse and select an image file.
- Analyze the selected image for specific colors.
- Generate an HTML report with detailed analysis and visualizations.
- Open the generated report in the default web browser.

## Requirements

- Python 3.x
- Required Python packages (install using `pip`):
  - `tkinter`
  - `pandas`
  - `Pillow`
  - `opencv-python`
  - `numpy`
  - `matplotlib`
  - `tqdm`
  - `jinja2`
  - `scipy`

## Installation

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the paths for templates, output directory, dataset, and sample image are correctly set in the `main.py` file:
   ```python
   # Global variables for folder and file paths
   TEMPLATES_DIR = "path/to/your/templates"
   OUTPUT_DIR = "output/report"
   SUGGESTED_DATASET_PATH = "path/to/your/dataset"
   SUGGESTED_SAMPLE_PATH = "path/to/your/sample"
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. The GUI will open. Follow these steps:
   - Click "Browse" next to "Dataset:" to select the dataset directory.
   - Click "Browse" next to "Image:" to select the image file.
   - Click "Analyze Image" to perform color analysis on the selected image.
   - Click "Generate Report" to generate an HTML report based on the analysis.

## Code Explanation

### main.py

- **Global Variables**: Paths for templates, output directory, dataset, and sample image.
- **ReportGeneratorGUI Class**: Handles the GUI operations.
  - `browse_dataset()`: Opens a dialog to select the dataset directory.
  - `browse_image()`: Opens a dialog to select the image file.
  - `analyze_image()`: Performs color analysis on the selected image.
  - `generate_report()`: Generates an HTML report based on the analysis.

### report.py

- **ReportConfig Class**: Configuration for report generation.
- **ImageReportGenerator Class**: Handles the report generation process.
  - `_setup_logging()`: Configures logging.
  - `_setup_template_environment()`: Sets up the Jinja2 template environment.
  - `analyze_image()`: Performs comprehensive image analysis.
  - `_generate_color_space_plot()`: Generates a color space plot with gradient.
  - `generate_report()`: Generates an HTML report from analysis results.
  - `process_batch()`: Processes all images in the configured image directory.

### image_analysis.py

- **ColorFinder Class**: Finds and highlights specific colors in images.
  - `find_color()`: Finds a color in an image and highlights it.
  - `calculate_percentage()`: Calculates the percentage of matched pixels.
  - `find_color_and_percentage()`: Finds and highlights a color in an image and calculates the percentage of pixels matching that color.
  - `plot_and_save_results()`: Plots and/or saves the original image, processed image, mask, pie chart, and bar chart.

## License

This project is licensed under the MIT License.