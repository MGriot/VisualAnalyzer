# VisualAnalyzer Documentation

## Introduction

This document provides a comprehensive guide to the VisualAnalyzer project, explaining the functionality of each script and the underlying classes.

## Project Structure

The project is structured as follows:

- **VisualAnalyzer:** This directory contains the core classes for image analysis and clustering.
    - **ImageCluster.py:** Provides methods for color clustering and analysis.
    - **ColorFinder.py:** Implements color detection and analysis functionalities.
    - **ImageAnalyzer.py:** (Not documented in this guide)
    - **ImageProcessor.py:** (Not documented in this guide)
    - **ManualImageAligner.py:** (Not documented in this guide)
    - **__init__.py:** Initializes the package.
- **Templates:** This directory contains the HTML template for report generation.
    - **Report.html:** Jinja2 template for the HTML report.
- **img:** This directory contains images used for testing and database.
    - **data:** Contains images to be processed.
    - **database:** Contains images used as a color database for analysis.
    - **logo:** Contains the logo used in the report.
- **output:** This directory will contain the generated plots from ImageCluster.
- **processed_images:** This directory will contain the processed images from ColorFinder.
- **main_cluster.py:** Example script demonstrating the usage of ImageCluster.
- **main_color_analysis.py:** Example script demonstrating the usage of ColorFinder.
- **report.py:** Script for generating HTML reports based on image analysis.
- **requirements.txt:** Lists the project's dependencies.
- **README.md:** This documentation file.

## Scripts

### report.py

This script analyzes images, generates visualizations, and creates an HTML report summarizing the findings.

**Functionality:**

1. **Image Analysis:**
   - Iterates through images in the `img/data` directory.
   - Extracts part number and thickness from the image file name.
   - Uses `ColorFinder` to identify dominant colors and calculate the percentage of matched pixels.
   - Generates a pie chart showing the proportion of matched and unmatched pixels.
   - Generates a color space plot visualizing the color range used for matching.
2. **Report Generation:**
   - Uses the `Report.html` template from the `Templates` directory.
   - Populates the template with image paths, analysis results, and metadata (logo, author, etc.).
   - Saves the generated HTML report in the current working directory.

**Requirements:**

- Python 3.x
- Libraries listed in `requirements.txt` (install using `pip install -r requirements.txt`)
- Images in `img/data` directory
- Color database images in `img/database` directory
- `Templates/Report.html` template file

**Usage:**

Run the script from the command line: `python report.py`

### main_cluster.py

This script demonstrates the usage of the `ImageCluster` class for color clustering and analysis.

**Functionality:**

1. **Image Preprocessing:**
   - Loads an image using `ImageProcessor`.
   - Applies blurring and resizing to the image.
2. **Color Clustering:**
   - Creates an `ImageCluster` object.
   - Removes transparent pixels.
   - Performs KMeans clustering to group similar colors.
3. **Visualization and Analysis:**
   - Plots the clustered image, high contrast clustered image, and cluster distribution pie chart.
   - Extracts and prints information about each cluster (color, pixel count, percentage).
   - Prints the dominant color of the image.

**Requirements:**

- Python 3.x
- Libraries listed in `requirements.txt`

**Usage:**

Run the script from the command line: `python main_cluster.py`

### main_color_analysis.py

This script demonstrates the usage of the `ColorFinder` class for color detection and analysis.

**Functionality:**

1. **Color Limits:**
   - Creates a `ColorFinder` object.
   - Sets color limits based on a dataset of images in the `img/database` directory.
2. **Color Detection:**
   - Processes an image to find and highlight areas matching the defined color limits.
   - Calculates the percentage of pixels matching the color.
3. **Output:**
   - Prints the selected color and the percentage of matched pixels.
   - Displays the processed image with highlighted regions.

**Requirements:**

- Python 3.x
- Libraries listed in `requirements.txt`
- Images in `img/database` directory

**Usage:**

Run the script from the command line: `python main_color_analysis.py`

## Classes

### VisualAnalyzer/ImageCluster.py

This class provides methods for color clustering and analysis of images.

**Attributes:**

- `image_input`: The input image (file path or PIL.Image object).
- `n_clusters`: The number of clusters to form.
- `initial_clusters`: Initial cluster centers (optional).
- `img_array`: The image data as a NumPy array.
- `data`: Reshaped image data for clustering.
- `removeTransparent`: Flag indicating if transparent pixels have been removed.
- `labels_full`: Cluster labels for all pixels.
- `mask`: Boolean mask indicating non-transparent pixels.
- `clustered_img`: The clustered image.
- `cluster_infos`: Information about each cluster.

**Methods:**

- `remove_transparent()`: Removes transparent pixels from the image.
- `filter_alpha()`: Returns a boolean mask indicating non-transparent pixels.
- `cluster()`: Performs color clustering using KMeans.
- `create_clustered_image()`: Creates an image where each pixel is replaced with its cluster's color.
- `create_clustered_image_with_ids()`: Creates an image where each pixel is replaced with its cluster's ID.
- `extract_cluster_info()`: Extracts information about the clusters.
- `calculate_brightness()`: Calculates the brightness of a color.
- `plot_original_image()`: Displays the original image.
- `plot_clustered_image()`: Displays the clustered image.
- `plot_clustered_image_high_contrast()`: Displays the clustered image with high contrast.
- `plot_cluster_pie()`: Displays a pie chart of cluster distribution.
- `plot_cluster_bar()`: Displays a bar chart of cluster distribution.
- `plot_cumulative_barchart()`: Displays a cumulative bar chart of cluster distribution.
- `plot_images()`: Displays the original, clustered, and high contrast clustered images.
- `plot_image_with_grid()`: Displays the original image with a grid overlaid.
- `save_plots()`: Saves all generated plots to a directory.
- `get_dominant_color()`: Returns the dominant color of the image.

### VisualAnalyzer/ColorFinder.py

This class implements color detection and analysis functionalities.

**Attributes:**

- `lower_limit`: Lower HSV color limit.
- `upper_limit`: Upper HSV color limit.
- `center`: Center HSV color.

**Methods:**

- `get_color_limits_from_dataset()`: Calculates color limits based on a dataset of images.
- `get_color_limits_from_hsv()`: Calculates color limits based on a given HSV color and percentages.
- `process_webcam()`: Processes video from the webcam to identify and highlight areas matching the color limits.
- `process_image()`: Processes an image to identify and highlight areas matching the color limits.
- `find_color_and_percentage()`: Finds and highlights a color in an image and calculates the percentage of pixels matching that color.

## HTML Template

The `Templates/Report.html` file is a Jinja2 template used by `report.py` to generate the HTML report. It contains placeholders for image paths, analysis results, and metadata, which are dynamically populated by the script.

## Conclusion

This documentation provides a detailed overview of the VisualAnalyzer project. By understanding the functionality of each script and class, you can effectively use and extend this project for your image analysis needs.
