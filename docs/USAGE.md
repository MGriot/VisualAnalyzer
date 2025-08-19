# Visual Analyzer - Detailed Usage Guide

This document provides a more in-depth guide on how to use the Visual Analyzer, covering project setup, sample management, running analyses, and understanding the debug output.

## 1. Project Setup

Each analysis in Visual Analyzer is organized around a 'project'. A project is simply a directory within the `data/projects/` folder that contains all the necessary configuration and image files for a specific analysis task.

### Project Directory Structure

```
VisualAnalyzer/
├── data/
│   ├── projects/
│   │   └── <your_project_name>/
│   │       ├── project_config.json
│   │       ├── sample_processing_config.json (Optional)
│   │       ├── reference_color_checker.png (Or whatever you name it in project_config.json)
│   │       ├── your_drawing.png (For alignment)
│   │       ├── samples/ (Directory for sample images, optional)
│   │       │   └── sample_image_1.png
│   │       │   └── sample_image_2.jpg
│   │       └── other_sample_image.png (Sample images can also be directly in the project folder)
```

### `project_config.json`

This file defines the core settings for your project. It **must** be present in your project folder.

```json
{
    "reference_color_checker_filename": "reference_color_checker.png",
    "colorchecker_reference_for_project": [
        "reference_color_checker.png" 
    ],
    # Visual Analyzer - Detailed Usage Guide

This document provides a more in-depth guide on how to use the Visual Analyzer, covering project setup, sample management, running analyses, and understanding the debug output.

## 1. Project Setup

Each analysis in Visual Analyzer is organized around a 'project'. A project is simply a directory within the `data/projects/` folder that contains all the necessary configuration and image files for a specific analysis task.

### Project Directory Structure

```
VisualAnalyzer/
├── data/
│   ├── projects/
│   │   └── <your_project_name>/
│   │       ├── project_config.json
│   │       ├── sample_processing_config.json (Optional)
│   │       ├── reference_color_checker.png (Or whatever you name it in project_config.json)
│   │       ├── your_drawing.png (For alignment)
│   │       ├── samples/ (Directory for sample images, optional)
│   │       │   └── sample_image_1.png
│   │       │   └── sample_image_2.jpg
│   │       └── other_sample_image.png (Sample images can also be directly in the project folder)
```

### `project_config.json`

This file defines the core settings for your project. It **must** be present in your project folder.

```json
{
    "reference_color_checker_filename": "reference_color_checker.png",
    "colorchecker_reference_for_project": [
        "reference_color_checker.png" 
    ],
    "technical_drawing_filename": "your_drawing.png"
}
```

*   `reference_color_checker_filename`: Specifies the filename of the *ideal* color checker image for your project. This image serves as the target for color alignment. It should be located directly within your project folder.
*   `colorchecker_reference_for_project`: A list of paths (relative to the project folder) to images that contain a color checker. These images are used to calculate the color alignment matrix for the project. The matrix is calculated once and cached.
*   `technical_drawing_filename` (Optional): The filename of the technical drawing to be used for alignment. This should be a black image with the object's profile in white.

### `sample_processing_config.json` (Optional)

This file allows you to define how individual sample images contribute to the color range calculation. If this file is not present, all discovered sample images will be processed using the `full_average` method by default.

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

*   `image_configs`: A list of configurations for individual sample images.
    *   `filename`: The filename of the sample image (relative to the project folder).
    *   `method`: Specifies how the color should be extracted from this image:
        *   `"full_average"`: The average color of all non-transparent pixels in the image will be used.
        *   `"points"`: The average color will be calculated from specific points within the image. This method requires the `points` field.
    *   `points` (Required if `method` is `"points"`): A list of dictionaries, each representing a point:
        *   `x`, `y`: Coordinates of the point.
        *   `radius`: (Optional) The radius in pixels around the point to average. Defaults to 7 pixels if not specified.

### Sample Images Discovery

All image files (`.png`, `.jpg`, `.jpeg`) directly within your project folder or its `samples/` subdirectory will be automatically discovered as sample images. Files listed in `project_config.json` as `reference_color_checker_filename` or within `colorchecker_reference_for_project` will be excluded from sample image discovery.

## 2. Sample Management

To configure `sample_processing_config.json` interactively, especially for selecting points, use the `sample_manager_main.py` script.

Activate your virtual environment:

```bash
.\.venv\Scripts\activate
```

Run the sample manager:

```bash
python src/sample_manager_main.py --project <your_project_name>
```

Replace `<your_project_name>` with the name of your project. The script will iterate through each discovered sample image:

*   If an image has no entry in `sample_processing_config.json` or its `method` is set to `"points"`, a GUI window will open.
*   **GUI Interaction:**
    *   Click on the image to add points. A red circle will appear around each selected point.
    *   You can add multiple points.
    *   Click "Clear Points" to remove all selected points for the current image.
    *   Click "Save Points" to save the selected points (or an empty list if cleared) to `sample_processing_config.json` for that image and close the GUI. The script will then proceed to the next sample image.

After managing all samples, the `sample_processing_config.json` will be updated. This action also invalidates the project's cache, forcing a recalculation of the color range on the next `main.py` run.

## 3. Running the Analysis

Once your project is set up and sample processing is configured, you can run the main analysis script.

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

*   **Process a single image with color alignment:**

    ```bash
    python src/main.py --project <project_name> --image <path_to_image.png> --color-alignment
    ```

    This will enable color correction for the image.

*   **Process an image with perspective alignment:**

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

### Command-Line Options

*   `--project <project_name>`: (Optional) Specifies the project to use. If omitted, you will be prompted to select one.
*   `--image <path>`: (Optional) Path to a single image file for analysis.
*   `--video <path>`: (Optional) Path to a video file for analysis.
*   `--camera`: (Optional) Use live camera stream for analysis.
*   `--alignment`: (Optional) Enable image alignment with a technical drawing.
*   `--color-alignment`: (Optional) Enable color alignment (correction).
*   `--debug`: (Optional) Enable debug mode for verbose console output and detailed debug reports.
*   `--aggregate`: (Optional) Enable aggregation of nearby matched pixel areas. This applies morphological closing to the mask to merge close regions and fill small holes.
*   `--blur`: (Optional) Enable blurring of the input image before color matching. This applies a Gaussian blur (5x5 kernel) to the image.

## 4. Understanding Reports

After processing, detailed HTML and PDF reports are generated in the `output/<project_name>/` directory. The report content varies based on whether debug mode is enabled.

### Default Report (without `--debug`)

This concise report focuses on the final analysis results:
*   Project characteristics (P/N, Thickness).
*   Original Input Image.
*   Matched Pixels image.
*   Pie Chart showing matched vs. unmatched pixels.

### Debug Report (with `--debug`)

This detailed report provides extensive information for troubleshooting and in-depth understanding:
*   All content from the Default Report.
*   Original vs. Analyzed Image comparison (showing the effect of color correction).
*   Color Space Plot (visualizing the calculated HSV range).
*   Mask image.
*   Negative Mask image.
*   **If `--aggregate` is used:** Mask Before Aggregation image.
*   **If `--blur` is used:** Blurred Image.
*   **Debug Information Table:** A table summarizing key parameters and results, such as HSV ranges, matched pixel counts, and flags for blur/aggregation.

## 5. Source Code Treemap

```
VisualAnalyzer/
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
│   ├── sample_manager/
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   └── gui.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── video_utils.py
│   ├── main.py
│   └── sample_manager_main.py
```

## Future Improvements

*   **Advanced Color Correction:** Explore more sophisticated color correction algorithms.
*   **Batch Processing:** Add functionality to process multiple images or videos in a single run.
*   **Customizable Grid Detection:** Allow users to define the grid size (e.g., 5x5, 7x7) for color checkers other than the standard 6x4.
*   **Configuration File for Projects:** Implement a more robust way to define project settings (e.g., JSON or YAML files) instead of relying solely on directory structure.
```

*   `reference_color_checker_filename`: Specifies the filename of the *ideal* color checker image for your project. This image serves as the target for color alignment. It should be located directly within your project folder.
*   `colorchecker_reference_for_project`: A list of paths (relative to the project folder) to images that contain a color checker. These are the *actual* color checker images taken under specific conditions that represent the 'standard' for your project. The color alignment matrix is calculated using the first image in this list against the `reference_color_checker_filename`. This matrix is then cached for persistent use.
*   `technical_drawing_filename` (Optional): The filename of the technical drawing to be used for alignment. This should be a black image with the object's profile in white.

### `sample_processing_config.json` (Optional)

This file allows you to define how individual sample images contribute to the color range calculation. If this file is not present, all discovered sample images will be processed using the `full_average` method by default.

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

*   `image_configs`: A list of configurations for individual sample images.
    *   `filename`: The filename of the sample image (relative to the project folder).
    *   `method`: Specifies how the color should be extracted from this image:
        *   `"full_average"`: The average color of all non-transparent pixels in the image will be used.
        *   `"points"`: The average color will be calculated from specific points within the image. This method requires the `points` field.
    *   `points` (Required if `method` is `"points"`): A list of dictionaries, each representing a point:
        *   `x`, `y`: Coordinates of the point.
        *   `radius`: (Optional) The radius in pixels around the point to average. Defaults to 7 pixels if not specified.

### Sample Images Discovery

All image files (`.png`, `.jpg`, `.jpeg`) directly within your project folder or its `samples/` subdirectory will be automatically discovered as sample images. Files listed in `project_config.json` as `reference_color_checker_filename` or within `colorchecker_reference_for_project` will be excluded from sample image discovery.

## 2. Sample Management

To configure `sample_processing_config.json` interactively, especially for selecting points, use the `sample_manager_main.py` script.

Activate your virtual environment:

```bash
.\.venv\Scripts\activate
```

Run the sample manager:

```bash
python src/sample_manager_main.py --project <your_project_name>
```

Replace `<your_project_name>` with the name of your project. The script will iterate through each discovered sample image:

*   If an image has no entry in `sample_processing_config.json` or its `method` is set to `"points"`, a GUI window will open.
*   **GUI Interaction:**
    *   Click on the image to add points. A red circle will appear around each selected point.
    *   You can add multiple points.
    *   Click "Clear Points" to remove all selected points for the current image.
    *   Click "Save Points" to save the selected points (or an empty list if cleared) to `sample_processing_config.json` for that image and close the GUI. The script will then proceed to the next sample image.

After managing all samples, the `sample_processing_config.json` will be updated. This action also invalidates the project's cache, forcing a recalculation of the color range on the next `main.py` run.

## 3. Running the Analysis

Once your project is set up and sample processing is configured, you can run the main analysis script.

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

### Image Alignment (New Feature)

The image alignment feature is designed to correct perspective distortion and align the object in the image with a technical drawing. This is particularly useful when analyzing objects from non-perpendicular camera angles. The alignment is a two-step process:

1.  **Perspective Correction:** The system uses a chessboard pattern in the background of the image to calculate the perspective transformation and "flatten" the image, as if it were taken from directly above.
2.  **Object Alignment:** After perspective correction, the system aligns the object with a provided technical drawing. The alignment is done by matching the position and rotation of the object's contour with the contour of the drawing.

#### Setup for Image Alignment

To use the alignment feature, you need to:

1.  **Use a Chessboard Pattern:** When taking photos of your objects, place them on a printed chessboard pattern. The system will use this pattern to correct the perspective. You can generate a reference chessboard image using the `generate_chessboard_image` function in `src/alignment/aligner.py`.

2.  **Provide a Technical Drawing:** You need to provide a technical drawing of the object. This should be a black image with the object's profile in white. This drawing will be used for the final alignment and as a mask to remove the background.

3.  **Update `project_config.json`:** Add the `technical_drawing_filename` key to your project's `project_config.json` file:

    ```json
    {
        "reference_color_checker_filename": "reference_color_checker.png",
        "colorchecker_reference_for_project": [
            "reference_color_checker.png"
        ],
        "technical_drawing_filename": "your_drawing.png"
    }
    ```

    Replace `"your_drawing.png"` with the actual filename of your technical drawing, which should be located in your project's directory.

#### Running Analysis with Alignment

To run the analysis with the image alignment feature, use the `--alignment` flag in the command line:

```bash
python src/main.py --project <project_name> --image <path_to_image.png> --alignment
```

The application will then perform the alignment at the beginning of the image processing pipeline, before any color correction or analysis.

### Command-Line Options

*   `--project <project_name>`: (Optional) Specifies the project to use. If omitted, you will be prompted to select one.
*   `--image <path>`: (Optional) Path to a single image file for analysis.
*   `--video <path>`: (Optional) Path to a video file for analysis.
*   `--camera`: (Optional) Use live camera stream for analysis.
*   `--alignment`: (Optional) Enable image alignment with a technical drawing.
*   `--debug`: (Optional) Enable debug mode for verbose console output and detailed debug reports.
*   `--aggregate`: (Optional) Enable aggregation of nearby matched pixel areas. This applies morphological closing to the mask to merge close regions and fill small holes.
*   `--blur`: (Optional) Enable blurring of the input image before color matching. This applies a Gaussian blur (5x5 kernel) to the image.

## 4. Understanding Reports

After processing, detailed HTML and PDF reports are generated in the `output/<project_name>/` directory. The report content varies based on whether debug mode is enabled.

### Default Report (without `--debug`)

This concise report focuses on the final analysis results:
*   Project characteristics (P/N, Thickness).
*   Original Input Image.
*   Matched Pixels image.
*   Pie Chart showing matched vs. unmatched pixels.

### Debug Report (with `--debug`)

This detailed report provides extensive information for troubleshooting and in-depth understanding:
*   All content from the Default Report.
*   Original vs. Analyzed Image comparison (showing the effect of color correction).
*   Color Space Plot (visualizing the calculated HSV range).
*   Mask image.
*   Negative Mask image.
*   **If `--aggregate` is used:** Mask Before Aggregation image.
*   **If `--blur` is used:** Blurred Image.
*   **Debug Information Table:** A table summarizing key parameters and results, such as HSV ranges, matched pixel counts, and flags for blur/aggregation.

## 5. Source Code Treemap

```
VisualAnalyzer/
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
│   ├── sample_manager/
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   └── gui.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── video_utils.py
│   ├── main.py
│   └── sample_manager_main.py
```

## Future Improvements

*   **Advanced Color Correction:** Explore more sophisticated color correction algorithms.
*   **Batch Processing:** Add functionality to process multiple images or videos in a single run.
*   **Customizable Grid Detection:** Allow users to define the grid size (e.g., 5x5, 7x7) for color checkers other than the standard 6x4.
*   **Configuration File for Projects:** Implement a more robust way to define project settings (e.g., JSON or YAML files) instead of relying solely on directory structure.
