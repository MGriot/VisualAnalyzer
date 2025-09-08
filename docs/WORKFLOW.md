Here is the correct workflow to use the Visual Analyzer without encountering errors:

### Step 1: Create a New Project

First, create a new project using the `create_project.py` script. This will set up the necessary directory structure and configuration files.

```bash
python src/create_project.py --name <your_project_name>
```

When you create a new project, a default ArUco reference image (`default_aruco_reference.png`) will be automatically generated in the `aruco` folder.

### Step 2: Add Your Project Assets

After creating the project, you need to add your files to the generated directories inside `data/projects/<your_project_name>/`:

1.  **Reference Color Checker**: Place your reference color checker image (e.g., `my_reference_color_checker.png`) inside the `dataset/colorchecker/` directory.

2.  **ArUco Reference (Optional)**: The `create_project.py` script generates a default ArUco reference sheet. If you have your own, place it in the `aruco` directory.

3.  **Update Project Configuration**: Open the `project_config.json` file and update the paths to your reference files.

    ```json
    {
        "reference_color_checker_path": "dataset/colorchecker/my_reference_color_checker.png",
        "aruco_reference_path": "aruco/default_aruco_reference.png",
        "technical_drawing_path": null
    }
    ```

    **Optional: Project-Level Color Calibration**

    If you want to apply a consistent color correction to all analyses within this project, you can add the `colorchecker_reference_for_project` field to your `project_config.json`. This is useful if you have a standard photo of a color checker taken in the project's typical lighting conditions.

    ```json
    {
        "reference_color_checker_path": "dataset/colorchecker/my_reference_color_checker.png",
        "colorchecker_reference_for_project": [
            "path/to/your/project_color_checker.png"
        ],
        "aruco_reference_path": "aruco/default_aruco_reference.png",
        "technical_drawing_path": null
    }
    ```

    This is different from the `--sample-color-checker` argument used during analysis, which performs color correction on a single image.

4.  **Dataset Samples**: Add the images that represent the colors you want to analyze into the `dataset/` directory (for existing projects) or `dataset/samples/` (for new projects created with `create_project.py`).

5.  **Analysis Image**: Place the image you want to perform the analysis on into the `sample/` directory.

### Step 3: (Optional) Define Specific Sample Areas

If you want to define specific points or areas on your dataset images for color extraction (instead of using the entire image), run the `dataset_manager_main.py` script:

```bash
python src/dataset_manager_main.py --project <your_project_name>
```

This will launch a GUI that allows you to select points on each image in the `dataset/` or `dataset/samples/` directory. Your selections will be saved automatically.

### Step 4: Run the Analysis

Finally, you can run the main analysis pipeline using `src/main.py`.

Here is an example command to run the analysis with geometrical alignment using the ArUco reference:

```bash
python src/main.py --project <your_project_name> --image data/projects/<your_project_name>/sample/<your_sample_image.png> --alignment
```

If you also want to perform on-the-fly color alignment, you can add the `--color-alignment` and `--sample-color-checker` flags:

```bash
python src/main.py --project <your_project_name> --image data/projects/<your_project_name>/sample/<your_sample_image.png> --alignment --color-alignment --sample-color-checker data/projects/<your_project_name>/sample/<your_sample_color_checker.png>
```

**Important Notes:**

*   When using the `--alignment` flag, the script will look for the `aruco_reference_path` in your `project_config.json` and use it for alignment.
*   When using the `--color-alignment` flag, you must also provide the `--sample-color-checker` argument with the path to the color checker image that is present in your sample image.
*   The file `src/sample_manager_main.py` does not exist. The correct script is `src/dataset_manager_main.py`.

### Command-Line Options for `src/main.py`

Here is a list of all the available command-line options for the main analysis script (`src/main.py`):

*   `--project <project_name>`: (Required) The name of the project to use for the analysis.
*   `--image <path>`: The path to a single image file or a directory of images to be analyzed.
*   `--video <path>`: The path to a single video file or a directory of videos to be analyzed.
*   `--camera`: Use the live camera stream as input for the analysis.
*   `--debug`: Enable debug mode, which will generate a detailed report with intermediate images and data.
*   `--aggregate`: Enable aggregation of matched pixel areas. This is useful for getting a single percentage match for the entire image.
*   `--blur`: Enable blurring of the input image before analysis. This can help reduce noise.
*   `--alignment`: Enable geometrical alignment of the input image. This requires either an `aruco_reference_path` in your `project_config.json` or an `aruco_marker_map` and `aruco_output_size`.
*   `--drawing <path>`: The path to a technical drawing (black and white image) to be used as a mask.
*   `--color-alignment`: Enable on-the-fly color correction of the input image. This is different from the project-level color calibration.
*   `--sample-color-checker <path>`: The path to a color checker image that is present in the sample image. This is required when using `--color-alignment`.
*   `--symmetry`: Enable symmetry analysis of the input image.
