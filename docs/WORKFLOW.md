Here is the correct workflow to use the Visual Analyzer without encountering errors:

### Step 1: Create a New Project

First, create a new project using the `create_project.py` script. This will set up the necessary directory structure and configuration files.

```bash
python src/create_project.py --name <your_project_name>
```

When you create a new project, a default ArUco reference image (`default_aruco_reference.png`) will be automatically generated in the `dataset/aruco` folder.

### Step 2: Add Your Project Assets

After creating the project, you need to add your files to the generated directories inside `data/projects/<your_project_name>/`:

1.  **Reference Color Checker**: Place your reference color checker image (e.g., `my_reference_color_checker.png`) inside the `dataset/colorchecker/` directory. The script will automatically find the first image in this directory.

2.  **ArUco Reference (Optional)**: The `create_project.py` script generates a default ArUco reference sheet. If you have your own, place it in the `dataset/aruco` directory. The script will automatically find the first image in this directory.

3.  **Training Images**: Place the images that will be used to calculate the color range in the `dataset/training` directory.

4.  **Dataset Samples**: Add the images that represent the colors you want to analyze into the `dataset/` directory.

5.  **Analysis Samples**: The `samples` directory is created to hold different batches or situations for analysis. By default, a `test` folder is created with a `colorchecker` and a `sample` subfolder. You can create your own subfolders inside `samples` to organize your analysis images. For each sample folder, you can have a `colorchecker` subfolder to store the color checker image for that specific sample.

6.  **Update Project Configuration**: Open the `project_config.json` file and update the paths to your reference files.

    ```json
    {
        "reference_color_checker_path": "dataset/colorchecker",
        "training_path": "dataset/training",
        "aruco_reference_path": "dataset/aruco",
        "technical_drawing_path": null
    }
    ```

### Step 3: (Optional) Define Specific Sample Areas

If you want to define specific points or areas on your training images for color extraction (instead of using the entire image), run the `dataset_manager_main.py` script:

```bash
python src/dataset_manager_main.py --project <your_project_name>
```

This will launch a GUI that allows you to select points on each image in the `dataset/training/` directory. Your selections will be saved automatically.

### Step 4: Run the Analysis

Finally, you can run the main analysis pipeline using `src/main.py`.

Here is an example command to run the analysis with geometrical alignment using the ArUco reference:

```bash
python src/main.py --project <your_project_name> --image data/projects/<your_project_name>/samples/test/<your_sample_image.png> --alignment
```

If you also want to perform on-the-fly color alignment, you can add the `--color-alignment` and `--sample-color-checker` flags:

```bash
python src/main.py --project <your_project_name> --image data/projects/<your_project_name>/samples/test/<your_sample_image.png> --alignment --color-alignment --sample-color-checker data/projects/<your_project_name>/samples/test/colorchecker/<your_sample_color_checker.png>
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
