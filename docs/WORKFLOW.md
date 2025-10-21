# Visual Analyzer - Project Workflow

This document outlines the standard workflow for setting up and running an analysis with the Visual Analyzer.

### Step 1: Create a New Project

First, create a new project using the GUI by running `python src/gui.py` and using the "Create Project" tab. This will set up the necessary directory structure and default configuration files.

This command creates a new folder under `data/projects/` and automatically generates two key reference files inside the `dataset` subfolder:
1.  `default_geometric_align_reference.png`: A large sheet with ArUco markers for perspective correction.
2.  `default_color_checker_reference.png`: An ideal, digitally perfect color checker with its own ArUco markers.

### Step 2: Add Your Project-Specific Assets

After creating the project, add your unique files to the generated directories:

1.  **Project-Specific Color Checker**: Place a photo of your color checker, taken under your project's specific lighting conditions, into the `dataset/` folder. **You must name this file `project_color_checker.png`** for the default configuration to work.
2.  **Object Reference**: If using object alignment, place your reference image in `dataset/` and name it `object_reference.png`.
3.  **Drawing Layers**: If using masking, place your drawing files in `dataset/drawing_layers/`.
4.  **Training Images**: Place the images that will be used to calculate the target color range in the `dataset/training_images/` directory.
5.  **Analysis Images & Metadata**: Place the images you want to analyze into the `samples/` directory.

    > **IMPORTANT**: The filename of your analysis images is used to automatically extract the **Part Number (PN)** and **Thickness** for the final report.
    >
    > The system expects the following format:
    > `PREFIX_ANYTHING_PARTNUMBER_THICKNESS_... .ext`
    >
    > - **Part Number**: The 3rd element separated by underscores.
    > - **Thickness**: The 4th element separated by underscores.
    >
    > **Example:**
    > For an image named `PROJECTA_SAMPLE_PN12345_1.5mm_test.jpg`:
    > - The Part Number will be `PN12345`.
    > - The Thickness will be `1.5mm`.
    >
    > If the filename does not follow this structure, the entire filename becomes the Part Number and Thickness is marked "N/A".

*Tip: The "Manage Dataset" tab in the main GUI provides a "File Placer" tool to help you copy and rename these files into the correct locations. This tool also provides **instant validation** for the `default_color_checker_reference.png` file, confirming if its patches can be automatically detected.*

### Step 3: Update Project Configuration

Open the `project_config.json` file. The script will have pre-filled some paths. Ensure they point to the correct files relative to the project directory. A complete configuration looks like this:

```json
{
    "training_path": "dataset/training_images",
    "object_reference_path": "dataset/object_reference.png",
    "color_correction": {
        "reference_color_checker_path": "dataset/reference_color_checker.png",
        "project_specific_color_checker_path": "dataset/project_color_checker.png"
    },
    "geometrical_alignment": {
        "reference_path": "dataset/default_aruco_reference.png",
        "marker_map": {},
        "output_size": [
            1000,
            1000
        ]
    },
    "masking": {
        "drawing_layers": {
            "1": "dataset/drawing_layers/layer1.png",
            "2": "dataset/drawing_layers/layer2.png",
            "3": "dataset/drawing_layers/layer3.png"
        }
    }
}
```

### Step 4: (Optional) Define Specific Sample Areas

If you want to define specific points on your training images for color extraction (instead of using the whole image), launch the main GUI (`python src/gui.py`), go to the "Manage Dataset" tab, select your project, and click "Launch Point Selector".

This opens a dedicated GUI to select points on each training image. Your selections are saved in `dataset_item_processing_config.json`.

### Step 5: Run the Analysis

Finally, you can run the main analysis pipeline using the Tkinter GUI (`python src/gui.py`) or the CLI (`src/main.py`).

> **Note on Output Folders:** Each analysis run will save its output (debug images, charts, and the `.gri` archive) into a unique, self-contained folder. This folder will be created in `output/<project_name>/` and named using the format: `<timestamp>_<part-number>_<thickness>`.

**Pipeline Order:**
1.  Color Correction (`--color-alignment`)
2.  Geometrical Alignment (`--alignment`)
3.  Object Alignment (`--object-alignment`)
4.  Masking / Background Removal (`--apply-mask`)
5.  Blur (`--blur`)
6.  Color Analysis
7.  Symmetry Analysis (`--symmetry`)

**Example Command:**

This command runs a comprehensive pipeline from the command line.

```bash
python src/main.py \
    --project <your_project_name> \
    --image data/projects/<your_project_name>/samples/<your_image.png> \
    --debug \
    --color-alignment \
    --color-correction-method polynomial \
    --alignment \
    --object-alignment \
    --apply-mask
```

To treat white pixels in the drawing as background, add the `--mask-bg-is-white` flag:

```bash
python src/main.py \
    --project <your_project_name> \
    --image <path_to_your_image> \
    --debug \
    --apply-mask \
    --mask-bg-is-white
```

*Note on Alignment: If you run a pipeline with color or geometrical alignment and the automatic ArUco marker detection fails, a GUI window will automatically open, allowing you to manually select the corners of the checker or alignment sheet to ensure the analysis can proceed.*