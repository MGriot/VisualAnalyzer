<<<<<<< Updated upstream
version https://git-lfs.github.com/spec/v1
oid sha256:c77573839e9942aa5fe5714e1f1d7efb41f96625d9b9aae6d44078e9c00b56f0
size 4231
=======
# Visual Analyzer - Project Workflow

This document outlines the standard workflow for setting up and running an analysis with the Visual Analyzer.

### Step 1: Create a New Project

First, create a new project using the `create_project.py` script. This will set up the necessary directory structure and default configuration files.

```bash
python src/create_project.py --name <your_project_name>
```

This command creates a new folder under `data/projects/` with the following structure:

```
<your_project_name>/
├── dataset/
│   ├── colorchecker/       # For the ideal color checker reference
│   ├── training/           # For images used to define the target color
│   ├── aruco/              # For the ArUco reference sheet
│   ├── drawing/            # For the drawing(s) to be used as a mask
│   └── object/             # For the object reference/template image
├── samples/
│   └── test/
│       ├── colorchecker/   # For color checkers found in a specific sample
│       └── sample/         # For the actual images to be analyzed
├── project_config.json
└── dataset_item_processing_config.json
```

### Step 2: Add Your Project Assets

After creating the project, add your files to the generated directories:

1.  **Reference Color Checker**: Place your ideal, canonical color checker image inside the `dataset/colorchecker/` directory.
2.  **Sample Color Checker**: For color correction, place an image of your color checker taken under the sample lighting conditions in a relevant folder, like `samples/test/colorchecker/`.
3.  **Training Images**: Place the images that will be used to calculate the target color range in the `dataset/training/` directory.
4.  **ArUco Reference (Optional)**: The script generates a default ArUco sheet. If you have a custom one, place it in `dataset/aruco/`.
5.  **Object Reference (Optional)**: For object alignment, place your template/reference image of the object in the `dataset/object/` directory.
6.  **Technical Drawing(s) (Optional)**: For background removal, place your drawing file(s) (e.g., a PNG with transparency) in the `dataset/drawing/` directory.
7.  **Analysis Images**: Place the images you want to analyze into a subfolder within the `samples/` directory (e.g., `samples/test/sample/`).

### Step 3: Update Project Configuration

Open the `project_config.json` file. The script will have pre-filled some paths. Ensure they point to the correct files relative to the project directory. A complete configuration looks like this:

```json
{
    "reference_color_checker_path": "dataset/colorchecker/colorchecker.png",
    "training_path": "dataset/training",
    "colorchecker_reference_for_project": [
        "samples/test/colorchecker/sample_checker.png"
    ],
    "object_reference_path": "dataset/object/object.png",
    "technical_drawing_path_layer_1": "dataset/drawing/mask_layer1.png",
    "technical_drawing_path_layer_2": null,
    "technical_drawing_path_layer_3": null,
    "aruco_reference_path": "dataset/aruco/default_aruco_reference.png",
    "aruco_marker_map": {},
    "aruco_output_size": [
        1000,
        1000
    ]
}
```

### Step 4: (Optional) Define Specific Sample Areas

If you want to define specific points on your training images for color extraction (instead of using the whole image), run the `dataset_gui.py` script found in `src/sample_manager/`:

```bash
python src/sample_manager/dataset_gui.py --project <your_project_name>
```

This launches a GUI to select points on each training image. Your selections are saved in `dataset_item_processing_config.json`.

### Step 5: Run the Analysis

Finally, you can run the main analysis pipeline using `streamlit_app.py` (GUI) or `src/main.py` (CLI). The pipeline includes several optional steps that are executed in a specific order if enabled.

**Pipeline Order:**
1.  Color Correction (`--color-alignment`)
2.  Geometrical Alignment (`--alignment`)
3.  **Object Alignment (`--object-alignment`)**
4.  **Masking / Background Removal (`--apply-mask`)**
5.  Blur (`--blur`)
6.  Color Analysis
7.  Symmetry Analysis (`--symmetry`)

**Note on Object Alignment**: When `--object-alignment` is used, the pipeline's default behavior is to use the `geometric_shape` method. This method attempts to find the object's contour and fit a pentagon or quadrilateral to it for a robust alignment, which is different from traditional feature-matching.

**Example Command (using all new features):**

This command runs the full pipeline, including both alignment steps and the new masking feature.

```bash
python src/main.py \
    --project <your_project_name> \
    --image data/projects/<your_project_name>/samples/test/sample/<your_image.png> \
    --debug \
    --color-alignment \
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
>>>>>>> Stashed changes
