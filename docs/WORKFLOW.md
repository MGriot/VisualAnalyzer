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
│   ├── reference_color_checker.png
│   ├── project_color_checker.png
│   ├── training_images/
│   ├── drawing_layers/
│   └── object_reference.png
├── samples/
│   └── README.md
├── project_config.json
└── dataset_item_processing_config.json
```

### Step 2: Add Your Project Assets

After creating the project, add your files to the generated directories:

1.  **Reference Files**: Place your ideal `reference_color_checker.png`, the `project_color_checker.png` (shot in your project's lighting), the `object_reference.png`, and any drawing layer images into the `dataset/` sub-folders as named in the default config.
2.  **Training Images**: Place the images that will be used to calculate the target color range in the `dataset/training_images/` directory.
3.  **Analysis Images**: Place the images you want to analyze into the `samples/` directory. The README.md inside explains its purpose.

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