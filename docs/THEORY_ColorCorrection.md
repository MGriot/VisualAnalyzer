# Theory: Color Correction

The color correction module is one of the most critical components of the Visual Analyzer. Its goal is to standardize colors in a photograph by comparing a color checker in the image (the *source*) to an ideal, digitally-perfect version (the *reference*). This process involves two major phases: **Patch Detection**, which employs a robust, multi-tiered approach to locate color patches, and **Patch Matching**, which intelligently aligns detected patches to their reference counterparts.

## Phase 1: Patch Detection

This phase is responsible for locating the 24 individual color patches on the color checker within the source and reference images. Since the source image can be subject to perspective distortion, poor lighting, and other photographic issues, a robust, tiered fallback system is used.

### Tier 1: ArUco-Based Alignment

-   **Theory (Fiducial Markers & Homography):** This method leverages **fiducial markers** (in this case, ArUco markers) printed on the color checker. These markers have unique, easily identifiable patterns. By finding the pixel coordinates of the marker corners in the distorted source image and knowing their ideal coordinates on the reference image, we can compute a **homography matrix**. A homography is a 3x3 transformation matrix from projective geometry that maps points from one plane to another.
-   **Process:**
    1.  The system detects all ArUco markers in the source image.
    2.  It matches them to the markers on the ideal reference checker.
    3.  The corresponding corner points are used to calculate the homography matrix `H`.
    4.  This matrix is used to apply a perspective warp to the source image, effectively "straightening out" the color checker to match the reference.
    5.  Because the checker is now perfectly aligned, the 24 patches can be extracted by simply slicing a known grid.
-   **Result:** This is the most precise method, as it corrects for perspective distortion before patch extraction.

### Tier 2: Robust OpenCV (Hough Transform)

-   **Theory (Edge & Line Detection):** If ArUco markers are not found, the system falls back to analyzing the structure of the color checker itself. The boundaries between color patches form a grid. This grid can be identified by finding strong edges and inferring the lines that form them.
-   **Process:**
    1.  The image is converted to grayscale.
    2.  **Canny edge detection** is applied to find high-gradient pixels (edges).
    3.  A **Hough Line Transform** is then used on the edge map. This algorithm can detect lines by converting points from Cartesian space to a parameter space (Hough space) and finding intersections.
    4.  The detected lines are clustered into horizontal and vertical groups to define the rows and columns of the grid.
    5.  Patches are extracted from the cells of this detected grid.
-   **Result:** This method is effective for well-lit color checkers, even those without physical gaps between patches.

### Tier 3: YOLOv8 Object Detection

-   **Theory (Convolutional Neural Networks):** If the grid structure is not clear enough for the Hough Transform, a deep learning approach is used. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. A pre-trained YOLOv8 model is used to directly identify the location of each color patch.
-   **Process:** The image is passed to the YOLOv8 model, which returns the bounding boxes for all color patches it recognizes.
-   **Result:** This method is robust to some rotation and distortion but is dependent on the quality and variety of the data the model was trained on.

### Tier 4: Simple Grid Slicing

-   **Theory:** This is the simplest fallback and assumes the color checker is the largest contour in the image and is oriented correctly.
-   **Process:** It finds the bounding box of the largest object and divides it into a uniform 4x6 grid.
-   **Result:** This method is fast but not robust to any rotation or perspective distortion.

## Phase 2: Patch Matching & Alignment

After Phase 1, we have two lists of colors: one from the source image and one from the reference. We cannot assume they are in the same order due to potential detection errors. Phase 2 intelligently matches them.

-   **Theory (The Assignment Problem & Perceptual Color Models):** The task of matching the detected source colors to the correct reference colors is a classic combinatorial optimization challenge known as the **assignment problem**. To solve it effectively, we must compare colors in a way that mimics human perception.
-   **Process:**
    1.  **CIELAB Color Space:** All RGB colors are converted to the **CIELAB (L\*a\*b\*) color space**. Unlike RGB, CIELAB is designed to be *perceptually uniform*. This means the geometric distance (Euclidean distance) between any two colors in this space is directly proportional to the humanly perceived difference between them. This distance is called **Delta E (ΔE\*)**.
    2.  **Cost Matrix Construction:** A cost matrix is created where the entry at `(i, j)` is the Delta E value between `source_color[i]` and `reference_color[j]`. This matrix represents the "cost" of assigning any source patch to any reference patch.
    3.  **The Hungarian Algorithm:** The **Hungarian algorithm** (or more generally, algorithms that solve the linear sum assignment problem, like the one in `scipy.optimize`) is used on the cost matrix. It efficiently finds the set of one-to-one pairings that minimizes the total cost (i.e., minimizes the total perceptual color difference across all matches).
-   **Result:** This produces two new lists of colors, `aligned_source_colors` and `aligned_reference_colors`, where we can be highly confident that `aligned_source_colors[i]` corresponds to `aligned_reference_colors[i]`. These perfectly matched lists are then used to calculate the final color correction matrix.
