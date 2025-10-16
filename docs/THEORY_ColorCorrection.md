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
-   **Result:** This is the most precise method, as it corrects for perspective distortion before patch extraction.

### Tier 2: Manual GUI Fallback

-   **Theory (User-in-the-Loop):** If automatic marker detection fails, the system falls back to a manual, user-guided approach.
-   **Process:**
    1.  A GUI window automatically opens, displaying the color checker image.
    2.  The user is prompted to click on the four corners of the color checker grid.
    3.  These four manually selected points are used to compute the homography matrix and perform the perspective warp.
-   **Result:** This provides a robust fallback that guarantees the alignment can be completed even when automatic methods fail due to poor lighting, obstructions, or missing markers.

### Tier 3: Simple Grid Sampling (Primary Detection)

-   **Theory:** After alignment, the primary method for patch detection assumes the checker is now a perfect, oriented rectangle and uses a simple grid overlay.
-   **Process:** The aligned image is divided into a 4x6 grid. To avoid edge artifacts or the black borders between patches, the color is not sampled from the entire grid cell. Instead, the average color is calculated from only the central 50% of each cell.
-   **Result:** This method is extremely fast and accurate, provided the initial alignment was successful.

### Tier 4: Advanced Fallbacks (Contour & Mask-Based)

-   **Theory:** If the simple grid sampling fails (which is unlikely after a successful alignment), the system falls back to more complex contour-based methods. These methods analyze the image to find the outlines of the 24 patches, using properties like area and shape to identify them.
-   **Process:** These fallbacks use adaptive thresholding and contour analysis to find the patches, and can include mask-based extraction for higher precision.
-   **Result:** These methods are more computationally intensive but provide a final layer of robustness.

## Phase 2: Patch Matching & Alignment

After Phase 1, we have two lists of colors: one from the source image and one from the reference. We cannot assume they are in the same order due to potential detection errors. Phase 2 intelligently matches them.

-   **Theory (The Assignment Problem & Perceptual Color Models):** The task of matching the detected source colors to the correct reference colors is a classic combinatorial optimization challenge known as the **assignment problem**. To solve it effectively, we must compare colors in a way that mimics human perception.
-   **Process:**
    1.  **CIELAB Color Space:** All RGB colors are converted to the **CIELAB (L\*a\*b\*) color space**. Unlike RGB, CIELAB is designed to be *perceptually uniform*. This means the geometric distance (Euclidean distance) between any two colors in this space is directly proportional to the humanly perceived difference between them. This distance is called **Delta E (ΔE\*)**.
    2.  **Cost Matrix Construction:** A cost matrix is created where the entry at `(i, j)` is the Delta E value between `source_color[i]` and `reference_color[j]`. This matrix represents the "cost" of assigning any source patch to any reference patch.
    3.  **The Hungarian Algorithm:** The **Hungarian algorithm** (or more generally, algorithms that solve the linear sum assignment problem, like the one in `scipy.optimize`) is used on the cost matrix. It efficiently finds the set of one-to-one pairings that minimizes the total cost (i.e., minimizes the total perceptual color difference across all matches).
-   **Result:** This produces two new lists of colors, `aligned_source_colors` and `aligned_reference_colors`, where we can be highly confident that `aligned_source_colors[i]` corresponds to `aligned_reference_colors[i]`. These perfectly matched lists are then used to calculate the final color correction matrix.
