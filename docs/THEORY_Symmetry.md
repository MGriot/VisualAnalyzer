# Theory: Symmetry Analysis

Symmetry is a fundamental concept in geometry and art, describing an object's invariance under certain transformations. The Visual Analyzer quantifies the symmetry of an object's 2D binary mask by testing its resilience to the four fundamental **isometries** (rigid transformations) of the Euclidean plane, providing a comprehensive analysis of its structural regularity.

-   **Theory (Group Theory & Isometries):** In mathematics, the set of transformations that leave an object unchanged forms a **symmetry group**. This analysis checks for invariance under discrete versions of these transformations. The analysis is performed on a binary mask of the object, and the symmetry "score" is a normalized similarity metric (typically `1 - L1 Norm`), where a score of 1.0 indicates perfect symmetry under that transformation.

### 1. Reflectional Symmetry

-   **Concept:** Also known as mirror symmetry, this is an object's invariance to being reflected ("flipped") across a line or axis.
-   **Process:**
    -   **Vertical Reflection:** The image mask is split vertically down the center. The right half is flipped horizontally and compared to the left half.
    -   **Horizontal Reflection:** The image is split horizontally. The bottom half is flipped vertically and compared to the top half.

### 2. Rotational Symmetry

-   **Concept:** An object has rotational symmetry if it looks the same after being rotated by some angle around a central point.
-   **Process:**
    1.  A rotation matrix `M` for a given angle (e.g., 90°, 180°) is computed.
    2.  This matrix is used to apply an **affine transformation** (a rotation) to the original image, producing a rotated version.
    3.  **Masking:** A crucial step is to create an `intersection_mask`. When an image is rotated on a discrete grid, the corners move out of the frame, creating empty areas. Comparing these empty areas would unfairly penalize the similarity score. The `intersection_mask` defines the region where both the original and rotated images contain valid pixels, ensuring the comparison is only performed on the overlapping area.
    4.  The similarity between the original and rotated image is calculated within this intersection mask.

### 3. Translational Symmetry

-   **Concept:** This symmetry describes a pattern that repeats itself when translated or "slid" by a certain distance. It is characteristic of friezes and wallpaper patterns.
-   **Process (Cross-Correlation via Template Matching):**
    1.  The analysis is based on **cross-correlation**, a fundamental signal processing technique for measuring the similarity of two signals as a function of the displacement of one relative to the other.
    2.  In 2D, this is implemented via **template matching** (`cv2.matchTemplate`).
    3.  A "template" (a small portion of the image, e.g., the left 25%) is selected.
    4.  This template is slid across a larger "search area" (the rest of the image).
    5.  At each location, the **Normalized Cross-Correlation Coefficient** is calculated. This score peaks where the template finds a region that is highly similar to itself.
    6.  The maximum coefficient found across the entire search is the score for translational symmetry. A high score indicates the presence of a repeating motif.

### 4. Glide-Reflection Symmetry

-   **Concept:** This is a composite isometry, less intuitive than the others. It consists of a reflection across an axis followed by a translation parallel to that same axis. The footprints left by walking in the snow are a classic example.
-   **Process:** The implementation directly follows the definition.
    1.  A portion of the image is selected as a template (e.g., the top half).
    2.  This template is **reflected** across an axis (e.g., flipped vertically).
    3.  This *new, flipped template* is then used in a **template matching** search across the other half of the image.
    4.  The resulting maximum correlation score indicates how well the reflected-and-translated part matches the other half, quantifying the glide-reflection symmetry.
