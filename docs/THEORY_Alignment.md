# Theory: Image Alignment

The Visual Analyzer employs a sophisticated two-stage alignment process to ensure that the object of interest is correctly oriented and scaled before analysis. This process corrects for both camera perspective and object position, utilizing distinct methodologies for each stage.

## Stage 1: Geometrical Alignment

This stage corrects for large-scale perspective distortion, such as when a photo is taken from an angle rather than head-on.

-   **Theory (Projective Geometry & Homography):** A camera performs a projective transformation, mapping the 3D world onto a 2D image plane. When viewing a planar surface (like a sheet of paper on a table) from an angle, the resulting image is a projectively distorted version of the original. The mathematical relationship between the original planar surface and its distorted image is a **homography**, a 3x3 matrix that maps points from one plane to another.

-   **ArUco Markers (Fiducial Markers):** To calculate the homography, we need to identify at least four corresponding points between the source (distorted) image and the reference (ideal) plane. **ArUco markers** are a type of **fiducial marker**—an artificial object placed in the scene to act as a point of reference. They are binary square patterns with unique IDs, making them easy to detect programmatically.

-   **Process:**
    1.  The `ArucoAligner` detects all ArUco markers present in both the source image and the ideal reference image (a digitally generated sheet with markers at known locations).
    2.  It finds the set of markers common to both images.
    3.  It extracts the pixel coordinates of the corners of these markers from both images. We now have a set of corresponding points.
    4.  Using these point correspondences, it employs a robust algorithm like **RANSAC (Random Sample Consensus)** to compute the homography matrix `H` that best maps the source points to the reference points.
    5.  Applying the inverse of this matrix `H` to the source image performs a **perspective warp**, transforming the distorted image so that it appears as if it were taken from a direct, head-on viewpoint.

## Stage 2: Object Alignment

After the overall scene perspective is corrected, this stage fine-tunes the alignment by focusing on the specific object of interest, aligning it with a reference or "template" version of the object.

-   **Theory (Contour-based Pose Estimation):** This method aligns two objects by matching the shape of their silhouettes or **contours**. By finding a geometric transformation that maps the contour of the object in the source image onto the contour of the reference object, we can align them with high precision.

-   **Process:**
    1.  **Shadow Removal & Binarization:** To reliably extract a clean contour, shadows and lighting variations are minimized. The default `clahe` method (**Contrast Limited Adaptive Histogram Equalization**) is applied to the L\* channel of the CIELAB color space to enhance local contrast without amplifying noise. The image is then binarized.
    2.  **Contour Detection:** The largest contour is extracted from both the source and reference images.
    3.  **Polygon Approximation:** The raw contour, which can have thousands of points, is simplified into a polygon using the **Ramer-Douglas-Peucker algorithm** (`cv2.approxPolyDP`). This algorithm reduces the number of vertices while keeping the overall shape intact.
    4.  **Pentagon-First Approach (5-Point Homography):** The logic first attempts to approximate the contour as a 5-sided polygon (a pentagon). A homography can be computed from five pairs of corresponding points. Using five points instead of four provides more constraints and can lead to a more stable and accurate transformation, especially for non-rigid or complex shapes.
    5.  **Quadrilateral Fallback (4-Point Homography):** If a stable pentagon cannot be resolved from both contours, the system falls back to a more standard method. It calculates the **minimum area bounding rectangle** for each contour and uses the four corners of these rectangles as the corresponding points to compute the homography.
    6.  **Final Warp:** The computed homography is applied to the source image to align the object precisely with the reference object.
