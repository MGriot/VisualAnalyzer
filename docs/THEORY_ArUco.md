# Theory: ArUco Markers

ArUco markers are a type of **fiducial marker** – artificial patterns designed for easy and robust detection by computer vision systems. They are square, binary patterns with a thick black border and an inner binary matrix that encodes a unique ID. This design makes them highly reliable for various tasks, even under challenging imaging conditions.

## Purpose in Visual Analyzer

In the Visual Analyzer, ArUco markers serve two primary purposes:

1.  **Geometrical Alignment (Perspective Correction):** They act as reliable reference points to correct for perspective distortions in images. When a camera captures a flat object from an angle, the object appears warped. By placing ArUco markers at known positions on a reference plane, the system can detect these markers in the distorted image and calculate a **homography** (a mathematical transformation). This homography is then used to "unwarp" the image, making it appear as if it was captured head-on. This is crucial for accurate measurements and comparisons.

2.  **Color Checker Patch Detection:** For color correction, ArUco markers can be embedded directly onto a color checker. This provides a highly robust method for precisely locating the color patches, even if the color checker itself is viewed under perspective distortion. Once the markers are detected, the color checker can be aligned, and the individual color patches can be extracted with high accuracy.

## Generation of ArUco Sheets

The Visual Analyzer includes utilities to generate printable ArUco marker sheets. These sheets are designed to be printed and used as physical reference targets in your imaging setup.

Key features of the generation process include:

*   **Customizable Layouts:** Markers can be arranged in a grid pattern or placed specifically at the corners of a page (e.g., for a simple alignment target).
*   **Adjustable Size and Density:** You can specify the size of individual markers and the overall density on the page.
*   **High Resolution for Printing:** Sheets are generated at a high DPI (dots per inch) to ensure print quality, making them suitable for precise applications.
*   **Unique IDs:** Each marker has a unique identifier, allowing the system to distinguish between them and establish correspondences.

The generation process typically involves:
1.  Defining the desired marker IDs and the ArUco dictionary (a set of predefined marker patterns).
2.  Calculating the pixel dimensions for the sheet based on physical size (e.g., A4 paper) and DPI.
3.  Drawing each marker onto a digital canvas.
4.  Optionally adding text labels (like marker IDs) for human readability.
5.  Saving the final sheet as an image file (e.g., PNG).

This ensures that you can create custom, high-quality ArUco targets tailored to your specific experimental setup.
