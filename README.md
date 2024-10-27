# VisualAnalyzer

## Introduction

VisualAnalyzer is a Python project that provides tools for analyzing and processing images. It includes functionalities for color clustering, image comparison, and various image processing techniques.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

The main script `main_cluster.py` demonstrates the usage of the classes in the `VisualAnalyzer` module. It performs the following steps:

1. **Image Processing:**
   - Loads an image using the `ImageProcessor` class.
   - Applies a blur filter to the image.
   - Resizes the image.
2. **Image Analysis:**
   - Analyzes the image using the `ImageAnalyzer` class.
   - Compares the image with an ideal image.
3. **Image Clustering:**
   - Clusters the image using the `ImageCluster` class.
   - Removes transparent pixels.
   - Plots the clustered image.
   - Extracts cluster information.
   - Plots the clustered image with high contrast.
   - Plots the images.
   - Plots the cluster pie chart.

## Technical Details

### ImageProcessor Class

The `ImageProcessor` class provides methods for processing images. It includes methods for:

- **`__init__(self, image_path)`:** Initializes the `ImageProcessor` object with the path of the image.
- **`load_image(self)`:** Loads the image from the specified path. If the image file does not exist or cannot be opened, an appropriate message will be printed.
- **`blur_filter(self, filter_type, **kwargs)`:** Applies a blur filter to the image. Available filters are: 'GaussianBlur', 'BoxBlur', 'MedianFilter'.
- **`increase_brightness(self, factor=1.2)`:** Increases the brightness of the image by a certain factor.
- **`increase_saturation(self, factor=1.2)`:** Increases the saturation of the image by a certain factor.
- **`increase_contrast(self, factor=1.2)`:** Increases the contrast of the image by a certain factor.
- **`resize(self, size=None, factor=None, maintain_aspect_ratio=False)`:** Resizes the image to the specified size or downsamples it by a certain factor.
- **`rotate(self, angle)`:** Rotates the image by a certain angle.
- **`crop(self, box)`:** Crops the image to the specified box.
- **`to_grayscale(self)`:** Converts the image to grayscale.
- **`normalize(self)`:** Normalizes the image by scaling the pixel values to the range 0-1.
- **`equalize(self)`:** Equalizes the image by applying histogram equalization.
- **`add_noise(self, radius=1.0)`:** Adds noise to the image by randomly redistributing pixel values within a certain neighborhood.
- **`flip(self, direction)`:** Flips the image horizontally or vertically.
- **`show_image(self, title="Image", use="Matplotlib")`:** Shows the image using Matplotlib or PIL.

### ImageAnalyzer Class

The `ImageAnalyzer` class provides methods for analyzing and comparing images. It includes methods for:

- **`__init__(self, img_input, ideal_img_input, ideal_img_processed=None)`:** Initializes the `ImageAnalyzer` object with the input image and the ideal image.
- **`load_image(self, input)`:** Loads an image from a file or a NumPy array.
- **`calculate_histogram(self, img)`:** Calculates the histogram of an image.
- **`compare_histograms(self, hist1, hist2)`:** Compares two histograms using correlation, chi-square, intersection, and Bhattacharyya distance.
- **`compare_images(self)`:** Compares the input image with the ideal image using histogram comparison and mean squared error (MSE).
- **`draw_histograms(self, img, title)`:** Draws the histograms of an image for each color channel (blue, green, red).
- **`analyze(self)`:** Analyzes the input image and compares it with the ideal image by drawing histograms and printing comparison results.

### ImageCluster Class

The `ImageCluster` class provides methods for performing color clustering on an image. It includes methods for:

- **`__init__(self, image_input)`:** Initializes the `ImageCluster` object with the image path or PIL Image object.
- **`remove_transparent(self, alpha_threshold=250)`:** Removes transparent pixels from the image based on the alpha threshold.
- **`filter_alpha(self)`:** Returns a boolean mask indicating non-transparent pixels.
- **`cluster(self, n_clusters=None, initial_clusters=None, merge_similar=False, threshold=10)`:** Performs KMeans clustering on the image's colors.
- **`create_clustered_image(self)`:** Creates a new image where each pixel is colored based on its cluster.
- **`create_clustered_image_with_ids(self)`:** Creates a new image where each pixel's value represents its cluster ID.
- **`extract_cluster_info(self)`:** Extracts information about each cluster, including color, pixel count, and percentage.
- **`calculate_brightness(self, color)`:** Calculates the brightness of a given color.
- **`plot_original_image(self, ax=None, max_size=(1024, 1024))`:** Plots the original image.
- **`plot_clustered_image(self, ax=None, max_size=(1024, 1024))`:** Plots the clustered image.
- **`plot_clustered_image_high_contrast(self, style='jet', show_percentage=True, dpi=100, ax=None)`:** Plots the clustered image with high contrast colors.
- **`plot_cluster_pie(self, ax=None, dpi=100)`:** Plots a pie chart showing the distribution of pixels in each cluster.
- **`plot_cluster_bar(self, ax=None, dpi=100)`:** Plots a bar chart showing the distribution of pixels in each cluster.
- **`plot_cumulative_barchart(self, ax=None, dpi=100)`:** Plots a cumulative bar chart showing the distribution of pixels in each cluster.
- **`plot_images(self, max_size=(1024, 1024))`:** Plots the original, clustered, and high contrast clustered images side-by-side.
- **`plot_image_with_grid(self, grid_size=50, color='white', max_size=(1024, 1024), dpi=100)`:** Plots the original image with a grid overlaid.
- **`save_plots(self)`:** Saves all generated plots to a directory.

### ColorFinder Class

The `ColorFinder` class provides methods for finding and highlighting specific colors in images and video streams. It uses color ranges in the HSV color space to identify regions of interest.

- **`__init__(self, base_color=(30, 255, 255), hue_percentage=3, saturation_percentage=70, value_percentage=70)`:** Initializes the `ColorFinder` object with a base color (in HSV) and percentage ranges for hue, saturation, and value. These ranges define the color limits used for identifying the target color.
- **`process_webcam(self)`:** Processes video from the webcam in real-time. It identifies regions in each frame that match the specified color limits and highlights them with rectangles. The lower and upper color limits are displayed on the screen for reference. Press 'q' to exit the webcam processing.
- **`process_image(self, image_path)`:** Processes a single image from the given path. It identifies regions in the image that match the specified color limits and highlights them with rectangles. The lower and upper color limits are displayed on the image for reference. The processed image and a mask showing the identified regions are displayed in separate windows.

**Example Usage:**

```python
from VisualAnalyzer.ColorFinder import ColorFinder

# Initialize ColorFinder with a yellow base color and specified percentage ranges
color_finder = ColorFinder(base_color=(30, 255, 255), hue_percentage=3, saturation_percentage=70, value_percentage=70)

# Process video from the webcam
color_finder.process_webcam()

# Process a single image
color_finder.process_image("path/to/your/image.jpg")
```

### colore_recognition

- **`get_colors(image_path, num_colors=5, show_chart=True)`:** Extracts the dominant colors from an image using KMeans clustering and optionally displays a bar chart of the color distribution.
- **`rgb_to_hex(rgb)`:** Converts an RGB color tuple to its hexadecimal representation.

### image_contour

- **`image_contour(image_path, edge_detection_method='Canny', filter_type='GaussianBlur', filter_radius=4, use_matplotlib=False, debug=False, **kwargs)`:** Applies an edge detection method (Canny, Sobel, or Laplacian) to an image after optionally applying a blur filter. It can display the results using either Matplotlib or OpenCV.

### Manual Image Alignment

The `ManualImageAligner` class provides a way to manually align two images by selecting corresponding points on both images. This is useful when automatic alignment methods fail or when precise alignment is required.

**`ManualImageAligner` Class:**

- **`__init__(self, image1_path, image2_path)`:** Initializes the `ManualImageAligner` with the paths to the two images to be aligned.
- **`select_points(self)`:** Opens two windows displaying the images and allows the user to select corresponding points on both images by clicking. At least four pairs of points are required for alignment.
- **`align_images(self)`:** Calculates the perspective transformation matrix based on the selected points and warps the first image to align with the second. Returns the aligned image, the transformed original image, and the transformation matrix.

**Example:**

```python
from VisualAnalyzer.ManualImageAligner import ManualImageAligner

image1_path = "path/to/image1.jpg"
image2_path = "path/to/image2.jpg"

aligner = ManualImageAligner(image1_path, image2_path)
aligner.select_points()
aligned_image, transformed_original, matrix = aligner.align_images()

if aligned_image is not None:
    # Display or save the aligned image
    cv2.imshow("Aligned Image", aligned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```


## Examples

The `main_cluster.py` script provides an example of how to use the `ImageProcessor`, `ImageAnalyzer`, and `ImageCluster` classes. 

**Example using `colore_recognition`:**

```python
from VisualAnalyzer.colore_recognition import get_colors

image_path = "path/to/your/image.jpg"
num_colors = 5
hex_colors, percentages = get_colors(image_path, num_colors)

print("Hex Colors:", hex_colors)
print("Percentages:", percentages)
```

**Example using `image_contour`:**

```python
from VisualAnalyzer.image_contour import image_contour

image_path = "path/to/your/image.jpg"
edges = image_contour(image_path, edge_detection_method="Canny", debug=True)
```

## Advanced Example

The following code demonstrates a more comprehensive example using the `ImageProcessor`, `ImageAnalyzer`, and `ImageCluster` classes, as shown in `main_cluster.py`:

```python
from VisualAnalyzer.ImageAnalyzer import ImageAnalyzer
from VisualAnalyzer.ImageCluster import ImageCluster
from VisualAnalyzer.ImageProcessor import ImageProcessor
import numpy as np

path = r"path/to/your/image.jpg"


# Uso della classe ImageProcessor
processor = ImageProcessor(path)
# processor.equalize()
processor.blur_filter("GaussianBlur", radius=5)
processor.show_image()
processor.resize(size=400, maintain_aspect_ratio=True)
processor.show_image()
blurred_img = processor.img

# Uso della classe ImageAnalyzer
analyzer = ImageAnalyzer(
    np.array(blurred_img),
    np.array(blurred_img),
)
analyzer.analyze()

# Uso della classe ImageCluster
c = ImageCluster(blurred_img)
c.remove_transparent()
c.cluster(n_clusters=3)
c.plot_clustered_image()
c.extract_cluster_info()
c.plot_clustered_image_high_contrast()
c.plot_images()
c.plot_cluster_pie()
