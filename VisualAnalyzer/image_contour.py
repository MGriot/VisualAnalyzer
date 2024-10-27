import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ImageProcessor import ImageProcessor


def image_contour(
    image_path,
    edge_detection_method="Canny",
    filter_type="GaussianBlur",
    filter_radius=4,
    use_matplotlib=False,
    debug=False,
    **kwargs,
):
    """
    This function applies an edge detection method to an image and optionally displays the result.

    Parameters:
    image_path (str): The path to the image file.
    edge_detection_method (str): The edge detection method to use ("Canny", "Sobel", or "Laplace").
    filter_type (str): The type of filter to apply before edge detection.
    filter_radius (int): The radius of the filter to apply.
    use_matplotlib (bool): Whether to use Matplotlib to display the image. If False, OpenCV is used.
    debug (bool): If True, displays the image with detected edges.

    **kwargs:
    For "Canny" method:
    canny_threshold1 (int): First threshold for the hysteresis procedure. Default is 100.
    canny_threshold2 (int): Second threshold for the hysteresis procedure. Default is 1500.

    For "Sobel" method:
    dx (int): Order of the derivative x. Default is 1.
    dy (int): Order of the derivative y. Default is 1.
    ksize (int): Size of the extended Sobel kernel; it must be 1, 3, 5, or 7. Default is 5.
    scale (int): Optional scale factor for the computed derivative values. Default is 1.
    delta (int): Optional delta value that is added to the results prior to storing them in dst. Default is 0.

    For "Laplacian" method:
    ddepth (int): Desired depth of the destination image. Default is cv2.CV_64F.
    ksize (int): Aperture size used to compute the second-derivative filters. Default is 1.
    scale (int): Optional scale factor for the computed Laplacian values. Default is 1.
    delta (int): Optional delta value that is added to the results prior to storing them in dst. Default is 0.
    borderType (int): Pixel extrapolation method. Default is cv2.BORDER_DEFAULT.

    Returns:
    edges (ndarray): The image data with edge detection applied.
    """
    # Instantiate the ImageProcessor class
    processor = ImageProcessor(image_path)

    # Apply the selected filter to the image
    processor.blur_filter(filter_type, radius=filter_radius)

    # Convert the PIL image to a NumPy array
    image = np.array(processor.img)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the selected edge detection method
    if edge_detection_method == "Canny":
        edges = cv2.Canny(
            gray_image,
            kwargs.get("canny_threshold1", 100),
            kwargs.get("canny_threshold2", 1500),
            apertureSize=5,
            L2gradient=True,
        )
    elif edge_detection_method == "Sobel":
        edges = cv2.Sobel(
            gray_image,
            cv2.CV_64F,
            dx=kwargs.get("dx", 1),
            dy=kwargs.get("dy", 1),
            ksize=kwargs.get("ksize", 5),
            scale=kwargs.get("scale", 1),
            delta=kwargs.get("delta", 0),
        )
    elif edge_detection_method == "Laplacian":
        edges = cv2.Laplacian(
            src=gray_image,
            ddepth=cv2.CV_64F,
            ksize=kwargs.get("ksize", 1),
            scale=kwargs.get("scale", 1),
            delta=kwargs.get("delta", 0),
            borderType=kwargs.get("borderType", cv2.BORDER_DEFAULT),
        )
    else:
        print(
            "Invalid edge detection method. Please specify one of the following: 'Canny', 'Laplacian', or 'Sobel'."
        )

    # If debug is True, display the image with detected edges
    if debug:
        if use_matplotlib:
            plt.imshow(edges, cmap="gray")
            plt.title(f"Edge Detection Method: {edge_detection_method}")
            plt.show()
            plt.close()
        else:
            cv2.imshow(f"Edge Detection Method: {edge_detection_method}", edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Return the image with edge detection applied
    return edges
