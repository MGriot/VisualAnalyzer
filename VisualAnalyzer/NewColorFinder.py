# pylint: disable=no-member
import cv2
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from scipy import stats
import os
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap  # Import for creating custom colormaps


def get_average_color(image: np.ndarray) -> np.ndarray:
    """
    Calculate the average color of an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The average color of the image.
    """
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    return average_color


def remove_outliers(
    data: list,
    method: str = "zscore",
    threshold: float = 3.0,
    confidence_level: float = 0.95,
) -> Tuple[list, list]:
    """
    Remove outliers from a list of data using the specified method.

    Parameters:
    data (list): The input data.
    method (str): The method to use for outlier removal. Options: 'zscore', 'iqr', 'stddev', 'confidence_interval', 'grubbs'. Default: 'zscore'.
    threshold (float): The threshold for outlier removal. Default: 3.0.
    confidence_level (float): The confidence level for the confidence interval method. Default: 0.95.

    Returns:
    tuple: A tuple containing the filtered data and the outliers.
    """
    if method == "zscore":
        z = np.abs(stats.zscore(data))
        filtered_data = [x for i, x in enumerate(data) if z[i] < threshold]
        outliers = [x for i, x in enumerate(data) if z[i] >= threshold]
    elif method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
    elif method == "stddev":
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
    elif method == "confidence_interval":
        mean = np.mean(data)
        sem = stats.sem(data)
        interval = stats.t.interval(
            confidence_level, len(data) - 1, loc=mean, scale=sem
        )
        filtered_data = [x for x in data if interval[0] <= x <= interval[1]]
        outliers = [x for x in data if x < interval[0] or x > interval[1]]
    elif method == "grubbs":
        # Grubbs' test implementation provided by the user
        data_array = np.array(data)
        mean = np.mean(data_array)
        std_dev = np.std(data_array, ddof=1)
        N = len(data_array)
        G_calculated = max(abs(data_array - mean)) / std_dev
        t_critical = stats.t.isf(0.025 / (2 * N), N - 2)
        G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(
            t_critical**2 / (N - 2 + t_critical**2)
        )
        if G_calculated > G_critical:
            # Find the index of the outlier
            outlier_index = np.argmax(abs(data_array - mean))
            outliers = [data_array[outlier_index]]
            filtered_data = np.delete(data_array, outlier_index).tolist()
        else:
            filtered_data = data
            outliers = []
    else:
        raise ValueError("Invalid outlier removal method specified.")
    return filtered_data, outliers


def get_color_limits_from_dataset(
    dataset_path: str,
    outlier_removal_method: str = "zscore",
    outlier_removal_threshold: float = 3.0,
    show_plot: bool = False,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Calculate color limits (HSV) based on a dataset of images, removing outliers.

    Parameters:
    dataset_path (str): The path to the dataset of images.
    outlier_removal_method (str): The method to use for outlier removal. Options: 'zscore', 'iqr', 'stddev', 'confidence_interval', 'grubbs'. Default: 'zscore'.
    outlier_removal_threshold (float): The threshold for outlier removal. Default: 3.0.
    show_plot (bool): Whether to display a 3D scatter plot of the colors in the dataset. Default: False.
    confidence_level (float): The confidence level for the confidence interval and Grubbs' test methods. Default: 0.95.

    Returns:
    tuple: The lower and upper color limits in HSV, and the center of the distribution.
    """
    hues, saturations, values = [], [], []
    for filename in os.listdir(dataset_path):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            average_color = get_average_color(hsv_image)
            hues.append(average_color[0])
            saturations.append(average_color[1])
            values.append(average_color[2])

    # Remove outliers
    hues, outlier_hues = remove_outliers(
        hues,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )
    saturations, outlier_saturations = remove_outliers(
        saturations,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )
    values, outlier_values = remove_outliers(
        values,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )

    lower_limit = np.array([min(hues), min(saturations), min(values)], dtype=np.uint8)
    upper_limit = np.array([max(hues), max(saturations), max(values)], dtype=np.uint8)
    center = (np.mean(hues), np.mean(saturations), np.mean(values))

    if show_plot:
        # Create the 3D scatter plot for inliers
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = [
            cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0] / 255
            for h, s, v in zip(hues, saturations, values)
        ]
        ax.scatter(hues, saturations, values, c=colors, label="Inliers")

        # Plot the outliers
        plot_outliers(outlier_hues, outlier_saturations, outlier_values, ax)

        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")
        ax.set_zlabel("Value")
        plt.title("Color Distribution Scatter Plot")
        plt.legend()

        # Create the color space rectangle plot
        generate_color_space_plot(lower_limit, upper_limit, center)

        plt.show()

    return lower_limit, upper_limit, center


def generate_color_space_plot(
    lower_limit, upper_limit, center, gradient_height=25, num_lines=5, output_dir="."
):
    """
    Generates and saves a color space plot with a customizable gradient.

    Args:
        lower_limit (np.ndarray): Lower HSV color limit.
        upper_limit (np.ndarray): Upper HSV color limit.
        center (tuple): Center HSV color.
        gradient_height (int, optional): Height of the gradient in pixels. Defaults to 25.
        num_lines (int, optional): Number of gradient lines to stack. Defaults to 5.
        output_dir (str, optional): Directory to save the plot. Defaults to current directory.
    """

    # Create a gradient from lower to upper limit
    lower_rgb = cv2.cvtColor(np.uint8([[lower_limit]]), cv2.COLOR_HSV2BGR)
    lower_rgb = lower_rgb[0][0]
    upper_rgb = cv2.cvtColor(np.uint8([[upper_limit]]), cv2.COLOR_HSV2BGR)
    upper_rgb = upper_rgb[0][0]
    center_rgb = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2RGB)
    center_rgb = center_rgb[0][0]

    gradient = np.linspace(lower_rgb, upper_rgb, 256)
    gradient = gradient / 255  # Normalize to 0-1 range

    # Create the gradient with desired height and lines
    gradient_resized = np.repeat(gradient.reshape(1, -1, 3), gradient_height, axis=0)
    gradient_stacked = np.vstack([gradient_resized] * num_lines)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(gradient_stacked)
    ax.axis("off")

    # Highlight the center color with a filled rectangle
    center_x = np.interp(center[0], [lower_limit[0], upper_limit[0]], [0, 256])
    rect_width = 2
    rect = Rectangle(
        (center_x - rect_width / 2, 0),
        rect_width,
        gradient_height * num_lines,
        linewidth=0,
        edgecolor="none",
        facecolor=center_rgb / 255,
    )
    ax.add_patch(rect)

    plt.show()


def blur_image(image: Image.Image, blur_radius: int = 2) -> Image.Image:
    """
    Blurs the input image using a Gaussian filter.

    Parameters:
        image (Image.Image): The input image.
        blur_radius (int): The radius for the Gaussian blur.

    Returns:
        Image.Image: The blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def plot_outliers(outlier_hues, outlier_saturations, outlier_values, ax):
    """
    Plots the outlier colors in a 3D scatter plot.

    Parameters:
        outlier_hues (list): List of outlier hue values.
        outlier_saturations (list): List of outlier saturation values.
        outlier_values (list): List of outlier value values.
        ax (Axes3D): The 3D axes object to plot on.
    """
    outlier_colors = [
        cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0] / 255
        for h, s, v in zip(outlier_hues, outlier_saturations, outlier_values)
    ]
    ax.scatter(
        outlier_hues,
        outlier_saturations,
        outlier_values,
        c=outlier_colors,
        marker="x",
        label="Outliers",
    )


class ColorFinder:
    """
    A class to find and highlight specific colors in images and video streams.
    """

    def __init__(self):
        """
        Initialize the ColorFinder with no base color. Color limits will be set later.
        """
        self.lower_limit = None
        self.upper_limit = None
        self.center = None

    def find_color(
        self,
        image: np.ndarray,
        exclude_transparent: bool = False,
        adaptive_thresholding: bool = False,
        apply_morphology: bool = False,
        apply_blur: bool = False,
        blur_radius: int = 2,
    ) -> Tuple[np.ndarray, Dict[str, Any], int, int, np.ndarray, int]:
        """
        Finds a color in an image and highlights it.

        Parameters:
            image (np.ndarray): The input image.
            exclude_transparent (bool): Whether to exclude transparent pixels from the calculation.
            adaptive_thresholding (bool): Whether to apply adaptive thresholding to the image.
            apply_morphology (bool): Whether to apply morphological operations to the mask.
            apply_blur (bool): Whether to apply Gaussian blur to the image.
            blur_radius (int): The radius for the Gaussian blur.

        Returns:
            tuple: A tuple containing the processed image with highlighted regions,
                the selected color (dictionary with different color spaces),
                the total number of pixels matching the color,
                the total number of pixels in the image,
                the average color of non-selected pixels,
                and the count of non-selected pixels.
                Returns None if the image cannot be loaded.
        """

        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError(
                "Color limits not set. Please use get_color_limits_from_dataset or get_color_limits_from_hsv to set the limits."
            )

        if exclude_transparent:
            # Convert to RGBA for alpha channel access
            image = image.convert("RGBA")

            # Extract alpha channel, filter for non-transparent pixels
            bgra = np.array(image)
            if bgra.ndim == 4:  # Check if the image has an alpha channel
                # Apply transparency filter before reshaping
                non_transparent_pixels = bgra[
                    bgra[:, :, 3] > 250
                ]  # Threshold for transparency

                # Convert back to image mode
                image = Image.fromarray(non_transparent_pixels.astype(np.uint8))
                image = image.convert("RGB")  # Convert to RGB for OpenCV
                total_pixels = len(non_transparent_pixels)
            else:
                # Handle images without alpha channel
                total_pixels = image.width * image.height
        else:
            # Handle non-transparent or images without alpha channel
            total_pixels = image.width * image.height

        # Apply blur if requested
        if apply_blur:
            image = blur_image(image, blur_radius)

        # Convert image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply adaptive thresholding if requested
        if adaptive_thresholding:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Create a mask based on color limits
            mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        # Apply morphological operations if requested
        if apply_morphology:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Calculate percentage of matching pixels
        matched_pixels = cv2.countNonZero(mask)

        # Find contours and draw rectangles
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

        selected_color_bgr = cv2.cvtColor(
            np.uint8([[self.lower_limit]]), cv2.COLOR_HSV2BGR
        )[0][0]
        selected_color_hsv = self.center
        selected_color_rgb = cv2.cvtColor(
            np.uint8([[selected_color_bgr]]), cv2.COLOR_BGR2RGB
        )[0][0]

        selected_colors = {
            "BGR": selected_color_bgr,
            "HSV": selected_color_hsv,
            "RGB": selected_color_rgb,
        }

        # Calculate average color of non-selected pixels
        inverted_mask = cv2.bitwise_not(mask)
        non_selected_pixels = cv2.bitwise_and(image, image, mask=inverted_mask)
        average_non_selected_color = get_average_color(non_selected_pixels)

        # Count non-selected pixels
        non_selected_pixel_count = cv2.countNonZero(inverted_mask)

        return (
            image,
            selected_colors,
            matched_pixels,
            total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
        )

    def calculate_percentage(self, matched_pixels: int, total_pixels: int) -> float:
        """
        Calculates the percentage of matched pixels.

        Parameters:
            matched_pixels (int): Number of matched pixels.
            total_pixels (int): Total number of pixels.

        Returns:
            float: The percentage of matched pixels.
        """
        return (matched_pixels / total_pixels) * 100

    def find_color_and_percentage(
        self,
        image_path: str,
        exclude_transparent: bool = False,
        adaptive_thresholding: bool = False,
        apply_morphology: bool = False,
        apply_blur: bool = False,
        blur_radius: int = 2,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, int, int, np.ndarray, int]:
        """
        Finds and highlights a color in an image and calculates the percentage of pixels matching that color.

        Parameters:
            image_path (str): The path to the input image.
            exclude_transparent (bool): Whether to exclude transparent pixels from the calculation.
            adaptive_thresholding (bool): Whether to apply adaptive thresholding to the image.
            apply_morphology (bool): Whether to apply morphological operations to the mask.
            apply_blur (bool): Whether to apply Gaussian blur to the image.
            blur_radius (int): The radius for the Gaussian blur.

        Returns:
            tuple: A tuple containing the processed image with highlighted regions,
                the selected color (dictionary with different color spaces),
                the percentage of pixels matching the color,
                the total number of pixels matching the color,
                the total number of pixels in the image,
                the average color of non-selected pixels,
                and the count of non-selected pixels.
                Returns None if the image cannot be loaded.
        """

        try:
            # Open image with Pillow, handle potential errors
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error: Could not load image at {image_path} ({e})")
            return None

        (
            image,
            selected_colors,
            matched_pixels,
            total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
        ) = self.find_color(
            image,
            exclude_transparent,
            adaptive_thresholding,
            apply_morphology,
            apply_blur,
            blur_radius,
        )
        percentage = self.calculate_percentage(matched_pixels, total_pixels)

        return (
            image,
            selected_colors,
            percentage,
            matched_pixels,
            total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
        )


def test_threshold_values(
    image_path: str,
    lower_limit: np.ndarray,
    upper_limit: np.ndarray,
    exclude_transparent: bool = False,
    adaptive_thresholding: bool = False,
    apply_morphology: bool = False,
    apply_blur: bool = False,
    blur_radius: int = 2,
):
    """
    Tests different threshold values and displays the results.

    Parameters:
        image_path (str): The path to the input image.
        lower_limit (np.ndarray): Lower HSV color limit.
        upper_limit (np.ndarray): Upper HSV color limit.
        exclude_transparent (bool): Whether to exclude transparent pixels from the calculation.
        adaptive_thresholding (bool): Whether to apply adaptive thresholding to the image.
        apply_morphology (bool): Whether to apply morphological operations to the mask.
        apply_blur (bool): Whether to apply Gaussian blur to the image.
        blur_radius (int): The radius for the Gaussian blur.
    """
    try:
        # Open image with Pillow
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error: Could not load image at {image_path} ({e})")
        return

    # Convert image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a ColorFinder instance and set the color limits
    color_finder = ColorFinder()
    color_finder.lower_limit = lower_limit
    color_finder.upper_limit = upper_limit

    # Find the color and calculate the percentage
    (
        processed_image,
        _,
        _,
        _,
        _,
        average_non_selected_color,
        non_selected_pixel_count,
    ) = color_finder.find_color(
        image,
        exclude_transparent,
        adaptive_thresholding,
        apply_morphology,
        apply_blur,
        blur_radius,
    )

    # Display the original image, mask, and processed image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    cv2.imshow("Original Image", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Average color of non-selected pixels: {average_non_selected_color}")
    print(f"Count of non-selected pixels: {non_selected_pixel_count}")


if __name__ == "__main__":
    # Example usage:
    color_finder = ColorFinder()

    # Set color limits using either method:
    # 1. From dataset:
    dataset_path = "img\\database"  # "path/to/your/dataset"
    lower_limit, upper_limit, center = get_color_limits_from_dataset(
        dataset_path, show_plot=True, outlier_removal_method="grubbs"
    )
    color_finder.lower_limit = lower_limit
    color_finder.upper_limit = upper_limit
    color_finder.center = center

    # 2. From HSV:
    # base_color = (30, 255, 255)
    # hue_percentage = 3
    # saturation_percentage = 70
    # value_percentage = 70
    # lower_limit, upper_limit, center = color_finder.get_color_limits_from_hsv(
    #     base_color, hue_percentage, saturation_percentage, value_percentage
    # )
    # print(f"Lower Limit: {lower_limit}")
    # print(f"Upper Limit: {upper_limit}")
    # print(f"Center: {center}")

    image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\.old\img\j.png"
    # color_finder.process_webcam()
    # color_finder.process_image(image_path)
    results = color_finder.find_color_and_percentage(
        image_path,
        exclude_transparent=True,
        adaptive_thresholding=True,
        apply_morphology=True,
        apply_blur=True,
    )

    if results:
        (
            processed_image,
            selected_colors,
            percentage,
            matched_pixels,
            total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
        ) = results
        print(f"Selected Colors: {selected_colors}")
        print(f"Percentage of matched pixels: {percentage:.2f}%")
        print(f"Number of matched pixels: {matched_pixels}")
        print(f"Average color of non-selected pixels: {average_non_selected_color}")
        print(f"Count of non-selected pixels: {non_selected_pixel_count}")
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Test different threshold values
    test_threshold_values(
        image_path,
        lower_limit,
        upper_limit,
        exclude_transparent=True,
        adaptive_thresholding=False,
        apply_morphology=False,
        apply_blur=True,
    )
