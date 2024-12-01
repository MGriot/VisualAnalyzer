import cv2
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy import stats
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle


def get_average_color(image: np.ndarray) -> np.ndarray:
    """Calculate the average color of an image."""
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    return average_color


def remove_outliers(
    data: list,
    method: str = "zscore",
    threshold: float = 3.0,
    confidence_level: float = 0.95,
) -> Tuple[list, list]:
    """Remove outliers from a list of data using the specified method."""
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
    """Calculate color limits (HSV) based on a dataset of images, removing outliers."""
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

    hues, _ = remove_outliers(
        hues,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )
    saturations, _ = remove_outliers(
        saturations,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )
    values, _ = remove_outliers(
        values,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )

    lower_limit = np.array([min(hues), min(saturations), min(values)], dtype=np.uint8)
    upper_limit = np.array([max(hues), max(saturations), max(values)], dtype=np.uint8)
    center = (np.mean(hues), np.mean(saturations), np.mean(values))

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = [
            cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0][0] / 255
            for h, s, v in zip(hues, saturations, values)
        ]
        ax.scatter(hues, saturations, values, c=colors, label="dataset")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")
        ax.set_zlabel("Value")
        plt.title("Color Distribution Scatter Plot")
        plt.legend()
        generate_color_space_plot(lower_limit, upper_limit, center)
        plt.show()

    return lower_limit, upper_limit, center


def generate_color_space_plot(
    lower_limit, upper_limit, center, gradient_height=25, num_lines=5
):
    """Generates and displays a color space plot with a customizable gradient,
    including individual channel gradients for HSV."""
    lower_rgb = cv2.cvtColor(np.uint8([[lower_limit]]), cv2.COLOR_HSV2BGR)[0][0]
    upper_rgb = cv2.cvtColor(np.uint8([[upper_limit]]), cv2.COLOR_HSV2BGR)[0][0]
    center_rgb = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2RGB)[0][0]

    gradient = np.linspace(lower_rgb, upper_rgb, 256) / 255
    gradient_resized = np.repeat(gradient.reshape(1, -1, 3), gradient_height, axis=0)
    gradient_stacked = np.vstack([gradient_resized] * num_lines)

    # Create individual HSV channel gradients
    hsv_lower = np.array([lower_limit])
    hsv_upper = np.array([upper_limit])
    h_gradient = np.linspace(hsv_lower[:, 0], hsv_upper[:, 0], 256).reshape(1, -1, 1)
    s_gradient = np.linspace(hsv_lower[:, 1], hsv_upper[:, 1], 256).reshape(1, -1, 1)
    v_gradient = np.linspace(hsv_lower[:, 2], hsv_upper[:, 2], 256).reshape(1, -1, 1)

    # Normalize and repeat for visualization
    h_gradient_vis = np.repeat(h_gradient / 179, gradient_height, axis=0)  # Hue is 0-179
    s_gradient_vis = np.repeat(s_gradient / 255, gradient_height, axis=0)
    v_gradient_vis = np.repeat(v_gradient / 255, gradient_height, axis=0)



    fig, axes = plt.subplots(4, 1, figsize=(10, 12))  # 4 subplots now

    axes[0].imshow(gradient_stacked)
    axes[0].axis("off")
    axes[0].set_title("Combined RGB Gradient")

    axes[1].imshow(h_gradient_vis, cmap='hsv', vmin=0, vmax=1)  # Use hsv cmap for Hue
    axes[1].axis("off")
    axes[1].set_title("Hue (H) Gradient")

    axes[2].imshow(s_gradient_vis, cmap='gray')
    axes[2].axis("off")
    axes[2].set_title("Saturation (S) Gradient")

    axes[3].imshow(v_gradient_vis, cmap='gray')
    axes[3].axis("off")
    axes[3].set_title("Value (V) Gradient")



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
    axes[0].add_patch(rect) # Add rect to the correct axes

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()


def blur_image(image: Image.Image, blur_radius: int = 2) -> Image.Image:
    """Blurs the input image using a Gaussian filter."""
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def plot_outliers(outlier_hues, outlier_saturations, outlier_values, ax):
    """Plots the outlier colors in a 3D scatter plot."""
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


def blur_image(image: Image.Image, blur_radius: int = 2) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def plot_outliers(outlier_hues, outlier_saturations, outlier_values, ax):
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
    """A class to find and highlight specific colors in images and video streams."""

    def __init__(self):
        """Initialize the ColorFinder with no base color."""
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
        """Finds a color in an image and highlights it."""
        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError("Color limits not set.")

        if exclude_transparent:
            image = image.convert("RGBA")
            bgra = np.array(image)
            if bgra.ndim == 4:
                non_transparent_pixels = bgra[bgra[:, :, 3] > 250]
                image = Image.fromarray(non_transparent_pixels.astype(np.uint8))
                image = image.convert("RGB")
                total_pixels = len(non_transparent_pixels)
            else:
                total_pixels = image.width * image.height
        else:
            total_pixels = image.width * image.height

        if apply_blur:
            image = blur_image(image, blur_radius)

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if adaptive_thresholding:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        if apply_morphology:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        matched_pixels = cv2.countNonZero(mask)
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

        inverted_mask = cv2.bitwise_not(mask)
        non_selected_pixels = cv2.bitwise_and(image, image, mask=inverted_mask)
        average_non_selected_color = get_average_color(non_selected_pixels)
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
        """Calculates the percentage of matched pixels."""
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
        """Finds and highlights a color in an image and calculates the percentage of pixels matching that color."""
        try:
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

    def plot_and_save_results(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        mask: np.ndarray,
        percentage: float,
        matched_pixels: int,
        total_pixels: int,
        output_dir: str = ".",
        save: bool = False,
        show: bool = True,
    ):
        """Plots and/or saves the original image, processed image, mask, pie chart, and bar chart."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert images to RGB for plotting
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Create a mask in RGB for visualization
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Plot original image
        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        plt.imshow(original_image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        # Plot processed image
        plt.subplot(2, 2, 2)
        plt.imshow(processed_image_rgb)
        plt.title("Processed Image")
        plt.axis("off")

        # Plot mask
        plt.subplot(2, 2, 3)
        plt.imshow(mask_rgb)
        plt.title("Mask (Black and White)")
        plt.axis("off")

        # Plot pie chart
        plt.subplot(2, 2, 4)
        labels = ["Matched Pixels", "Non-Matched Pixels"]
        sizes = [matched_pixels, total_pixels - matched_pixels]
        center_rgb = (
            cv2.cvtColor(
                np.uint8(
                    [[[int(self.center[0]), int(self.center[1]), int(self.center[2])]]]
                ),
                cv2.COLOR_HSV2RGB,
            )[0][0]
            / 255
        )
        colors = [center_rgb, "#41424C"]
        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
        plt.title("Pixel Distribution")

        # Save the plots
        if save:
            plt.savefig(os.path.join(output_dir, "results.png"), bbox_inches="tight")

        # Show the plots
        if show:
            plt.show()

        # Plot bar chart
        plt.figure(figsize=(8, 6))
        labels = ["Matched Pixels", "Non-Matched Pixels"]
        sizes = [matched_pixels, total_pixels - matched_pixels]

        colors = [center_rgb, "#41424C"]
        plt.bar(labels, sizes, color=colors)
        plt.title("Pixel Distribution")
        plt.ylabel("Number of Pixels")
        plt.ylim(0, total_pixels)

        # Save the bar chart
        if save:
            plt.savefig(os.path.join(output_dir, "bar_chart.png"), bbox_inches="tight")

        # Show the bar chart
        if show:
            plt.show()


# Save the ColorFinder object and color range
def save_color_finder(color_finder, filename="color_finder.pkl"):
    data = {
        "lower_limit": color_finder.lower_limit,
        "upper_limit": color_finder.upper_limit,
        "center": color_finder.center,
        "color_finder": color_finder,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)


# Load the ColorFinder object and color range
def load_color_finder(filename="color_finder.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    color_finder = data["color_finder"]
    color_finder.lower_limit = data["lower_limit"]
    color_finder.upper_limit = data["upper_limit"]
    color_finder.center = data["center"]
    return color_finder


if __name__ == "__main__":
    color_finder = ColorFinder()
    dataset_path_main = "img\\database"
    lower_limit_main, upper_limit_main, center_main = get_color_limits_from_dataset(
        dataset_path_main, show_plot=True, outlier_removal_method="grubbs"
    )
    color_finder.lower_limit = lower_limit_main
    color_finder.upper_limit = upper_limit_main
    color_finder.center = center_main

    image_path_main = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\.old\img\j.png"
    results_main = color_finder.find_color_and_percentage(
        image_path_main,
        exclude_transparent=True,
        adaptive_thresholding=False,
        apply_morphology=True,
        apply_blur=True,
    )

    if results_main:
        (
            processed_image_main,
            selected_colors_main,
            percentage_main,
            matched_pixels_main,
            total_pixels_main,
            average_non_selected_color_main,
            non_selected_pixel_count_main,
        ) = results_main

        original_image_main = cv2.imread(image_path_main)
        hsv_image_main = cv2.cvtColor(processed_image_main, cv2.COLOR_BGR2HSV)
        mask_main = cv2.inRange(
            hsv_image_main, color_finder.lower_limit, color_finder.upper_limit
        )

        # ... (print statements)

        color_finder.plot_and_save_results(
            original_image_main,
            processed_image_main,
            mask_main,
            percentage_main,
            matched_pixels_main,
            total_pixels_main,
            output_dir="output",
            save=False,
            show=True,
        )
