import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy import stats
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

class ColorFinder:
    """A class to find and highlight specific colors in images and video streams."""

    def __init__(self):
        """Initialize the ColorFinder with no base color."""
        self.lower_limit = None
        self.upper_limit = None
        self.center = None
        self.original_image = None
        self.processed_image = None
        self.mask = None
        self.initial_mask = None
        self.matched_pixels = None
        self.total_pixels = None

    def find_color(
        self,
        image: np.ndarray,
        exclude_transparent: bool = False,
        adaptive_thresholding: bool = False,
        apply_morphology: bool = False,
        apply_blur: bool = False,
        blur_radius: int = 2,
        apply_pixelation: bool = False,
        pixelation_size: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, Any], int, int, np.ndarray, int]:
        """Finds a color in an image and highlights it."""
        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError("Color limits not set.")

        if exclude_transparent:
            image = image.convert("RGBA")
            bgra = np.array(image)
            if bgra.shape[2] == 4:
                non_transparent_pixels = bgra[bgra[:, :, 3] > 240]
                total_pixels = len(non_transparent_pixels)
                plt.figure()
                plt.title("Alpha Channel")
                plt.imshow(bgra[:, :, 3], cmap="gray")
                plt.show()
            else:
                total_pixels = image.width * image.height
        else:
            total_pixels = image.width * image.height
        print(f"total pixels : {total_pixels}")
        if apply_blur:
            image = blur_image(image, blur_radius)
        if apply_pixelation:
            image = pixelate_image(image, pixelation_size)
        img = np.array(image)
        if img.shape[2] == 4:
            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except:
                image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 3 and image.dtype == np.uint8:
            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except:
                image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if adaptive_thresholding:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        self.initial_mask = mask.copy()

        if apply_morphology:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        matched_pixels = cv2.countNonZero(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
        plt.figure()
        plt.title("Mask before inversion")
        plt.imshow(mask, cmap="gray")
        plt.show()
        inverted_mask = cv2.bitwise_not(mask)
        plt.figure()
        plt.title("Inverted Mask before setting transparent pixels to zero")
        plt.imshow(inverted_mask, cmap="gray")
        plt.show()
        if exclude_transparent:
            unique_alpha_values = np.unique(bgra[:, :, 3])
            plt.figure()
            plt.title("Alpha Channel")
            plt.imshow(bgra[:, :, 3], cmap="gray")
            plt.show()
            inverted_mask[bgra[:, :, 3] <= 250] = 0
        plt.figure()
        plt.title("Inverted Mask after setting transparent pixels to zero")
        plt.imshow(inverted_mask, cmap="gray")
        plt.show()
        non_selected_pixels = cv2.bitwise_and(image, image, mask=inverted_mask)
        average_non_selected_color = get_average_color(non_selected_pixels)
        non_selected_pixel_count = cv2.countNonZero(inverted_mask)
        print(f"Matched pixels : {matched_pixels}")
        return (
            image,
            selected_colors,
            matched_pixels,
            total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
            mask,
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
        apply_pixelation: bool = False,
        pixelation_size: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, int, int, np.ndarray, int]:
        """Finds and highlights a color in an image and calculates the percentage of pixels matching that color."""
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error: Could not load image at {image_path} ({e})")
            return None

        (
            self.processed_image,
            selected_colors,
            self.matched_pixels,
            self.total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
            self.mask,
        ) = self.find_color(
            image,
            exclude_transparent,
            adaptive_thresholding,
            apply_morphology,
            apply_blur,
            blur_radius,
            apply_pixelation,
            pixelation_size,
        )
        percentage = self.calculate_percentage(self.matched_pixels, self.total_pixels)
        self.original_image = cv2.imread(image_path)

        return (
            self.processed_image,
            selected_colors,
            percentage,
            self.matched_pixels,
            self.total_pixels,
            average_non_selected_color,
            non_selected_pixel_count,
            self.mask,
        )

    def plot_and_save_results(
        self,
        output_dir: str = ".",
        save: bool = False,
        show: bool = True,
    ):
        """Plots and/or saves the original image, processed image, mask, pie chart, and bar chart."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.original_image.shape[2] == 4:  # Check if the image has an alpha channel
            original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGRA2RGBA)
        else:
            original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        processed_image_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(self.initial_mask, cv2.COLOR_GRAY2RGB)

        custom_plot = np.zeros_like(mask_rgb)
        custom_plot[self.initial_mask == 255] = [0, 255, 0]

        if self.original_image.shape[2] == 4:
            alpha_channel = self.original_image[:, :, 3]
            custom_plot[alpha_channel < 250] = [255, 255, 255]

        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        plt.imshow(original_image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(processed_image_rgb)
        plt.title("Processed Image")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(custom_plot)
        plt.title("Mask (Black, White and Green)")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        labels = ["Matched Pixels", "Non-Matched Pixels"]
        sizes = [self.matched_pixels, self.total_pixels - self.matched_pixels]
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

        if save:
            plt.savefig(os.path.join(output_dir, "results.png"), bbox_inches="tight")

        if show:
            plt.show()

        plt.figure(figsize=(8, 6))
        labels = ["Matched Pixels", "Non-Matched Pixels"]
        sizes = [self.matched_pixels, self.total_pixels - self.matched_pixels]
        colors = [center_rgb, "#41424C"]
        plt.bar(labels, sizes, color=colors)
        plt.title("Pixel Distribution")
        plt.ylabel("Number of Pixels")
        plt.ylim(0, self.total_pixels)

        if save:
            plt.savefig(os.path.join(output_dir, "bar_chart.png"), bbox_inches="tight")

        if show:
            plt.show()

        plt.figure()
        plt.title("Custom Plot: Green=Matched, White=Transparent, Black=Non-Matched")
        plt.imshow(custom_plot)
        plt.axis("off")
        if save:
            plt.savefig(os.path.join(output_dir, "custom_plot.png"))
        if show:
            plt.show()

def get_average_color(image: np.ndarray) -> np.ndarray:
    """Calculate the average color of an image."""
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    return average_color

def remove_outliers(
    data: List[Tuple[float, float, float]],
    method: str = "zscore",
    threshold: float = 3.0,
    confidence_level: float = 0.95,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Remove outliers from a list of data using the specified method.

    Parameters:
    - data: List of tuples containing the data points (hue, saturation, value).
    - method: Method to use for outlier detection ("zscore", "iqr", "stddev", "confidence_interval", "grubbs").
    - threshold: Threshold value for outlier detection.
    - confidence_level: Confidence level for the confidence interval method.

    Returns:
    - filtered_data: List of tuples containing the data points after outlier removal.
    - outliers: List of tuples containing the detected outliers.
    """
    data_array = np.array(data)

    if method == "zscore":
        z = np.abs(stats.zscore(data_array, axis=0))
        filtered_data = [
            tuple(x) for i, x in enumerate(data) if np.all(z[i] < threshold)
        ]
        outliers = [tuple(x) for i, x in enumerate(data) if np.any(z[i] >= threshold)]
    elif method == "iqr":
        q1 = np.percentile(data_array, 25, axis=0)
        q3 = np.percentile(data_array, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = [
            tuple(x)
            for x in data
            if np.all(lower_bound <= x) and np.all(x <= upper_bound)
        ]
        outliers = [
            tuple(x) for x in data if np.any(x < lower_bound) or np.any(x > upper_bound)
        ]
    elif method == "stddev":
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        filtered_data = [
            tuple(x)
            for x in data
            if np.all(lower_bound <= x) and np.all(x <= upper_bound)
        ]
        outliers = [
            tuple(x) for x in data if np.any(x < lower_bound) or np.any(x > upper_bound)
        ]
    elif method == "confidence_interval":
        mean = np.mean(data_array, axis=0)
        sem = stats.sem(data_array, axis=0)
        interval = stats.t.interval(
            confidence_level, len(data_array) - 1, loc=mean, scale=sem
        )
        filtered_data = [
            tuple(x)
            for x in data
            if np.all(interval[0] <= x) and np.all(x <= interval[1])
        ]
        outliers = [
            tuple(x) for x in data if np.any(x < interval[0]) or np.any(x > interval[1])
        ]
    elif method == "grubbs":
        mean = np.mean(data_array, axis=0)
        std_dev = np.std(data_array, ddof=1, axis=0)
        N = len(data_array)
        G_calculated = np.max(np.abs(data_array - mean), axis=0) / std_dev
        t_critical = stats.t.isf(0.025 / (2 * N), N - 2)
        G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(
            t_critical**2 / (N - 2 + t_critical**2)
        )
        if np.any(G_calculated > G_critical):
            outlier_indices = np.argmax(np.abs(data_array - mean), axis=0)
            outliers = [tuple(data_array[outlier_indices])]
            filtered_data = [
                tuple(x) for i, x in enumerate(data) if i not in outlier_indices
            ]
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

    # Remove outliers consistently across all channels
    all_channels_combined = list(zip(hues, saturations, values))
    all_channels_combined_filtered, all_channels_outliers = remove_outliers(
        all_channels_combined,
        method=outlier_removal_method,
        threshold=outlier_removal_threshold,
        confidence_level=confidence_level,
    )

    hues_filtered, saturations_filtered, values_filtered = zip(
        *all_channels_combined_filtered
    )
    try:
        hues_outliers, saturations_outliers, values_outliers = zip(
            *all_channels_outliers
        )
    except:
        hues_outliers, saturations_outliers, values_outliers = None, None, None
        print("No outliers found")

    lower_limit = np.array(
        [min(hues_filtered), min(saturations_filtered), min(values_filtered)],
        dtype=np.uint8,
    )
    upper_limit = np.array(
        [max(hues_filtered), max(saturations_filtered), max(values_filtered)],
        dtype=np.uint8,
    )
    center = (
        np.mean(hues_filtered),
        np.mean(saturations_filtered),
        np.mean(values_filtered),
    )

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot filtered data with star markers
        colors_filtered = [
            cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2RGB)[0][0] / 255
            for hue, sat, val in zip(
                hues_filtered, saturations_filtered, values_filtered
            )
        ]

        ax.scatter(
            hues_filtered,
            saturations_filtered,
            values_filtered,
            c=colors_filtered,
            marker="*",
            label="Filtered Data",
            alpha=0.6,
        )

        # Plot outliers with triangle markers
        try:
            colors_outliers = [
                cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2RGB)[0][0]
                / 255
                for hue, sat, val in zip(
                    hues_outliers, saturations_outliers, values_outliers
                )
            ]

            ax.scatter(
                hues_outliers,
                saturations_outliers,
                values_outliers,
                c=colors_outliers,
                marker="^",
                label="Outliers",
                alpha=0.6,
            )
        except:
            print("No outliers can be plot")

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
    h_gradient_vis = np.repeat(
        h_gradient / 179, gradient_height, axis=0
    )  # Hue is 0-179
    s_gradient_vis = np.repeat(s_gradient / 255, gradient_height, axis=0)
    v_gradient_vis = np.repeat(v_gradient / 255, gradient_height, axis=0)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))  # 4 subplots now

    axes[0].imshow(gradient_stacked)
    axes[0].axis("off")
    axes[0].set_title("Combined RGB Gradient")

    axes[1].imshow(h_gradient_vis, cmap="hsv", vmin=0, vmax=1)  # Use hsv cmap for Hue
    axes[1].axis("off")
    axes[1].set_title("Hue (H) Gradient")

    axes[2].imshow(s_gradient_vis, cmap="gray")
    axes[2].axis("off")
    axes[2].set_title("Saturation (S) Gradient")

    axes[3].imshow(v_gradient_vis, cmap="gray")
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
    axes[0].add_patch(rect)  # Add rect to the correct axes

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def blur_image(image: Image.Image, blur_radius: int = 2) -> Image.Image:
    """Blurs the input image using a Gaussian filter."""
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def pixelate_image(image: Image.Image, pixelation_size: int = 10) -> Image.Image:
    """Pixelates the input image by segmenting it into squares and replacing each square with its dominant color."""
    image = image.resize(
        (image.width // pixelation_size, image.height // pixelation_size),
        Image.NEAREST,
    )
    image = image.resize((image.width * pixelation_size, image.height * pixelation_size), Image.NEAREST)
    return image

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

def save_color_finder(color_finder, filename="color_finder.pkl"):
    """Save the ColorFinder object and color range."""
    data = {
        "lower_limit": color_finder.lower_limit,
        "upper_limit": color_finder.upper_limit,
        "center": color_finder.center,
        "color_finder": color_finder,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_color_finder(filename="color_finder.pkl"):
    """Load the ColorFinder object and color range."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    color_finder = data["color_finder"]
    color_finder.lower_limit = data["lower_limit"]
    color_finder.upper_limit = data["upper_limit"]
    color_finder.center = data["center"]
    return color_finder


if __name__ == "__main__":
    color_finder = ColorFinder()
    dataset_path_main = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\database"
    lower_limit_main, upper_limit_main, center_main = get_color_limits_from_dataset(
        dataset_path_main, show_plot=True, outlier_removal_method="stddev"
    )
    color_finder.lower_limit = lower_limit_main
    color_finder.upper_limit = upper_limit_main
    color_finder.center = center_main

    image_path_main = (
        r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\data\A12345_2mm.png"
    )
    results_main = color_finder.find_color_and_percentage(
        image_path_main,
        exclude_transparent=True,
        adaptive_thresholding=False,
        apply_morphology=True,
        apply_blur=False,
        blur_radius=3,
        apply_pixelation=True,
        pixelation_size=10,
    )

    if results_main:
        color_finder.plot_and_save_results(
            output_dir="output",
            save=False,
            show=True,
        )
