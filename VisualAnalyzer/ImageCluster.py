import colorsys
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.cluster import KMeans


class ImageCluster:
    """
    Class to cluster an image based on its colors.
    """

    def __init__(self, image_input):
        """
        Initialize the ImageCluster object.

        Args:
            image_input (str or numpy.ndarray): Path to the image file or the image itself.
        """
        self.image = self.load_image(image_input)
        self.clustered_image = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.cluster_counts = None
        self.cluster_percentages = None
        self.cluster_info = None

    def load_image(self, input):
        """
        Load an image from a file or directly from a numpy array.

        Args:
            input (str or numpy.ndarray): Path to the image file or the image itself.

        Returns:
            numpy.ndarray: Loaded image.
        """
        if isinstance(input, str):
            return np.array(Image.open(input))
        elif isinstance(input, np.ndarray):
            return input
        else:
            raise ValueError("Invalid input type. Must be a string or a numpy array.")

    def remove_transparent(self, alpha_threshold=250):
        """
        Remove transparent pixels from the image.

        Args:
            alpha_threshold (int): Threshold for alpha channel to consider a pixel transparent.
        """
        if self.image.shape[2] == 4:
            alpha_channel = self.image[:, :, 3]
            self.image = self.image[alpha_channel >= alpha_threshold]
            self.image = self.image[:, :, :3]

    def filter_alpha(self):
        """
        Filter out pixels with low alpha values.
        """
        if self.image.shape[2] == 4:
            alpha_channel = self.image[:, :, 3]
            self.image = self.image[alpha_channel > 0]

    def cluster(
        self, n_clusters=5, remove_transparent_pixels=True, alpha_threshold=250
    ):
        """
        Cluster the image based on its colors.

        Args:
            n_clusters (int): Number of clusters to create.
            remove_transparent_pixels (bool): Whether to remove transparent pixels before clustering.
            alpha_threshold (int): Threshold for alpha channel to consider a pixel transparent.
        """
        if remove_transparent_pixels:
            self.remove_transparent(alpha_threshold)

        pixels = self.image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(pixels)

        self.cluster_centers = kmeans.cluster_centers_
        self.cluster_labels = kmeans.labels_
        self.cluster_counts = np.bincount(self.cluster_labels)
        self.cluster_percentages = self.cluster_counts / len(self.cluster_labels)

        self.create_clustered_image()
        self.extract_cluster_info()

    def create_clustered_image(self):
        """
        Create a new image where each pixel is replaced with its cluster center color.
        """
        new_image = np.zeros_like(self.image)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel_cluster = self.cluster_labels[i * self.image.shape[1] + j]
                new_image[i, j] = self.cluster_centers[pixel_cluster]
        self.clustered_image = new_image

    def create_clustered_image_with_ids(self):
        """
        Create a new image where each pixel is replaced with its cluster ID.
        """
        new_image = np.zeros_like(self.image)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel_cluster = self.cluster_labels[i * self.image.shape[1] + j]
                new_image[i, j] = pixel_cluster
        self.clustered_image = new_image

    def extract_cluster_info(self):
        """
        Extract information about each cluster, including its RGB and HEX color, percentage, and brightness.
        """
        self.cluster_info = []
        for i, center in enumerate(self.cluster_centers):
            rgb_color = tuple(map(int, center))
            hex_color = self.rgb_to_hex(rgb_color)
            percentage = self.cluster_percentages[i]
            brightness = self.calculate_brightness(rgb_color)
            self.cluster_info.append(
                {
                    "id": i,
                    "rgb": rgb_color,
                    "hex": hex_color,
                    "percentage": percentage,
                    "brightness": brightness,
                }
            )

    def calculate_brightness(self, color):
        """
        Calculate the brightness of a color.

        Args:
            color (tuple): RGB color tuple.

        Returns:
            float: Brightness value (0-1).
        """
        r, g, b = color
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255

    def plot_original_image(self, ax=None, max_size=(1024, 1024)):
        """
        Plot the original image.

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            max_size (tuple): Maximum size of the image to display.
        """
        fig, ax = plt.subplots()
        if ax is None:
            
            pass
        ax.imshow(self.image)
        ax.set_title("Original Image")
        ax.axis("off")

        # Resize the figure to fit the image within max_size
        width, height = self.image.shape[1], self.image.shape[0]
        if width > max_size[0] or height > max_size[1]:
            if width / height > max_size[0] / max_size[1]:
                new_width = max_size[0]
                new_height = int(height * (new_width / width))
            else:
                new_height = max_size[1]
                new_width = int(width * (new_height / height))
            fig.set_size_inches(new_width / fig.dpi, new_height / fig.dpi)

    def plot_clustered_image(self, ax=None, max_size=(1024, 1024)):
        """
        Plot the clustered image.

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            max_size (tuple): Maximum size of the image to display.
        """
        fig, ax = plt.subplots()
        if ax is None:
            
            pass
        ax.imshow(self.clustered_image)
        ax.set_title("Clustered Image")
        ax.axis("off")

        # Resize the figure to fit the image within max_size
        width, height = self.clustered_image.shape[1], self.clustered_image.shape[0]
        if width > max_size[0] or height > max_size[1]:
            if width / height > max_size[0] / max_size[1]:
                new_width = max_size[0]
                new_height = int(height * (new_width / width))
            else:
                new_height = max_size[1]
                new_width = int(width * (new_height / height))
            fig.set_size_inches(new_width / fig.dpi, new_height / fig.dpi)

    def plot_clustered_image_high_contrast(
        self, ax=None, max_size=(1024, 1024), dpi=100
    ):
        """
        Plot the clustered image with high contrast colors.

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            max_size (tuple): Maximum size of the image to display.
            dpi (int): DPI of the plot.
        """
        fig, ax = plt.subplots(dpi=dpi)
        if ax is None:
            
            pass

        high_contrast_colors = [
            self.rgb_to_hex(self.closest_color(center))
            for center in self.cluster_centers
        ]

        new_image = np.zeros_like(self.image)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel_cluster = self.cluster_labels[i * self.image.shape[1] + j]
                new_image[i, j] = self.hex_to_rgb(high_contrast_colors[pixel_cluster])

        ax.imshow(new_image)
        ax.set_title("Clustered Image (High Contrast)")
        ax.axis("off")

        # Resize the figure to fit the image within max_size
        width, height = new_image.shape[1], new_image.shape[0]
        if width > max_size[0] or height > max_size[1]:
            if width / height > max_size[0] / max_size[1]:
                new_width = max_size[0]
                new_height = int(height * (new_width / width))
            else:
                new_height = max_size[1]
                new_width = int(width * (new_height / height))
            fig.set_size_inches(new_width / fig.dpi, new_height / fig.dpi)

    def plot_cluster_pie(self, ax=None, dpi=100):
        """
        Plot a pie chart of the cluster percentages.

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            dpi (int): DPI of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        labels = [info["hex"] for info in self.cluster_info]
        sizes = [info["percentage"] for info in self.cluster_info]

        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title("Cluster Percentages")

    def plot_cluster_bar(self, ax=None, dpi=100):
        """
        Plot a bar chart of the cluster percentages.

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            dpi (int): DPI of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        labels = [info["hex"] for info in self.cluster_info]
        sizes = [info["percentage"] for info in self.cluster_info]

        ax.bar(labels, sizes)
        ax.set_xlabel("Clusters")
        ax.set_ylabel("Percentage")
        ax.set_title("Cluster Percentages")

    def plot_cumulative_barchart(self, ax=None, dpi=100):
        """
        Plot a cumulative bar chart of the cluster percentages, sorted by brightness.

        Args:
            ax (matplotlib.axes.Axes): Axes object to plot on.
            dpi (int): DPI of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        sorted_info = sorted(self.cluster_info, key=lambda x: x["brightness"])
        labels = [info["hex"] for info in sorted_info]
        sizes = [info["percentage"] for info in sorted_info]
        cumulative_sizes = np.cumsum(sizes)

        ax.bar(labels, cumulative_sizes, label="Cumulative Percentage")
        ax.bar(labels, sizes, label="Percentage")

        ax.set_xlabel("Clusters (Sorted by Brightness)")
        ax.set_ylabel("Percentage")
        ax.set_title("Cumulative Cluster Percentages")
        ax.legend()

    def plot_images(self, max_size=(1024, 1024)):
        """
        Plot the original and clustered images side by side.

        Args:
            max_size (tuple): Maximum size of the images to display.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.plot_original_image(ax1, max_size)
        self.plot_clustered_image(ax2, max_size)
        plt.show()

    def plot_image_with_grid(
        self, grid_size=(10, 10), max_size=(1024, 1024), dpi=100
    ):
        """
        Plot the original image with a grid overlay, where each grid cell shows the dominant cluster color.

        Args:
            grid_size (tuple): Number of rows and columns in the grid.
            max_size (tuple): Maximum size of the image to display.
            dpi (int): DPI of the plot.
        """
        fig, ax = plt.subplots(dpi=dpi)
        self.plot_original_image(ax, max_size)

        width, height = self.image.shape[1], self.image.shape[0]
        grid_width = width // grid_size[1]
        grid_height = height // grid_size[0]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y_start = i * grid_height
                y_end = (i + 1) * grid_height
                x_start = j * grid_width
                x_end = (j + 1) * grid_width

                grid_cell = self.image[y_start:y_end, x_start:x_end]
                pixels = grid_cell.reshape((-1, 3))

                if len(pixels) > 0:
                    kmeans = KMeans(n_clusters=1)
                    kmeans.fit(pixels)
                    dominant_color = self.rgb_to_hex(tuple(map(int, kmeans.cluster_centers_[0])))

                    rect = plt.Rectangle(
                        (x_start, y_start),
                        grid_width,
                        grid_height,
                        linewidth=1,
                        edgecolor="black",
                        facecolor=dominant_color,
                        alpha=0.5,
                    )
                    ax.add_patch(rect)

        plt.show()

    def save_plots(self, output_dir="output", filename_prefix="cluster_analysis"):
        """
        Save the generated plots to files.

        Args:
            output_dir (str): Directory to save the plots in.
            filename_prefix (str): Prefix for the filenames.
        """
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.plot_original_image(ax1)
        self.plot_clustered_image(ax2)
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_images.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        self.plot_cluster_pie(ax)
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_piechart.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        self.plot_cluster_bar(ax)
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_barchart.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        self.plot_clustered_image_high_contrast(ax)
        plt.savefig(
            os.path.join(output_dir, f"{filename_prefix}_cluster_image.png")
        )
        plt.close(fig)

    def rgb_to_hex(self, color):
        """
        Convert an RGB color tuple to a HEX color string.

        Args:
            color (tuple): RGB color tuple.

        Returns:
            str: HEX color string.
        """
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def rgba_to_hex(self, color):
        """
        Convert an RGBA color tuple to a HEX color string.

        Args:
            color (tuple): RGBA color tuple.

        Returns:
            str: HEX color string.
        """
        return "#{:02x}{:02x}{:02x}{:02x}".format(
            int(color[0]), int(color[1]), int(color[2]), int(color[3])
        )

    def hex_to_rgb(self, hex_color):
        """
        Convert a HEX color string to an RGB color tuple.

        Args:
            hex_color (str): HEX color string.

        Returns:
            tuple: RGB color tuple.
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def HEX_COLORS(self):
        """
        Get a list of predefined HEX color strings.

        Returns:
            list: List of HEX color strings.
        """
        return [
            "#000000",
            "#FFFFFF",
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#00FFFF",
            "#FF00FF",
            "#C0C0C0",
            "#808080",
            "#800000",
            "#808000",
            "#008000",
            "#800080",
            "#008080",
            "#000080",
        ]

    def RGB_COLORS(self):
        """
        Get a list of predefined RGB color tuples.

        Returns:
            list: List of RGB color tuples.
        """
        return [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (192, 192, 192),
            (128, 128, 128),
            (128, 0, 0),
            (128, 128, 0),
            (0, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
            (0, 0, 128),
        ]

    def get_color_names(self):
        """
        Get a list of predefined color names.

        Returns:
            list: List of color names.
        """
        return [
            "black",
            "white",
            "red",
            "lime",
            "blue",
            "yellow",
            "cyan",
            "magenta",
            "silver",
            "gray",
            "maroon",
            "olive",
            "green",
            "purple",
            "teal",
            "navy",
        ]

    def closest_color(self, requested_color):
        """
        Find the closest predefined color to a given RGB color.

        Args:
            requested_color (tuple): RGB color tuple.

        Returns:
            tuple: RGB color tuple of the closest predefined color.
        """
        min_colors = {}
        for key, name in enumerate(self.RGB_COLORS()):
            r_c, g_c, b_c = name
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with the path to your image
    clusterer = ImageCluster(image_path)
    clusterer.cluster(n_clusters=5)
    clusterer.save_plots()
