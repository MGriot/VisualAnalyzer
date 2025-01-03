import cv2  # type: ignore
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, Dict, Any  # Import Tuple and Dict for type hinting
import matplotlib.pyplot as plt
from PIL import Image


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


def remove_outliers(data: list) -> list:
    """
    Remove outliers using the Interquartile Range (IQR) method.

    Parameters:
    data (list): The input data.

    Returns:
    list: The data with outliers removed.
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


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

    def get_color_limits_from_dataset(
        self, dataset_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """
        Calculate color limits (HSV) based on a dataset of images, removing outliers.

        Parameters:
        dataset_path (str): The path to the dataset of images.

        Returns:
        tuple: The lower and upper color limits in HSV, and the center of the distribution.
        """
        hues, saturations, values = [], [], []
        image_files = [
            f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png"))
        ]
        num_images = len(image_files)

        print(f"Processing {num_images} images in dataset...")

        for filename in tqdm(image_files):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            average_color = get_average_color(hsv_image)
            hues.append(average_color[0])  # Hue component
            saturations.append(average_color[1])  # Saturation component
            values.append(average_color[2])  # Value component

            print(f"Image: {filename}, Average Color (HSV): {average_color}")

        # Remove outliers
        hues = remove_outliers(hues)
        saturations = remove_outliers(saturations)
        values = remove_outliers(values)

        self.lower_limit = np.array(
            [min(hues), min(saturations), min(values)], dtype=np.uint8
        )
        self.upper_limit = np.array(
            [max(hues), max(saturations), max(values)], dtype=np.uint8
        )
        self.center = (np.mean(hues), np.mean(saturations), np.mean(values))

        print(f"Lower Limit (HSV): {self.lower_limit}")
        print(f"Upper Limit (HSV): {self.upper_limit}")
        print(f"Center (HSV): {self.center}")

        return self.lower_limit, self.upper_limit, self.center

    def get_color_limits_from_hsv(
        self,
        base_color: Tuple[int, int, int],
        hue_percentage: float,
        saturation_percentage: float,
        value_percentage: float,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
        """
        Calculate color limits (HSV) based on a given color and user-provided percentages.

        Parameters:
        base_color (tuple): The base color in HSV.
        hue_percentage (float): The percentage range for hue.
        saturation_percentage (float): The percentage range for saturation.
        value_percentage (float): The percentage range for value.

        Returns:
        tuple: The lower and upper color limits in HSV, and the center of the distribution.
        """
        hue, saturation, value = base_color

        hue_range = 255 * hue_percentage / 100
        saturation_range = 255 * saturation_percentage / 100
        value_range = 255 * value_percentage / 100

        self.lower_limit = np.array(
            [
                max(0, hue - hue_range),
                max(0, saturation - saturation_range),
                max(0, value - value_range),
            ],
            dtype=np.uint8,
        )
        self.upper_limit = np.array(
            [
                min(255, hue + hue_range),
                min(255, saturation + saturation_range),
                min(255, value + value_range),
            ],
            dtype=np.uint8,
        )

        self.center = (hue, saturation, value)

        return self.lower_limit, self.upper_limit, self.center

    def process_webcam(self):
        """
        Process video from the webcam to identify and highlight areas matching the color limits.
        """
        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError(
                "Color limits not set. Please use get_color_limits_from_dataset or get_color_limits_from_hsv to set the limits."
            )

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a mask based on color limits
            mask = cv2.inRange(hsv_frame, self.lower_limit, self.upper_limit)

            # Find contours of areas that match the color range
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw rectangles around each found contour
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # Add color limit legend to the main frame
            lower_color_bgr = cv2.cvtColor(
                np.uint8([[self.lower_limit]]), cv2.COLOR_HSV2BGR
            )[0][0]
            upper_color_bgr = cv2.cvtColor(
                np.uint8([[self.upper_limit]]), cv2.COLOR_HSV2BGR
            )[0][0]

            cv2.rectangle(frame, (10, 10), (30, 30), lower_color_bgr.tolist(), -1)
            cv2.putText(
                frame,
                "Lower Limit",
                (35, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.rectangle(frame, (10, 40), (30, 60), upper_color_bgr.tolist(), -1)
            cv2.putText(
                frame,
                "Upper Limit",
                (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Add color limit legend to the mask
            mask_with_legend = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(
                mask_with_legend, (10, 10), (30, 30), lower_color_bgr.tolist(), -1
            )
            cv2.putText(
                mask_with_legend,
                "Lower Limit",
                (35, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.rectangle(
                mask_with_legend, (10, 40), (30, 60), upper_color_bgr.tolist(), -1
            )
            cv2.putText(
                mask_with_legend,
                "Upper Limit",
                (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Show the frame and mask with legend
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask_with_legend)

            # Exit the loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, image_path: str):
        """
        Process an image to identify and highlight areas matching the color limits.

        Parameters:
        image_path (str): The path to the input image.
        """
        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError(
                "Color limits not set. Please use get_color_limits_from_dataset or get_color_limits_from_hsv to set the limits."
            )

        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask based on color limits
        mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        # Find contours of areas that match the color range
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around each found contour
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Add color limit legend to the main image
        lower_color_bgr = cv2.cvtColor(
            np.uint8([[self.lower_limit]]), cv2.COLOR_HSV2BGR
        )[0][0]
        upper_color_bgr = cv2.cvtColor(
            np.uint8([[self.upper_limit]]), cv2.COLOR_HSV2BGR
        )[0][0]

        cv2.rectangle(image, (10, 10), (30, 30), lower_color_bgr.tolist(), -1)
        cv2.putText(
            image,
            "Lower Limit",
            (35, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.rectangle(image, (10, 40), (30, 60), upper_color_bgr.tolist(), -1)
        cv2.putText(
            image,
            "Upper Limit",
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show the image with legend
        cv2.imshow("image", image)
        cv2.imshow("mask", mask)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_color_and_percentage(
        self,
        image_path: str,
        save_images: bool = False,
        exclude_transparent: bool = False,
        output_dir: str = "processed_images",
    ) -> Tuple[np.ndarray, Dict[str, Any], float, int, int]:
        """
        Finds and highlights a color in an image and calculates the percentage of pixels matching that color.

        Combines process_image and get_color_percentage for a single call.

        Parameters:
            image_path (str): The path to the input image.
            exclude_transparent (bool): Whether to exclude transparent pixels from the calculation.
            output_dir (str): The directory to save the processed images to.

        Returns:
            tuple: A tuple containing the processed image with highlighted regions,
                the selected color (dictionary with different color spaces),
                the percentage of pixels matching the color,
                the total number of pixels matching the color,
                and the total number of pixels in the image (including or excluding transparent pixels based on exclude_transparent).
                Returns None if the image cannot be loaded.
        """

        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError(
                "Color limits not set. Please use get_color_limits_from_dataset or get_color_limits_from_hsv to set the limits."
            )

        try:
            # Open image with Pillow, handle potential errors
            image = Image.open(image_path)
            if exclude_transparent:
                # Convert to RGBA for alpha channel access
                image = image.convert("RGBA")

                # Extract alpha channel, filter for non-transparent pixels
                bgra = np.array(image)
                if bgra.ndim == 4:  # Check if the image has an alpha channel
                    # Apply transparency filter before reshaping
                    non_transparent_pixels = bgra[bgra[:, :, 3] > 250]  # Threshold for transparency

                    # Convert back to image mode
                    image = Image.fromarray(non_transparent_pixels.astype(np.uint8))
                    image = image.convert("RGB")  # Convert to RGB for OpenCV
                    total_pixels = len(non_transparent_pixels)
                else:
                    image = np.asarray(image)
                    total_pixels = image.shape[0] * image.shape[1]
            else:
                # Handle non-transparent or images without alpha channel
                image = np.asarray(image)
                total_pixels = image.shape[0] * image.shape[1]
        except Exception as e:
            print(f"Error: Could not load image at {image_path} ({e})")
            return None

        # Convert image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask based on color limits
        mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        # Calculate percentage of matching pixels
        matched_pixels = cv2.countNonZero(mask)
        percentage = (matched_pixels / total_pixels) * 100

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

        # Add the color legend (from process_image) - slightly modified to avoid redundancy
        if (
            not exclude_transparent or image.shape[2] != 4
        ):  # Only add legend if not excluding transparent pixels or no alpha channel
            cv2.rectangle(
                image, (10, 10), (30, 30), selected_color_bgr.tolist(), -1
            )  # Reusing selected_color_bgr
            cv2.putText(
                image,
                "Color",
                (35, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        if save_images:
            # Create a folder "processed_images" if it doesn't exist
            # save_dir = "processed_images"
            os.makedirs(output_dir, exist_ok=True)

            # Save original image, processed image, and mask (if desired)
            cv2.imwrite(os.path.join(output_dir, "original_image.png"), image)
            cv2.imwrite(os.path.join(output_dir, "processed_image.png"), image)
            cv2.imwrite(
                os.path.join(output_dir, "mask.png"), mask
            )  # Add mask saving if needed   processed_image,
            return (
                image,
                selected_colors,
                percentage,
                matched_pixels,
                total_pixels,
            )

        else:
            return (
                image,
                selected_colors,
                percentage,
                matched_pixels,
                total_pixels,
            )  # Return original image if not saving


if __name__ == "__main__":
    # Example usage:
    color_finder = ColorFinder()

    # Set color limits using either method:
    # 1. From dataset:
    # dataset_path = "path/to/your/dataset"
    # color_finder.get_color_limits_from_dataset(dataset_path)

    # 2. From HSV:
    base_color = (30, 255, 255)
    hue_percentage = 3
    saturation_percentage = 70
    value_percentage = 70
    lower_limit, upper_limit, center = color_finder.get_color_limits_from_hsv(
        base_color, hue_percentage, saturation_percentage, value_percentage
    )
    print(f"Lower Limit: {lower_limit}")
    print(f"Upper Limit: {upper_limit}")
    print(f"Center: {center}")

    image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\.old\img\j.png"
    # color_finder.process_webcam()
    # color_finder.process_image(image_path)
    results = color_finder.find_color_and_percentage(
        image_path, exclude_transparent=True
    )

    if results:
        processed_image, selected_colors, percentage, matched_pixels, total_pixels = results
        print(f"Selected Colors: {selected_colors}")
        print(f"Percentage of matched pixels: {percentage:.2f}%")
        print(f"Number of matched pixels: {matched_pixels}")
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
