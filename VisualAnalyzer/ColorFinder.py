import cv2
import numpy as np
import os


class ColorFinder:
    """
    A class to find and highlight specific colors in images and video streams.
    """

    def __init__(
        self,
        base_color: tuple = (30, 255, 255),
        hue_percentage: float = 3,
        saturation_percentage: float = 70,
        value_percentage: float = 70,
    ):
        """
        Initialize the ColorFinder with base color and percentage ranges.

        Parameters:
        base_color (tuple): The base color in HSV.
        hue_percentage (float): The percentage range for hue.
        saturation_percentage (float): The percentage range for saturation.
        value_percentage (float): The percentage range for value.
        """
        self.lower_limit, self.upper_limit = self.get_color_limits_from_hsv(
            base_color, hue_percentage, saturation_percentage, value_percentage
        )

    def get_average_color(self, image: np.ndarray) -> np.ndarray:
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

    def remove_outliers(self, data: list) -> list:
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

    def get_color_limits_from_dataset(self, dataset_path: str) -> tuple:
        """
        Calculate color limits (HSV) based on a dataset of images, removing outliers.

        Parameters:
        dataset_path (str): The path to the dataset of images.

        Returns:
        tuple: The lower and upper color limits in HSV.
        """
        hues, saturations, values = [], [], []

        for filename in os.listdir(dataset_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(dataset_path, filename)
                image = cv2.imread(image_path)
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                average_color = self.get_average_color(hsv_image)
                hues.append(average_color[0])  # Hue component
                saturations.append(average_color[1])  # Saturation component
                values.append(average_color[2])  # Value component

        # Remove outliers
        hues = self.remove_outliers(hues)
        saturations = self.remove_outliers(saturations)
        values = self.remove_outliers(values)

        lower_limit = np.array(
            [min(hues), min(saturations), min(values)], dtype=np.uint8
        )
        upper_limit = np.array(
            [max(hues), max(saturations), max(values)], dtype=np.uint8
        )

        return lower_limit, upper_limit

    def get_color_limits_from_hsv(
        self,
        base_color: tuple,
        hue_percentage: float,
        saturation_percentage: float,
        value_percentage: float,
    ) -> tuple:
        """
        Calculate color limits (HSV) based on a given color and user-provided percentages.

        Parameters:
        base_color (tuple): The base color in HSV.
        hue_percentage (float): The percentage range for hue.
        saturation_percentage (float): The percentage range for saturation.
        value_percentage (float): The percentage range for value.

        Returns:
        tuple: The lower and upper color limits in HSV.
        """
        hue, saturation, value = base_color

        hue_range = 255 * hue_percentage / 100
        saturation_range = 255 * saturation_percentage / 100
        value_range = 255 * value_percentage / 100

        lower_limit = np.array(
            [
                max(0, hue - hue_range),
                max(0, saturation - saturation_range),
                max(0, value - value_range),
            ],
            dtype=np.uint8,
        )
        upper_limit = np.array(
            [
                min(255, hue + hue_range),
                min(255, saturation + saturation_range),
                min(255, value + value_range),
            ],
            dtype=np.uint8,
        )

        return lower_limit, upper_limit

    def process_webcam(self):
        """
        Process video from the webcam to identify and highlight areas matching the color limits.
        """
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

    def find_color_and_percentage(self, image_path: str) -> tuple:
        """
        Finds and highlights a color in an image and calculates the percentage of pixels matching that color.

        Combines process_image and get_color_percentage for a single call.

        Parameters:
            image_path (str): The path to the input image.

        Returns:
            tuple: A tuple containing the processed image with highlighted regions, the selected color (BGR),
                   the percentage of pixels matching the color, and the total number of pixels matching the color.
                   Returns None if the image cannot be loaded.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask based on color limits
        mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        # Find contours and draw rectangles (as in process_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

        # Calculate percentage (as in get_color_percentage)
        matched_pixels = cv2.countNonZero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        percentage = (matched_pixels / total_pixels) * 100

        selected_color_bgr = cv2.cvtColor(
            np.uint8([[self.lower_limit]]), cv2.COLOR_HSV2BGR
        )[0][0]

        # Add the color legend (from process_image) - slightly modified to avoid redundancy
        cv2.rectangle(
            image, (10, 10), (30, 30), selected_color_bgr.tolist(), -1
        )  # Reusing selected_color_bgr
        cv2.putText(
            image, "Color", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        return image, selected_color_bgr, percentage, matched_pixels


if __name__ == "__main__":
    # Example usage:
    color_finder = ColorFinder(
        base_color=(30, 255, 255),
        hue_percentage=3,
        saturation_percentage=70,
        value_percentage=70,
    )

    image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\.old\img\j.png"
    # color_finder.process_webcam()
    #color_finder.process_image(r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\.old\img\j.png")
    results = color_finder.find_color_and_percentage(image_path)

    if results:
        processed_image, color_bgr, percentage, matched_pixels = results
        print(f"Selected Color (BGR): {color_bgr}")
        print(f"Percentage of matched pixels: {percentage:.2f}%")
        print(f"Number of matched pixels: {matched_pixels}")
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
