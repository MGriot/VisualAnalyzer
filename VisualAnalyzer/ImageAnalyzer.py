import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageAnalyzer:
    """
    Class to analyze and compare images.
    """

    def __init__(self, img_input, ideal_img_input, ideal_img_processed=None):
        """
        Initialize the ImageAnalyzer object.

        Args:
            img_input (str or numpy.ndarray): Path to the image file or the image itself.
            ideal_img_input (str or numpy.ndarray): Path to the ideal image file or the ideal image itself.
            ideal_img_processed (numpy.ndarray, optional): The already processed ideal image. Defaults to None.
        """
        self.img = self.load_image(img_input)
        if ideal_img_processed is not None:
            self.ideal_img = ideal_img_processed
        else:
            self.ideal_img = self.load_image(ideal_img_input)

    def load_image(self, input):
        """
        Load an image from a file or directly from a numpy array.

        Args:
            input (str or numpy.ndarray): Path to the image file or the image itself.

        Returns:
            numpy.ndarray: Loaded image.
        """
        if isinstance(input, str):
            # pylint: disable=no-member
            return cv2.imread(input)
        elif isinstance(input, np.ndarray):
            return input
        else:
            raise ValueError("Invalid input type. Must be a string or a numpy array.")

    def calculate_histogram(self, img):
        """
        Calculate the histogram of an image.

        Args:
            img (numpy.ndarray): Image to calculate the histogram for.

        Returns:
            numpy.ndarray: Histogram of the image.
        """
        # pylint: disable=no-member
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return hist

    def compare_histograms(self, hist1, hist2):
        """
        Compare two histograms using different metrics.

        Args:
            hist1 (numpy.ndarray): First histogram.
            hist2 (numpy.ndarray): Second histogram.

        Returns:
            dict: Dictionary containing the comparison results.
        """
        # pylint: disable=no-member
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        return {
            "Correlation": correlation,
            "Chi-Square": chi_square,
            "Intersection": intersection,
            "Bhattacharyya": bhattacharyya,
        }

    def compare_images(self):
        """
        Compare the input image with the ideal image using different metrics.

        Returns:
            dict: Dictionary containing the comparison results.
        """
        hist1 = self.calculate_histogram(self.img)
        hist2 = self.calculate_histogram(self.ideal_img)
        histogram_comparison = self.compare_histograms(hist1, hist2)

        # pylint: disable=no-member
        gray1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.ideal_img, cv2.COLOR_BGR2GRAY)
        mse_score = np.mean((gray1 - gray2) ** 2)

        return {
            "Histogram Comparison": histogram_comparison,
            "MSE": mse_score,
        }

    def draw_histograms(self, img, title):
        """
        Draw the histogram of an image.

        Args:
            img (numpy.ndarray): Image to draw the histogram for.
            title (str): Title of the histogram plot.
        """
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            # pylint: disable=no-member
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title(title)
        plt.show()

    def analyze(self):
        """
        Analyze the input image and compare it with the ideal image.
        """
        self.draw_histograms(self.img, "Histogram of Input Image")
        self.draw_histograms(self.ideal_img, "Histogram of Ideal Image")

        comparison_results = self.compare_images()
        print("Comparison Results:")
        for metric, value in comparison_results.items():
            print(f"- {metric}: {value}")


if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"  # Replace with the path to your image
    ideal_img_path = "path/to/your/ideal_image.jpg"  # Replace with the path to your ideal image
    analyzer = ImageAnalyzer(img_path, ideal_img_path)
    analyzer.analyze()
