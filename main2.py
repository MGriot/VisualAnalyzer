import cv2
import numpy as np


from VisualAnalyzer.image_contour import image_contour
from VisualAnalyzer.ManualImageAligner import ManualImageAligner
from VisualAnalyzer.ImageCluster import ImageCluster


def process_image(image_path, reference_drawing_path):
    """
    Processes an image by applying contour detection, manual alignment,
    image clustering, and color recognition.

    Args:
        image_path (str): Path to the input image.
        reference_drawing_path (str): Path to the reference drawing for alignment.
        third_image_path (str): Path to the third image to transform.
    """

    # Apply contour detection
    contoured_image = image_contour(image_path)
    cv2.imwrite("temp_contoured_image.png", contoured_image)  # pylint: disable=no-member

    aligner = ManualImageAligner("temp_contoured_image.png", reference_drawing_path)
    aligner.select_points()
    aligned_image, transformed_original, matrix = aligner.align_images()
    cv2.imshow("aligned_image", aligned_image)  # Display image
    cv2.imshow("transformed_original", transformed_original)  # Display image

    if matrix is not None:
        # Align the original image using the transformation matrix
        original_image = cv2.imread(image_path)  # pylint: disable=no-member
        aligned_original_image = cv2.warpPerspective(original_image, matrix, (original_image.shape[1], original_image.shape[0]))  # pylint: disable=no-member

        # Save or display the aligned original image (choose one)
        cv2.imwrite("temp_aligned_original_image.png", aligned_original_image)  # pylint: disable=no-member  # Save to file
        cv2.imshow("Aligned Original Image", aligned_original_image)  # Display image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Cluster the aligned image based on color
    clusterer = ImageCluster(aligned_original_image)
    clustered_image = clusterer.cluster_image()
    cv2.imwrite("temp_clustered_image.png", clustered_image)  # pylint: disable=no-member


if __name__ == "__main__":
    image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test_.png"  # Replace with the path to your image
    reference_drawing_path = (
        r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test.png"
    )
    process_image(image_path, reference_drawing_path)
