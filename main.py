import cv2
import numpy as np
from VisualAnalyzer.ImageCluster import ImageCluster
from VisualAnalyzer.colore_recognition import get_colors
from VisualAnalyzer.ManualImageAligner import ManualImageAligner
from VisualAnalyzer.image_contour import image_contour


def process_image(image_path, reference_drawing_path, third_image_path):
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

    # Save the contoured image to a temporary file
    cv2.imwrite("temp_contoured_image.png", contoured_image)

    # Align the contoured image with the reference drawing
    aligner = ManualImageAligner("temp_contoured_image.png", reference_drawing_path)
    aligner.select_points()
    aligned_image, transformed_original, transformed_third_image = aligner.align_images(
        image_to_transform_path=third_image_path
    )

    if aligned_image is not None:
        # Save the aligned image to a temporary file
        cv2.imwrite("temp_aligned_image.png", aligned_image)

        # Save the transformed original image to a temporary file
        cv2.imwrite("temp_transformed_original.png", transformed_original)

        # Save the transformed third image to a temporary file
        cv2.imwrite("temp_transformed_third_image.png", transformed_third_image)

        # Perform image clustering
        clusterer = ImageCluster("temp_transformed_original.png")
        clusterer.cluster(n_clusters=5)
        clusterer.save_plots()

        # Perform color recognition
        hex_colors, percentages = get_colors("temp_transformed_original.png", num_colors=5)
        print("Hex Colors:", hex_colors)
        print("Percentages:", percentages)
    else:
        print("Image alignment failed.")


if __name__ == "__main__":
    image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test_.png"  # Replace with the path to your image
    reference_drawing_path = (
        r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test.png"
    )
    third_image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test.png"
    process_image(image_path, reference_drawing_path, third_image_path)
