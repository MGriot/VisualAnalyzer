#!/usr/bin/env python

import cv2
import numpy as np

class ManualImageAligner:
    """
    Class for manually aligning two images.
    """

    def __init__(self, image1_path, image2_path):
        """
        Initialize the ManualImageAligner object.

        Args:
            image1_path (str): Path to the first image.
            image2_path (str): Path to the second image.
        """
        self.image1 = cv2.imread(image1_path)
        self.image2 = cv2.imread(image2_path)
        self.points1 = []
        self.points2 = []
        self.current_image = 1  # 1 for image1, 2 for image2

    def select_points(self):
        """
        Allow the user to select points on both images.
        """
        cv2.namedWindow("Image 1")
        cv2.namedWindow("Image 2")
        cv2.setMouseCallback("Image 1", self.mouse_callback, param=1)
        cv2.setMouseCallback("Image 2", self.mouse_callback, param=2)

        while True:
            cv2.imshow("Image 1", self.image1)
            cv2.imshow("Image 2", self.image2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or len(self.points1) >= 4 and len(self.points2) >= 4:
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events on both images.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if param == 1:
                self.points1.append((x, y))
                cv2.circle(self.image1, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.image1, str(len(self.points1)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif param == 2:
                self.points2.append((x, y))
                cv2.circle(self.image2, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.image2, str(len(self.points2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def align_images(self):
        """
        Align the first image to the second image based on the selected points.

        Returns:
            tuple: A tuple containing:
                - The aligned image
                - The transformed original image (image1)
                - The transformation matrix
        """
        if len(self.points1) >= 4 and len(self.points2) >= 4:
            pts1 = np.float32(self.points1)
            pts2 = np.float32(self.points2)
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            aligned_image = cv2.warpPerspective(self.image1, matrix, (self.image2.shape[1], self.image2.shape[0]))
            transformed_original = cv2.warpPerspective(self.image1, matrix, (self.image1.shape[1], self.image1.shape[0]))

            return aligned_image, transformed_original, matrix
        else:
            print("Not enough points selected for alignment.")
            return None, None, None

if __name__ == "__main__":
    image1_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test_.png"
    image2_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\old\img\test.png"

    aligner = ManualImageAligner(image1_path, image2_path)
    aligner.select_points()
    aligned_image, transformed_original, matrix = aligner.align_images()

    if aligned_image is not None:
        cv2.imshow("Aligned Image", aligned_image)
        cv2.imshow("Transformed Original", transformed_original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Transformation Matrix:")
    print(matrix)
