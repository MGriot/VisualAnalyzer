import cv2
import numpy as np
import os

class Aligner:
    """
    A class to handle image alignment tasks, including perspective correction and object alignment.
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initializes the Aligner.

        Args:
            debug_mode (bool): If True, enables debug output.
        """
        self.debug_mode = debug_mode

    def generate_chessboard_image(self, pattern_size=(9, 6), square_size_px=50, output_path="chessboard.png"):
        """
        Generates and saves a chessboard image.

        Args:
            pattern_size (tuple): The number of inner corners in the grid pattern (width, height).
            square_size_px (int): The size of each square in pixels.
            output_path (str): The path to save the generated chessboard image.
        """
        board_width = pattern_size[0] + 1
        board_height = pattern_size[1] + 1
        img_width = board_width * square_size_px
        img_height = board_height * square_size_px

        chessboard = np.zeros((img_height, img_width), dtype=np.uint8)
        for y in range(board_height):
            for x in range(board_width):
                if (x + y) % 2 == 0:
                    chessboard[y*square_size_px:(y+1)*square_size_px, x*square_size_px:(x+1)*square_size_px] = 255
        
        cv2.imwrite(output_path, chessboard)
        if self.debug_mode:
            print(f"[DEBUG] Generated chessboard image saved to: {output_path}")

    def align_image(self, image_path: str, drawing_path: str, pattern_size=(9, 6)):
        """
        Aligns an image with a technical drawing using a chessboard for perspective correction.

        Args:
            image_path (str): The path to the input image containing the object and chessboard.
            drawing_path (str): The path to the technical drawing (black background, white profile).
            pattern_size (tuple): The number of inner corners of the chessboard (width, height).

        Returns:
            np.ndarray: The aligned and masked image, or None if alignment fails.
        """
        if self.debug_mode:
            print(f"[DEBUG] Starting alignment for image: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        drawing = cv2.imread(drawing_path, cv2.IMREAD_GRAYSCALE)
        if drawing is None:
            raise ValueError(f"Could not load drawing from path: {drawing_path}")

        # 1. Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, None)

        if not ret:
            if self.debug_mode:
                print("[DEBUG] Chessboard corners not found in the image.")
            return None

        # 2. Generate ideal chessboard points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        # 3. Calculate homography
        homography, _ = cv2.findHomography(objp[:, :2], corners)

        # 4. Warp the image
        rectified_img = cv2.warpPerspective(image, np.linalg.inv(homography), (drawing.shape[1], drawing.shape[0]))
        if self.debug_mode:
            cv2.imwrite("rectified_debug.png", rectified_img)

        # 5. Align object with drawing
        # Find contour of the drawing
        contours_drawing, _ = cv2.findContours(drawing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_drawing:
            if self.debug_mode:
                print("[DEBUG] No contours found in the drawing.")
            return None
        drawing_contour = max(contours_drawing, key=cv2.contourArea)

        # Find contour of the object in the rectified image
        gray_rectified = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
        _, thresh_rectified = cv2.threshold(gray_rectified, 50, 255, cv2.THRESH_BINARY)
        contours_object, _ = cv2.findContours(thresh_rectified, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_object:
            if self.debug_mode:
                print("[DEBUG] No contours found in the rectified image.")
            return None
        object_contour = max(contours_object, key=cv2.contourArea)

        # Align by moments and rotation
        M_drawing = cv2.moments(drawing_contour)
        M_object = cv2.moments(object_contour)

        if M_drawing['m00'] == 0 or M_object['m00'] == 0:
            if self.debug_mode:
                print("[DEBUG] Cannot calculate moments for alignment.")
            return None

        cx_drawing = int(M_drawing['m10'] / M_drawing['m00'])
        cy_drawing = int(M_drawing['m01'] / M_drawing['m00'])
        cx_object = int(M_object['m10'] / M_object['m00'])
        cy_object = int(M_object['m01'] / M_object['m00'])

        # Get rotation angle
        _, _, angle_drawing = cv2.fitEllipse(drawing_contour)
        _, _, angle_object = cv2.fitEllipse(object_contour)
        angle_diff = angle_drawing - angle_object

        # Rotation and Translation
        rot_mat = cv2.getRotationMatrix2D((cx_object, cy_object), angle_diff, 1)
        
        # Adjust translation part of the rotation matrix
        rot_mat[0, 2] += cx_drawing - cx_object
        rot_mat[1, 2] += cy_drawing - cy_object

        aligned_rectified = cv2.warpAffine(rectified_img, rot_mat, (rectified_img.shape[1], rectified_img.shape[0]))
        if self.debug_mode:
            cv2.imwrite("aligned_rectified_debug.png", aligned_rectified)

        # 6. Mask the image
        mask = cv2.bitwise_not(drawing)
        final_image = cv2.bitwise_and(aligned_rectified, aligned_rectified, mask=drawing)
        if self.debug_mode:
            cv2.imwrite("final_image_debug.png", final_image)

        if self.debug_mode:
            print("[DEBUG] Alignment successful.")

        return final_image
