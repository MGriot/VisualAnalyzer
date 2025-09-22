"""
This module provides utility functions for generating chessboard patterns.

These patterns are commonly used in camera calibration for detecting corners
and estimating camera intrinsic and extrinsic parameters.
"""

import cv2
import numpy as np
import os

def generate_chessboard_image(pattern_size: tuple = (9, 6), square_size_px: int = 50, padding: int = 100, output_path: str = "chessboard.png") -> np.ndarray:
    """
    Generates and saves a chessboard image with specified dimensions and properties.

    Args:
        pattern_size (tuple): A tuple (width, height) representing the number of
                              inner corners in the grid pattern. Defaults to (9, 6).
        square_size_px (int): The size of each square in pixels. Defaults to 50.
        padding (int): The amount of white padding around the chessboard in pixels.
                       Defaults to 100.
        output_path (str): The file path where the generated chessboard image will be saved.
                           Defaults to "chessboard.png".

    Returns:
        np.ndarray: The generated chessboard image as a NumPy array.
    """
    board_width = pattern_size[0] + 1
    board_height = pattern_size[1] + 1
    img_width = board_width * square_size_px + 2 * padding
    img_height = board_height * square_size_px + 2 * padding

    chessboard = np.zeros((img_height, img_width), dtype=np.uint8)
    chessboard.fill(255) # White background
    for y in range(board_height):
        for x in range(board_width):
            if (x + y) % 2 == 0:
                chessboard[padding + y*square_size_px:padding + (y+1)*square_size_px, padding + x*square_size_px:padding + (x+1)*square_size_px] = 128 # Gray squares

    cv2.imwrite(output_path, chessboard)
    print(f"Generated chessboard image saved to: {output_path}")
    return chessboard # Return the chessboard numpy array

if __name__ == "__main__":
    # Example usage:
    # This will save a chessboard.png in the current directory
    generate_chessboard_image(output_path="my_chessboard.png")
    # To save in templates folder, you'd need to adjust the path
    # from src import config
    # generate_chessboard_image(output_path=str(config.TEMPLATES_DIR / "chessboard_template.png"))
