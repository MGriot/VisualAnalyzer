import cv2
import numpy as np
import os

def generate_chessboard_image(pattern_size=(9, 6), square_size_px=50, padding=100, output_path="chessboard.png"):
    """
    Generates and saves a chessboard image.

    Args:
        pattern_size (tuple): The number of inner corners in the grid pattern (width, height).
        square_size_px (int): The size of each square in pixels.
        padding (int): Padding around the chessboard.
        output_path (str): The path to save the generated chessboard image.
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
