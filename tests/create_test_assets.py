import cv2
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.chessboard_utils import generate_chessboard_image # Import the new utility

# Create the test project directory if it doesn't exist
test_project_dir = "output/test_assets/alignment_test"
os.makedirs(test_project_dir, exist_ok=True)

# 1. Generate a chessboard image using the utility function
pattern_size = (9, 6)
square_size_px = 50
padding = 100
chessboard_path = os.path.join(test_project_dir, "chessboard.png")
chessboard = generate_chessboard_image(pattern_size=pattern_size, square_size_px=square_size_px, padding=padding, output_path=chessboard_path)

# 2. Create a technical drawing of a rectangle
drawing_width = 400
drawing_height = 300
drawing = np.zeros((drawing_height, drawing_width), dtype=np.uint8)
rect_x, rect_y, rect_w, rect_h = 50, 50, 300, 200
cv2.rectangle(drawing, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 255, -1)
drawing_path = os.path.join(test_project_dir, "drawing.png")
cv2.imwrite(drawing_path, drawing)

# 3. Create a test image
# Create an image with the rectangle on the chessboard
test_image_base = cv2.cvtColor(chessboard, cv2.COLOR_GRAY2BGR)

# Get img_height and img_width from the generated chessboard
img_height, img_width = chessboard.shape[:2]

# Create a colored rectangle to place on the board
rectangle_image = np.zeros((150, 250, 3), dtype=np.uint8)
rectangle_image[:,:] = (0, 0, 255) # Red rectangle
# Place the rectangle on the board
test_image_base[200:350, 200:450] = rectangle_image

# Apply perspective transform
pts1 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
pts2 = np.float32([[100, 50], [img_width-100, 0], [50, img_height-50], [img_width-50, img_height-100]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
test_image_warped = cv2.warpPerspective(test_image_base, matrix, (img_width, img_height))
test_image_path = os.path.join(test_project_dir, "test_image.png")
cv2.imwrite(test_image_path, test_image_warped)
cv2.imwrite("output/test_assets/test_image_for_inspection.png", test_image_warped) # Save for inspection

print(f"Test assets created in {test_project_dir}")