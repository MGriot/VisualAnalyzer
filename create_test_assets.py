import cv2
import numpy as np
import os

# Create the test project directory if it doesn't exist
test_project_dir = "data/projects/alignment_test"
os.makedirs(test_project_dir, exist_ok=True)

# 1. Generate a chessboard image
pattern_size = (9, 6)
square_size_px = 50
board_width = pattern_size[0] + 1
board_height = pattern_size[1] + 1
padding = 100
img_width = board_width * square_size_px + 2 * padding
img_height = board_height * square_size_px + 2 * padding

chessboard = np.zeros((img_height, img_width), dtype=np.uint8)
chessboard.fill(255)
for y in range(board_height):
    for x in range(board_width):
        if (x + y) % 2 == 0:
            chessboard[padding + y*square_size_px:padding + (y+1)*square_size_px, padding + x*square_size_px:padding + (x+1)*square_size_px] = 128

chessboard_path = os.path.join(test_project_dir, "chessboard.png")
cv2.imwrite(chessboard_path, chessboard)

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
cv2.imwrite("test_image_for_inspection.png", test_image_warped) # Save for inspection

print(f"Test assets created in {test_project_dir}")
