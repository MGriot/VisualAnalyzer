from PIL import Image
import matplotlib.pyplot as plt

image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\data\A12345_2mm.png"

# Open image with Pillow
try:
    image = Image.open(image_path)
except Exception as e:
    print(f"Error: Could not open image: {e}")
    exit()

# Display original image
plt.imshow(image)
plt.title("Original Image")
plt.show()

# Convert to RGBA (if necessary)
if image.mode != "RGBA":
    image = image.convert("RGBA")

# Display converted image (if necessary)
if image.mode == "RGBA":
    plt.imshow(image)
    plt.title("Image Converted to RGBA")
    plt.show()

# Further processing here (assuming you don't need to display intermediate steps)
# ... (your code)
