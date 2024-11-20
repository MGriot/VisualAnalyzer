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

# Convert to RGBA if necessary
if image.mode != "RGBA":
    image = image.convert("RGBA")

# Convert image to RGB channels
r, g, b, a = image.split()

# Plot individual RGB channels
plt.figure(figsize=(10, 6))  # Adjust figure size as needed

plt.subplot(2, 2, 1)  # Top left subplot
plt.imshow(r, cmap="gray")  # Convert to grayscale for clarity
plt.title("Red Channel")
plt.axis("off")  # Hide axes for cleaner presentation

plt.subplot(2, 2, 2)  # Top right subplot
plt.imshow(g, cmap="gray")
plt.title("Green Channel")
plt.axis("off")

plt.subplot(2, 2, 3)  # Bottom left subplot
plt.imshow(b, cmap="gray")
plt.title("Blue Channel")
plt.axis("off")

if image.mode == "RGBA":  # Only show alpha channel if it exists
    plt.subplot(2, 2, 4)  # Bottom right subplot
    plt.imshow(a, cmap="gray")
    plt.title("Alpha Channel (Transparency)")
    plt.axis("off")

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
