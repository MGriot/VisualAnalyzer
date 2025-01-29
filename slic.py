import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import KDTree

from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float

image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\models\ColourChecker\train\images\b053ef14-pixelfactory-colorchecker-target1.jpg"

image = Image.open(image_path)
image_np = np.array(image)
img = img_as_float(image_np[::2, ::2])

n_segments = 150
compactness=15
# Check if the image is grayscale
if img.ndim == 2:
    segments_slic = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=1,
        channel_axis=None,
    )
else:
    segments_slic = slic(
        img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1
    )

print(f"SLIC number of segments: {len(np.unique(segments_slic))}")

# Extract colors from the segmented areas
unique_segments = np.unique(segments_slic)
colors = []
for segment in unique_segments:
    mask = segments_slic == segment
    segment_colors = img[mask]
    average_color = np.mean(segment_colors, axis=0)
    colors.append(average_color)

# Merge areas with closest colors
def merge_closest_colors(segments, colors, threshold=0.2):
    tree = KDTree(colors)
    merged_segments = segments.copy()
    for i, color in enumerate(colors):
        distances, indices = tree.query(color, k=2)
        if distances[1] < threshold:
            merged_segments[segments == unique_segments[i]] = unique_segments[indices[1]]
    return merged_segments

merged_segments_slic = merge_closest_colors(segments_slic, colors)
merged_unique_segments = np.unique(merged_segments_slic)
merged_colors = [np.mean(img[merged_segments_slic == segment], axis=0) for segment in merged_unique_segments]

print(f"Number of merged segments: {len(merged_unique_segments)}")
print("Merged colors:", merged_colors)

fig, ax = plt.subplots()
ax.imshow(mark_boundaries(img, merged_segments_slic))
ax.set_title("SLIC with Merged Colors")
plt.tight_layout()
plt.show()
