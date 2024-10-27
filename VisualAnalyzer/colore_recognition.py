import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_colors(image_path, num_colors=5, show_chart=True):

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape image to be a list of pixels
    pixels = img.reshape((-1, 3))

    # Cluster the pixels using KMeans
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    # Convert the RGB values to hex codes
    hex_colors = [rgb_to_hex(color) for color in colors]

    # Get the percentage of each color
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels)

    # Create a bar chart of the colors and their percentages
    if show_chart:
        plt.figure(figsize=(8, 6))
        plt.bar(hex_colors, percentages)
        plt.xlabel("Colors")
        plt.ylabel("Percentage")
        plt.title("Color Distribution")
        plt.show()

    return hex_colors, percentages


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with the path to your image
    num_colors = 5  # Number of colors to extract
    hex_colors, percentages = get_colors(image_path, num_colors)

    print("Hex Colors:", hex_colors)
    print("Percentages:", percentages)
