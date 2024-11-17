import cv2
from VisualAnalyzer.ColorFinder import ColorFinder
import os


def main():
    """
    Main function to process images in a folder, extract colors, and find color percentages.
    """
    database = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\database"  # Replace with the actual folder path
    image_path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\data\j.png"

    color_finder = ColorFinder()
    color_finder.get_color_limits_from_dataset(dataset_path=database)
    results = color_finder.find_color_and_percentage(image_path, save_images=True)

    if results:
        processed_image, selected_colors, percentage, matched_pixels, image_width, image_height = results
        print(f"Selected Colors: {selected_colors}")
        print(f"Percentage of matched pixels: {percentage:.2f}%")
        print(f"Number of matched pixels: {matched_pixels}")
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
