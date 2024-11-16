from VisualAnalyzer.ColorFinder import ColorFinder
import os


def main():
    """
    Main function to process images in a folder, extract colors, and find color percentages.
    """
    database = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\database\image.png"  # Replace with the actual folder path
    image = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\data\j.png"

    color_finder = ColorFinder(
        base_color=(30, 255, 255),
        hue_percentage=3,
        saturation_percentage=70,
        value_percentage=70,
    )
    color_finder.find_color_and_percentage(
        image_path=image,
    )

if __name__ == "__main__":
    main()
