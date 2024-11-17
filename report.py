from jinja2 import Environment, FileSystemLoader
import datetime
import cv2
from VisualAnalyzer.ColorFinder import ColorFinder
import os
import matplotlib.pyplot as plt

# --- Constants and global variables ---
TEMPLATES_DIR = "Templates"
DATABASE_PATH = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\img\database"
IMAGE_PATH = r"img\data\j.png"
OUTPUT_DIR = "processed_images"
REPORT_HTML_PATH = "report1.html"

# --- Report metadata ---
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
AUTHOR = "Griot Matteo"
DEPARTMENT = "Global Quality"
REPORT_TITLE = "Under Layer Report"
PART_NUMBER = "xxxxxxxxx"
THICKNESS = "2 mm"

# --- Initialize Jinja2 environment ---
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
template = env.get_template("Report.html")


def analyze_image(image_path, database_path):
    """
    Analyzes an image to find dominant colors and generate a pie chart.

    Args:
        image_path (str): Path to the image to analyze.
        database_path (str): Path to the color database.

    Returns:
        tuple: A tuple containing the processed image, selected colors,
               percentage of matched pixels, number of matched pixels,
               image width, and image height.
    """
    color_finder = ColorFinder()
    color_finder.get_color_limits_from_dataset(dataset_path=database_path)
    results = color_finder.find_color_and_percentage(image_path, save_images=True, exclude_transparent=True)
    return results


def generate_pie_chart(matched_pixels, image_width, image_height, selected_colors):
    """
    Generates and saves a pie chart showing matched vs. unmatched pixels.

    Args:
        matched_pixels (int): Number of matched pixels.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        selected_colors (dict): Dictionary containing RGB values of selected colors.
    """
    unmatched_pixels = image_width * image_height - matched_pixels
    labels = ["Matched Pixels", "Unmatched Pixels"]
    sizes = [matched_pixels, unmatched_pixels]
    colors = [selected_colors["RGB"] / 255, "darkgray"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")
    plt.savefig(os.path.join(OUTPUT_DIR, "pie_chart.png"))
    plt.close()


def generate_report(image_path, processed_image_path, mask_path, pie_chart_path, part_number, thickness, today, author, department, report_title):
    """
    Generates an HTML report using the provided data.

    Args:
        image_path (str): Path to the original image.
        processed_image_path (str): Path to the processed image.
        mask_path (str): Path to the mask image.
        pie_chart_path (str): Path to the pie chart image.
        part_number (str): Part number.
        thickness (str): Thickness.
        today (str): Today's date.
        author (str): Author.
        department (str): Department.
        report_title (str): Report title.
    """
    html_content = template.render(
        image_path=image_path,
        image1_path=processed_image_path,
        image2_path=mask_path,
        image3_path=pie_chart_path,
        part_number=part_number,
        thickness=thickness,
        today=today,
        author=author,
        department=department,
        report_title=report_title,
    )
    with open(REPORT_HTML_PATH, "w") as html_file:
        html_file.write(html_content)
    print(f"HTML report saved to {REPORT_HTML_PATH}")


if __name__ == "__main__":
    # --- Analyze the image ---
    results = analyze_image(IMAGE_PATH, DATABASE_PATH)
    processed_image, selected_colors, percentage, matched_pixels, image_width, image_height = results
    print(f"Selected Colors: {selected_colors}")
    print(f"Percentage of matched pixels: {percentage:.2f}%")
    print(f"Number of matched pixels: {matched_pixels}")

    # --- Generate the pie chart ---
    generate_pie_chart(matched_pixels, image_width, image_height, selected_colors)

    # --- Generate the report ---
    processed_image_path = os.path.join(OUTPUT_DIR, "processed_image.png")
    mask_path = os.path.join(OUTPUT_DIR, "mask.png")
    pie_chart_path = os.path.join(OUTPUT_DIR, "pie_chart.png")
    generate_report(IMAGE_PATH, processed_image_path, mask_path, pie_chart_path, PART_NUMBER, THICKNESS, TODAY, AUTHOR, DEPARTMENT, REPORT_TITLE)
