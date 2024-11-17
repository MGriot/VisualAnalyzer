from jinja2 import Environment, FileSystemLoader
import datetime
import cv2 as cv
from VisualAnalyzer.ColorFinder import ColorFinder
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.patches import Rectangle
import shutil

# --- Constants and global variables ---
TEMPLATES_DIR = "Templates"
DATABASE_PATH = "img/database"
IMAGE_DIR = "img/data"  # Directory containing images to process
OUTPUT_DIR = "output/report"

# --- Report metadata ---
LOGO = "img/logo/logo.png"
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
AUTHOR = "Griot Matteo"
DEPARTMENT = "Global Quality"
REPORT_TITLE = "Under Layer Report"

# --- Initialize Jinja2 environment ---
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
template = env.get_template("Report.html")


def analyze_image(image_path, database_path, output_dir):
    """
    Analyzes an image to find dominant colors and generate a pie chart.

    Args:
        image_path (str): Path to the image to analyze.
        database_path (str): Path to the color database.
        output_dir (str): Directory to save the processed images.

    Returns:
        tuple: A tuple containing the processed image, selected colors,
               percentage of matched pixels, number of matched pixels,
               image width, image height, and color space plot path.
    """
    color_finder = ColorFinder()
    lower_limit, upper_limit, center = color_finder.get_color_limits_from_dataset(
        dataset_path=database_path
    )
    results = color_finder.find_color_and_percentage(
        image_path, save_images=True, exclude_transparent=True, output_dir=output_dir
    )

    # --- Generate color space plot ---
    color_space_plot_path = generate_color_space_plot(
        lower_limit, upper_limit, center, output_dir=output_dir
    )

    return results + (color_space_plot_path,)


def generate_pie_chart(
    matched_pixels, image_width, image_height, selected_colors, output_dir
):
    """
    Generates and saves a pie chart showing matched vs. unmatched pixels.

    Args:
        matched_pixels (int): Number of matched pixels.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        selected_colors (dict): Dictionary containing RGB values of selected colors.
        output_dir (str): Directory to save the pie chart.
    """
    unmatched_pixels = image_width * image_height - matched_pixels
    labels = ["Matched Pixels", "Unmatched Pixels"]
    sizes = [matched_pixels, unmatched_pixels]
    colors = [selected_colors["RGB"] / 255, "darkgray"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")
    plt.savefig(os.path.join(output_dir, "pie_chart.png"))
    plt.close()


def generate_color_space_plot(
    lower_limit, upper_limit, center, gradient_height=25, num_lines=5, output_dir="."
):
    """
    Generates and saves a color space plot with a customizable gradient.

    Args:
        lower_limit (np.ndarray): Lower HSV color limit.
        upper_limit (np.ndarray): Upper HSV color limit.
        center (tuple): Center HSV color.
        gradient_height (int, optional): Height of the gradient in pixels. Defaults to 25.
        num_lines (int, optional): Number of gradient lines to stack. Defaults to 5.
        output_dir (str, optional): Directory to save the plot. Defaults to current directory.

    Returns:
        str: Path to the saved color space plot image.
    """

    # Create a gradient from lower to upper limit
    lower_rgb = cv.cvtColor(np.uint8([[lower_limit]]), cv.COLOR_HSV2RGB)
    lower_rgb = lower_rgb[0][0]
    upper_rgb = cv.cvtColor(np.uint8([[upper_limit]]), cv.COLOR_HSV2RGB)
    upper_rgb = upper_rgb[0][0]
    center_rgb = cv.cvtColor(np.uint8([[center]]), cv.COLOR_HSV2RGB)
    center_rgb = center_rgb[0][0]

    gradient = np.linspace(lower_rgb, upper_rgb, 256)
    gradient = gradient / 255  # Normalize to 0-1 range

    # Create the gradient with desired height and lines
    gradient_resized = np.repeat(gradient.reshape(1, -1, 3), gradient_height, axis=0)
    gradient_stacked = np.vstack([gradient_resized] * num_lines)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(gradient_stacked)
    ax.axis("off")

    # Highlight the center color with a filled rectangle
    center_x = np.interp(center[0], [lower_limit[0], upper_limit[0]], [0, 256])
    rect_width = 2
    rect = Rectangle(
        (center_x - rect_width / 2, 0),
        rect_width,
        gradient_height * num_lines,
        linewidth=0,
        edgecolor="none",
        facecolor=center_rgb / 255,
    )
    ax.add_patch(rect)

    # Save the plot
    color_space_plot_path = os.path.join(output_dir, "color_space_plot.png")
    plt.savefig(color_space_plot_path)
    plt.close()

    return color_space_plot_path


def generate_report(
    original_image,
    processed_image_path,
    mask_path,
    pie_chart_path,
    color_space_plot_path,
    part_number,
    thickness,
    logo,
    today,
    author,
    department,
    report_title,
    report_html_path,
):
    """
    Generates an HTML report using the provided data.

    Args:
        original_image (str): Path to the original image.
        processed_image_path (str): Path to the processed image.
        mask_path (str): Path to the mask image.
        pie_chart_path (str): Path to the pie chart image.
        color_space_plot_path (str): Path to the color space plot image.
        part_number (str): Part number.
        thickness (str): Thickness.
        logo (str): Path to the logo image.
        today (str): Today's date.
        author (str): Author.
        department (str): Department.
        report_title (str): Report title.
        report_html_path (str): Path to save the HTML report.
    """
    html_content = template.render(
        image_path=original_image,
        image1_path=processed_image_path,
        image2_path=mask_path,
        image3_path=pie_chart_path,
        color_space_plot_path=color_space_plot_path,
        part_number=part_number,
        thickness=thickness,
        logo=logo,
        today=today,
        author=author,
        department=department,
        report_title=report_title,
    )
    with open(report_html_path, "w") as html_file:
        html_file.write(html_content)
    print(f"HTML report saved to {report_html_path}")


if __name__ == "__main__":
    for image_file in tqdm(os.listdir(IMAGE_DIR), desc="Processing images"):
        if image_file.endswith((".png", ".jpg", ".jpeg")):  # Process only image files
            image_path = os.path.join(IMAGE_DIR, image_file)

            # --- Extract part number and thickness from file name ---
            file_name_without_ext = os.path.splitext(image_file)[0]
            part_number, thickness = file_name_without_ext.split("_")

            # --- Create output directory if it doesn't exist ---
            output_dir = os.path.join(OUTPUT_DIR, f"{part_number}_{thickness}")
            os.makedirs(output_dir, exist_ok=True)

            # --- Analyze the image ---
            results = analyze_image(image_path, DATABASE_PATH, output_dir)
            (
                processed_image,
                selected_colors,
                percentage,
                matched_pixels,
                image_width,
                image_height,
                color_space_plot_path,
            ) = results
            print(f"Selected Colors: {selected_colors}")
            print(f"Percentage of matched pixels: {percentage:.2f}%")
            print(f"Number of matched pixels: {matched_pixels}")

            # --- Generate the pie chart ---
            generate_pie_chart(
                matched_pixels, image_width, image_height, selected_colors, output_dir
            )

            # --- Move original image to output directory ---
            original_image = os.path.join(output_dir, image_file)
            shutil.move(image_path, original_image)
            original_image = image_file

            # --- Generate the report ---
            report_html_path = os.path.join(
                output_dir, f"{file_name_without_ext}.html"
            )  # HTML file name based on image name
            processed_image_path = os.path.join("processed_image.png")
            mask_path = os.path.join("mask.png")
            pie_chart_path = os.path.join("pie_chart.png")
            color_space_plot_path = os.path.join("color_space_plot.png")  # Update path

            generate_report(
                original_image,
                processed_image_path,
                mask_path,
                pie_chart_path,
                color_space_plot_path,
                part_number,
                thickness,
                LOGO,
                TODAY,
                AUTHOR,
                DEPARTMENT,
                REPORT_TITLE,
                report_html_path,
            )
