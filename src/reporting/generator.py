import os
import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from matplotlib.patches import Rectangle
import shutil

from src import config

class ReportGenerator:
    """
    A class to generate analysis reports in HTML and PDF formats.
    """

    def __init__(self, project_name: str, debug_mode: bool = False):
        """
        Initializes the ReportGenerator.

        Args:
            project_name (str): The name of the project for which the report is generated.
            debug_mode (bool): If True, loads the debug report template.
        """
        self.project_name = project_name
        self.project_output_dir = config.OUTPUT_DIR / project_name
        os.makedirs(self.project_output_dir, exist_ok=True)

        env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR))
        if debug_mode:
            self.template = env.get_template("Report_Debug.html")
        else:
            self.template = env.get_template("Report_Default.html")

    def _generate_pie_chart(self, matched_pixels: int, total_pixels: int, selected_colors: dict) -> str:
        """
        Generates and saves a pie chart showing matched vs. unmatched pixels.
        """
        labels = ["Matched Pixels", "Unmatched Pixels"]
        sizes = [matched_pixels, total_pixels - matched_pixels]
        colors = [selected_colors["RGB"] / 255, "darkgray"]
        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
        plt.axis("equal")
        pie_chart_path = self.project_output_dir / "pie_chart.png"
        plt.savefig(pie_chart_path)
        plt.close()
        return "pie_chart.png"

    def _generate_color_space_plot(self, lower_limit: np.ndarray, upper_limit: np.ndarray, center: tuple, gradient_height=25, num_lines=5) -> str:
        """
        Generates and saves a color space plot.
        """
        lower_rgb = cv.cvtColor(np.uint8([[lower_limit]]), cv.COLOR_HSV2RGB)[0][0]
        upper_rgb = cv.cvtColor(np.uint8([[upper_limit]]), cv.COLOR_HSV2RGB)[0][0]
        center_rgb = cv.cvtColor(np.uint8([[center]]), cv.COLOR_HSV2RGB)[0][0]

        gradient = np.linspace(lower_rgb, upper_rgb, 256) / 255
        gradient_resized = np.repeat(gradient.reshape(1, -1, 3), gradient_height, axis=0)
        gradient_stacked = np.vstack([gradient_resized] * num_lines)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(gradient_stacked)
        ax.axis("off")

        center_x = np.interp(center[0], [lower_limit[0], upper_limit[0]], [0, 256])
        rect = Rectangle((center_x - 1, 0), 2, gradient_height * num_lines, linewidth=0, facecolor=center_rgb / 255)
        ax.add_patch(rect)

        color_space_plot_path = self.project_output_dir / "color_space_plot.png"
        plt.savefig(color_space_plot_path)
        plt.close()
        return "color_space_plot.png"

    def plot_hue_saturation_diagram(self, image_bgr: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray, output_path: str) -> str:
        """
        Generates and saves a 2D Hue-Saturation diagram of the image's color space.

        Args:
            image_bgr (np.ndarray): The input image in BGR format.
            lower_hsv (np.ndarray): The lower bound of the HSV color range.
            upper_hsv (np.ndarray): The upper bound of the HSV color range.
            output_path (str): The relative path to save the generated plot.

        Returns:
            str: The relative path to the saved plot.
        """
        hsv_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_image)

        # Flatten the arrays to get a list of pixels
        h_flat = h.flatten()
        s_flat = s.flatten()
        v_flat = v.flatten()
        
        # Sample a subset of pixels to avoid plotting millions of points, which can be slow
        sample_size = min(len(h_flat), 20000) # Plot up to 20,000 pixels
        indices = np.random.choice(len(h_flat), sample_size, replace=False)
        
        h_sample = h_flat[indices]
        s_sample = s_flat[indices]
        v_sample = v_flat[indices]

        # Assemble the sampled HSV colors and convert them to RGB for plotting
        # This is how we get the "real" color for each point
        sampled_hsv = np.stack((h_sample, s_sample, v_sample), axis=-1)
        sampled_rgb = cv.cvtColor(np.uint8([sampled_hsv]), cv.COLOR_HSV2RGB)[0] / 255.0

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot of the image's colors, using the actual pixel colors
        ax.scatter(h_sample, s_sample, c=sampled_rgb, alpha=0.5, s=10, label='Image Pixels')

        # Draw a rectangle for the selected color range
        rect_width = upper_hsv[0] - lower_hsv[0]
        rect_height = upper_hsv[1] - lower_hsv[1]
        selection_rect = Rectangle(
            (lower_hsv[0], lower_hsv[1]), 
            rect_width, 
            rect_height,
            linewidth=2, 
            edgecolor='r', 
            facecolor='none',
            label='Selected Range'
        )
        ax.add_patch(selection_rect)

        ax.set_xlabel('Hue (0-179)')
        ax.set_ylabel('Saturation (0-255)')
        ax.set_title('Hue-Saturation Color Distribution')
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 256)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plot_full_path = self.project_output_dir / output_path
        plt.savefig(plot_full_path)
        plt.close(fig)
        
        return output_path

    def generate_report(self, analysis_results: dict, metadata: dict, debug_data: dict = None) -> None:
        """
        Generates an HTML and PDF report from the analysis results.
        """
        total_pixels = analysis_results['total_pixels']
        matched_pixels = analysis_results['matched_pixels']

        pie_chart_path_relative = self._generate_pie_chart(
            matched_pixels,
            total_pixels,
            analysis_results['selected_colors']
        )

        color_space_plot_path_relative = self._generate_color_space_plot(
            analysis_results['lower_limit'],
            analysis_results['upper_limit'],
            analysis_results['center_color']
        )

        # Generate Hue-Saturation Diagram
        chromaticity_diagram_path_relative = self.plot_hue_saturation_diagram(
            image_bgr=analysis_results['original_image'],
            lower_hsv=analysis_results['lower_limit'],
            upper_hsv=analysis_results['upper_limit'],
            output_path="hue_saturation_diagram.png"
        )

        # Copy original input image to output directory and get relative path
        original_input_image_filename = os.path.basename(analysis_results['original_image_path'])
        original_input_image_dest_path = self.project_output_dir / original_input_image_filename
        shutil.copy(analysis_results['original_image_path'], original_input_image_dest_path)
        original_input_image_path_relative = original_input_image_filename

        analyzed_image_path_relative = os.path.basename(analysis_results['analyzed_image_path'])

        logo_filename = os.path.basename(config.LOGO_PATH)
        logo_dest_path = self.project_output_dir / logo_filename
        logo_path_relative = ""
        try:
            shutil.copy(config.LOGO_PATH, logo_dest_path)
            logo_path_relative = logo_filename
        except FileNotFoundError:
            print(f"[WARNING] Logo file not found at {config.LOGO_PATH}. Report will be generated without a logo.")
        except Exception as e:
            print(f"[WARNING] An unexpected error occurred while copying logo: {e}. Report will be generated without a logo.")

        processed_image_path_relative = os.path.basename(analysis_results['processed_image_path'])
        mask_path_relative = os.path.basename(analysis_results['mask_path'])
        negative_mask_path_relative = os.path.basename(analysis_results['negative_mask_path'])
        
        mask_pre_aggregation_path_relative = os.path.basename(analysis_results['mask_pre_aggregation_path']) if analysis_results['mask_pre_aggregation_path'] else None
        blurred_image_path_relative = os.path.basename(analysis_results['blurred_image_path']) if analysis_results['blurred_image_path'] else None

        template_vars = {
            "author": config.AUTHOR,
            "department": config.DEPARTMENT,
            "report_title": f"{config.REPORT_TITLE} - {self.project_name}",
            "logo": logo_path_relative, 
            "today": datetime.datetime.now().strftime("%Y-%m-%d"),
            "part_number": metadata.get("part_number", "N/A"),
            "thickness": metadata.get("thickness", "N/A"),
            "image_path": original_input_image_path_relative,
            "analyzed_image_path": analyzed_image_path_relative,
            "color_space_plot_path": color_space_plot_path_relative,
            "hsv_diagram_path": chromaticity_diagram_path_relative, # New key for the template
            "image1_path": processed_image_path_relative,
            "image2_path": mask_path_relative,
            "image3_path": pie_chart_path_relative,
            "negative_mask_path": negative_mask_path_relative,
            "mask_pre_aggregation_path": mask_pre_aggregation_path_relative,
            "blurred_image_path": blurred_image_path_relative,
            "debug_data": debug_data
        }

        html_content = self.template.render(template_vars)
        report_html_path = self.project_output_dir / f"{metadata.get('part_number', 'report')}.html"
        report_pdf_path = self.project_output_dir / f"{metadata.get('part_number', 'report')}.pdf"

        with open(report_html_path, "w") as html_file:
            html_file.write(html_content)
        print(f"HTML report saved to {report_html_path}")

        HTML(string=html_content, base_url=str(self.project_output_dir)).write_pdf(report_pdf_path)
        print(f"PDF report saved to {report_pdf_path}")