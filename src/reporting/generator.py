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
from src.color_analysis.project_manager import ProjectManager

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

        # Ensure the assets directory exists
        config.REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

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
        return "pie_chart.png" # Return relative path

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
        return "color_space_plot.png" # Return relative path

    def plot_hsv_color_space_3d(self, hsv_colors: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray, output_path: str) -> str:
        """
        Generates and saves a 3D plot of the HSV color space.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Sample the colors to avoid plotting too many points
        sample_size = min(len(hsv_colors), 10000)
        sampled_colors = hsv_colors[np.random.choice(len(hsv_colors), sample_size, replace=False)]

        # Convert HSV to RGB for plotting
        rgb_colors = cv.cvtColor(np.uint8([sampled_colors]), cv.COLOR_HSV2RGB)[0] / 255.0

        ax.scatter(sampled_colors[:, 0], sampled_colors[:, 1], sampled_colors[:, 2], c=rgb_colors, marker='o')

        # Draw wireframe box for the color range
        l, u = lower_hsv, upper_hsv
        x = [l[0], u[0], u[0], l[0], l[0], u[0], u[0], l[0]]
        y = [l[1], l[1], u[1], u[1], l[1], l[1], u[1], u[1]]
        z = [l[2], l[2], l[2], l[2], u[2], u[2], u[2], u[2]]
        ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], color='r')
        ax.plot([x[1], x[2]], [y[1], y[2]], [z[1], z[2]], color='r')
        ax.plot([x[2], x[3]], [y[2], y[3]], [z[2], z[3]], color='r')
        ax.plot([x[3], x[0]], [y[3], y[0]], [z[3], z[0]], color='r')

        ax.plot([x[4], x[5]], [y[4], y[5]], [z[4], z[5]], color='r')
        ax.plot([x[5], x[6]], [y[5], y[6]], [z[5], z[6]], color='r')
        ax.plot([x[6], x[7]], [y[6], y[7]], [z[6], z[7]], color='r')
        ax.plot([x[7], x[4]], [y[7], y[4]], [z[7], z[4]], color='r')

        ax.plot([x[0], x[4]], [y[0], y[4]], [z[0], z[4]], color='r')
        ax.plot([x[1], x[5]], [y[1], y[5]], [z[1], z[5]], color='r')
        ax.plot([x[2], x[6]], [y[2], y[6]], [z[2], z[6]], color='r')
        ax.plot([x[3], x[7]], [y[3], y[7]], [z[3], z[7]], color='r')

        ax.set_xlabel('Hue')
        ax.set_ylabel('Saturation')
        ax.set_zlabel('Value')
        ax.set_title('HSV Color Space')

        plot_path = self.project_output_dir / output_path
        plt.savefig(plot_path)
        plt.close()
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

        # Generate 3D HSV plot
        project_manager = ProjectManager()
        file_paths = project_manager.get_project_file_paths(self.project_name)
        hsv_colors = project_manager.get_hsv_colors_from_samples(file_paths['sample_image_configs'])
        project_data = project_manager.get_project_data(self.project_name)
        lower_hsv = project_data['lower_hsv']
        upper_hsv = project_data['upper_hsv']

        hsv_3d_plot_path_relative = self.plot_hsv_color_space_3d(hsv_colors, lower_hsv, upper_hsv, "hsv_3d_plot.png")

        # Copy original input image to output directory and get relative path
        original_input_image_filename = os.path.basename(analysis_results['original_input_image_path'])
        original_input_image_dest_path = self.project_output_dir / original_input_image_filename
        shutil.copy(analysis_results['original_input_image_path'], original_input_image_dest_path)
        original_input_image_path_relative = original_input_image_filename

        # Analyzed image is already saved in output_dir by ColorAnalyzer, just get relative path
        analyzed_image_path_relative = os.path.basename(analysis_results['analyzed_image_path'])

        # Copy logo to output directory and get relative path
        logo_filename = os.path.basename(config.LOGO_PATH)
        logo_dest_path = self.project_output_dir / logo_filename
        shutil.copy(config.LOGO_PATH, logo_dest_path)
        logo_path_relative = logo_filename

        # Get relative paths for other images saved by ColorAnalyzer
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
            "image_path": original_input_image_path_relative, # Original input image
            "analyzed_image_path": analyzed_image_path_relative, # Image after correction, before analysis
            "color_space_plot_path": color_space_plot_path_relative,
            "hsv_3d_plot_path": hsv_3d_plot_path_relative,
            "image1_path": processed_image_path_relative, # Matched pixels
            "image2_path": mask_path_relative, # Mask
            "image3_path": pie_chart_path_relative, # Pie chart
            "negative_mask_path": negative_mask_path_relative, # Negative mask
            "mask_pre_aggregation_path": mask_pre_aggregation_path_relative, # New: Path to mask before aggregation
            "blurred_image_path": blurred_image_path_relative, # New: Path to blurred image
            "debug_data": debug_data # New: Debug information
        }

        html_content = self.template.render(template_vars)
        report_html_path = self.project_output_dir / f"{metadata.get('part_number', 'report')}.html"
        report_pdf_path = self.project_output_dir / f"{metadata.get('part_number', 'report')}.pdf"

        with open(report_html_path, "w") as html_file:
            html_file.write(html_content)
        print(f"HTML report saved to {report_html_path}")

        # Use the project_output_dir as base_url for WeasyPrint to resolve relative paths
        HTML(string=html_content, base_url=str(self.project_output_dir)).write_pdf(report_pdf_path)
        print(f"PDF report saved to {report_pdf_path}")
