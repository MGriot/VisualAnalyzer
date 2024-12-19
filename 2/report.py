import os
import datetime
import logging
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from image_analysis import ColorFinder, get_color_limits_from_dataset

@dataclass
class ReportConfig:
    """
    Configuration class for report generation.

    Attributes:
        templates_dir (str): Directory containing Jinja2 templates
        database_path (str): Path to color database
        image_dir (str): Directory containing images to process
        output_dir (str): Directory to save processed reports
        logo (str): Path to logo image
        author (str): Report author
        department (str): Department name
        report_title (str): Title of the report
    """

    templates_dir: str = "UL project/Templates"  # Ensure this path is correct
    database_path: str = "img/database/Production"
    image_dir: str = "img/data"
    output_dir: str = "output/report"
    logo: str = "logo.png"
    author: str = "Griot Matteo"
    department: str = "Global Quality"
    report_title: str = "Under Layer Report"
    today: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d")
    )
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))


class ImageReportGenerator:
    """
    A comprehensive image report generation utility.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the report generator.

        Args:
            config (Optional[ReportConfig]): Configuration for report generation
        """
        self.config = config or ReportConfig()
        self._setup_logging()
        self._setup_template_environment()
        self.color_finder = ColorFinder()

    def _setup_logging(self):
        """Configure logging for the report generator."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_template_environment(self):
        """
        Set up Jinja2 template environment with error handling.

        Raises:
            TemplateNotFound: If template directory or file is invalid
        """
        try:
            # Ensure the templates directory path is correct
            self.env = Environment(loader=FileSystemLoader(self.config.templates_dir))
            self.template = self.env.get_template("Report.html")
        except (TemplateNotFound, FileNotFoundError) as e:
            self.config.logger.error(f"Template setup failed: {e}")
            raise

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive image analysis.

        Args:
            image_path (str): Path to the image to analyze

        Returns:
            Dict containing analysis results
        """
        try:
            lower_limit, upper_limit, center = get_color_limits_from_dataset(
                self.config.database_path
            )
            self.color_finder.lower_limit = lower_limit
            self.color_finder.upper_limit = upper_limit
            self.color_finder.center = center

            results = self.color_finder.find_color_and_percentage(
                image_path, exclude_transparent=True
            )

            color_space_plot = self._generate_color_space_plot(
                lower_limit, upper_limit, center
            )

            return {**results, "color_space_plot": color_space_plot}

        except Exception as e:
            self.config.logger.error(f"Image analysis failed for {image_path}: {e}")
            return {}

    def _generate_color_space_plot(
        self,
        lower_limit: np.ndarray,
        upper_limit: np.ndarray,
        center: Tuple[float, float, float],
        gradient_height: int = 25,
        num_lines: int = 5,
    ) -> str:
        """
        Generate a color space plot with gradient.

        Args:
            lower_limit (np.ndarray): Lower HSV color limit
            upper_limit (np.ndarray): Upper HSV color limit
            center (Tuple): Center color
            gradient_height (int): Height of gradient lines
            num_lines (int): Number of gradient lines

        Returns:
            str: Path to generated plot
        """
        try:
            # Convert color spaces
            lower_rgb = cv2.cvtColor(np.uint8([[lower_limit]]), cv2.COLOR_HSV2RGB)[0][0]
            upper_rgb = cv2.cvtColor(np.uint8([[upper_limit]]), cv2.COLOR_HSV2RGB)[0][0]
            center_rgb = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2RGB)[0][0]

            # Create gradient
            gradient = np.linspace(lower_rgb, upper_rgb, 256) / 255
            gradient_resized = np.repeat(
                gradient.reshape(1, -1, 3), gradient_height, axis=0
            )
            gradient_stacked = np.vstack([gradient_resized] * num_lines)

            # Plot
            plt.figure(figsize=(10, 3))
            plt.imshow(gradient_stacked)
            plt.axis("off")

            # Save plot
            output_path = os.path.join(self.config.output_dir, "color_space_plot.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()

            return output_path

        except Exception as e:
            self.config.logger.error(f"Color space plot generation failed: {e}")
            return ""

    def generate_report(self, analysis_results: Dict[str, Any], image_path: str) -> str:
        """
        Generate an HTML report from analysis results.

        Args:
            analysis_results (Dict): Results from image analysis
            image_path (str): Path to original image

        Returns:
            str: Path to generated HTML report
        """
        try:
            # Extract filename details
            filename = os.path.basename(image_path)
            name_without_ext, _ = os.path.splitext(filename)
            part_number, thickness = name_without_ext.split("_")

            # Prepare output directory
            output_dir = os.path.join(
                self.config.output_dir, f"{part_number}_{thickness}"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Render HTML report
            html_content = self.template.render(
                image_path=filename,
                part_number=part_number,
                thickness=thickness,
                logo=self.config.logo,
                today=self.config.today,
                author=self.config.author,
                department=self.config.department,
                report_title=self.config.report_title,
                **analysis_results,
            )

            # Save report
            report_path = os.path.join(output_dir, f"{name_without_ext}.html")
            with open(report_path, "w") as report_file:
                report_file.write(html_content)

            return report_path

        except Exception as e:
            self.config.logger.error(f"Report generation failed: {e}")
            return ""

    def process_batch(self):
        """
        Process all images in the configured image directory.
        """
        for image_file in tqdm(
            os.listdir(self.config.image_dir), desc="Processing images"
        ):
            if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    image_path = os.path.join(self.config.image_dir, image_file)

                    # Analyze image
                    analysis_results = self.analyze_image(image_path)

                    # Generate report
                    self.generate_report(analysis_results, image_path)

                except Exception as e:
                    self.config.logger.error(f"Failed to process {image_file}: {e}")


def main():
    """
    Main entry point for the report generation process.
    """
    # Optional: customize configuration
    config = ReportConfig(image_dir="path/to/your/images", output_dir="path/to/output")

    # Initialize and run
    report_generator = ImageReportGenerator(config)
    report_generator.process_batch()


if __name__ == "__main__":
    main()
