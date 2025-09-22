"""
This module provides the `ReportGenerator` class, which is responsible for
creating comprehensive analysis reports in various formats (HTML, PDF).

It integrates with Jinja2 for HTML templating, WeasyPrint for HTML to PDF conversion,
and ReportLab for an alternative PDF generation method. It also handles the
generation of various plots and visualizations for the reports.
"""

import os
import datetime
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from pathlib import Path

from src import config
from src.reporting.archiver import ReportArchiver

# Graceful imports for reporting libraries
try:
    from jinja2 import Environment, FileSystemLoader
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from matplotlib.patches import Rectangle

class ReportGenerator:
    """
    A class to generate analysis reports in HTML and PDF formats.

    This class orchestrates the creation of detailed reports by:
    - Managing output directories.
    - Generating various plots (pie charts, color space diagrams, hue-saturation diagrams).
    - Processing debug information and dataset details for inclusion.
    - Rendering HTML reports using Jinja2 templates.
    - Converting HTML reports to PDF using WeasyPrint.
    - Generating alternative PDF reports using ReportLab.
    - Archiving report data for later regeneration.
    """

    def __init__(self, project_name: str, sample_name: str = None, debug_mode: bool = False):
        """
        Initializes the ReportGenerator.

        Sets up output directories and initializes Jinja2 environment for HTML templating.

        Args:
            project_name (str): The name of the project for which the report is being generated.
            sample_name (str, optional): The name of the specific sample being analyzed.
                                         If provided, output will be organized under a sample-specific directory.
                                         Defaults to None.
            debug_mode (bool, optional): If True, enables debug-specific features in the report
                                         (e.g., more detailed sections, debug template).
                                         Defaults to False.
        """
        self.project_name = project_name
        self.sample_name = sample_name
        self.debug_mode = debug_mode
        
        if sample_name:
            self.project_output_dir = config.OUTPUT_DIR / project_name / sample_name
        else:
            self.project_output_dir = config.OUTPUT_DIR / project_name
        
        os.makedirs(self.project_output_dir, exist_ok=True)

        if HAS_WEASYPRINT:
            self.env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR))
            self.env.filters['basename'] = os.path.basename
            template_name = "Report_Debug.html" if debug_mode else "Report_Default.html"
            try:
                self.template = self.env.get_template(template_name)
            except Exception as e:
                print(f"[WARNING] Could not load template {template_name}: {e}")
                self.template = None
        else:
            self.env = None
            self.template = None

    def get_step_output_dir(self, step_name: str) -> Path:
        """
        Creates and returns the path to a subdirectory for a specific analysis step.

        This helps organize output files generated during different stages of the pipeline.

        Args:
            step_name (str): The name of the analysis step (e.g., "color_correction", "alignment").

        Returns:
            Path: The absolute path to the output directory for the specified step.
        """
        step_dir = self.project_output_dir / step_name
        os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def _generate_pie_chart(self, matched_pixels: int, total_pixels: int, selected_colors: list) -> str:
        """
        Generates a pie chart visualizing the percentage of matched pixels versus unmatched pixels.

        The chart is saved as a PNG image in the reporting output directory.

        Args:
            matched_pixels (int): The number of pixels that matched the target color range.
            total_pixels (int): The total number of pixels considered in the analysis.
            selected_colors (list): A list of dictionaries containing color information.
                                    The first item's "rgb" value is used for the pie chart.

        Returns:
            str: The relative path to the generated pie chart image.
        """
        step_dir = self.get_step_output_dir("reporting")
        labels = ["Matched Pixels", "Unmatched Pixels"]
        sizes = [matched_pixels, total_pixels - matched_pixels]
        
        # Use the rgb value from the first selected color
        pie_color = [0.5, 0.5, 0.5] # Default gray
        if selected_colors and 'rgb' in selected_colors[0]:
            pie_color = np.array(selected_colors[0]['rgb']) / 255.0

        colors_pie = [pie_color, "darkgray"]
        plt.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=140)
        plt.axis("equal")
        pie_chart_path = step_dir / "pie_chart.png"
        plt.savefig(pie_chart_path)
        plt.close()
        return os.path.relpath(pie_chart_path, self.project_output_dir)

    def _generate_color_space_plot(self, lower_limit: np.ndarray, upper_limit: np.ndarray, center: tuple, gradient_height=25, num_lines=5) -> str:
        """
        Generates a visual representation of the defined HSV color space.

        This plot shows a gradient from the lower to the upper HSV limit, with the
        center color highlighted. It is saved as a PNG image.

        Args:
            lower_limit (np.ndarray): NumPy array [H, S, V] representing the lower HSV bounds.
            upper_limit (np.ndarray): NumPy array [H, S, V] representing the upper HSV bounds.
            center (tuple): Tuple (H, S, V) representing the center of the HSV color range.
            gradient_height (int, optional): Height of the color gradient strip in pixels. Defaults to 25.
            num_lines (int, optional): Number of times the gradient strip is stacked. Defaults to 5.

        Returns:
            str: The relative path to the generated color space plot image.
        """
        step_dir = self.get_step_output_dir("reporting")
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

        color_space_plot_path = step_dir / "color_space_plot.png"
        plt.savefig(color_space_plot_path)
        plt.close()
        return os.path.relpath(color_space_plot_path, self.project_output_dir)

    def plot_hue_saturation_diagram(self, image_bgr: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray, output_path: str) -> str:
        """
        Generates a scatter plot showing the Hue-Saturation distribution of an image's pixels.

        The plot highlights the defined HSV color range with a red rectangle.
        A random sample of pixels is used for plotting to improve performance for large images.

        Args:
            image_bgr (np.ndarray): The input image in BGR format.
            lower_hsv (np.ndarray): NumPy array [H, S, V] representing the lower HSV bounds.
            upper_hsv (np.ndarray): NumPy array [H, S, V] representing the upper HSV bounds.
            output_path (str): The filename for the generated plot image.

        Returns:
            str: The relative path to the generated hue-saturation diagram image.
        """
        step_dir = self.get_step_output_dir("reporting")
        hsv_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_image)

        h_flat, s_flat = h.flatten(), s.flatten()
        sample_size = min(len(h_flat), 20000)
        indices = np.random.choice(len(h_flat), sample_size, replace=False)
        
        h_sample, s_sample, v_sample = h_flat[indices], s_flat[indices], v.flatten()[indices]

        sampled_hsv = np.stack((h_sample, s_sample, v_sample), axis=-1)
        sampled_rgb = cv.cvtColor(np.uint8([sampled_hsv]), cv.COLOR_HSV2RGB)[0] / 255.0

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(h_sample, s_sample, c=sampled_rgb, alpha=0.5, s=10, label='Image Pixels')

        rect_width, rect_height = int(upper_hsv[0]) - int(lower_hsv[0]), int(upper_hsv[1]) - int(lower_hsv[1])
        selection_rect = Rectangle((lower_hsv[0], lower_hsv[1]), rect_width, rect_height, linewidth=2, edgecolor='r', facecolor='none', label='Selected Range')
        ax.add_patch(selection_rect)

        ax.set(xlabel='Hue (0-179)', ylabel='Saturation (0-255)', title='Hue-Saturation Color Distribution', xlim=(0, 180), ylim=(0, 256))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plot_full_path = step_dir / output_path
        plt.savefig(plot_full_path)
        plt.close(fig)
        return os.path.relpath(plot_full_path, self.project_output_dir)

    def _generate_dataset_color_space_plot(self, dataset_debug_info: list, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> str:
        """
        Generates a scatter plot visualizing the color distribution of the training dataset.

        Each point represents an extracted color from a training image, colored by its average RGB.
        The calculated HSV range is overlaid as a red rectangle.

        Args:
            dataset_debug_info (list): A list of dictionaries containing debug information
                                       about the dataset, including extracted HSV colors.
            lower_hsv (np.ndarray): NumPy array [H, S, V] representing the lower HSV bounds
                                    of the calculated range.
            upper_hsv (np.ndarray): NumPy array [H, S, V] representing the upper HSV bounds
                                    of the calculated range.

        Returns:
            str: The relative path to the generated dataset color space plot image.
        """
        step_dir = self.get_step_output_dir("reporting")
        fig, ax = plt.subplots(figsize=(12, 8))

        for item in dataset_debug_info:
            hsv_colors = item.get('hsv_colors', [])
            if not hsv_colors: continue

            h_vals, s_vals = [c[0] for c in hsv_colors], [c[1] for c in hsv_colors]
            avg_hsv = np.uint8([[[np.mean(h_vals), np.mean(s_vals), np.mean([c[2] for c in hsv_colors])]]])
            avg_rgb = cv.cvtColor(avg_hsv, cv.COLOR_HSV2RGB)[0][0] / 255.0

            ax.scatter(h_vals, s_vals, color=avg_rgb, label=os.path.basename(item['path']), s=50, alpha=0.8, edgecolors='black')

        rect_width, rect_height = int(upper_hsv[0]) - int(lower_hsv[0]), int(upper_hsv[1]) - int(lower_hsv[1])
        selection_rect = Rectangle((lower_hsv[0], lower_hsv[1]), rect_width, rect_height, linewidth=2, edgecolor='r', facecolor='none', label='Calculated Range')
        ax.add_patch(selection_rect)

        ax.set(xlabel='Hue (0-179)', ylabel='Saturation (0-255)', title='Training Data Color Space Definition', xlim=(0, 180), ylim=(0, 256))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        plot_path = step_dir / "dataset_color_space.png"
        plt.savefig(plot_path)
        plt.close(fig)
        return os.path.relpath(plot_path, self.project_output_dir)

    def _process_dataset_debug_info(self, dataset_debug_info: list) -> list:
        """
        Processes raw dataset debug information to prepare it for reporting.

        This includes drawing sample points on images, generating color palettes,
        and converting color formats for display in the report.

        Args:
            dataset_debug_info (list): A list of dictionaries containing raw debug
                                       information for each dataset item.

        Returns:
            list: A list of processed dictionaries, with added paths to images
                  with points, and detailed color information for reporting.
        """
        step_dir = self.get_step_output_dir("reporting")
        processed_items = []
        for i, item in enumerate(dataset_debug_info):
            print(f"--- Debugging item in _process_dataset_debug_info ---")
            print(f"Item type: {type(item)}")
            print(f"Item content: {item}")
            print(f"-----------------------------------------------------")
            processed_item = item.copy()
            img = cv.imread(item['path'])
            if img is None: continue

            # Draw points on the image
            img_with_points = img.copy()
            if item.get('method') == 'points' and item.get('points'):
                for point in item['points']:
                    cv.circle(img_with_points, (point['x'], point['y']), point.get('radius', 7), (0, 0, 255), 2)
            
            img_path = step_dir / f"dataset_{i}_with_points.png"
            cv.imwrite(str(img_path), img_with_points)
            processed_item['image_with_points_path'] = os.path.relpath(img_path, self.project_output_dir)

            # Create color details for each sample color
            processed_item['color_details'] = []
            bgr_colors = item.get('bgr_colors', [])
            hsv_colors = item.get('hsv_colors', [])

            for j, (bgr_color, hsv_color) in enumerate(zip(bgr_colors, hsv_colors)):
                # Create palette image
                palette = np.full((100, 100, 3), tuple(map(int, bgr_color)), np.uint8)
                palette_path = step_dir / f"dataset_{i}_palette_{j}.png"
                cv.imwrite(str(palette_path), palette)

                # Get RGB version for hex code
                h, s, v = hsv_color
                rgb_color = cv.cvtColor(np.uint8([[[h, s, v]]]), cv.COLOR_HSV2RGB)[0][0]

                # Store all details
                processed_item['color_details'].append({
                    'palette_path': os.path.relpath(palette_path, self.project_output_dir),
                    'hsv': f"({int(h)}, {int(s)}, {int(v)})",
                    'bgr': f"({int(bgr_color[0])}, {int(bgr_color[1])}, {int(bgr_color[2])})",
                    'rgb': f"({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})",
                    'hex': f"#{int(rgb_color[0]):02x}{int(rgb_color[1]):02x}{int(rgb_color[2]):02x}"
                })
            
            processed_items.append(processed_item)
        return processed_items

    def _generate_reportlab_pdf(self, report_data: dict, base_dir: Path, pdf_path: str):
        """
        Generates a PDF report using the ReportLab library.

        This method constructs the PDF document by adding various elements like
        text, tables, and images based on the provided `report_data`.

        Args:
            report_data (dict): A dictionary containing all the data required for the report.
            base_dir (Path): The base directory from which relative image paths are resolved.
            pdf_path (str): The full path where the generated PDF will be saved.
        """
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=inch/2, bottomMargin=inch/2)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = styles['h1']
        h2_style = styles['h2']
        h3_style = styles['h3']
        body_style = styles['BodyText']
        code_style = styles['Code']
        path_style = ParagraphStyle('path_style', parent=styles['Italic'], fontSize=8, textColor=colors.grey)

        # Enhanced add_image helper function
        def add_image(path_key, caption_text, width=6*inch):
            img_path_str = report_data.get(path_key)
            # Handle cases where the path_key itself is the path (for pipeline images)
            if not img_path_str and isinstance(path_key, str) and (path_key.endswith('.png') or path_key.endswith('.jpg')):
                img_path_str = path_key

            story.append(Paragraph(caption_text, h3_style))
            if not img_path_str:
                story.append(Paragraph(f"<i>(Image path not found in report data for key: {path_key})</i>", body_style))
                story.append(Spacer(1, 0.1*inch))
                return

            img_path = base_dir / img_path_str
            if img_path.is_file():
                try:
                    img = Image(str(img_path), width=width, height=width/1.5, kind='proportional')
                    story.append(img)
                    story.append(Paragraph(f"Path: {img_path_str}", path_style))
                    story.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    story.append(Paragraph(f"<i>Could not load image: {os.path.basename(img_path_str)} ({e})</i>", body_style))
            else:
                story.append(Paragraph(f"<i>(Image not found at {img_path_str})</i>", body_style))
                story.append(Spacer(1, 0.1*inch))

        # --- Main Report Content ---
        story.append(Paragraph(report_data.get('report_title', 'Analysis Report'), title_style))
        story.append(Spacer(1, 0.2*inch))

        meta_table = Table([
            ['Author:', report_data.get('author', 'N/A'), 'Part Number:', report_data.get('part_number', 'N/A')],
            ['Date:', report_data.get('today', 'N/A'), 'Thickness:', report_data.get('thickness', 'N/A')]
        ], colWidths=[1*inch, 2.5*inch, 1*inch, 2.5*inch])
        story.append(meta_table)
        story.append(Spacer(1, 0.2*inch))

        add_image('analyzed_image_path', 'Image Before Color Analysis')
        add_image('contours_image_path', 'Image After Color Analysis (with Contours)')
        add_image('pie_chart_path', 'Pixel-Percentage Pie Chart', width=4*inch)

        # --- Debug Section ---
        if report_data.get("debug_data"):
            debug_data = report_data["debug_data"]
            story.append(PageBreak())
            story.append(Paragraph("Debug Report", title_style))
            story.append(Spacer(1, 0.2*inch))

            story.append(Paragraph("Analysis Details", h2_style))
            debug_table_data = []
            for key, value in debug_data.items():
                if key in ['image_pipeline', 'dataset_debug_info', 'symmetry_visualizations', 'symmetry_results']:
                    continue
                
                try:
                    value_str = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                except TypeError:
                    value_str = "<Contains non-serializable data>"

                debug_table_data.append([Paragraph(str(key), body_style), Paragraph(value_str.replace('\n', '<br/>'), code_style)])

            if debug_table_data:
                tbl = Table(debug_table_data, colWidths=[2*inch, 4.5*inch], repeatRows=1)
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ]))
                story.append(tbl)
            story.append(Spacer(1, 0.2*inch))

            if 'image_pipeline' in debug_data:
                story.append(PageBreak())
                story.append(Paragraph("Image Processing Pipeline", h2_style))
                story.append(Spacer(1, 0.2*inch))
                for step in debug_data['image_pipeline']:
                    add_image(step['path'], step['title'])
            
            if debug_data.get('symmetry_results'):
                story.append(PageBreak())
                story.append(Paragraph("Symmetry Analysis", h2_style))
                story.append(Paragraph("Scores:", h3_style))
                scores_only = {k: v.get('score', 'N/A') for k, v in debug_data['symmetry_results'].items()}
                symmetry_scores = [[k,v] for k,v in scores_only.items()]
                if symmetry_scores:
                    tbl = Table(symmetry_scores, colWidths=[2.5*inch, 4*inch])
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                        ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ]))
                    story.append(tbl)
                story.append(Spacer(1, 0.2*inch))
                
                if debug_data.get('symmetry_visualizations'):
                    story.append(Paragraph("Visualizations:", h3_style))
                    for step in debug_data['symmetry_visualizations']:
                        add_image(step['path'], step['title'])

            if report_data.get('dataset_color_space_plot_path'):
                story.append(PageBreak())
                story.append(Paragraph("Dataset Color Space Definition", h2_style))
                add_image('dataset_color_space_plot_path', 'Training Data Distribution')
                if report_data.get('dataset_debug_info'):
                    for item in report_data['dataset_debug_info']:
                        story.append(Paragraph(f"Sample: {os.path.basename(item['path'])}", h3_style))
                        add_image(item['image_with_points_path'], 'Image with Sample Points', width=4*inch)
                        
                        color_details = item.get('color_details', [])
                        if color_details:
                            story.append(Paragraph("Sampled Colors:", body_style))
                            color_table_data = [['Palette', 'HSV', 'BGR', 'RGB', 'HEX']]
                            for detail in color_details:
                                try:
                                    palette_img = Image(base_dir / detail['palette_path'], width=0.5*inch, height=0.5*inch)
                                    color_table_data.append([
                                        palette_img,
                                        Paragraph(detail['hsv'], code_style),
                                        Paragraph(detail['bgr'], code_style),
                                        Paragraph(detail['rgb'], code_style),
                                        Paragraph(detail['hex'], code_style)
                                    ])
                                except Exception:
                                    color_table_data.append(['(Img Fail)', detail['hsv'], detail['bgr'], detail['rgb'], detail['hex']])
                            
                            color_tbl = Table(color_table_data, colWidths=[0.7*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.3*inch])
                            color_tbl.setStyle(TableStyle([
                                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                                ('GRID', (0,0), (-1,-1), 1, colors.black),
                                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                            ]))
                            story.append(color_tbl)
                        story.append(Spacer(1, 0.2*inch))

        doc.build(story)
        if self.debug_mode:
            print(f"ReportLab PDF report saved to {pdf_path}")

    def generate_report(self, analysis_results: dict, metadata: dict, debug_data: dict = None, report_type: str = 'all'):
        """
        Generates all specified report files (HTML, PDF) from the analysis results.

        This is the main method for report generation, orchestrating the creation
        of plots, processing data, rendering templates, and archiving the report.

        Args:
            analysis_results (dict): A dictionary containing the results of the image analysis.
            metadata (dict): A dictionary containing metadata for the report (e.g., part number, thickness).
            debug_data (dict, optional): A dictionary containing additional debug information
                                         to be included in the report. Defaults to None.
            report_type (str, optional): The type of report(s) to generate. Can be 'html',
                                         'reportlab', or 'all'. Defaults to 'all'.

        Returns:
            dict: The dictionary of template variables used to generate the report.
        """
        pie_chart_path = self._generate_pie_chart(analysis_results['matched_pixels'], analysis_results['total_pixels'], analysis_results['selected_colors'])
        color_space_plot_path = self._generate_color_space_plot(analysis_results['lower_limit'], analysis_results['upper_limit'], analysis_results['center_color'])
        chromaticity_diagram_path = self.plot_hue_saturation_diagram(analysis_results['original_image'], analysis_results['lower_limit'], analysis_results['upper_limit'], "hue_saturation_diagram.png")

        reporting_dir = self.get_step_output_dir("reporting")
        orig_img_path = reporting_dir / Path(analysis_results['original_image_path']).name
        shutil.copy(analysis_results['original_image_path'], orig_img_path)

        logo_path = ""
        if config.LOGO_PATH.is_file():
            logo_dest = reporting_dir / config.LOGO_PATH.name
            shutil.copy(config.LOGO_PATH, logo_dest)
            logo_path = os.path.relpath(logo_dest, self.project_output_dir)

        processed_dataset_info, dataset_color_space_plot_path = None, None
        if debug_data and 'dataset_debug_info' in debug_data:
            processed_dataset_info = self._process_dataset_debug_info(debug_data['dataset_debug_info'])
            dataset_color_space_plot_path = self._generate_dataset_color_space_plot(debug_data['dataset_debug_info'], analysis_results['lower_limit'], analysis_results['upper_limit'])

        template_vars = {
            "project_name": self.project_name,
            "author": config.AUTHOR, "department": config.DEPARTMENT,
            "report_title": f"{config.REPORT_TITLE} - {self.project_name}",
            "logo": logo_path,
            "today": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S" if self.debug_mode else "%Y-%m-%d"),
            "part_number": metadata.get("part_number", "N/A"),
            "thickness": metadata.get("thickness", "N/A"),
            "image_path": os.path.relpath(orig_img_path, self.project_output_dir),
            "analyzed_image_path": os.path.relpath(analysis_results['input_to_analysis_path'], self.project_output_dir),
            "contours_image_path": os.path.relpath(analysis_results['contours_image_path'], self.project_output_dir),
            "color_space_plot_path": color_space_plot_path,
            "hsv_diagram_path": chromaticity_diagram_path,
            "pie_chart_path": pie_chart_path,
            "debug_data": debug_data,
            "dataset_debug_info": processed_dataset_info,
            "dataset_color_space_plot_path": dataset_color_space_plot_path,
            "report_type": report_type,
        }

        template_vars['analysis_results_raw'] = analysis_results

        self.generate_from_archived_data(template_vars, self.project_output_dir, is_regeneration=False)

        # Create a serializable object that includes the numpy arrays
        serializable_data = template_vars.copy()
        serializable_data['analysis_results_raw'] = analysis_results
        serializable_data['numpy_images'] = {
            'original': analysis_results.get('original_image'),
            'processed': analysis_results.get('processed_image'),
            'mask': analysis_results.get('mask'),
            'negative_mask': analysis_results.get('negative_mask'),
        }

        report_archiver = ReportArchiver(self.project_output_dir, debug_mode=self.debug_mode)
        report_archiver.archive_report(serializable_data)

        return template_vars

    def generate_from_archived_data(self, report_data: dict, base_dir: Path, is_regeneration: bool = True):
        """
        Generates HTML and PDF reports from a pre-existing data dictionary (e.g., from an archive).

        This method is used for regenerating reports without re-running the full analysis pipeline.

        Args:
            report_data (dict): A dictionary containing all the data required for the report.
            base_dir (Path): The base directory from which relative image paths are resolved.
            is_regeneration (bool, optional): If True, prefixes output filenames with "regenerated_".
                                             Defaults to True.
        """
        report_type = report_data.get('report_type', 'all')
        prefix = ""
        if self.debug_mode:
            prefix += "debug_"
        if is_regeneration:
            prefix += "regenerated_"
        part_number = report_data.get('part_number', 'report')

        # HTML and WeasyPrint PDF Generation
        if report_type in ['html', 'all']:
            if HAS_WEASYPRINT and self.template:
                html_content = self.template.render(report_data)
                report_html_path = self.project_output_dir / f"{prefix}{part_number}.html"
                report_pdf_path = self.project_output_dir / f"{prefix}{part_number}.pdf"

                with open(report_html_path, "w", encoding='utf-8') as f: f.write(html_content)
                if self.debug_mode:
                    print(f"HTML report saved to {report_html_path}")

                HTML(string=html_content, base_url=str(base_dir)).write_pdf(report_pdf_path)
                if self.debug_mode:
                    print(f"PDF report saved to {report_pdf_path}")
            elif not HAS_WEASYPRINT:
                print("[WARNING] jinja2 and/or weasyprint not installed. Skipping HTML/WeasyPrint PDF generation.")

        # ReportLab PDF Generation
        if report_type in ['reportlab', 'all']:
            if HAS_REPORTLAB:
                reportlab_pdf_path = self.project_output_dir / f"{prefix}{part_number}_reportlab.pdf"
                try:
                    self._generate_reportlab_pdf(report_data, base_dir, reportlab_pdf_path)
                except Exception as e:
                    print(f"[WARNING] Failed to generate ReportLab PDF: {e}")
            else:
                print("[WARNING] reportlab not installed. Skipping ReportLab PDF generation.")