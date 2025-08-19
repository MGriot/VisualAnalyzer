import argparse
import os
import cv2
import numpy as np
import warnings

# Suppress NumPy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy')

from src.color_analysis.project_manager import ProjectManager
from src.color_analysis.analyzer import ColorAnalyzer
from src.color_correction.corrector import ColorCorrector
from src.reporting.generator import ReportGenerator
from src.utils.image_utils import load_image, save_image
from src.utils.video_utils import process_video_stream
from src import config

def main():
    parser = argparse.ArgumentParser(description="Visual Analyzer for Color Correction and Analysis.")
    parser.add_argument("--project", type=str, help="Name of the project to use.")
    parser.add_argument("--image", type=str, help="Path to a single image file for analysis.")
    parser.add_argument("--video", type=str, help="Path to a video file for analysis.")
    parser.add_argument("--camera", action="store_true", help="Use live camera stream for analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output.")
    parser.add_argument("--aggregate", action="store_true", help="Enable aggregation of nearby matched pixel areas.")
    parser.add_argument("--blur", action="store_true", help="Enable blurring of the input image before color matching.")
    parser.add_argument("--alignment", action="store_true", help="Enable image alignment with a technical drawing.")
    parser.add_argument("--color-alignment", action="store_true", help="Enable color alignment (correction).")

    args = parser.parse_args()

    project_manager = ProjectManager()
    available_projects = project_manager.list_projects()

    if not available_projects:
        print("No projects found. Please create a project in the 'data/projects' directory.")
        return

    selected_project = args.project
    if not selected_project:
        print("Available projects:")
        for i, project in enumerate(available_projects):
            print(f"{i+1}. {project}")
        while True:
            try:
                choice = input("Select a project by number or name: ")
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_projects):
                        selected_project = available_projects[choice_idx]
                        break
                    else:
                        print("Invalid project number.")
                elif choice in available_projects:
                    selected_project = choice
                    break
                else:
                    print("Invalid project name or number.")
            except EOFError:
                print("\nExiting due to no further input.")
                return

    try:
        project_data = project_manager.get_project_data(selected_project, debug_mode=args.debug)
        correction_matrix = project_data['correction_matrix']
        lower_hsv = project_data['lower_hsv']
        upper_hsv = project_data['upper_hsv']
        center_hsv = project_data['center_hsv']

        print(f"Loaded project '{selected_project}' with HSV range: {lower_hsv} - {upper_hsv}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading project: {e}")
        return

    color_corrector = ColorCorrector()
    color_analyzer = ColorAnalyzer()
    report_generator = ReportGenerator(selected_project, debug_mode=args.debug)

    # Convert center_hsv to RGB for reporting
    center_rgb = cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

    if args.image:
        print(f"Processing single image: {args.image}")
        try:
            project_files = project_manager.get_project_file_paths(selected_project, debug_mode=args.debug)
            technical_drawing_path = project_files.get("technical_drawing")

            original_input_image_bgr, _ = load_image(args.image) # Load the original input image
            if original_input_image_bgr is None:
                raise ValueError(f"Could not load image {args.image}")

            # Apply color correction if a matrix was calculated
            image_to_analyze = original_input_image_bgr.copy()
            if args.color_alignment and correction_matrix is not None:
                image_to_analyze = color_corrector.apply_color_correction(original_input_image_bgr, correction_matrix)

            if args.debug: print(f"[DEBUG] Shape of image being analyzed: {image_to_analyze.shape}")

            # Save the image to analyze for reporting (it's either original or corrected)
            analyzed_image_filename = f"analyzed_{os.path.basename(args.image)}"
            analyzed_image_path_for_report = report_generator.project_output_dir / analyzed_image_filename
            save_image(str(analyzed_image_path_for_report), image_to_analyze)

            analysis_results = color_analyzer.process_image(
                image=image_to_analyze,
                image_path=args.image,
                lower_hsv=lower_hsv,
                upper_hsv=upper_hsv,
                output_dir=str(report_generator.project_output_dir),
                debug_mode=args.debug,
                aggregate_mode=args.aggregate,
                blur_mode=args.blur,
                alignment_mode=args.alignment,
                drawing_path=str(technical_drawing_path) if technical_drawing_path else None
            )

            # Prepare metadata for reporting
            analysis_results['original_input_image_path'] = args.image # Path to the original input image
            analysis_results['analyzed_image_path'] = str(analyzed_image_path_for_report) # Path to the image that was analyzed (original or corrected)
            analysis_results['selected_colors'] = {"RGB": center_rgb} # Use RGB for reporting
            analysis_results['lower_limit'] = lower_hsv
            analysis_results['upper_limit'] = upper_hsv
            analysis_results['center_color'] = center_hsv

            # Collect debug data for the report
            debug_data_for_report = {}
            if args.debug:
                debug_data_for_report["Project Name"] = selected_project
                debug_data_for_report["HSV Range (Lower)"] = str(lower_hsv.tolist())
                debug_data_for_report["HSV Range (Upper)"] = str(upper_hsv.tolist())
                debug_data_for_report["HSV Range (Center)"] = str(center_hsv.tolist())
                debug_data_for_report["Matched Pixels (Raw)"] = analysis_results['matched_pixels']
                debug_data_for_report["Percentage Matched (Raw)"] = f"{analysis_results['percentage']:.2f}%"
                if args.aggregate:
                    # To show original matched pixels before aggregation, ColorAnalyzer needs to return it
                    # For now, we'll just show the final matched pixels
                    pass # Placeholder for more detailed aggregation debug data
                if args.blur:
                    debug_data_for_report["Blur Applied"] = "Yes"
                    # Add blur kernel size if it becomes configurable

            # Extract part_number and thickness from filename if possible
            file_name_without_ext = os.path.splitext(os.path.basename(args.image))[0]
            part_number = file_name_without_ext
            thickness = "N/A" # Default, can be extracted if naming convention allows
            if '_' in file_name_without_ext:
                parts = file_name_without_ext.split('_')
                if len(parts) >= 2:
                    part_number = parts[0]
                    thickness = parts[1]

            metadata = {"part_number": part_number, "thickness": thickness}
            report_generator.generate_report(analysis_results, metadata, debug_data=debug_data_for_report)

        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing image: {e}")

    elif args.video:
        print(f"Processing video file: {args.video}")
        def video_frame_processor(frame):
            process_and_display_frame(frame, str(report_generator.project_output_dir), correction_matrix)

        process_video_stream(args.video, video_frame_processor)

    elif args.camera:
        print("Starting live camera stream. Press 'q' to quit.")
        def camera_frame_processor(frame):
            process_and_display_frame(frame, str(report_generator.project_output_dir), correction_matrix)

        process_video_stream(0, camera_frame_processor)

    else:
        print("Please specify an input: --image <path>, --video <path>, or --camera.")

if __name__ == "__main__":
    main()
