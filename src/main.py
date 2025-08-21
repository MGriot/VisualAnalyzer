import argparse
import os
import cv2
import numpy as np
import warnings

# Suppress NumPy warnings that might arise from operations like empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy')

# Import core modules for the Visual Analyzer application
from src.color_analysis.project_manager import ProjectManager
from src.color_analysis.analyzer import ColorAnalyzer
from src.color_correction.corrector import ColorCorrector
from src.alignment.aligner import Aligner
from src.reporting.generator import ReportGenerator
from src.utils.image_utils import load_image, save_image
from src.utils.video_utils import process_video_stream
from src import config

def main():
    """
    Main function to parse command-line arguments and initiate the analysis.
    """
    parser = argparse.ArgumentParser(description="Visual Analyzer for Color Correction and Analysis.")
    parser.add_argument("--project", type=str, help="Name of the project to use.")
    parser.add_argument("--image", type=str, help="Path to a single image file for analysis.")
    parser.add_argument("--video", type=str, help="Path to a video file for analysis.")
    parser.add_argument("--camera", action="store_true", help="Use live camera stream for analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--aggregate", action="store_true", help="Enable aggregation of matched pixel areas.")
    parser.add_argument("--blur", action="store_true", help="Enable blurring of the input image.")
    parser.add_argument("--alignment", action="store_true", help="Enable geometrical alignment.")
    parser.add_argument("--drawing", type=str, help="Path to a technical drawing for masking.")
    parser.add_argument("--color-alignment", action="store_true", help="Enable color correction.")

    args = parser.parse_args()
    run_analysis_from_gui(args)

def run_analysis_from_gui(args):
    """
    Core analysis function that orchestrates the image processing pipeline.
    """
    project_manager = ProjectManager()
    try:
        project_data = project_manager.get_project_data(args.project, debug_mode=args.debug)
        correction_matrix, lower_hsv, upper_hsv, center_hsv = (
            project_data['correction_matrix'],
            project_data['lower_hsv'],
            project_data['upper_hsv'],
            project_data['center_hsv'],
        )
        print(f"Loaded project '{args.project}' with HSV range: {lower_hsv} - {upper_hsv}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading project: {e}")
        return

    color_corrector = ColorCorrector()
    color_analyzer = ColorAnalyzer()
    report_generator = ReportGenerator(args.project, debug_mode=args.debug)
    center_rgb = cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

    if args.image:
        print(f"Processing single image: {args.image}")
        try:
            debug_image_pipeline = []
            pipeline_step_counter = 1

            original_input_image_bgr, _ = load_image(args.image)
            if original_input_image_bgr is None:
                raise ValueError(f"Could not load image {args.image}")
            
            if args.debug:
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Original Input", "path": os.path.basename(args.image)})
                pipeline_step_counter += 1

            image_for_processing = original_input_image_bgr.copy()
            alignment_data = None

            if args.color_alignment and correction_matrix is not None:
                image_for_processing = color_corrector.apply_color_correction(image_for_processing, correction_matrix)
                if args.debug:
                    path = str(report_generator.project_output_dir / "color_corrected_debug.png")
                    save_image(path, image_for_processing)
                    debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Color Correction", "path": os.path.basename(path)})
                    pipeline_step_counter += 1

            if args.alignment:
                project_files = project_manager.get_project_file_paths(args.project, debug_mode=args.debug)
                aruco_marker_map = project_files.get("aruco_marker_map")
                aruco_output_size = project_files.get("aruco_output_size")
                if aruco_marker_map and aruco_output_size:
                    temp_aligner = Aligner(debug_mode=args.debug, output_dir=str(report_generator.project_output_dir))
                    alignment_result = temp_aligner.align_image(image=image_for_processing, marker_map=aruco_marker_map, output_size_wh=tuple(aruco_output_size))
                    if alignment_result:
                        image_for_processing, alignment_data = alignment_result
                        if args.debug:
                            path = str(report_generator.project_output_dir / "geometrically_aligned_debug.png")
                            save_image(path, image_for_processing)
                            debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Geometrical Alignment", "path": os.path.basename(path)})
                            pipeline_step_counter += 1
                    else:
                        print("[WARNING] Image alignment failed.")
                else:
                    print("[WARNING] Alignment enabled but ArUco config not found.")

            if args.drawing:
                try:
                    drawing_mask_image, _ = load_image(args.drawing, handle_transparency=False)
                    if drawing_mask_image is not None:
                        if len(drawing_mask_image.shape) == 3:
                            drawing_mask_image = cv2.cvtColor(drawing_mask_image, cv2.COLOR_BGR2GRAY)
                        _, drawing_mask_binary = cv2.threshold(drawing_mask_image, 128, 255, cv2.THRESH_BINARY)
                        if drawing_mask_binary.shape[:2] != image_for_processing.shape[:2]:
                            drawing_mask_binary = cv2.resize(drawing_mask_binary, (image_for_processing.shape[1], image_for_processing.shape[0]), interpolation=cv2.INTER_NEAREST)
                        image_for_processing = cv2.bitwise_and(image_for_processing, image_for_processing, mask=drawing_mask_binary)
                        if args.debug:
                            path = str(report_generator.project_output_dir / "masked_image_debug.png")
                            save_image(path, image_for_processing)
                            debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Masking with Drawing", "path": os.path.basename(path)})
                            pipeline_step_counter += 1
                    else:
                        print(f"[WARNING] Could not load drawing image for masking from {args.drawing}.")
                except Exception as e:
                    print(f"[WARNING] Error applying mask from drawing {args.drawing}: {e}.")

            analysis_results = color_analyzer.process_image(
                image=image_for_processing, image_path=args.image, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
                output_dir=str(report_generator.project_output_dir), debug_mode=args.debug,
                aggregate_mode=args.aggregate, blur_mode=args.blur,
                alignment_mode=False, drawing_path=None # These are handled externally now
            )

            analyzed_image_filename = f"analyzed_{os.path.basename(args.image)}"
            analyzed_image_path_for_report = report_generator.project_output_dir / analyzed_image_filename
            save_image(str(analyzed_image_path_for_report), analysis_results['processed_image'])
            analysis_results['analyzed_image_path'] = str(analyzed_image_path_for_report)

            if args.debug:
                if analysis_results.get('blurred_image_path'):
                    debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Blurred for Analysis", "path": os.path.basename(analysis_results['blurred_image_path'])})
                    pipeline_step_counter += 1
                if analysis_results.get('mask_pre_aggregation_path'):
                    debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Mask (Before Aggregation)", "path": os.path.basename(analysis_results['mask_pre_aggregation_path'])})
                    pipeline_step_counter += 1
                
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Final Color Mask", "path": os.path.basename(analysis_results['mask_path'])})
                pipeline_step_counter += 1
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Matched Pixels Overlay", "path": os.path.basename(analysis_results['processed_image_path'])})
                pipeline_step_counter += 1
                
                if analysis_results.get('contours_image_path'):
                    debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Final Result with Contours", "path": os.path.basename(analysis_results['contours_image_path'])})
                    pipeline_step_counter += 1

            analysis_results['lower_limit'] = lower_hsv
            analysis_results['upper_limit'] = upper_hsv
            analysis_results['center_color'] = center_hsv
            analysis_results['selected_colors'] = {'RGB': center_rgb}

            file_name_without_ext = os.path.splitext(os.path.basename(args.image))[0]
            part_number, thickness = (file_name_without_ext.split('_')[:2] + ["N/A"][:2-len(file_name_without_ext.split('_'))]) if '_' in file_name_without_ext else (file_name_without_ext, "N/A")
            metadata = {"part_number": part_number, "thickness": thickness}
            
            debug_data_for_report = {}
            if args.debug:
                debug_data_for_report["--- Project Info ---"] = ""
                debug_data_for_report["Project Name"] = args.project
                debug_data_for_report["HSV Range (Lower)"] = str(lower_hsv.tolist())
                debug_data_for_report["HSV Range (Upper)"] = str(upper_hsv.tolist())
                debug_data_for_report["HSV Range (Center)"] = str(center_hsv.tolist())
                
                debug_data_for_report["--- Analysis Settings ---"] = ""
                debug_data_for_report["Color Alignment"] = args.color_alignment
                debug_data_for_report["Geometrical Alignment"] = args.alignment
                debug_data_for_report["Masking with Drawing"] = f"Enabled (using {os.path.basename(args.drawing)})" if args.drawing else "Disabled"
                debug_data_for_report["Blur"] = args.blur
                if args.blur and analysis_results.get('blurred_kernel_size'):
                    debug_data_for_report["Blur Kernel Size"] = str(analysis_results['blurred_kernel_size'])
                debug_data_for_report["Aggregation"] = args.aggregate

                debug_data_for_report["--- Analysis Results ---"] = ""
                debug_data_for_report["Matched Pixels"] = f"{analysis_results['matched_pixels']:,}"
                debug_data_for_report["Total Pixels"] = f"{analysis_results['total_pixels']:,}"
                debug_data_for_report["Percentage Matched"] = f"{analysis_results['percentage']:.2f}%"
                if alignment_data:
                    debug_data_for_report["Detected ArUco IDs"] = alignment_data.get('detected_ids')
                
                debug_data_for_report["image_pipeline"] = debug_image_pipeline

            report_generator.generate_report(analysis_results, metadata, debug_data=debug_data_for_report)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
