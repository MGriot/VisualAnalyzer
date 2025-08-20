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
    Defines all available command-line arguments for controlling the analysis workflow.
    """
    parser = argparse.ArgumentParser(description="Visual Analyzer for Color Correction and Analysis.")
    parser.add_argument("--project", type=str, help="Name of the project to use. This project defines color space/range and ArUco marker configurations.")
    parser.add_argument("--image", type=str, help="Path to a single image file for analysis. If provided, video/camera inputs are ignored.")
    parser.add_argument("--video", type=str, help="Path to a video file for analysis. If provided, camera input is ignored.")
    parser.add_argument("--camera", action="store_true", help="Use live camera stream for analysis. Requires a connected camera.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output and saving intermediate images for inspection.")
    parser.add_argument("--aggregate", action="store_true", help="Enable aggregation of nearby matched pixel areas during color analysis to form larger regions.")
    parser.add_argument("--blur", action="store_true", help="Enable blurring of the input image before color matching to reduce noise.")
    parser.add_argument("--alignment", action="store_true", help="Enable geometrical alignment using ArUco markers. Requires 'aruco_marker_map' and 'aruco_output_size' in project config.")
    parser.add_argument("--drawing", type=str, help="Path to a technical drawing image (e.g., black background, white profile). Used as a mask applied *after* all alignments and corrections, but *before* color analysis.")
    parser.add_argument("--color-alignment", action="store_true", help="Enable color correction (alignment) based on reference color checker data defined in the project.")

    args = parser.parse_args()
    # Call the core analysis function with parsed arguments
    run_analysis_from_gui(args)

def run_analysis_from_gui(args):
    """
    Core analysis function that orchestrates the image processing pipeline.
    This function is designed to be called from both CLI and GUI.

    Args:
        args: An object containing all configuration parameters (from argparse or GUI).
    """
    # --- 1. Project Initialization and Data Loading ---
    # Initializes the ProjectManager to handle project-specific configurations and data.
    project_manager = ProjectManager()
    available_projects = project_manager.list_projects()

    if not available_projects:
        print("Error: No projects found. Please create a project in the 'data/projects' directory.")
        return

    selected_project = args.project
    if not selected_project:
        print("Error: No project selected. Please select a project in the GUI or provide --project argument in CLI.")
        return

    try:
        # Load project-specific data: color correction matrix, and HSV color range for analysis.
        # Input: selected_project (str), debug_mode (bool)
        # Output: correction_matrix (np.ndarray), lower_hsv (np.ndarray), upper_hsv (np.ndarray), center_hsv (np.ndarray)
        project_data = project_manager.get_project_data(selected_project, debug_mode=args.debug)
        correction_matrix = project_data['correction_matrix']
        lower_hsv = project_data['lower_hsv']
        upper_hsv = project_data['upper_hsv']
        center_hsv = project_data['center_hsv']

        print(f"Loaded project '{selected_project}' with HSV range: {lower_hsv} - {upper_hsv}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading project: {e}")
        return

    # --- 2. Module Initialization ---
    # Initializes instances of the core processing modules.
    color_corrector = ColorCorrector()
    color_analyzer = ColorAnalyzer()
    report_generator = ReportGenerator(selected_project, debug_mode=args.debug)

    # Prepare center HSV color in RGB format for reporting purposes
    center_rgb = cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

    # --- 3. Image Processing Workflow (for single image input) ---
    # This block handles the processing of a single image file.
    # Input: args.image (path to input image)
    # Output: Processed image data, analysis results, and generated report files.
    if args.image:
        print(f"Processing single image: {args.image}")
        try:
            # Load the original input image
            # Input: args.image (str)
            # Output: original_input_image_bgr (np.ndarray)
            original_input_image_bgr, _ = load_image(args.image)
            if original_input_image_bgr is None:
                raise ValueError(f"Could not load image {args.image}")
            if args.debug:
                print(f"[DEBUG] Image loaded. Shape: {original_input_image_bgr.shape}")

            # 'image_for_processing' will hold the image as it goes through various transformations.
            image_for_processing = original_input_image_bgr.copy()
            aligned_image_for_report = None # Stores the image after geometrical alignment for reporting
            alignment_data = None # Stores data from geometrical alignment

            # --- 3.1. Apply Color Correction (Optional) ---
            # Applies color correction if enabled and a correction matrix is available.
            # Input: image_for_processing (np.ndarray), correction_matrix (np.ndarray)
            # Output: image_for_processing (np.ndarray) - color corrected
            if args.color_alignment and correction_matrix is not None:
                image_for_processing = color_corrector.apply_color_correction(image_for_processing, correction_matrix)
                if args.debug:
                    print(f"[DEBUG] Color correction applied. Shape: {image_for_processing.shape}")
                    save_image(str(report_generator.project_output_dir / "color_corrected_debug.png"), image_for_processing)

            # --- 3.2. Apply Geometrical Alignment (Optional) ---
            # Performs image alignment using ArUco markers if enabled.
            # Input: image_for_processing (np.ndarray), aruco_marker_map (dict), aruco_output_size (tuple)
            # Output: image_for_processing (np.ndarray) - geometrically aligned, alignment_data (dict)
            if args.alignment:
                # Retrieve ArUco specific configuration from project files
                project_files = project_manager.get_project_file_paths(selected_project, debug_mode=args.debug)
                aruco_marker_map = project_files.get("aruco_marker_map")
                aruco_output_size = project_files.get("aruco_output_size")

                if not aruco_marker_map or not aruco_output_size:
                    print("[WARNING] Alignment enabled but ArUco marker map or output size not provided in project config. Skipping alignment.")
                else:
                    # Initialize Aligner and perform alignment
                    temp_aligner = Aligner(debug_mode=args.debug, output_dir=str(report_generator.project_output_dir))
                    alignment_result = temp_aligner.align_image(image=image_for_processing, marker_map=aruco_marker_map, output_size_wh=tuple(aruco_output_size))
                    
                    if alignment_result is None: # Check if alignment was successful
                        print("[WARNING] Image alignment failed. Proceeding without alignment.")
                    else:
                        image_for_processing, alignment_data = alignment_result
                        # Save a copy of the aligned image specifically for reporting purposes
                        aligned_image_for_report = image_for_processing.copy() 
                        if args.debug:
                            print(f"[DEBUG] Geometrical alignment applied. Shape: {image_for_processing.shape}")
                            save_image(str(report_generator.project_output_dir / "geometrically_aligned_debug.png"), image_for_processing)

            # --- 3.3. Apply Masking with --drawing (Optional) ---
            # Applies a mask from a technical drawing to the image.
            # Input: image_for_processing (np.ndarray), args.drawing (path to drawing image)
            # Output: image_for_processing (np.ndarray) - masked
            if args.drawing:
                try:
                    # Load the drawing image (e.g., black background, white profile)
                    drawing_mask_image, _ = load_image(args.drawing, handle_transparency=False) 
                    if drawing_mask_image is None:
                        print(f"[WARNING] Could not load drawing image for masking from {args.drawing}. Skipping masking.")
                    else:
                        # Convert drawing to grayscale if it's a color image
                        if len(drawing_mask_image.shape) == 3: 
                            drawing_mask_image = cv2.cvtColor(drawing_mask_image, cv2.COLOR_BGR2GRAY)
                        
                        # Threshold to ensure it's a binary mask (0 for black, 255 for white)
                        _, drawing_mask_binary = cv2.threshold(drawing_mask_image, 128, 255, cv2.THRESH_BINARY)

                        # Resize mask to match the current image dimensions if they differ
                        if drawing_mask_binary.shape[:2] != image_for_processing.shape[:2]:
                            drawing_mask_binary = cv2.resize(drawing_mask_binary, (image_for_processing.shape[1], image_for_processing.shape[0]), interpolation=cv2.INTER_NEAREST)

                        # Apply the mask: only pixels corresponding to white areas in the mask are kept.
                        # For BGR images, the mask is applied to each channel.
                        image_for_processing = cv2.bitwise_and(image_for_processing, image_for_processing, mask=drawing_mask_binary)
                        if args.debug:
                            print(f"[DEBUG] Mask applied using drawing: {args.drawing}. Shape: {image_for_processing.shape}")
                            save_image(str(report_generator.project_output_dir / "masked_image_debug.png"), image_for_processing)

                except Exception as e:
                    print(f"[WARNING] Error applying mask from drawing {args.drawing}: {e}. Skipping masking.")

            # --- 3.4. Perform Color Analysis ---
            # Analyzes the processed image for color zones and calculates statistics.
            # Input: image_for_processing (np.ndarray), HSV color range, various analysis flags.
            # Output: analysis_results (dict) - contains processed image, masks, statistics.
            analysis_results = color_analyzer.process_image(
                image=image_for_processing,
                image_path=args.image, # Original image path for reference in report
                lower_hsv=lower_hsv,
                upper_hsv=upper_hsv,
                output_dir=str(report_generator.project_output_dir),
                debug_mode=args.debug,
                aggregate_mode=args.aggregate,
                blur_mode=args.blur,
                alignment_mode=False, # Alignment is handled externally in main.py now
                drawing_path=None # Drawing is handled externally in main.py now
            )

            # Update analysis_results with geometrical alignment data if alignment was performed
            if alignment_data:
                analysis_results['alignment_data'] = alignment_data
                # Save the aligned image for reporting (if alignment was performed)
                aligned_input_image_path_for_report = report_generator.project_output_dir / f"aligned_input_{os.path.basename(args.image)}"
                save_image(str(aligned_input_image_path_for_report), aligned_image_for_report)
                analysis_results['aligned_input_image_path'] = str(aligned_input_image_path_for_report)

            # Save the final processed image (after all corrections and masking) for the report.
            # This is the image that was actually analyzed for colors.
            analyzed_image_filename = f"analyzed_{os.path.basename(args.image)}"
            analyzed_image_path_for_report = report_generator.project_output_dir / analyzed_image_filename
            save_image(str(analyzed_image_path_for_report), analysis_results['processed_image'])

            # Prepare metadata for the report (e.g., part number, thickness from filename)
            file_name_without_ext = os.path.splitext(os.path.basename(args.image))[0]
            part_number = file_name_without_ext
            thickness = "N/A" # Default value
            if '_' in file_name_without_ext:
                parts = file_name_without_ext.split('_')
                if len(parts) >= 2:
                    part_number = parts[0]
                    thickness = parts[1]

            metadata = {"part_number": part_number, "thickness": thickness}
            
            # --- 3.5. Generate Report ---
            # Creates the HTML and PDF reports based on all collected analysis results and metadata.
            # Input: analysis_results (dict), metadata (dict), debug_data (dict)
            # Output: HTML and PDF report files in the project's output directory.
            debug_data_for_report = {}
            if args.debug:
                debug_data_for_report["Project Name"] = selected_project
                debug_data_for_report["HSV Range (Lower)"] = str(lower_hsv.tolist())
                debug_data_for_report["HSV Range (Upper)"] = str(upper_hsv.tolist())
                debug_data_for_report["HSV Range (Center)"] = str(center_hsv.tolist())
                debug_data_for_report["Matched Pixels (Raw)"] = analysis_results['matched_pixels']
                debug_data_for_report["Percentage Matched (Raw)"] = f"{analysis_results['percentage']:.2f}%"
                if args.aggregate:
                    debug_data_for_report["Aggregation Applied"] = "Yes"
                if args.blur:
                    debug_data_for_report["Blur Applied"] = "Yes"
                if alignment_data: # Check if alignment_data is not None before adding
                    debug_data_for_report["Alignment Data"] = alignment_data
                if analysis_results.get('aligned_input_image_path'):
                    debug_data_for_report["Aligned Input Image Path"] = analysis_results['aligned_input_image_path']
                if args.drawing: # Check if drawing was used for masking
                    debug_data_for_report["Masking Applied"] = "Yes"
                    debug_data_for_report["Mask Drawing Path"] = args.drawing

            report_generator.generate_report(analysis_results, metadata, debug_data=debug_data_for_report)

        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing image: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during image processing: {e}")

    # --- 4. Video/Camera Stream Processing (Placeholder) ---
    # This section is for future implementation of real-time video or camera stream analysis.
    # It defines a helper function that would process individual frames.
    def process_and_display_frame(frame, output_dir, correction_matrix):
        """
        Helper function to process and display a single frame from a video stream.
        This is a placeholder for real-time analysis logic.
        """
        processed_frame = frame.copy()
        if args.color_alignment and correction_matrix is not None:
            processed_frame = color_corrector.apply_color_correction(processed_frame, correction_matrix)

            # Example: Display the frame (this will open a new window)
            cv2.imshow("Live Analysis", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False # Signal to stop the stream

            return True # Signal to continue the stream

        elif args.video:
            print(f"Processing video file: {args.video}")
            process_video_stream(args.video, lambda frame: process_and_display_frame(frame, str(report_generator.project_output_dir), correction_matrix))

        elif args.camera:
            print("Starting live camera stream. Press 'q' to quit.\n")
            process_video_stream(0, lambda frame: process_and_display_frame(frame, str(report_generator.project_output_dir), correction_matrix))

        else:
            print("Please specify an input: --image <path>, --video <path>, or --camera.")

if __name__ == "__main__":
    main()