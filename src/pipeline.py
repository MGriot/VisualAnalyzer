import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Suppress NumPy warnings that might arise from operations like empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy')

# Import core modules for the Visual Analyzer application
from src.color_analysis.project_manager import ProjectManager
from src.color_analysis.analyzer import ColorAnalyzer
from src.color_correction.corrector import ColorCorrector
from src.alignment.aligner import Aligner
from src.reporting.generator import ReportGenerator
from src.utils.image_utils import load_image, save_image, blur_image
from src.utils.video_utils import process_video_stream
from src import config
from src.symmetry_analysis.symmetry import SymmetryAnalyzer

def run_analysis(args):
    try:
        project_manager = ProjectManager()
        project_data = project_manager.get_project_data(args.project, debug_mode=args.debug)
        correction_matrix, lower_hsv, upper_hsv, center_hsv, dataset_debug_info = (
            project_data['correction_matrix'],
            project_data['lower_hsv'],
            project_data['upper_hsv'],
            project_data['center_hsv'],
            project_data['dataset_debug_info'],
        )
        print(f"Loaded project '{args.project}' with HSV range: {lower_hsv} - {upper_hsv}")

        color_corrector = ColorCorrector()
        color_analyzer = ColorAnalyzer()

        if args.image:
            if os.path.isdir(args.image):
                for filename in os.listdir(args.image):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(args.image, filename)
                        process_image(image_path, args, project_manager, color_corrector, color_analyzer, lower_hsv, upper_hsv, center_hsv, correction_matrix, dataset_debug_info)
            else:
                process_image(args.image, args, project_manager, color_corrector, color_analyzer, lower_hsv, upper_hsv, center_hsv, correction_matrix, dataset_debug_info)

        if args.video:
            if os.path.isdir(args.video):
                for filename in os.listdir(args.video):
                    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                        video_path = os.path.join(args.video, filename)
                        process_video(video_path, args, color_corrector, color_analyzer, None, lower_hsv, upper_hsv, center_hsv, correction_matrix, dataset_debug_info)
            else:
                process_video(args.video, args, color_corrector, color_analyzer, None, lower_hsv, upper_hsv, center_hsv, correction_matrix, dataset_debug_info)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_image(image_path, args, project_manager, color_corrector, color_analyzer, lower_hsv, upper_hsv, center_hsv, correction_matrix, dataset_debug_info):
    print(f"Processing single image: {image_path}")
    try:
        # Extract sample name from image path
        sample_name = None
        path_parts = image_path.split(os.sep)
        if 'samples' in path_parts:
            sample_index = path_parts.index('samples') + 1
            if sample_index < len(path_parts):
                sample_name = path_parts[sample_index]

        report_generator = ReportGenerator(args.project, sample_name=sample_name, debug_mode=args.debug)

        debug_image_pipeline = []
        pipeline_step_counter = 1

        original_input_image_bgr, _ = load_image(image_path)
        if original_input_image_bgr is None:
            raise ValueError(f"Could not load image {image_path}")
        
        debug_data_for_report = {}
        if args.debug:
            debug_data_for_report['dataset_debug_info'] = dataset_debug_info

        if args.debug:
            debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Original Input", "path": os.path.basename(image_path)})
            pipeline_step_counter += 1

        image_for_processing = original_input_image_bgr.copy()
        alignment_data = None

        if args.color_alignment and correction_matrix is not None:
            step_dir = report_generator.get_step_output_dir("color_correction")
            image_for_processing = color_corrector.apply_color_correction(image_for_processing, correction_matrix)
            if args.debug:
                path = os.path.join(step_dir, "color_corrected_debug.png")
                save_image(path, image_for_processing)
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Color Correction", "path": os.path.relpath(path, report_generator.project_output_dir)})
                pipeline_step_counter += 1

        if args.alignment:
            step_dir = report_generator.get_step_output_dir("alignment")
            project_files = project_manager.get_project_file_paths(args.project, debug_mode=args.debug)
            aruco_reference_path = project_files.get("aruco_reference")
            aruco_marker_map = project_files.get("aruco_marker_map")
            aruco_output_size = project_files.get("aruco_output_size")

            temp_aligner = Aligner(debug_mode=args.debug, output_dir=step_dir)
            alignment_result = None

            if aruco_reference_path:
                alignment_result = temp_aligner.align_image(image=image_for_processing, aruco_reference_path=str(aruco_reference_path))
            elif aruco_marker_map and aruco_output_size:
                alignment_result = temp_aligner.align_image(image=image_for_processing, marker_map=aruco_marker_map, output_size_wh=tuple(aruco_output_size))
            else:
                print("[WARNING] Alignment enabled but no ArUco reference image or marker map found in project config.")

            if alignment_result:
                image_for_processing, alignment_data = alignment_result
                if image_for_processing is not None:
                    if args.debug:
                        path = os.path.join(step_dir, "geometrically_aligned_debug.png")
                        save_image(path, image_for_processing)
                        debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Geometrical Alignment", "path": os.path.relpath(path, report_generator.project_output_dir)})
                        pipeline_step_counter += 1
                else:
                    print("[WARNING] Image alignment failed.")
            else:
                print("[WARNING] Image alignment failed.")

        blurred_kernel_size = None
        if args.blur:
            step_dir = report_generator.get_step_output_dir("blur")
            image_for_processing, blurred_kernel_size = blur_image(image_for_processing)
            if args.debug:
                path = os.path.join(step_dir, "blurred_debug.png")
                save_image(path, image_for_processing)
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Blurring", "path": os.path.relpath(path, report_generator.project_output_dir)})
                pipeline_step_counter += 1

        if args.drawing:
            step_dir = report_generator.get_step_output_dir("masking")
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
                        path = os.path.join(step_dir, "masked_image_debug.png")
                        save_image(path, image_for_processing)
                        debug_image_pipeline.append({"title": f"{pipeline_step_counter}. After Masking with Drawing", "path": os.path.relpath(path, report_generator.project_output_dir)})
                        pipeline_step_counter += 1
                else:
                    print(f"[WARNING] Could not load drawing image for masking from {args.drawing}.")
            except Exception as e:
                print(f"[WARNING] Error applying mask from drawing {args.drawing}: {e}.")

        step_dir = report_generator.get_step_output_dir("color_analysis")
        analysis_results = color_analyzer.process_image(
            image=image_for_processing, image_path=image_path, lower_hsv=lower_hsv, upper_hsv=upper_hsv,
            output_dir=step_dir, debug_mode=args.debug,
            aggregate_mode=args.aggregate,
            alignment_mode=False, drawing_path=None # These are handled externally now
        )
        analysis_results['blurred_kernel_size'] = blurred_kernel_size

        analyzed_image_filename = f"analyzed_{os.path.basename(image_path)}"
        analyzed_image_path_for_report = os.path.join(step_dir, analyzed_image_filename)
        save_image(str(analyzed_image_path_for_report), analysis_results['processed_image'])
        analysis_results['analyzed_image_path'] = str(analyzed_image_path_for_report)

        if args.debug:
            if analysis_results.get('mask_pre_aggregation_path'):
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Mask (Before Aggregation)", "path": os.path.relpath(analysis_results['mask_pre_aggregation_path'], report_generator.project_output_dir)})
                pipeline_step_counter += 1
            
            debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Final Color Mask", "path": os.path.relpath(analysis_results['mask_path'], report_generator.project_output_dir)})
            pipeline_step_counter += 1
            debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Matched Pixels Overlay", "path": os.path.relpath(analysis_results['processed_image_path'], report_generator.project_output_dir)})
            pipeline_step_counter += 1
            
            if analysis_results.get('contours_image_path'):
                debug_image_pipeline.append({"title": f"{pipeline_step_counter}. Final Result with Contours", "path": os.path.relpath(analysis_results['contours_image_path'], report_generator.project_output_dir)})
                pipeline_step_counter += 1

        # --- Symmetry Analysis ---
        symmetry_results = {}
        if args.symmetry:
            step_dir = report_generator.get_step_output_dir("symmetry_analysis")
            try:
                # Ensure the image is in a suitable format (grayscale) for symmetry analysis
                # The SymmetryAnalyzer expects a single-channel image
                if len(original_input_image_bgr.shape) == 3:
                    gray_image_for_symmetry = cv2.cvtColor(original_input_image_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image_for_symmetry = original_input_image_bgr.copy()

                symmetry_analyzer = SymmetryAnalyzer(gray_image_for_symmetry)
                symmetry_analyzer.analyze_all()
                symmetry_results = symmetry_analyzer.results
                print("[DEBUG] Symmetry analysis completed.")
                

            except Exception as e:
                print(f"[WARNING] Error during symmetry analysis: {e}")
        # --- End Symmetry Analysis ---

        analysis_results['lower_limit'] = lower_hsv
        analysis_results['upper_limit'] = upper_hsv
        analysis_results['center_color'] = center_hsv
        analysis_results['selected_colors'] = {'RGB': cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2RGB)[0][0]}

        file_name_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        file_parts = file_name_without_ext.split('_')
        if len(file_parts) >= 3:
            part_number = file_parts[2]
            thickness = file_parts[3] if len(file_parts) >= 4 else "N/A"
        else:
            part_number = file_name_without_ext
            thickness = "N/A"
        metadata = {"part_number": part_number, "thickness": thickness}
        
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

            # --- Symmetry Analysis Results (Moved) ---
            if args.symmetry:
                step_dir = report_generator.get_step_output_dir("symmetry_analysis")
                debug_data_for_report["--- Symmetry Analysis Results ---"] = ""
                debug_data_for_report["symmetry_visualizations"] = []
                for key, value in symmetry_results.items():
                    debug_data_for_report[f"Symmetry: {key.replace('_', ' ').title()}"] = f"{value['score']:.4f}"
                    # Generate and save visualization for each symmetry type
                    if 'chunks' in value or 'reconstruction' in value:
                        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                        fig.suptitle(f"{key.replace('_', ' ').title()} (Score: {value['score']:.4f})")
                        axes[0].imshow(symmetry_analyzer.processed_image, cmap='gray')
                        axes[0].set_title('Original Processed')
                        if 'reconstruction' in value:
                            axes[1].imshow(value['reconstruction'], cmap='gray')
                            axes[1].set_title('Ideal Reconstruction')
                        else:
                            chunks = list(value['chunks'].values())
                            axes[1].imshow(chunks[1], cmap='gray')
                            axes[1].set_title(list(value['chunks'].keys())[1].title())
                        for ax in axes: ax.axis('off')
                        
                        # Save the plot
                        plot_filename = f"symmetry_{key}.png"
                        plot_path = os.path.join(step_dir, plot_filename)
                        plt.savefig(plot_path)
                        plt.close(fig)
                        if os.path.exists(plot_path):
                            print(f"[DEBUG] Saved symmetry plot: {plot_path}")
                            debug_data_for_report["symmetry_visualizations"].append({"title": f"{key.replace('_', ' ').title()} (Score: {value['score']:.4f})", "path": os.path.relpath(plot_path, report_generator.project_output_dir)})
                            print(f"[DEBUG] Appended to symmetry_visualizations: {debug_data_for_report['symmetry_visualizations'][-1]}")
                        else:
                            print(f"[WARNING] Failed to save symmetry plot: {plot_path}")

        if args.debug:
            print(f"[DEBUG] Contents of debug_data_for_report before report generation: {debug_data_for_report}")
            print(f"[DEBUG] Symmetry visualizations: {debug_data_for_report.get('symmetry_visualizations')}")

        report_generator.generate_report(analysis_results, metadata, debug_data=debug_data_for_report)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_video(video_path, args, color_corrector, color_analyzer, report_generator, lower_hsv, upper_hsv, center_hsv, correction_matrix, dataset_debug_info):
    print(f"Processing video: {video_path}")
    # Placeholder for video processing logic
    pass
