import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle

# Suppress NumPy warnings that might arise from operations like empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Import core modules for the Visual Analyzer application
from src.project_manager import ProjectManager
from src.color_analysis.analyzer import ColorAnalyzer
from src.color_correction.corrector import ColorCorrector
from src.alignment.aligner import Aligner
from src.object_alignment.object_aligner import AdvancedAligner
from src.masking.creator import MaskCreator
from src.reporting.generator import ReportGenerator
from src.utils.image_utils import load_image, save_image, blur_image
from src.utils.video_utils import process_video_stream
from src import config
from src.symmetry_analysis.symmetry import SymmetryAnalyzer

class Pipeline:
    def __init__(self, args):
        self.args = args
        self.project_manager = ProjectManager()
        self.color_corrector = ColorCorrector()
        self.color_analyzer = ColorAnalyzer()
        self.project_data = None
        self.image_path = None
        self.original_input_image_bgr = None
        self.image_to_be_processed = None
        self.analysis_results = None
        self.metadata = None
        self.debug_data_for_report = {}
        self.debug_image_pipeline = []
        self.pipeline_step_counter = 1
        self.pipeline_image_stages = {} # To store image output of each step


    def load_project_data(self):
        self.project_data = self.project_manager.get_project_data(
            self.args.project, debug_mode=self.args.debug
        )
        if self.args.debug:
            print(
                f"Loaded project '{self.args.project}' with HSV range: {self.project_data['lower_hsv']} - {self.project_data['upper_hsv']}"
            )

    def process_image(self, image_path):
        self.image_path = image_path
        if self.args.debug:
            print(f"[DEBUG] Inside process_image. args.object_alignment = {self.args.object_alignment}")
            print(f"Processing single image: {self.image_path}")

        sample_name = None
        path_parts = self.image_path.split(os.sep)
        if "samples" in path_parts:
            sample_index = path_parts.index("samples") + 1
            if sample_index < len(path_parts):
                sample_name = path_parts[sample_index]

        self.report_generator = ReportGenerator(
            self.args.project, sample_name=sample_name, debug_mode=self.args.debug
        )

        self.original_input_image_bgr, _ = load_image(self.image_path)
        if self.original_input_image_bgr is None:
            raise ValueError(f"Could not load image {self.image_path}")

        # Store the original image as the first stage
        self.pipeline_image_stages['original'] = self.original_input_image_bgr.copy()

        if self.args.debug:
            import shutil
            from pathlib import Path
            # Copy the original image to the report directory to make paths relative and self-contained.
            copied_input_filename = f"00_original_input{Path(self.image_path).suffix}"
            copied_input_path = self.report_generator.project_output_dir / copied_input_filename
            shutil.copy(self.image_path, copied_input_path)

            self.debug_data_for_report["dataset_debug_info"] = self.project_data["dataset_debug_info"]
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. Original Input",
                    "path": copied_input_filename, # Use the relative path (filename)
                }
            )
            self.pipeline_step_counter += 1

        self.image_to_be_processed = self.original_input_image_bgr.copy()
        
        # The rest of the processing steps from the original process_image function will be here
        # ... (color correction, alignment, masking, etc.)
        self.run_full_pipeline()


    def run_full_pipeline(self):
        if self.args.color_alignment and self.project_data["correction_matrix"] is not None:
            self._perform_color_correction()

        if self.args.alignment:
            self._perform_geometrical_alignment()

        if self.args.object_alignment:
            self._perform_object_alignment()

        if self.args.apply_mask:
            self._apply_masking()

        if self.args.blur:
            self._perform_blur()
        
        if not self.args.skip_color_analysis:
            self._perform_color_analysis()

        if self.args.symmetry:
            self._perform_symmetry_analysis()
        
        self._extract_metadata()

        if not self.args.skip_report_generation:
            self.generate_report()


    def _perform_color_correction(self):
        step_dir = self.report_generator.get_step_output_dir("color_correction")
        
        correction_result = self.color_corrector.apply_color_correction(
            self.image_to_be_processed, 
            self.project_data["correction_matrix"],
            output_dir=step_dir
        )

        self.image_to_be_processed = correction_result['image']
        self.pipeline_image_stages['color_corrected'] = self.image_to_be_processed.copy()

        if self.args.debug and correction_result['debug_path']:
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. After Color Correction",
                    "path": os.path.relpath(
                        correction_result['debug_path'], self.report_generator.project_output_dir
                    ),
                }
            )
            self.pipeline_step_counter += 1

    def _perform_geometrical_alignment(self):
        step_dir = self.report_generator.get_step_output_dir("geometrical_alignment")
        aligner = Aligner(debug_mode=self.args.debug, output_dir=step_dir)
        project_files = self.project_manager.get_project_file_paths(self.args.project, self.args.debug)
        
        aruco_ref_path = project_files.get("aruco_reference")
        marker_map = project_files.get("aruco_marker_map")
        output_size = project_files.get("aruco_output_size")

        result = aligner.align_image(
            image=self.image_to_be_processed,
            aruco_reference_path=str(aruco_ref_path) if aruco_ref_path else None,
            marker_map=marker_map if marker_map else None,
            output_size_wh=output_size
        )

        if result and result.get('image') is not None:
            self.image_to_be_processed = result['image']
            self.pipeline_image_stages['geometrically_aligned'] = self.image_to_be_processed.copy()
            self.debug_data_for_report["geometrical_alignment_data"] = result.get('alignment_data')
            
            if self.args.debug and result.get('debug_paths'):
                # Define titles for known debug images
                path_titles = {
                    'detected_markers': "Detected ArUco Markers",
                    'final_aligned': "After Geometrical Alignment"
                }
                for key, path in result['debug_paths'].items():
                    title = path_titles.get(key, key.replace('_', ' ').title())
                    self.debug_image_pipeline.append({
                        "title": f"{self.pipeline_step_counter}. GA: {title}",
                        "path": os.path.relpath(path, self.report_generator.project_output_dir)
                    })
                    self.pipeline_step_counter += 1
        elif self.args.debug:
            print("[WARNING] Geometrical alignment failed. Proceeding without alignment.")

    def _perform_object_alignment(self):
        step_dir = self.report_generator.get_step_output_dir("object_alignment")
        project_files = self.project_manager.get_project_file_paths(self.args.project, self.args.debug)
        object_ref_path = project_files.get("object_reference_path")

        if not object_ref_path:
            if self.args.debug:
                print("[DEBUG] No object reference path specified. Skipping object alignment.")
            return

        ref_image, _ = load_image(str(object_ref_path))
        if ref_image is None:
            if self.args.debug:
                print(f"[WARNING] Could not load object reference image at {object_ref_path}. Skipping object alignment.")
            return

        advanced_aligner = AdvancedAligner(debug_mode=self.args.debug, output_dir=step_dir)
        result = advanced_aligner.align(self.image_to_be_processed, ref_image, method="feature_orb")

        if result and result.get('image') is not None:
            self.image_to_be_processed = result['image']
            self.pipeline_image_stages['object_aligned'] = self.image_to_be_processed.copy()

            if self.args.debug and result.get('debug_paths'):
                debug_paths = result['debug_paths']
                # Store statistics
                if 'good_feature_matches' in debug_paths:
                    self.debug_data_for_report['object_alignment_stats'] = {
                        'good_feature_matches': debug_paths['good_feature_matches']
                    }

                # Store images for report
                path_titles = {
                    'feature_matches_image': "Feature Matches",
                    'final_aligned': "After Object Alignment"
                }
                for key, path in debug_paths.items():
                    if not isinstance(path, str): continue # Skip non-path values like stats
                    title = path_titles.get(key, key.replace('_', ' ').title())
                    self.debug_image_pipeline.append({
                        "title": f"{self.pipeline_step_counter}. OA: {title}",
                        "path": os.path.relpath(path, self.report_generator.project_output_dir)
                    })
                    self.pipeline_step_counter += 1
        elif self.args.debug:
            print("[WARNING] Object alignment failed. Proceeding with unaligned image.")

    def _apply_masking(self):
        step_dir = self.report_generator.get_step_output_dir("masking")
        project_files = self.project_manager.get_project_file_paths(self.args.project, self.args.debug)
        
        # The masking_order argument is now handled inside the new function.
        masking_order = []
        if self.args.masking_order:
            masking_order = [layer for layer in self.args.masking_order.split('-') if layer]

        if not masking_order:
            raise ValueError("Masking is enabled (--apply-mask), but no valid layers were specified via --masking-order. Please provide a value like '1' or '1-2'.")

        # Defer all logic to the self-contained function in the masking module.
        from src.masking.creator import create_and_apply_mask_from_layers
        result = create_and_apply_mask_from_layers(
            image_to_be_processed=self.image_to_be_processed,
            project_files=project_files,
            masking_order=masking_order,
            mask_bg_is_white=self.args.mask_bg_is_white,
            output_dir=step_dir,
            debug_mode=self.args.debug
        )

        self.image_to_be_processed = result['image']
        self.pipeline_image_stages['masked'] = self.image_to_be_processed.copy()
        if result.get('stats'):
            self.debug_data_for_report['masking_stats'] = result['stats']

        if self.args.debug and result['debug_paths']:
            for debug_info in result['debug_paths']:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. {debug_info['title']}",
                        "path": os.path.relpath(debug_info['path'], self.report_generator.project_output_dir),
                    }
                )
                self.pipeline_step_counter += 1

    def _perform_color_analysis(self):
        step_dir = self.report_generator.get_step_output_dir("color_analysis")

        if self.args.debug:
            debug_path = os.path.join(step_dir, "image_sent_to_analyzer.png")
            save_image(debug_path, self.image_to_be_processed)
            print(f"[DEBUG] Diagnostic image saved: the exact image being sent to ColorAnalyzer is at {debug_path}")

        self.analysis_results = self.color_analyzer.process_image(
            image=self.image_to_be_processed,
            original_image_path=self.image_path,
            lower_hsv=self.project_data["lower_hsv"],
            upper_hsv=self.project_data["upper_hsv"],
            center_hsv=self.project_data["center_hsv"],
            output_dir=step_dir,
            debug_mode=self.args.debug,
            aggregate_mode=self.args.aggregate,
            agg_kernel_size=self.args.agg_kernel_size,
            agg_min_area=self.args.agg_min_area,
            agg_density_thresh=self.args.agg_density_thresh,
        )
        self.pipeline_image_stages['color_analyzed'] = self.analysis_results['processed_image'].copy()
        
        if self.args.debug and self.analysis_results.get('debug_info'):
            for debug_item in self.analysis_results['debug_info']:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. CA: {debug_item['title']}",
                        "path": os.path.relpath(debug_item['path'], self.report_generator.project_output_dir),
                    }
                )
                self.pipeline_step_counter += 1
    def _perform_symmetry_analysis(self):
        """
        Performs symmetry analysis on the binary mask from color analysis.
        """
        if not self.analysis_results or 'binary_mask' not in self.analysis_results:
            if self.args.debug:
                print("[WARNING] Skipping symmetry analysis because color analysis results are not available.")
            return

        binary_mask = self.analysis_results['binary_mask']
        if binary_mask is None or np.count_nonzero(binary_mask) == 0:
            if self.args.debug:
                print("[WARNING] Skipping symmetry analysis because the binary mask is empty.")
            return
            
        step_dir = self.report_generator.get_step_output_dir("symmetry_analysis")

        symmetry_analyzer = SymmetryAnalyzer(
            binary_mask, 
            output_dir=step_dir, 
            debug_mode=self.args.debug
        )
        symmetry_analyzer.analyze_all()
        self.debug_data_for_report["symmetry_results"] = symmetry_analyzer.results

        if self.args.debug and 'visualizations' in symmetry_analyzer.results:
            for viz_info in symmetry_analyzer.results['visualizations']:
                self.debug_image_pipeline.append({
                    "title": f"{self.pipeline_step_counter}. {viz_info['title']}",
                    "path": os.path.relpath(viz_info['path'], self.report_generator.project_output_dir)
                })
                self.pipeline_step_counter += 1

    def _perform_blur(self):
        step_dir = self.report_generator.get_step_output_dir("blur")
        
        blur_result = blur_image(
            self.image_to_be_processed, 
            self.args.blur_kernel,
            output_dir=step_dir
        )
        
        self.image_to_be_processed = blur_result['image']
        self.pipeline_image_stages['blurred'] = self.image_to_be_processed.copy()
        self.debug_data_for_report["blur_kernel_used"] = blur_result['kernel_used']

        if self.args.debug and blur_result['debug_path']:
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. After Blur (Kernel: {blur_result['kernel_used']})",
                    "path": os.path.relpath(blur_result['debug_path'], self.report_generator.project_output_dir),
                }
            )
            self.pipeline_step_counter += 1
    
    def _extract_metadata(self):
        file_name_without_ext = os.path.splitext(os.path.basename(self.image_path))[0]
        file_parts = file_name_without_ext.split("_")
        if len(file_parts) >= 3:
            part_number = file_parts[2]
            thickness = file_parts[3] if len(file_parts) >= 4 else "N/A"
        else:
            part_number = file_name_without_ext
            thickness = "N/A"
        self.metadata = {"part_number": part_number, "thickness": thickness}


    def generate_report(self):
        if self.analysis_results is None:
            print("[WARNING] No analysis results to generate report from.")
            return None

        # Add the collected pipeline images to the main debug data dictionary
        self.debug_data_for_report['image_pipeline'] = self.debug_image_pipeline

        report_data = self.report_generator.generate_report(
            self.analysis_results,
            self.metadata,
            debug_data=self.debug_data_for_report,
            report_type=self.args.report_type,
        )
        return report_data

    def save_state(self, path):
        """Saves the pipeline state to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        if self.args.debug:
            print(f"Pipeline state saved to {path}")

    @staticmethod
    def load_state(path):
        """Loads a pipeline state from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def run_analysis(args):
    """
    Main entry point for the analysis pipeline.
    """
    try:
        if args.load_state_from:
            pipeline = Pipeline.load_state(args.load_state_from)
            pipeline.args = args # Update args in case they changed
        else:
            pipeline = Pipeline(args)
            pipeline.load_project_data()

        if args.image:
            if os.path.isdir(args.image):
                results = []
                for filename in os.listdir(args.image):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(args.image, filename)
                        pipeline.process_image(image_path)
                        results.append(pipeline.analysis_results)
                return results
            else:
                pipeline.process_image(args.image)
                if args.save_state_to:
                    pipeline.save_state(args.save_state_to)
                return pipeline.analysis_results

        if args.video:
            # Video processing is not fully integrated into the new class structure yet.
            # This part needs to be adapted.
            process_video(args, pipeline)
        
        return None

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def process_video(args, pipeline):
    # This function needs to be adapted to the new class structure
    if args.debug:
        print(f"Processing video: {args.video}")
    # ... (original process_video logic adapted to use the pipeline object)