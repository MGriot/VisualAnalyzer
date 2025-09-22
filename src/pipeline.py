import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle

# Suppress NumPy warnings that might arise from operations like empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Import core modules for the Visual Analyzer application
from src.color_analysis.project_manager import ProjectManager
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

        if self.args.debug:
            self.debug_data_for_report["dataset_debug_info"] = self.project_data["dataset_debug_info"]
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. Original Input",
                    "path": os.path.basename(self.image_path),
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
        self.image_to_be_processed = self.color_corrector.apply_color_correction(
            self.image_to_be_processed, self.project_data["correction_matrix"]
        )
        path = os.path.join(step_dir, "color_corrected.png")
        save_image(path, self.image_to_be_processed)
        if self.args.debug:
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. After Color Correction",
                    "path": os.path.relpath(
                        path, self.report_generator.project_output_dir
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

        # Pass all available alignment parameters to the aligner.
        # The aligner will prioritize the reference path if available.
        aligned_image, alignment_data = aligner.align_image(
            image=self.image_to_be_processed,
            aruco_reference_path=str(aruco_ref_path) if aruco_ref_path else None,
            marker_map=marker_map if marker_map else None, # Pass None if empty dict
            output_size_wh=output_size
        )

        if aligned_image is not None:
            self.image_to_be_processed = aligned_image
            self.debug_data_for_report["geometrical_alignment_data"] = alignment_data
            path = os.path.join(step_dir, "geometrically_aligned.png")
            save_image(path, self.image_to_be_processed)
            if self.args.debug:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. After Geometrical Alignment",
                        "path": os.path.relpath(path, self.report_generator.project_output_dir),
                    }
                )
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

        advanced_aligner = AdvancedAligner()
        aligned_image = advanced_aligner.align(self.image_to_be_processed, ref_image, method="feature_orb")

        if aligned_image is not None:
            self.image_to_be_processed = aligned_image
            path = os.path.join(step_dir, "object_aligned.png")
            save_image(path, self.image_to_be_processed)
            if self.args.debug:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. After Object Alignment",
                        "path": os.path.relpath(path, self.report_generator.project_output_dir),
                    }
                )
                self.pipeline_step_counter += 1
        elif self.args.debug:
            print("[WARNING] Object alignment failed. Proceeding with unaligned image.")

    def _apply_masking(self):
        step_dir = self.report_generator.get_step_output_dir("masking")
        project_files = self.project_manager.get_project_file_paths(self.args.project, self.args.debug)
        mask_creator = MaskCreator()
        
        masking_order = self.args.masking_order.split('-')
        final_mask = None

        for layer_num in masking_order:
            layer_key = f"technical_drawing_layer_{layer_num}"
            drawing_path = project_files.get(layer_key)
            if drawing_path:
                mask = mask_creator.create_mask(str(drawing_path), self.args.mask_bg_is_white)
                if mask is not None:
                    if final_mask is None:
                        final_mask = mask
                    else:
                        final_mask = cv2.bitwise_and(final_mask, mask)
        
        if final_mask is not None:
            self.image_to_be_processed = mask_creator.apply_mask(self.image_to_be_processed, final_mask)
            path = os.path.join(step_dir, "masked_image.png")
            save_image(path, self.image_to_be_processed)
            if self.args.debug:
                mask_path = os.path.join(step_dir, "final_mask.png")
                save_image(mask_path, final_mask)
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. Final Applied Mask",
                        "path": os.path.relpath(mask_path, self.report_generator.project_output_dir),
                    }
                )
                self.pipeline_step_counter += 1
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. After Masking",
                        "path": os.path.relpath(path, self.report_generator.project_output_dir),
                    }
                )
                self.pipeline_step_counter += 1

    def _perform_color_analysis(self):
        step_dir = self.report_generator.get_step_output_dir("color_analysis")
        self.analysis_results = self.color_analyzer.process_image(
            image=self.image_to_be_processed,
            image_path=self.image_path,
            lower_hsv=self.project_data["lower_hsv"],
            upper_hsv=self.project_data["upper_hsv"],
            center_hsv=self.project_data["center_hsv"],
            output_dir=step_dir,
            debug_mode=self.args.debug,
            aggregate_mode=self.args.aggregate,
            alignment_mode=False,
            drawing_path=None,
            agg_kernel_size=self.args.agg_kernel_size,
            agg_min_area=self.args.agg_min_area,
            agg_density_thresh=self.args.agg_density_thresh,
        )
        # Save the final processed image from color analysis for clarity in the report
        path = os.path.join(step_dir, "final_color_analyzed_image.png")
        save_image(path, self.analysis_results['processed_image'])
        if self.args.debug:
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. Color Analysis Result",
                    "path": os.path.relpath(path, self.report_generator.project_output_dir),
                }
            )
            self.pipeline_step_counter += 1

    def _perform_symmetry_analysis(self):
        step_dir = self.report_generator.get_step_output_dir("symmetry_analysis")
        symmetry_analyzer = SymmetryAnalyzer(self.image_to_be_processed)
        symmetry_analyzer.analyze_all()
        self.debug_data_for_report["symmetry_results"] = symmetry_analyzer.results

        if self.args.debug:
            self.debug_data_for_report['symmetry_visualizations'] = []
            for analysis_type, result in symmetry_analyzer.results.items():
                if 'reconstruction' in result:
                    vis_img = result['reconstruction']
                    fname = f"symmetry_{analysis_type}.png"
                    path = os.path.join(step_dir, fname)
                    save_image(path, vis_img)
                    self.debug_data_for_report['symmetry_visualizations'].append({
                        'title': f"Symmetry: {analysis_type.replace('_', ' ').title()}",
                        'path': os.path.relpath(path, self.report_generator.project_output_dir)
                    })

    def _perform_blur(self):
        step_dir = self.report_generator.get_step_output_dir("blur")
        
        # The blur_kernel argument from argparse will be None if not provided
        # The blur_image function handles the None case by calculating an adaptive kernel size
        blurred_image, kernel_used = blur_image(self.image_to_be_processed, self.args.blur_kernel)
        
        self.image_to_be_processed = blurred_image
        self.debug_data_for_report["blur_kernel_used"] = kernel_used

        path = os.path.join(step_dir, "blurred.png")
        save_image(path, self.image_to_be_processed)
        if self.args.debug:
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. After Blur (Kernel: {kernel_used})",
                    "path": os.path.relpath(path, self.report_generator.project_output_dir),
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

        # ... (logic from original process_image for debug_data_for_report)

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