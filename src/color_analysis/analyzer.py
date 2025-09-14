import os
import cv2
import numpy as np
from typing import Tuple
import datetime

from src.utils.image_utils import load_image, save_image, blur_image
from src.alignment.aligner import Aligner

class ColorAnalyzer:
    """
    Analyzes images or video frames to find zones matching a specified HSV color range.
    Generates masks, negative images, and calculates statistics.
    """

    def __init__(self):
        """
        Initializes the ColorAnalyzer.
        """
        pass

    def find_color_zones(self, image: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray, alpha_channel: np.ndarray = None, debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds color zones within an image that fall within the specified HSV range.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        if alpha_channel is not None:
            mask = cv2.bitwise_and(mask, mask, mask=alpha_channel)

        negative_mask = cv2.bitwise_not(mask)

        if debug_mode: print(f"[DEBUG] Color analysis performed with HSV range: {lower_hsv} - {upper_hsv}")

        return mask, negative_mask

    def _aggregate_mask_improved(self, mask: np.ndarray, kernel_size: int, min_area_ratio: float, agg_density_thresh: float, debug_mode: bool = False) -> np.ndarray:
        """
        Aggregates nearby matched pixel areas in a binary mask.
        """
        if debug_mode: print(f"[DEBUG] Improved aggregating mask with kernel_size={kernel_size}, min_area_ratio={min_area_ratio}, density_thresh={agg_density_thresh}")

        dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_mask, 8, cv2.CV_32S)

        aggregated_mask = np.zeros_like(mask)
        total_image_area = mask.shape[0] * mask.shape[1]

        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]
            if component_area >= total_image_area * min_area_ratio:
                component_mask = (labels == i).astype(np.uint8) * 255
                
                # Density Check
                original_pixels_in_component = cv2.countNonZero(cv2.bitwise_and(mask, component_mask))
                density = original_pixels_in_component / component_area if component_area > 0 else 0

                if density >= agg_density_thresh:
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(aggregated_mask, contours, -1, 255, cv2.FILLED)
                    if debug_mode: print(f"[DEBUG]   Kept component {i} with area {component_area} and density {density:.2f}.")
                else:
                    if debug_mode: print(f"[DEBUG]   Rejected component {i} with area {component_area} due to low density ({density:.2f} < {agg_density_thresh}).")
            else:
                if debug_mode: print(f"[DEBUG]   Filtered out component {i} with area {component_area} (too small).")
        
        final_aggregated_mask = cv2.bitwise_or(aggregated_mask, mask)
        
        if debug_mode: print(f"[DEBUG] Improved aggregation complete. Original matched pixels: {cv2.countNonZero(mask)}, Aggregated matched pixels: {cv2.countNonZero(final_aggregated_mask)}")

        return final_aggregated_mask

    def calculate_statistics(self, mask: np.ndarray, total_pixels: int, debug_mode: bool = False) -> Tuple[float, int]:
        """
        Calculates the percentage and number of matched pixels.
        """
        matched_pixels = cv2.countNonZero(mask)
        percentage = (matched_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        if debug_mode:
            print(f"[DEBUG] Matched Pixels: {matched_pixels}")
            print(f"[DEBUG] Total Pixels (non-transparent): {total_pixels}")
            print(f"[DEBUG] Percentage of Matched Pixels: {percentage:.2f}%")

        return percentage, matched_pixels

    def process_image(self, image: np.ndarray = None, image_path: str = None, lower_hsv: np.ndarray = None, upper_hsv: np.ndarray = None, center_hsv: np.ndarray = None, output_dir: str = None, debug_mode: bool = False, aggregate_mode: bool = False, alignment_mode: bool = False, drawing_path: str = None, agg_kernel_size: int = 7, agg_min_area: float = 0.0005, agg_density_thresh: float = 0.5) -> dict:
        """
        Processes a single image for color analysis.
        """
        if image is None and image_path is None:
            raise ValueError("Either 'image' or 'image_path' must be provided.")

        alignment_data = None
        if alignment_mode:
            # This block is currently not hit from main.py but kept for potential future use
            if not drawing_path:
                raise ValueError("Drawing path must be provided for alignment.")
            if not image_path:
                raise ValueError("Image path must be provided for alignment.")
            aligner = Aligner(debug_mode=debug_mode, output_dir=output_dir)
            alignment_result = aligner.align_image(image_path=image_path, drawing_path=drawing_path)
            if alignment_result is None:
                print("[WARNING] Image alignment failed. Proceeding without alignment.")
                image = None
                alignment_data = None
            else:
                aligned_image, alignment_data = alignment_result
                image = aligned_image

        if image is not None:
            original_image = image
            alpha_channel = None
            if original_image.shape[2] == 4:
                alpha_channel = original_image[:, :, 3]
                original_image = original_image[:, :, :3]
        else:
            original_image, alpha_channel = load_image(image_path, handle_transparency=True)
            if original_image is None:
                raise ValueError(f"Could not load image {image_path}")

        if lower_hsv is None or upper_hsv is None:
            raise ValueError("lower_hsv and upper_hsv must be provided.")
        if output_dir is None:
            raise ValueError("output_dir must be provided.")

        image_for_analysis = original_image.copy()
        
        total_pixels = image_for_analysis.shape[0] * image_for_analysis.shape[1]
        if alpha_channel is not None:
            total_pixels = cv2.countNonZero(alpha_channel)

        mask, negative_mask = self.find_color_zones(image_for_analysis, lower_hsv, upper_hsv, alpha_channel, debug_mode=debug_mode)

        mask_pre_aggregation_path = None
        if aggregate_mode:
            mask_pre_aggregation_path = os.path.join(output_dir, f"mask_pre_aggregation_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
            save_image(mask_pre_aggregation_path, mask)
            if debug_mode: print(f"[DEBUG] Mask before aggregation saved to {mask_pre_aggregation_path}")
            mask = self._aggregate_mask_improved(mask, kernel_size=agg_kernel_size, min_area_ratio=agg_min_area, agg_density_thresh=agg_density_thresh, debug_mode=debug_mode)
            negative_mask = cv2.bitwise_not(mask)

        percentage, matched_pixels = self.calculate_statistics(mask, total_pixels, debug_mode=debug_mode)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

        processed_image_path = os.path.join(output_dir, f"processed_image_{timestamp}.png")
        mask_path = os.path.join(output_dir, f"mask_{timestamp}.png")
        negative_mask_path = os.path.join(output_dir, f"negative_mask_{timestamp}.png")

        processed_image = original_image.copy()
        processed_image[mask == 0] = [0, 0, 0]

        save_image(processed_image_path, processed_image)
        save_image(mask_path, mask)
        save_image(negative_mask_path, negative_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = original_image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)
        contours_image_path = os.path.join(output_dir, f"contours_{timestamp}.png")
        save_image(contours_image_path, image_with_contours)

        return {
            "original_image": original_image,
            "processed_image": processed_image,
            "mask": mask,
            "negative_mask": negative_mask,
            "original_image_path": image_path,
            "processed_image_path": processed_image_path,
            "mask_path": mask_path,
            "negative_mask_path": negative_mask_path,
            "contours_image_path": contours_image_path,
            "percentage": percentage,
            "matched_pixels": matched_pixels,
            "total_pixels": total_pixels,
            "mask_pre_aggregation_path": mask_pre_aggregation_path,
            "alignment_data": alignment_data,
            "lower_limit": lower_hsv,
            "upper_limit": upper_hsv,
            "center_color": center_hsv,
            "selected_colors": [
                {"color_name": "Selected Area", "hsv": center_hsv if center_hsv is not None else [0,0,0], "rgb": [0,0,0]}
            ]
        }