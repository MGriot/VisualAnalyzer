import os
import cv2
import numpy as np
from typing import Tuple
import datetime

from src.utils.image_utils import load_image, save_image
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

    def _blur_image(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), debug_mode: bool = False) -> np.ndarray:
        """
        Applies Gaussian blur to the image.

        Args:
            image (np.ndarray): The input image.
            kernel_size (Tuple[int, int]): Size of the Gaussian kernel. (width, height).
            debug_mode (bool): If True, prints debug information.

        Returns:
            np.ndarray: The blurred image.
        """
        if debug_mode: print(f"[DEBUG] Applying Gaussian blur with kernel size: {kernel_size}")
        return cv2.GaussianBlur(image, kernel_size, 0)

    def find_color_zones(self, image: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray, alpha_channel: np.ndarray = None, debug_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds color zones within an image that fall within the specified HSV range.

        Args:
            image (np.ndarray): The input image (BGR format).
            lower_hsv (np.ndarray): The lower bound of the HSV color range.
            upper_hsv (np.ndarray): The upper bound of the HSV color range.
            alpha_channel (np.ndarray, optional): The alpha channel of the image for transparency handling.
            debug_mode (bool): If True, prints debug information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the mask image and the negative mask image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        if alpha_channel is not None:
            # Apply the alpha channel to the mask, so transparent areas are not considered
            mask = cv2.bitwise_and(mask, mask, mask=alpha_channel)

        negative_mask = cv2.bitwise_not(mask)

        if debug_mode: print(f"[DEBUG] Color analysis performed with HSV range: {lower_hsv} - {upper_hsv}")

        return mask, negative_mask

    def _aggregate_mask(self, mask: np.ndarray, kernel_size: int = 5, min_area_ratio: float = 0.001, debug_mode: bool = False) -> np.ndarray:
        """
        Aggregates nearby matched pixel areas in a binary mask.

        Args:
            mask (np.ndarray): The binary mask (255 for matched, 0 for unmatched).
            kernel_size (int): Size of the kernel for morphological operations. Controls aggregation extent.
            min_area_ratio (float): Minimum area of a connected component to keep, as a ratio of total image area.
            debug_mode (bool): If True, prints debug information.

        Returns:
            np.ndarray: The aggregated mask.
        """
        if debug_mode: print(f"[DEBUG] Aggregating mask with kernel_size={kernel_size}, min_area_ratio={min_area_ratio}")

        # Morphological Closing: Dilation followed by Erosion
        # This connects nearby white regions and fills small black holes.
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, 8, cv2.CV_32S)

        aggregated_mask = np.zeros_like(mask)
        total_image_area = mask.shape[0] * mask.shape[1]

        for i in range(1, num_labels): # Skip background label (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= total_image_area * min_area_ratio:
                aggregated_mask[labels == i] = 255
                if debug_mode: print(f"[DEBUG]   Kept component {i} with area {area}.")
            else:
                if debug_mode: print(f"[DEBUG]   Filtered out component {i} with area {area} (too small).")
        
        if debug_mode: print(f"[DEBUG] Aggregation complete. Original matched pixels: {cv2.countNonZero(mask)}, Aggregated matched pixels: {cv2.countNonZero(aggregated_mask)}")

        return aggregated_mask

    def calculate_statistics(self, mask: np.ndarray, total_pixels: int, debug_mode: bool = False) -> Tuple[float, int]:
        """
        Calculates the percentage and number of matched pixels.

        Args:
            mask (np.ndarray): The mask image (binary, 255 for matched, 0 for unmatched).
            total_pixels (int): The total number of pixels in the original image.
            debug_mode (bool): If True, prints debug information.

        Returns:
            Tuple[float, int]: A tuple containing the percentage of matched pixels and the
                               number of matched pixels.
        """
        matched_pixels = cv2.countNonZero(mask)
        percentage = (matched_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        if debug_mode:
            print(f"[DEBUG] Matched Pixels: {matched_pixels}")
            print(f"[DEBUG] Total Pixels (non-transparent): {total_pixels}")
            print(f"[DEBUG] Percentage of Matched Pixels: {percentage:.2f}%")

        return percentage, matched_pixels

    def process_image(self, image: np.ndarray = None, image_path: str = None, lower_hsv: np.ndarray = None, upper_hsv: np.ndarray = None, output_dir: str = None, debug_mode: bool = False, aggregate_mode: bool = False, blur_mode: bool = False, alignment_mode: bool = False, drawing_path: str = None) -> dict:
        """
        Processes a single image for color analysis.

        Args:
            image (np.ndarray, optional): The input image as a NumPy array. If provided, image_path is ignored.
            image_path (str, optional): Path to the input image. Used if 'image' is None.
            lower_hsv (np.ndarray): Lower HSV limit for color matching.
            upper_hsv (np.ndarray): Upper HSV limit for color matching.
            output_dir (str): Directory to save output images.
            debug_mode (bool): If True, prints debug information.
            aggregate_mode (bool): If True, aggregates nearby matched pixel areas.
            blur_mode (bool): If True, blurs the image before color matching.
            alignment_mode (bool): If True, performs image alignment.
            drawing_path (str): Path to the technical drawing for alignment.

        Returns:
            dict: A dictionary containing analysis results.
        """
        if image is None and image_path is None:
            raise ValueError("Either 'image' or 'image_path' must be provided.")

        if alignment_mode:
            if not drawing_path:
                raise ValueError("Drawing path must be provided for alignment.")
            if not image_path:
                raise ValueError("Image path must be provided for alignment.")
            aligner = Aligner(debug_mode=debug_mode)
            image = aligner.align_image(image_path=image_path, drawing_path=drawing_path)

        if image is not None:
            original_image = image
            alpha_channel = None # Assume no alpha if direct image is passed, or handle separately if needed
            if original_image.shape[2] == 4: # Check for RGBA
                alpha_channel = original_image[:, :, 3]
                original_image = original_image[:, :, :3] # Convert to BGR
        else:
            original_image, alpha_channel = load_image(image_path, handle_transparency=True)
            if original_image is None:
                raise ValueError(f"Could not load image {image_path}")

        if lower_hsv is None or upper_hsv is None:
            raise ValueError("lower_hsv and upper_hsv must be provided.")

        if output_dir is None:
            raise ValueError("output_dir must be provided.")

        image_for_analysis = original_image.copy()
        blurred_image_path = None
        if blur_mode:
            image_for_analysis = self._blur_image(image_for_analysis, debug_mode=debug_mode)
            blurred_image_path = os.path.join(output_dir, f"blurred_image_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png")
            save_image(blurred_image_path, image_for_analysis) # Save blurred image
            if debug_mode: print(f"[DEBUG] Blurred image saved to {blurred_image_path}")

        total_pixels = image_for_analysis.shape[0] * image_for_analysis.shape[1]
        if alpha_channel is not None:
            total_pixels = cv2.countNonZero(alpha_channel) # Only count solid pixels

        mask, negative_mask = self.find_color_zones(image_for_analysis, lower_hsv, upper_hsv, alpha_channel, debug_mode=debug_mode)

        mask_pre_aggregation_path = None
        if aggregate_mode:
            mask_pre_aggregation_path = os.path.join(output_dir, f"mask_pre_aggregation_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png")
            save_image(mask_pre_aggregation_path, mask) # Save mask before aggregation
            if debug_mode: print(f"[DEBUG] Mask before aggregation saved to {mask_pre_aggregation_path}")

            mask = self._aggregate_mask(mask, debug_mode=debug_mode)
            negative_mask = cv2.bitwise_not(mask) # Recalculate negative mask after aggregation

        percentage, matched_pixels = self.calculate_statistics(mask, total_pixels, debug_mode=debug_mode)

        # Save output images
        # Ensure output_dir exists
        os.makedirs(output_dir, exist_ok=True)

        # Use unique filenames for output images
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        processed_image_path = os.path.join(output_dir, f"processed_image_{timestamp}.png")
        mask_path = os.path.join(output_dir, f"mask_{timestamp}.png")
        negative_mask_path = os.path.join(output_dir, f"negative_mask_{timestamp}.png")

        # Create a visual representation of the processed image (e.g., original with mask overlay)
        processed_image = original_image.copy()
        processed_image[mask == 0] = [0, 0, 0] # Black out non-matched areas

        save_image(processed_image_path, processed_image)
        save_image(mask_path, mask)
        save_image(negative_mask_path, negative_mask)

        return {
            "original_image": original_image, # Return original image array
            "processed_image": processed_image, # Return processed image array
            "mask": mask, # Return mask array
            "negative_mask": negative_mask, # Return negative mask array
            "original_image_path": image_path, # Original path if provided
            "processed_image_path": processed_image_path,
            "mask_path": mask_path,
            "negative_mask_path": negative_mask_path,
            "percentage": percentage,
            "matched_pixels": matched_pixels,
            "total_pixels": total_pixels,
            "mask_pre_aggregation_path": mask_pre_aggregation_path, # New: Path to mask before aggregation
            "blurred_image_path": blurred_image_path # New: Path to blurred image
        }
