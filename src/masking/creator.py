"""
This module provides the `MaskCreator` class for generating and applying masks
from drawing images. It supports creating masks based on transparency and color,
and applying these masks to images to achieve background removal effects.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from src.utils.image_utils import load_image, save_image


class MaskCreator:
    """
    A class to create and apply masks from drawing files.

    It provides methods to generate a binary mask from an input drawing image
    (supporting transparency and white color as background) and to apply this
    mask to another image, typically to make the background transparent.
    """

    def create_mask(
        self,
        drawing_path: str,
        handle_alpha: bool = True,
        treat_white_as_bg: bool = False,
        debug_mode: bool = False,
        output_dir: str = None,
        image_for_debug: np.ndarray = None,
    ) -> np.ndarray:
        """
        Creates a binary mask from a drawing image.

        By default, if the image has an alpha channel, non-solid pixels (alpha < 255)
        are treated as background. This can be disabled by setting `handle_alpha` to False.

        Args:
            drawing_path (str): The path to the drawing image file.
            handle_alpha (bool): If True (default), use the alpha channel to create the mask,
                                 treating only fully opaque pixels (alpha=255) as foreground.
                                 If False, generates the mask from grayscale intensity.
            treat_white_as_bg (bool): If True, pure white pixels (255, 255, 255) in the
                                      drawing image are also treated as background and
                                      will be masked out.
            debug_mode (bool): If True, saves intermediate images of the mask creation process.
            output_dir (str): The directory to save debug images to. Required if debug_mode is True.
            image_for_debug (np.ndarray, optional): If provided in debug mode, intermediate masks
                                                will be applied to this image for visualization.

        Returns:
            np.ndarray: A single-channel 8-bit binary mask (foreground is 255, background is 0).
                        This is the correct, standard format for a mask.
                        Returns None if the image cannot be loaded from `drawing_path`.
        """
        drawing_img, alpha_channel = load_image(drawing_path, handle_transparency=True)

        if drawing_img is None:
            return None

        def _save_debug_and_apply(mask_to_save, step_name):
            if not (debug_mode and output_dir):
                return
            base_name = Path(drawing_path).stem
            mask_filename = f"{base_name}_{step_name}_mask.png"
            save_path = os.path.join(output_dir, mask_filename)
            save_image(save_path, mask_to_save)
            print(f"[DEBUG] Saved masking step (raw mask): {save_path}")

            image_to_visualize = (
                image_for_debug if image_for_debug is not None else drawing_img
            )
            if image_to_visualize is not None:
                # The apply_mask function already returns a 4-channel BGRA image
                applied_img = self.apply_mask(image_to_visualize.copy(), mask_to_save)
                applied_filename = (
                    f"{base_name}_{step_name}_applied_to_pipeline_image.png"
                )
                applied_save_path = os.path.join(output_dir, applied_filename)
                save_image(applied_save_path, applied_img)
                print(f"[DEBUG] Saved masked visualization: {applied_save_path}")

        if handle_alpha and alpha_channel is not None:
            # Threshold at 254 to treat only solid pixels (255) as foreground
            _, mask = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
            _save_debug_and_apply(mask, "1_from_solid_alpha")
        else:
            gray_img = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
            if debug_mode and output_dir:
                save_image(
                    os.path.join(
                        output_dir, f"{Path(drawing_path).stem}_1a_grayscale_input.png"
                    ),
                    gray_img,
                )
            # Original behavior for non-alpha images
            _, mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
            _save_debug_and_apply(mask, "1b_from_grayscale")

        if treat_white_as_bg:
            if len(drawing_img.shape) == 2:
                bgr_img = cv2.cvtColor(drawing_img, cv2.COLOR_GRAY2BGR)
            else:
                bgr_img = drawing_img

            white_mask = cv2.inRange(bgr_img, (255, 255, 255), (255, 255, 255))
            _save_debug_and_apply(white_mask, "2a_white_pixels")

            inverted_white_mask = cv2.bitwise_not(white_mask)
            mask = cv2.bitwise_and(mask, inverted_white_mask)
            _save_debug_and_apply(mask, "3_final_mask")

        # This function correctly returns the 1-channel binary mask.
        return mask

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies a binary mask to an image, making the masked-out regions transparent.

        The input image is converted to BGRA format, and its alpha channel is set
        according to the provided `mask`. If the mask dimensions do not match the
        image, the mask is resized using nearest-neighbor interpolation.

        Args:
            image (np.ndarray): The input BGR or grayscale image.
            mask (np.ndarray): The single-channel binary mask (0 for transparent, 255 for opaque).

        Returns:
            np.ndarray: A 4-CHANNEL (BGRA) image with a transparent background
                        where the mask was 0. If `mask` is None, the original
                        image is returned unchanged.
        """
        if mask is None:
            return image

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert the input image to 4 channels (BGRA) if it isn't already.
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Set the 4th channel (alpha) using the 1-channel mask.
        image[:, :, 3] = mask

        # This function correctly returns the final 4-channel image.
        return image

def create_and_apply_mask_from_layers(
    image_to_be_processed: np.ndarray,
    project_files: dict,
    masking_order: list,
    mask_bg_is_white: bool,
    output_dir: str,
    debug_mode: bool = False,
) -> dict:
    """
    Creates a composite mask from multiple drawing layers and applies it to an image.

    This function encapsulates the logic of iterating through specified drawing layers,
    generating a mask from each, combining them, and applying the final mask to the
    input image. It also handles saving intermediate and final outputs for debugging.

    Args:
        image_to_be_processed (np.ndarray): The input image (BGR or BGRA) to be masked.
        project_files (dict): A dictionary of project file paths.
        masking_order (list): A list of layer numbers (as strings) to combine for the mask.
        mask_bg_is_white (bool): Flag indicating if white should be treated as background.
        output_dir (str): The directory to save output and debug files.
        debug_mode (bool, optional): Enables saving of debug files. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'image' (np.ndarray | None): The masked image, or None if masking fails.
            - 'debug_paths' (list): A list of dictionaries with title and path for debug reports.
    """
    mask_creator = MaskCreator()
    final_mask = None
    debug_paths = []

    for layer_num in masking_order:
        layer_key = f"technical_drawing_layer_{layer_num}"
        drawing_path = project_files.get(layer_key)
        if drawing_path:
            if debug_mode:
                print(f"[DEBUG] Attempting to create mask from layer {layer_num} using file: {drawing_path}")
            mask = mask_creator.create_mask(
                str(drawing_path), 
                treat_white_as_bg=mask_bg_is_white, 
                debug_mode=debug_mode, 
                output_dir=output_dir, 
                image_for_debug=image_to_be_processed
            )
            if mask is not None:
                if final_mask is None:
                    final_mask = mask
                else:
                    final_mask = cv2.bitwise_and(final_mask, mask)
                if debug_mode:
                    print(f"[DEBUG] Successfully created and combined mask from layer {layer_num}.")
            elif debug_mode:
                print(f"[WARNING] Failed to create mask from layer {layer_num} file: {drawing_path}")
        elif debug_mode:
            print(f"[WARNING] No path found for technical_drawing_layer_{layer_num} in project configuration.")
    
    if final_mask is None:
        raise RuntimeError("Masking step was enabled but failed to generate a final mask.")

    if debug_mode:
        print(f"[DEBUG] Final mask created. Applying to image.")

    # If the image is grayscale, convert it to BGR. This leaves BGRA images untouched.
    if len(image_to_be_processed.shape) == 2:
        image_to_be_processed = cv2.cvtColor(image_to_be_processed, cv2.COLOR_GRAY2BGR)

    # Apply the mask. For BGRA images, this correctly makes masked-out areas
    # transparent black ([0, 0, 0, 0]). For BGR, it makes them black.
    masked_image = cv2.bitwise_and(image_to_be_processed, image_to_be_processed, mask=final_mask)

    # Calculate statistics
    kept_pixels = np.count_nonzero(final_mask)
    total_pixels = final_mask.shape[0] * final_mask.shape[1]
    kept_percentage = (kept_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    mask_stats = {
        'kept_pixels': kept_pixels,
        'total_pixels': total_pixels,
        'kept_percentage': kept_percentage
    }

    path = os.path.join(output_dir, "masked_image.png")
    save_image(path, masked_image)
    
    if debug_mode:
        mask_path = os.path.join(output_dir, "final_mask.png")
        save_image(mask_path, final_mask)
        debug_paths.append({'title': "Final Applied Mask", 'path': mask_path})
        debug_paths.append({'title': "After Masking", 'path': path})

    return {'image': masked_image, 'debug_paths': debug_paths, 'stats': mask_stats}
