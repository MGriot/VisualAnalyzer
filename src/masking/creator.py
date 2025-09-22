"""
This module provides the `MaskCreator` class for generating and applying masks
from drawing images. It supports creating masks based on transparency and color,
and applying these masks to images to achieve background removal effects.
"""

import cv2
import numpy as np
from src.utils.image_utils import load_image

class MaskCreator:
    """
    A class to create and apply masks from drawing files.

    It provides methods to generate a binary mask from an input drawing image
    (supporting transparency and white color as background) and to apply this
    mask to another image, typically to make the background transparent.
    """

    def create_mask(
        self, drawing_path: str, treat_white_as_bg: bool = False
    ) -> np.ndarray:
        """
        Creates a binary mask from a drawing image.

        The foreground is assumed to be non-transparent and, if specified, non-white.
        All other pixels are considered background (value 0).

        Args:
            drawing_path (str): The path to the drawing image file.
            treat_white_as_bg (bool): If True, pure white pixels (255, 255, 255) in the
                                      drawing image are also treated as background and
                                      will be masked out.

        Returns:
            np.ndarray: A single-channel 8-bit binary mask (foreground is 255, background is 0).
                        Returns None if the image cannot be loaded from `drawing_path`.
        """
        drawing_img, alpha_channel = load_image(drawing_path, handle_transparency=True)

        if drawing_img is None:
            return None

        if alpha_channel is not None:
            _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        else:
            gray_img = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        if treat_white_as_bg:
            if len(drawing_img.shape) == 2:
                bgr_img = cv2.cvtColor(drawing_img, cv2.COLOR_GRAY2BGR)
            else:
                bgr_img = drawing_img

            white_mask = cv2.inRange(bgr_img, (255, 255, 255), (255, 255, 255))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))

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
            np.ndarray: A BGRA image with a transparent background where the mask was 0.
                        If `mask` is None, the original image is returned unchanged.
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

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        image[:, :, 3] = mask
        return image
