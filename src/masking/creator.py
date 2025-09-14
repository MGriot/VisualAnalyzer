import cv2
import numpy as np
from src.utils.image_utils import load_image

class MaskCreator:
    """
    A class to create and apply masks from drawing files.
    """

    def create_mask(
        self, drawing_path: str, treat_white_as_bg: bool = False
    ) -> np.ndarray:
        """
        Creates a binary mask from a drawing image.

        The foreground is assumed to be non-transparent and, if specified, non-white.
        All other pixels are considered background (value 0).

        Args:
            drawing_path (str): The path to the drawing image.
            treat_white_as_bg (bool): If True, white pixels (255, 255, 255) are also
                                      treated as background.

        Returns:
            np.ndarray: A single-channel 8-bit binary mask (foreground is 255).
                        Returns None if the image cannot be loaded.
        """
        # Load the image, ensuring the alpha channel is loaded if present
        drawing_img, alpha_channel = load_image(drawing_path, handle_transparency=True)

        if drawing_img is None:
            return None

        # Case 1: Image has an alpha channel
        if alpha_channel is not None:
            # Use the alpha channel as the primary mask
            # Alpha > 0 is considered foreground
            _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        # Case 2: Image does not have an alpha channel
        else:
            # Assume the object is not pure white. The mask is initially all foreground.
            gray_img = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
            # We can't use transparency, so we assume non-black is foreground
            _, mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        # If treat_white_as_bg is True, remove white pixels from the foreground
        if treat_white_as_bg:
            # Ensure image is 3-channel BGR for color comparison
            if len(drawing_img.shape) == 2:
                bgr_img = cv2.cvtColor(drawing_img, cv2.COLOR_GRAY2BGR)
            else:
                bgr_img = drawing_img[:,:,:3]

            white_mask = cv2.inRange(bgr_img, (255, 255, 255), (255, 255, 255))
            # Invert the white mask (white pixels become 0)
            white_mask_inv = cv2.bitwise_not(white_mask)
            # Combine with the main mask
            mask = cv2.bitwise_and(mask, white_mask_inv)

        return mask

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies a mask to an image, making the background transparent.

        Args:
            image (np.ndarray): The input BGR image.
            mask (np.ndarray): The single-channel binary mask.

        Returns:
            np.ndarray: A BGRA image with a transparent background.
        """
        if mask is None:
            return image
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure image has 3 channels before converting to BGRA
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:,:,:3]

        # Resize mask to match image if necessary
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert image to BGRA and apply the mask to the alpha channel
        # bgra_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # bgra_image[:, :, 3] = mask
        # return bgra_image

        # Apply the mask to make the background black, preserving the 3-channel format
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
