"""
This module provides utility functions for common image processing operations
such as loading, saving, and blurring images using OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple

def load_image(image_path: str, handle_transparency: bool = True) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.
        handle_transparency (bool): If True, and the image has an alpha channel (RGBA),
                                    it separates the alpha channel and returns the image
                                    as BGR. Otherwise, it loads the image as is.

    Returns:
        Tuple[np.ndarray, np.ndarray | None]: A tuple containing:
            - The loaded image (NumPy array in BGR format).
            - The alpha channel (NumPy array) if `handle_transparency` is True and the
              image has an alpha channel, otherwise None.

    Raises:
        FileNotFoundError: If the image file does not exist at the specified path.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    alpha_channel = None
    if handle_transparency and img.shape[2] == 4:  # Check if image has an alpha channel
        alpha_channel = img[:, :, 3]
        img = img[:, :, :3]  # Convert to BGR

    return img, alpha_channel

def save_image(image_path: str, image: np.ndarray):
    """
    Saves a NumPy array image to the specified file path.

    Args:
        image_path (str): The full path, including filename and extension, where the image will be saved.
        image (np.ndarray): The image data as a NumPy array.
    """
    cv2.imwrite(image_path, image)

def blur_image(image: np.ndarray, kernel_size: Tuple[int, int] | None = None, output_dir: str | None = None) -> dict:
    """
    Applies a Gaussian blur filter to the input image, optionally saving the result.

    If `kernel_size` is not provided, it is adaptively calculated based on the
    smaller dimension of the image to ensure a reasonable blur effect.

    Args:
        image (np.ndarray): The input image to be blurred.
        kernel_size (Tuple[int, int] | None, optional): The size of the Gaussian kernel (width, height).
                                                        Both dimensions must be odd. If None, it's calculated automatically.
                                                        Defaults to None.
        output_dir (str | None, optional): If provided, the directory where the blurred image will be saved.
                                           Defaults to None.

    Returns:
        dict: A dictionary containing:
            - 'image' (np.ndarray): The blurred image.
            - 'kernel_used' (Tuple[int, int]): The kernel size used.
            - 'debug_path' (str | None): The path to the saved blurred image, or None if not saved.
    """
    if kernel_size is None:
        # Adaptively calculate kernel size based on the smaller dimension of the image.
        # This heuristic aims for a kernel size that is roughly 0.5% of the smaller dimension.
        min_dim = min(image.shape[0], image.shape[1])
        kernel_dim = max(3, int(min_dim * 0.005))

        # Ensure the kernel dimension is an odd number
        if kernel_dim % 2 == 0:
            kernel_dim += 1
        
        kernel_size = (kernel_dim, kernel_dim)

    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    
    debug_path = None
    if output_dir:
        import os
        debug_path = os.path.join(output_dir, "blurred.png")
        save_image(debug_path, blurred_image)

    return {
        'image': blurred_image,
        'kernel_used': kernel_size,
        'debug_path': debug_path
    }
