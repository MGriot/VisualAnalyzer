import cv2
import numpy as np
from typing import Tuple

def load_image(image_path: str, handle_transparency: bool = True):
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.
        handle_transparency (bool): If True, converts RGBA to RGB and creates a mask for transparent pixels.

    Returns:
        np.ndarray: The loaded image (RGB or BGR).
        np.ndarray or None: The alpha channel (mask) if handle_transparency is True and image has alpha, else None.
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
    Saves an image to the specified path.

    Args:
        image_path (str): The path to save the image file.
        image (np.ndarray): The image to save.
    """
    cv2.imwrite(image_path, image)

def blur_image(image: np.ndarray, kernel_size: Tuple[int, int] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Applies Gaussian blur to the image.
    If kernel_size is not provided, it is calculated adaptively based on image dimensions.

    Args:
        image (np.ndarray): The input image.
        kernel_size (Tuple[int, int], optional): Size of the Gaussian kernel.
                                                  If None, it's calculated automatically.
                                                  Defaults to None.

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: A tuple containing the blurred image and the kernel size used.
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

    return cv2.GaussianBlur(image, kernel_size, 0), kernel_size
