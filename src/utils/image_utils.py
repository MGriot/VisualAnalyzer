import cv2
import numpy as np

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
