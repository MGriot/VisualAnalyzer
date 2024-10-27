from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


class ImageProcessor:
    def __init__(self, image_path):
        """
        Initialize the ImageProcessor object with the path of the image.

        Parameters:
        image_path (str): The path of the image file.
        """
        self.image_path = image_path
        self.img = None
        self.load_image()

    def load_image(self):
        """
        Load the image from the specified path. If the image file does not exist or cannot be opened,
        an appropriate message will be printed.
        """
        if not os.path.exists(self.image_path):
            print(f"The image file {self.image_path} does not exist.")
            return

        self.img = Image.open(self.image_path)
        if self.img is None:
            print(f"The image file {self.image_path} cannot be opened.")
            return

    def blur_filter(self, filter_type, **kwargs):
        """
        Apply a filter to the image. Available filters are: 'GaussianBlur', 'BoxBlur', 'MedianFilter'.

        Parameters:
        filter_type (str): The type of the filter to be applied. It should be one of the following: 'GaussianBlur', 'BoxBlur', 'MedianFilter'.
        **kwargs: Additional parameters that might be needed for the filters. For 'GaussianBlur' and 'BoxBlur', you can specify 'radius'. For 'MedianFilter', you can specify 'size'.
        """
        if self.img is None:
            print("No image loaded.")
            return

        if filter_type == "GaussianBlur":
            self.img = self.img.filter(
                ImageFilter.GaussianBlur(kwargs.get("radius", 2))
            )
        elif filter_type == "BoxBlur":
            self.img = self.img.filter(ImageFilter.BoxBlur(kwargs.get("radius", 2)))
        elif filter_type == "MedianFilter":
            self.img = self.img.filter(ImageFilter.MedianFilter(kwargs.get("size", 3)))

    def increase_brightness(self, factor=1.2):
        """
        Increase the brightness of the image by a certain factor.

        Parameters:
        factor (float): The factor by which to increase the brightness. Default is 1.2.
        """
        if self.img is None:
            print("No image loaded.")
            return

        enhancer = ImageEnhance.Brightness(self.img)
        self.img = enhancer.enhance(factor)

    def increase_saturation(self, factor=1.2):
        """
        Increase the saturation of the image by a certain factor.

        Parameters:
        factor (float): The factor by which to increase the saturation. Default is 1.2.
        """
        if self.img is None:
            print("No image loaded.")
            return

        enhancer = ImageEnhance.Color(self.img)
        self.img = enhancer.enhance(factor)

    def increase_contrast(self, factor=1.2):
        """
        Increase the contrast of the image by a certain factor.

        Parameters:
        factor (float): The factor by which to increase the contrast. Default is 1.2.
        """
        if self.img is None:
            print("No image loaded.")
            return

        enhancer = ImageEnhance.Contrast(self.img)
        self.img = enhancer.enhance(factor)

    def resize(self, size=None, factor=None, maintain_aspect_ratio=False):
        """
        Resize the image to the specified size or downsample it by a certain factor.

        Parameters:
        size (tuple or int): The desired size of the image or the length of the longer side when maintain_aspect_ratio is True. Default is None.
        factor (int): The factor by which to downsample the image. Default is None.
        maintain_aspect_ratio (bool): If True, maintain the aspect ratio when resizing to a specific size. Default is False.
        """
        if self.img is None:
            print("No image loaded.")
            return

        if size is not None:
            if maintain_aspect_ratio:
                width, height = self.img.size
                aspect_ratio = width / height
                if width > height:
                    size = (size, int(size / aspect_ratio))
                else:
                    size = (int(size * aspect_ratio), size)
            else:
                if isinstance(size, int):
                    size = (size, size)
            self.img = self.img.resize(size)
        elif factor is not None:
            width, height = self.img.size
            self.img = self.img.resize((width // factor, height // factor))
        else:
            print("Please provide either size or factor.")

    def rotate(self, angle):
        """
        Rotate the image by a certain angle.

        Parameters:
        angle (float): The angle by which to rotate the image.
        """
        self.img = self.img.rotate(angle)

    def crop(self, box):
        """
        Crop the image to the specified box.

        Parameters:
        box (tuple): The box to which to crop the image.
        """
        self.img = self.img.crop(box)

    def to_grayscale(self):
        """
        Convert the image to grayscale.
        """
        self.img = self.img.convert("L")

    def normalize(self):
        """
        Normalize the image.
        This method scales the pixel values in the image to the range 0-1. This is done by dividing each pixel value by 255 (since images are 8-bit per channel, so the maximum value is 255).
        """
        self.img = self.img.point(lambda i: i / 255.0)

    def equalize(self):
        """
        Equalize the image.

        This method applies a histogram equalization to the image. Histogram equalization is a method in image processing of contrast adjustment using the image's histogram. This method usually increases the global contrast of many images, especially when the usable data of the image is represented by close contrast values.
        """
        if self.img is None:
            print("No image loaded.")
            return

        if self.img.mode == "RGBA":
            # Separate the alpha channel
            r, g, b, a = self.img.split()

            # Convert RGB channels to an image and equalize
            rgb_image = Image.merge("RGB", (r, g, b))
            equalized_rgb_image = ImageOps.equalize(rgb_image)

            # Merge equalized RGB channels and alpha channel back into an image
            r, g, b = equalized_rgb_image.split()
            self.img = Image.merge("RGBA", (r, g, b, a))
        else:
            self.img = ImageOps.equalize(self.img)

    def add_noise(self, radius=1.0):
        """
        Add noise to the image.

        This method applies a noise effect to the image. The effect randomly redistributes pixel values within a certain neighborhood around each pixel. The size of this neighborhood is defined by the radius parameter.

        Parameters:
        radius (float): The radius defining the neighborhood for the noise effect. Default is 1.0.
        """
        if self.img is None:
            print("No image loaded.")
            return

        self.img = self.img.effect_spread(radius)

    def flip(self, direction):
        """
        Flip the image in the specified direction.

        Parameters:
        direction (str): The direction in which to flip the image. It should be either 'horizontal' or 'vertical'.
        """
        if self.img is None:
            print("No image loaded.")
            return

        if direction == "horizontal":
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            self.img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            print(
                "Invalid direction. Please provide either 'horizontal' or 'vertical'."
            )

    def show_image(self, title="Image", use="Matplotlib"):
        """
        Show the image.

        Parameters:
        title (str): The title of the image. Default is "Image".
        use (str): The library to use for showing the image. Default is "Matplotlib".
        """
        if use == "Matplotlib":
            plt.imshow(self.img)
            plt.title(title)
            plt.show()
        else:
            self.img.show(title=title)
