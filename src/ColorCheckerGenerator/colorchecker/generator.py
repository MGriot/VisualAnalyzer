"""
generator.py

Main orchestration class ColorCheckerGenerator that composes the final card using other modules.
"""

from typing import Optional, Literal
import numpy as np
import cv2

from .unit_utils import parse_size
from .color_data import get_colorchecker_classic, get_calibration_colors
from .drawing_utils import draw_color_grid, hex_to_bgr, put_text_cv2
from .ruler_utils import generate_ruler_dual
from .aruco_utils import add_aruco_markers_around
from .export_utils import save_image
from .config import DEFAULT_DPI, DEFAULT_BG

class ColorCheckerGenerator:
    """
    High level generator class to build different calibration cards.

    Example:
        gen = ColorCheckerGenerator(size="250mm", dpi=300, checker_type="calibration_card")
        gen.build()
        gen.save("card.png")
    """

    def __init__(
        self,
        size: Optional[Literal[str,int]] = "200mm",
        dpi: int = DEFAULT_DPI,
        checker_type: Literal["classic","calibration_card","greyscale","combined"] = "classic",
        include_ruler: Literal["none","cm","inch","both"] = "both",
        include_aruco: bool = False,
        include_grayscale: bool = False,
        include_shadow: bool = False,
        include_labels: bool = True,
        logo_text: Optional[str] = None,
    ):
        self.size = size
        self.dpi = dpi
        self.checker_type = checker_type
        self.include_ruler = include_ruler
        self.include_aruco = include_aruco
        self.include_grayscale = include_grayscale
        self.include_shadow = include_shadow
        self.include_labels = include_labels
        self.logo_text = logo_text

        # Will be produced by build()
        self.canvas = None
        self.target_width_px = parse_size(size, dpi)

    def build(self):
        """
        Build the chart canvas (numpy array BGR).
        """
        if self.checker_type == "classic":
            self.canvas = self._build_classic()
        elif self.checker_type == "calibration_card":
            self.canvas = self._build_calibration_card()
        elif self.checker_type == "greyscale":
            self.canvas = self._build_greyscale_only()
        elif self.checker_type == "combined":
            self.canvas = self._build_combined()
        else:
            raise ValueError("Unknown checker_type")

        # Optionally add rulers
        if self.include_ruler != "none":
            ruler = generate_ruler_dual(self.canvas.shape[1], dpi=self.dpi, mode=self.include_ruler)
            self.canvas = np.vstack([ruler, self.canvas, ruler])

        # Optionally add ArUco markers after rulers so they sit outside everything
        if self.include_aruco:
            self.canvas = add_aruco_markers_around(self.canvas)

        # Finally scale to requested width if needed
        self._scale_to_target_width()

        # Optional overlay logo or text
        if self.logo_text:
            h = self.canvas.shape[0]
            self.canvas = put_text_cv2(self.canvas, self.logo_text, (20, h-30), font_size=24, color=(30,30,30))

        return self.canvas

    def save(self, filename: str):
        """
        Save the generated canvas to a file. Supports PNG/JPG/PDF via export_utils.save_image.
        """
        if self.canvas is None:
            raise RuntimeError("Canvas not built yet. Call build() first.")
        save_image(self.canvas, filename)

    # --------------------- Internal builders ---------------------

    def _scale_to_target_width(self):
        """
        Resize canvas keeping aspect ratio to target_width_px.
        """
        w = self.canvas.shape[1]
        if w == self.target_width_px:
            return
        scale = self.target_width_px / w
        new_h = int(round(self.canvas.shape[0] * scale))
        self.canvas = cv2.resize(self.canvas, (self.target_width_px, new_h), interpolation=cv2.INTER_AREA)

    def _build_classic(self):
        """
        Builds the classic 24-patch ColorChecker grid.
        The main `build` method is responsible for adding optional rulers,
        markers, and labels.
        """
        return self._build_color_grid()


    def _build_color_grid(self):
        """
        Build the 24-patch ColorChecker Classic grid using the
        color data defined in color_data.py.

        Returns
        -------
        numpy.ndarray
            The rendered color grid as an OpenCV BGR image.
        """
        import cv2
        import numpy as np
        from . import color_data

        # Get the color array from color_data
        color_array = color_data.get_colorchecker_classic()
        nrows, ncols, _ = color_array.shape

        # Define a base patch size; the final scaling will handle the target size.
        patch_w = 100 
        patch_h = 75

        # Create blank canvas
        grid = np.ones((nrows * patch_h, ncols * patch_w, 3), dtype=np.uint8) * 255

        # Draw each patch
        for i in range(nrows):
            for j in range(ncols):
                rgb = color_array[i, j]
                bgr = tuple(int(c) for c in reversed(rgb))
                x0 = j * patch_w
                y0 = i * patch_h
                x1 = x0 + patch_w
                y1 = y0 + patch_h
                cv2.rectangle(grid, (x0, y0), (x1, y1), bgr, -1)

        return grid

    def _build_calibration_card(self):
        """
        Build calibration card similar to user's example:
        - Top ruler(s are added later)
        - Color row with labels and hex codes
        - Optional grayscale strip below
        """
        colors = get_calibration_colors()
        patch_w = 220
        patch_h = 150
        n = len(colors)
        # Create top color strip (two rows: color + text area)
        strip_h = patch_h + 60
        strip = np.ones((strip_h, patch_w * n, 3), dtype=np.uint8) * 240  # background
        for i, (name, hexc) in enumerate(colors):
            bgr = hex_to_bgr(hexc)
            x0, x1 = i*patch_w, (i+1)*patch_w
            # color patch top
            strip[0:patch_h, x0:x1] = bgr
            # text labels area (below color patch)
            txt_y = patch_h + 20
            strip = put_text_cv2(strip, name, (x0+10, txt_y), font_size=22, color=(20,20,20))
            strip = put_text_cv2(strip, hexc, (x0+10, txt_y+28), font_size=18, color=(80,80,80))

        parts = [strip]
        if self.include_grayscale:
            gs = self._build_greyscale_strip(width_px=strip.shape[1], n_steps=14, height_px=80)
            parts.append(gs)
        return np.vstack(parts)

    def _build_greyscale_only(self):
        """
        Build only a grayscale strip.
        """
        width = self.target_width_px if self.target_width_px else 1600
        return self._build_greyscale_strip(width_px=width, n_steps=14, height_px=120)

    def _build_combined(self):
        """
        Compose classic + calibration card stacked vertically.
        """
        classic = self._build_classic()
        calib = self._build_calibration_card()
        # make widths equal by resizing smaller to match larger
        w = max(classic.shape[1], calib.shape[1])
        classic = self._resize_width(classic, w)
        calib = self._resize_width(calib, w)
        return np.vstack([calib, classic])

    # --------------------- small helpers ---------------------

    def _resize_width(self, img, new_w):
        import cv2
        h = int(round(img.shape[0] * (new_w / img.shape[1])))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    def _build_greyscale_strip(self, width_px: int, n_steps: int = 14, height_px: int = 80):
        """
        Build a grayscale strip with n_steps patches across width_px.
        """
        patch_w = int(round(width_px / n_steps))
        strip = np.ones((height_px, patch_w*n_steps, 3), dtype=np.uint8) * 255
        for i in range(n_steps):
            val = int(round(255 * (1 - i / (n_steps - 1))))
            x0, x1 = i*patch_w, (i+1)*patch_w
            strip[:, x0:x1] = (val, val, val)
            strip = put_text_cv2(strip, f"{i+1}", (x0 + 6, height_px - 6), font_size=14, color=(0,0,0))
        return strip

    def _build_shadow_strip(self, width_px: int, height_px: int = 80):
        """
        Create a combined gradient + dark/light blocks for shadow/highlight testing.
        """
        import numpy as np
        gradient = np.tile(np.linspace(255, 0, width_px, dtype=np.uint8), (height_px, 1))
        gradient = np.stack([gradient]*3, axis=2)
        dark = np.ones_like(gradient) * 20
        light = np.ones_like(gradient) * 235
        # stack vertically gradient, dark left, light right
        left = dark[:, :width_px//3]
        right = light[:, -width_px//3:]
        mid = gradient[:, width_px//3: 2*width_px//3]
        combined = np.hstack([left, mid, right])
        return combined
