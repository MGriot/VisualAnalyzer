import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from typing import List, Tuple

class ManualAlignerGUI:
    """A simple GUI to capture four corner points from an image."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.points = []
        self.window = tk.Tk()
        self.window.title("Manual Color Checker Alignment - Select 4 Corners & Close Window")

        # Resize image for display if it's too large
        max_h, max_w = 800, 1200
        h, w, _ = self.image.shape
        self.scale_factor = min(1.0, max_w / w, max_h / h)
        display_w, display_h = int(w * self.scale_factor), int(h * self.scale_factor)
        display_image = cv2.resize(self.image, (display_w, display_h))
        
        img_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.photo_image = ImageTk.PhotoImage(img_pil)

        self.canvas = tk.Canvas(self.window, width=display_w, height=display_h)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _on_canvas_click(self, event):
        """Callback to record a point when the user clicks the canvas."""
        if len(self.points) < 4:
            # Store point in original image coordinates
            x = int(event.x / self.scale_factor)
            y = int(event.y / self.scale_factor)
            self.points.append((x, y))

            # Draw feedback on canvas
            cv2.circle(self.image, (x, y), 10, (0, 0, 255), -1) # Draw on original for reference
            
            # Update display
            display_w, display_h = self.photo_image.width(), self.photo_image.height()
            display_image = cv2.resize(self.image, (display_w, display_h))
            img_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.photo_image = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

            if len(self.points) == 4:
                print("[INFO] 4 points selected. You can now close the GUI window.")

    def run(self) -> List[Tuple[int, int]]:
        """Runs the Tkinter main loop and returns points upon closing."""
        self.window.mainloop()
        return self.points

def get_corners_from_user(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Public function to launch the GUI and get four corner points from the user.

    Args:
        image: The image on which to select points.

    Returns:
        A list of four (x, y) tuples, or an empty list if selection is incomplete.
    """
    print("[INFO] Opening manual alignment GUI. Please click the 4 corners of the color checker.")
    print("[INFO] Start with the top-left corner and proceed clockwise. Close the window when done.")
    gui = ManualAlignerGUI(image)
    points = gui.run()
    if len(points) == 4:
        return points
    return []
