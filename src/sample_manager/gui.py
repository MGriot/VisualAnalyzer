from typing import List, Dict
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
from pathlib import Path

class PointSelectorGUI:
    def __init__(self, master, image_path: Path, config_file_path: Path, existing_points: List[Dict] = None):
        self.master = master
        self.image_path = image_path
        self.config_file_path = config_file_path
        self.points = existing_points if existing_points is not None else []
        self.radius = 7 # Default radius for points

        self.master.title(f"Select Points for {self.image_path.name}")

        self.image_bgr = cv2.imread(str(image_path))
        if self.image_bgr is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.image_rgb))

        self.canvas = tk.Canvas(master, width=self.image_rgb.shape[1], height=self.image_rgb.shape[0])
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_points()

        self.save_button = tk.Button(master, text="Save Points", command=self.save_points)
        self.save_button.pack()

        self.clear_button = tk.Button(master, text="Clear Points", command=self.clear_points)
        self.clear_button.pack()

    def on_click(self, event):
        x, y = event.x, event.y
        self.points.append({"x": x, "y": y, "radius": self.radius})
        self.draw_points()

    def draw_points(self):
        self.canvas.delete("points") # Clear existing points
        for point in self.points:
            x, y, r = point['x'], point['y'], point['radius']
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="red", tags="points")

    def save_points(self):
        # Load existing config or create new one
        config_data = {}
        if self.config_file_path.exists():
            with open(self.config_file_path, 'r') as f:
                config_data = json.load(f)
        
        # Find the image config or create a new one
        image_configs = config_data.get("image_configs", [])
        found = False
        for img_config in image_configs:
            if img_config.get("filename") == self.image_path.name:
                img_config["method"] = "points"
                img_config["points"] = self.points
                found = True
                break
        if not found:
            image_configs.append({
                "filename": self.image_path.name,
                "method": "points",
                "points": self.points
            })
        config_data["image_configs"] = image_configs

        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Points saved to {self.config_file_path}")
        self.master.destroy()

    def clear_points(self):
        self.points = []
        self.draw_points()

if __name__ == '__main__':
    # Example usage (for testing the GUI independently)
    root = tk.Tk()
    # Create a dummy image file for testing
    dummy_image_path = Path("dummy_image.png")
    dummy_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(dummy_image_path), dummy_image)

    # Create a dummy config file path
    dummy_config_path = Path("dummy_config.json")

    app = PointSelectorGUI(root, dummy_image_path, dummy_config_path)
    root.mainloop()

    # Clean up dummy files
    os.remove(dummy_image_path)
    if dummy_config_path.exists():
        os.remove(dummy_config_path)
