import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import json
from pathlib import Path
from typing import List, Dict

class DatasetManagerGUI:
    def __init__(self, master, image_paths: List[Path], config_file_path: Path):
        self.master = master
        self.image_paths = image_paths
        self.config_file_path = config_file_path
        self.current_image_index = 0
        self.image_points = {}

        self.master.title("Dataset Manager")

        # Load existing config
        self.load_config()

        # --- UI Elements ---
        # Frame for image canvas
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Frame for image info and navigation
        self.nav_frame = tk.Frame(master)
        self.nav_frame.pack(pady=5)

        self.prev_button = tk.Button(self.nav_frame, text="<< Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.image_label = tk.Label(self.nav_frame, text="")
        self.image_label.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.nav_frame, text="Next >>", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Frame for actions
        self.action_frame = tk.Frame(master)
        self.action_frame.pack(pady=10)

        self.clear_button = tk.Button(self.action_frame, text="Clear Points for this Image", command=self.clear_points)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(self.action_frame, text="Save and Close", command=self.save_and_close)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Load the first image
        self.load_image()

    def load_config(self):
        if self.config_file_path.exists():
            with open(self.config_file_path, 'r') as f:
                config_data = json.load(f)
                for img_config in config_data.get("image_configs", []):
                    self.image_points[img_config["filename"]] = img_config.get("points", [])

    def load_image(self):
        image_path = self.image_paths[self.current_image_index]
        self.image_label.config(text=f"Image {self.current_image_index + 1} / {len(self.image_paths)}: {image_path.name}")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            messagebox.showerror("Error", f"Could not load image: {image_path.name}")
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))

        self.canvas.config(width=image_rgb.shape[1], height=image_rgb.shape[0])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.draw_points()

    def on_canvas_click(self, event):
        image_name = self.image_paths[self.current_image_index].name
        if image_name not in self.image_points:
            self.image_points[image_name] = []
        
        self.image_points[image_name].append({"x": event.x, "y": event.y, "radius": 7})
        self.draw_points()

    def draw_points(self):
        self.canvas.delete("point")
        image_name = self.image_paths[self.current_image_index].name
        points = self.image_points.get(image_name, [])
        for point in points:
            x, y, r = point['x'], point['y'], point.get('radius', 7)
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="red", tags="point")

    def clear_points(self):
        image_name = self.image_paths[self.current_image_index].name
        if image_name in self.image_points:
            self.image_points[image_name] = []
            self.draw_points()

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def save_and_close(self):
        image_configs = []
        for image_path in self.image_paths:
            image_name = image_path.name
            points = self.image_points.get(image_name)

            if points is not None:
                # If there are points, the method is 'points'
                image_configs.append({
                    "filename": image_name,
                    "method": "points",
                    "points": points
                })
            else:
                # If no points were ever added for this image, keep its original config or default to full_average
                # For simplicity, we'll just not add it to the config if no points are selected, 
                # allowing the main app to default to full_average.
                pass

        # Preserve existing configs for images not in the current session
        if self.config_file_path.exists():
            with open(self.config_file_path, 'r') as f:
                existing_config_data = json.load(f)
                existing_configs = existing_config_data.get("image_configs", [])
                current_image_names = {p.name for p in self.image_paths}
                for cfg in existing_configs:
                    if cfg["filename"] not in current_image_names:
                        image_configs.append(cfg)

        config_data = {"image_configs": image_configs}
        with open(self.config_file_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        messagebox.showinfo("Success", f"Configuration saved to {self.config_file_path}")
        self.master.destroy()