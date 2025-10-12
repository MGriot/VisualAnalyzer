import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to sys.path to allow for module imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


# Import the core logic
try:
    from src.color_correction.patch_detector import ColorCheckerAligner, PatchInfo
except ImportError as e:
    print("Error: Failed to import core logic from 'src/color_correction/patch_detector.py'.")
    print("Please ensure the script is run from the project root or the path is configured correctly.")
    print(f"Original Error: {e}")
    sys.exit(1)


class ColorCheckerGUI:
    """GUI for color checker alignment and patch extraction"""

    def __init__(self, root):
        self.root = root
        self.root.title("Color Checker Alignment Tool")
        self.root.geometry("1400x900")

        # State variables
        self.image = None
        self.display_image = None
        self.photo_image = None
        self.aligner = None
        self.patches = None

        self.points = []
        self.max_points = 4
        self.scale_factor = 1.0

        # UI setup
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - Image display
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.setup_controls(left_panel)
        self.setup_canvas(right_panel)

    def setup_controls(self, parent):
        """Setup control panel"""
        # Title
        title = ttk.Label(parent, text="Color Checker Tool", font=("Arial", 16, "bold"))
        title.pack(pady=(0, 20))

        # Load Image button
        load_btn = ttk.Button(parent, text="Load Image", command=self.load_image)
        load_btn.pack(fill=tk.X, pady=5)

        # Points selection frame
        points_frame = ttk.LabelFrame(parent, text="Point Selection", padding=10)
        points_frame.pack(fill=tk.X, pady=10)

        ttk.Label(points_frame, text="Max Points:").pack(anchor=tk.W)

        self.max_points_var = tk.IntVar(value=4)
        points_spin = ttk.Spinbox(
            points_frame,
            from_=3,
            to=8,
            textvariable=self.max_points_var,
            command=self.update_max_points,
            width=10,
        )
        points_spin.pack(anchor=tk.W, pady=5)

        self.points_label = ttk.Label(points_frame, text="Points: 0/4")
        self.points_label.pack(anchor=tk.W, pady=5)

        clear_points_btn = ttk.Button(
            points_frame, text="Clear Points", command=self.clear_points
        )
        clear_points_btn.pack(fill=tk.X, pady=5)

        # Grid configuration
        grid_frame = ttk.LabelFrame(parent, text="Grid Configuration", padding=10)
        grid_frame.pack(fill=tk.X, pady=10)

        self.auto_detect_var = tk.BooleanVar(value=True)
        auto_check = ttk.Checkbutton(
            grid_frame,
            text="Auto-detect grid",
            variable=self.auto_detect_var,
            command=self.toggle_grid_inputs,
        )
        auto_check.pack(anchor=tk.W)

        # Manual grid inputs
        manual_frame = ttk.Frame(grid_frame)
        manual_frame.pack(fill=tk.X, pady=5)

        ttk.Label(manual_frame, text="Rows:").grid(row=0, column=0, sticky=tk.W)
        self.rows_var = tk.IntVar(value=4)
        self.rows_spin = ttk.Spinbox(
            manual_frame, from_=2, to=12, textvariable=self.rows_var, width=8
        )
        self.rows_spin.grid(row=0, column=1, padx=5)

        ttk.Label(manual_frame, text="Cols:").grid(row=1, column=0, sticky=tk.W)
        self.cols_var = tk.IntVar(value=6)
        self.cols_spin = ttk.Spinbox(
            manual_frame, from_=2, to=12, textvariable=self.cols_var, width=8
        )
        self.cols_spin.grid(row=1, column=1, padx=5)

        self.toggle_grid_inputs()

        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)

        self.auto_align_btn = ttk.Button(
            action_frame,
            text="Auto-Align with ArUco",
            command=self.auto_align,
            state=tk.DISABLED,
        )
        self.auto_align_btn.pack(fill=tk.X, pady=5)

        self.align_btn = ttk.Button(
            action_frame,
            text="Align Rectangle (Manual)",
            command=self.align_rectangle,
            state=tk.DISABLED,
        )
        self.align_btn.pack(fill=tk.X, pady=5)

        self.detect_btn = ttk.Button(
            action_frame,
            text="Detect Patches",
            command=self.detect_patches,
            state=tk.DISABLED,
        )
        self.detect_btn.pack(fill=tk.X, pady=5)

        # Export options
        export_frame = ttk.LabelFrame(parent, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            export_frame, text="Save Aligned Image", command=self.save_aligned_image
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            export_frame,
            text="Export Patch Data (JSON)",
            command=self.export_patch_data,
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            export_frame, text="Export Patch Data (CSV)", command=self.export_patch_csv
        ).pack(fill=tk.X, pady=2)

        # Status
        self.status_var = tk.StringVar(value="Load an image to begin")
        status_label = ttk.Label(
            parent, textvariable=self.status_var, wraplength=280, foreground="blue"
        )
        status_label.pack(side=tk.BOTTOM, pady=10)

    def setup_canvas(self, parent):
        """Setup image canvas"""
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        # Load image
        self.image = cv2.imread(file_path)
        if self.image is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        # Reset state
        self.clear_points()
        self.aligner = None
        self.patches = None
        self.detect_btn.config(state=tk.DISABLED)
        self.auto_align_btn.config(state=tk.NORMAL)

        # Display image
        self.display_original_image()
        self.status_var.set(f"Image loaded: {Path(file_path).name}")

    def auto_align(self):
        """Attempt to align the image automatically using ArUco markers."""
        if self.image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        try:
            # 1. Get the reference checker
            from src.color_correction.patch_detector import get_or_generate_reference_checker
            reference_checker_img = get_or_generate_reference_checker()
            if reference_checker_img is None:
                messagebox.showerror("Error", "Failed to load or generate the reference ArUco color checker.")
                return

            # 2. Create aligner and attempt alignment
            from src.color_correction.patch_detector import ColorCheckerAligner
            self.aligner = ColorCheckerAligner(self.image)
            
            self.status_var.set("Attempting automatic alignment with ArUco markers...")
            self.root.update_idletasks() # Force GUI update

            aligned_image = self.aligner.align_with_aruco(reference_checker_img)

            if aligned_image is not None:
                # 3. Success: Display result
                self.display_aligned_image()
                self.detect_btn.config(state=tk.NORMAL)
                self.status_var.set("Auto-alignment successful. Click 'Detect Patches'.")
            else:
                # 4. Failure: Inform user to use manual method
                messagebox.showwarning("Alignment Failed", "Automatic ArUco alignment failed. Please select the 4 corners of the color checker manually.")
                self.status_var.set("Auto-alignment failed. Select 4 points manually.")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during auto-alignment: {str(e)}")
            self.status_var.set("An error occurred.")

    def display_original_image(self):
        """Display the original image on canvas"""
        if self.image is None:
            return

        # Calculate scale to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.image.shape[:2]

        if canvas_w > 1 and canvas_h > 1:
            scale_w = canvas_w / img_w
            scale_h = canvas_h / img_h
            self.scale_factor = min(scale_w, scale_h, 1.0)
        else:
            self.scale_factor = 1.0

        # Resize for display
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        self.display_image = cv2.resize(self.image, (new_w, new_h))
        self.update_canvas()

    def display_aligned_image(self):
        """Display the aligned image"""
        if self.aligner is None or self.aligner.aligned_image is None:
            return

        # Calculate scale
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.aligner.aligned_image.shape[:2]

        if canvas_w > 1 and canvas_h > 1:
            scale_w = canvas_w / img_w
            scale_h = canvas_h / img_h
            self.scale_factor = min(scale_w, scale_h, 1.0)
        else:
            self.scale_factor = 1.0

        # Resize for display
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        self.display_image = cv2.resize(self.aligner.aligned_image, (new_w, new_h))
        self.update_canvas()

    def update_canvas(self):
        """Update canvas with current display image"""
        if self.display_image is None:
            return

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)

        # Draw points if in point selection mode
        if self.aligner is None and len(self.points) > 0:
            for i, (x, y) in enumerate(self.points):
                # Scale points
                sx = int(x * self.scale_factor)
                sy = int(y * self.scale_factor)

                # Draw circle
                cv2.circle(img_rgb, (sx, sy), 8, (255, 0, 0), -1)
                cv2.circle(img_rgb, (sx, sy), 10, (255, 255, 255), 2)

                # Draw number
                cv2.putText(
                    img_rgb,
                    str(i + 1),
                    (sx + 15, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            # Draw lines between points
            if len(self.points) > 1:
                scaled_points = [
                    (int(x * self.scale_factor), int(y * self.scale_factor))
                    for x, y in self.points
                ]
                for i in range(len(scaled_points)):
                    pt1 = scaled_points[i]
                    pt2 = scaled_points[(i + 1) % len(scaled_points)]
                    cv2.line(img_rgb, pt1, pt2, (0, 255, 0), 2)

        # Convert to PhotoImage
        pil_img = Image.fromarray(img_rgb)
        self.photo_image = ImageTk.PhotoImage(pil_img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_canvas_click(self, event):
        """Handle canvas click for point selection"""
        if self.image is None or self.aligner is not None:
            return

        if len(self.points) >= self.max_points:
            messagebox.showwarning(
                "Warning", f"Maximum {self.max_points} points reached"
            )
            return

        # Convert canvas coordinates to image coordinates
        x = int(event.x / self.scale_factor)
        y = int(event.y / self.scale_factor)

        self.points.append((x, y))
        self.points_label.config(text=f"Points: {len(self.points)}/{self.max_points}")

        # Enable align button when we have 4 points
        if len(self.points) == 4:
            self.align_btn.config(state=tk.NORMAL)
            self.status_var.set("4 points selected. Click 'Align Rectangle'")

        self.update_canvas()

    def on_canvas_motion(self, event):
        """Show coordinates on mouse motion"""
        if self.image is None:
            return

        x = int(event.x / self.scale_factor)
        y = int(event.y / self.scale_factor)

    def clear_points(self):
        """Clear all selected points"""
        self.points = []
        self.points_label.config(text=f"Points: 0/{self.max_points}")
        self.align_btn.config(state=tk.DISABLED)
        self.update_canvas()

    def update_max_points(self):
        """Update maximum points setting"""
        self.max_points = self.max_points_var.get()
        self.clear_points()

    def toggle_grid_inputs(self):
        """Enable/disable grid input fields"""
        if self.auto_detect_var.get():
            self.rows_spin.config(state=tk.DISABLED)
            self.cols_spin.config(state=tk.DISABLED)
        else:
            self.rows_spin.config(state=tk.NORMAL)
            self.cols_spin.config(state=tk.NORMAL)

    def align_rectangle(self):
        """Align the selected rectangle"""
        if len(self.points) != 4:
            messagebox.showerror("Error", "Please select exactly 4 points")
            return

        try:
            # Create aligner
            from src.color_correction.patch_detector import ColorCheckerAligner

            self.aligner = ColorCheckerAligner(self.image)

            # Align rectangle
            self.aligner.align_rectangle(self.points)

            # Display aligned image
            self.display_aligned_image()

            # Enable detect button
            self.detect_btn.config(state=tk.NORMAL)
            self.status_var.set("Rectangle aligned. Click 'Detect Patches'")

        except Exception as e:
            messagebox.showerror("Error", f"Alignment failed: {str(e)}")

    def detect_patches(self):
        """Detect patches in aligned image"""
        if self.aligner is None or self.aligner.aligned_image is None:
            messagebox.showerror("Error", "Please align rectangle first")
            return

        try:
            # Get grid configuration
            if self.auto_detect_var.get():
                grid_size = None
            else:
                grid_size = (self.rows_var.get(), self.cols_var.get())

            # Detect patches
            self.patches = self.aligner.detect_patches(
                grid_size=grid_size, adaptive=self.auto_detect_var.get()
            )

            # Visualize patches
            vis_image = self.aligner.visualize_patches(self.patches)

            # Calculate scale for display
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_h, img_w = vis_image.shape[:2]

            if canvas_w > 1 and canvas_h > 1:
                scale_w = canvas_w / img_w
                scale_h = canvas_h / img_h
                self.scale_factor = min(scale_w, scale_h, 1.0)

            new_w = int(img_w * self.scale_factor)
            new_h = int(img_h * self.scale_factor)

            self.display_image = cv2.resize(vis_image, (new_w, new_h))

            # Update canvas
            img_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            self.photo_image = ImageTk.PhotoImage(pil_img)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

            self.status_var.set(f"Detected {len(self.patches)} patches")

        except Exception as e:
            messagebox.showerror("Error", f"Patch detection failed: {str(e)}")

    def save_aligned_image(self):
        """Save the aligned image"""
        if self.aligner is None or self.aligner.aligned_image is None:
            messagebox.showwarning("Warning", "No aligned image to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Aligned Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            cv2.imwrite(file_path, self.aligner.aligned_image)
            messagebox.showinfo("Success", "Image saved successfully")

    def export_patch_data(self):
        """Export patch data to JSON"""
        if self.patches is None:
            messagebox.showwarning("Warning", "No patches to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Patch Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            data = {
                "num_patches": len(self.patches),
                "patches": [
                    {
                        "index": p.index,
                        "center": p.center,
                        "color_rgb": p.color_rgb,
                        "color_lab": p.color_lab,
                        "bounding_box": p.bounding_box,
                    }
                    for p in self.patches
                ],
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            messagebox.showinfo("Success", "Patch data exported successfully")

    def export_patch_csv(self):
        """Export patch data to CSV"""
        if self.patches is None:
            messagebox.showwarning("Warning", "No patches to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Patch Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if file_path:
            with open(file_path, "w") as f:
                # Write header
                f.write(
                    "index,center_x,center_y,r,g,b,L,a,b,bbox_x,bbox_y,bbox_w,bbox_h\n"
                )

                # Write data
                for p in self.patches:
                    f.write(f"{p.index},{p.center[0]},{p.center[1]},")
                    f.write(f"{p.color_rgb[0]},{p.color_rgb[1]},{p.color_rgb[2]},")
                    f.write(
                        f"{p.color_lab[0]:.2f},{p.color_lab[1]:.2f},{p.color_lab[2]:.2f},"
                    )
                    f.write(f"{p.bounding_box[0]},{p.bounding_box[1]},")
                    f.write(f"{p.bounding_box[2]},{p.bounding_box[3]}\n")

            messagebox.showinfo("Success", "Patch data exported to CSV successfully")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ColorCheckerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
