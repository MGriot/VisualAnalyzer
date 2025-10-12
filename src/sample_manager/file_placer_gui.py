import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import shutil
import os
import cv2

from src.project_manager import ProjectManager
from src.color_correction.corrector import ColorCorrector

class ProjectFilePlacerGUI(tk.Toplevel):
    def __init__(self, master, project_name):
        super().__init__(master)
        self.title(f"File Placer for '{project_name}'")
        self.project_name = project_name
        self.project_manager = ProjectManager()
        self.project_path = self.project_manager.projects_root / self.project_name
        self.geometry("600x750")
        self.grab_set()

        self.file_widgets = {}
        self.validation_widgets = {}
        self.create_widgets()
        self.refresh_statuses()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        self.target_files = self._get_target_files_from_config()

        for i, (key, info) in enumerate(self.target_files.items()):
            row_frame = ttk.LabelFrame(main_frame, text=info["label"])
            row_frame.pack(fill="x", expand=True, pady=5, padx=5)

            path_label = ttk.Label(row_frame, text=f"Expected: {info['rel_path']}", wraplength=550, justify="left")
            path_label.pack(side="top", fill="x", padx=5, pady=(2, 5))

            status_frame = ttk.Frame(row_frame)
            status_frame.pack(side="top", fill="x", padx=5, pady=2)

            status_text_label = ttk.Label(status_frame, text="File Status:")
            status_text_label.pack(side="left")

            status_indicator = ttk.Label(status_frame, text="Checking...", foreground="orange")
            status_indicator.pack(side="left", padx=5)

            # Add a new label for validation status, specifically for the checker
            if key == 'ideal_checker':
                validation_status_frame = ttk.Frame(row_frame)
                validation_status_frame.pack(side="top", fill="x", padx=5, pady=2)
                validation_text_label = ttk.Label(validation_status_frame, text="Validation Status:")
                validation_text_label.pack(side="left")
                validation_indicator = ttk.Label(validation_status_frame, text="Pending...", foreground="gray")
                validation_indicator.pack(side="left", padx=5)
                self.validation_widgets[key] = validation_indicator

            select_button = ttk.Button(
                row_frame,
                text="Select & Copy File...",
                command=lambda dest=info["abs_path"], k=key: self._select_and_copy(dest, k)
            )
            select_button.pack(side="bottom", fill="x", padx=5, pady=5)

            self.file_widgets[key] = status_indicator

        refresh_button = ttk.Button(main_frame, text="Refresh Statuses", command=self.refresh_statuses)
        refresh_button.pack(side="bottom", pady=10)

    def _get_target_files_from_config(self):
        try:
            config = self.project_manager._get_project_config(self.project_name)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", f"Could not load project configuration: {e}")
            self.destroy()
            return {}

        targets = {}
        
        # Color Correction
        cc_config = config.color_correction
        if cc_config.reference_color_checker_path:
            targets["ideal_checker"] = {
                "label": "Ideal Reference Color Checker",
                "abs_path": self.project_path / cc_config.reference_color_checker_path,
                "rel_path": cc_config.reference_color_checker_path
            }
        if cc_config.project_specific_color_checker_path:
            targets["project_checker"] = {
                "label": "Project-Specific Color Checker",
                "abs_path": self.project_path / cc_config.project_specific_color_checker_path,
                "rel_path": cc_config.project_specific_color_checker_path
            }

        # Object Reference
        if config.object_reference_path:
            targets["object_ref"] = {
                "label": "Object Reference Image",
                "abs_path": self.project_path / config.object_reference_path,
                "rel_path": config.object_reference_path
            }

        # Drawing Layers
        for key, path_str in config.masking.drawing_layers.items():
            targets[f"drawing_{key}"] = {
                "label": f"Drawing Layer '{key}'",
                "abs_path": self.project_path / path_str,
                "rel_path": path_str
            }
            
        # Geometrical Alignment
        geo_config = config.geometrical_alignment
        if geo_config.reference_path:
            targets["aruco_ref"] = {
                "label": "ArUco Reference Image",
                "abs_path": self.project_path / geo_config.reference_path,
                "rel_path": geo_config.reference_path
            }

        return targets

    def refresh_statuses(self):
        for key, info in self.target_files.items():
            widget = self.file_widgets.get(key)
            if not widget:
                continue
            
            abs_path = info.get("abs_path")
            if abs_path and abs_path.exists() and abs_path.is_file():
                widget.config(text="Found", foreground="green")
                # If the file is the ideal checker, validate it on refresh
                if key == 'ideal_checker':
                    self._validate_color_checker(abs_path)
            else:
                widget.config(text="Missing", foreground="red")
                if key == 'ideal_checker' and key in self.validation_widgets:
                    self.validation_widgets[key].config(text="Pending file...", foreground="gray")

    def _validate_color_checker(self, image_path: Path):
        validation_label = self.validation_widgets.get("ideal_checker")
        if not validation_label:
            return

        try:
            validation_label.config(text="Validating...", foreground="orange")
            self.update_idletasks() # Force GUI to update

            image = cv2.imread(str(image_path))
            if image is None:
                validation_label.config(text="Validation failed: Could not read image file.", foreground="red")
                return

            corrector = ColorCorrector()
            result = corrector.detect_color_checker_patches(image, debug_mode=False)
            num_patches = len(result.get("patches", []))
            method = result.get("detection_method", "unknown")

            if num_patches >= 20:
                validation_label.config(text=f"Success: Found {num_patches} patches via '{method}'.", foreground="green")
            else:
                validation_label.config(text=f"Warning: Only {num_patches} patches found. Automatic correction may fail.", foreground="orange")

        except Exception as e:
            validation_label.config(text=f"Validation error: {e}", foreground="red")

    def _select_and_copy(self, destination_path: Path, key: str):
        source_path_str = filedialog.askopenfilename(
            title="Select Source File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not source_path_str:
            return

        source_path = Path(source_path_str)

        if source_path.resolve() == destination_path.resolve():
            messagebox.showwarning("No Action Needed", "The selected source file is already the destination file. No copy was performed.")
            # Still run validation even if no copy is performed
            if key == 'ideal_checker':
                self._validate_color_checker(destination_path)
            return

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, destination_path)
            messagebox.showinfo("Success", f"File copied successfully to:\n{destination_path}")
        except shutil.SameFileError:
            messagebox.showwarning("No Action Needed", "The selected source file is the same as the destination file. No copy was performed.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy file: {e}")

        self.refresh_statuses()
