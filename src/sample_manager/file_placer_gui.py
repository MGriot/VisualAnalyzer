import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import shutil
import os
import cv2
from PIL import Image, ImageTk

from src.project_manager import ProjectManager
from src.color_correction.color_correction_pipeline import ColorCorrectionPipeline


class ProjectFilePlacerGUI(tk.Toplevel):
    def __init__(self, master, project_name):
        super().__init__(master)
        self.title(f"File Placer for '{project_name}'")
        self.project_name = project_name
        self.project_manager = ProjectManager()
        self.project_path = self.project_manager.projects_root / self.project_name
        self.training_images_path = self.project_path / "dataset" / "training_images"
        self.geometry("700x800")
        self.grab_set()

        self.file_widgets = {}
        self.validation_widgets = {}
        self.thumbnail_references = []  # To prevent garbage collection

        self.create_widgets()
        self.refresh_all_statuses()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill="both", expand=True)

        self.target_files = self._get_target_files_from_config()

        # --- Frame for Config Files ---
        config_files_frame = ttk.LabelFrame(self.main_frame, text="Project Configuration Files")
        config_files_frame.pack(fill="x", expand=True, pady=5, padx=5)

        for i, (key, info) in enumerate(self.target_files.items()):
            row_frame = ttk.Frame(config_files_frame, padding=5)
            row_frame.pack(fill="x", expand=True, pady=2)

            path_label = ttk.Label(row_frame, text=f"{info['label']}: {info['rel_path']}", wraplength=550, justify="left")
            path_label.pack(side="top", fill="x")

            status_frame = ttk.Frame(row_frame)
            status_frame.pack(side="top", fill="x", pady=2)

            status_indicator = ttk.Label(status_frame, text="Checking...", foreground="orange", font=("TkDefaultFont", 8))
            status_indicator.pack(side="left")

            if key == 'ideal_checker':
                validation_indicator = ttk.Label(status_frame, text="", foreground="gray", font=("TkDefaultFont", 8))
                validation_indicator.pack(side="left", padx=10)
                self.validation_widgets[key] = validation_indicator

            select_button = ttk.Button(
                row_frame,
                text="Select & Copy File...",
                command=lambda dest=info["abs_path"], k=key: self._select_and_copy(dest, k)
            )
            select_button.pack(side="right", padx=5)

            self.file_widgets[key] = status_indicator

        # --- Frame for Training Images ---
        self._create_training_images_manager()

        # --- Bottom Buttons ---
        bottom_frame = ttk.Frame(self.main_frame)
        bottom_frame.pack(side="bottom", fill="x", pady=10)
        refresh_button = ttk.Button(bottom_frame, text="Refresh All Statuses", command=self.refresh_all_statuses)
        refresh_button.pack()

    def _create_training_images_manager(self):
        training_frame = ttk.LabelFrame(self.main_frame, text="Training Images")
        training_frame.pack(fill="both", expand=True, pady=10, padx=5)

        # Add button
        add_button = ttk.Button(training_frame, text="Add Training Image(s)...", command=self._add_training_images)
        add_button.pack(pady=5, fill="x", padx=5)

        # Scrollable area for image previews
        canvas_container = ttk.Frame(training_frame)
        canvas_container.pack(fill="both", expand=True)
        canvas = tk.Canvas(canvas_container)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _refresh_training_images_list(self):
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_references.clear()

        self.training_images_path.mkdir(parents=True, exist_ok=True)
        image_files = [f for f in self.training_images_path.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')] # noqa

        if not image_files:
            ttk.Label(self.scrollable_frame, text="No training images found.").pack(pady=10)
            return

        for image_path in sorted(image_files):
            item_frame = ttk.Frame(self.scrollable_frame, padding=5, relief="solid", borderwidth=1)
            item_frame.pack(fill="x", pady=5, padx=5)

            try:
                # Create thumbnail
                img = Image.open(image_path)
                img.thumbnail((100, 100))
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_references.append(photo) # Keep reference

                preview_label = ttk.Label(item_frame, image=photo)
                preview_label.pack(side="left", padx=5)

                info_frame = ttk.Frame(item_frame)
                info_frame.pack(side="left", expand=True, fill="x", padx=5)

                filename_label = ttk.Label(info_frame, text=image_path.name, wraplength=400)
                filename_label.pack(anchor="w")

                delete_button = ttk.Button(item_frame, text="Delete", command=lambda p=image_path: self._delete_training_image(p))
                delete_button.pack(side="right", padx=5)
            except Exception as e:
                error_label = ttk.Label(item_frame, text=f"Error loading {image_path.name}: {e}", foreground="red")
                error_label.pack(padx=5, pady=5)

    def _add_training_images(self):
        source_paths = filedialog.askopenfilenames(
            title="Select Training Images to Add",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not source_paths:
            return

        copied_count = 0
        for source_path_str in source_paths:
            source_path = Path(source_path_str)
            destination_path = self.training_images_path / source_path.name
            try:
                shutil.copy(source_path, destination_path)
                copied_count += 1
            except Exception as e:
                messagebox.showwarning("Copy Error", f"Could not copy {source_path.name}: {e}")
        
        if copied_count > 0:
            messagebox.showinfo("Success", f"Successfully added {copied_count} training image(s).")
            self._refresh_training_images_list()

    def _delete_training_image(self, path_to_delete: Path):
        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to permanently delete {path_to_delete.name}?"):
            return
        
        try:
            os.remove(path_to_delete)
            self._refresh_training_images_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete file: {e}")

    def _get_target_files_from_config(self):
        try:
            config = self.project_manager._get_project_config(self.project_name)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", f"Could not load project configuration: {e}")
            self.destroy()
            return {}

        targets = {}
        cc_config = config.color_correction
        if cc_config.reference_color_checker_path:
            targets["ideal_checker"] = {"label": "Ideal Reference Color Checker", "abs_path": self.project_path / cc_config.reference_color_checker_path, "rel_path": cc_config.reference_color_checker_path}
        if cc_config.project_specific_color_checker_path:
            targets["project_checker"] = {"label": "Project-Specific Color Checker", "abs_path": self.project_path / cc_config.project_specific_color_checker_path, "rel_path": cc_config.project_specific_color_checker_path}
        if config.object_reference_path:
            targets["object_ref"] = {"label": "Object Reference Image", "abs_path": self.project_path / config.object_reference_path, "rel_path": config.object_reference_path}
        for key, path_str in config.masking.drawing_layers.items():
            targets[f"drawing_{key}"] = {"label": f"Drawing Layer '{key}'", "abs_path": self.project_path / path_str, "rel_path": path_str}
        geo_config = config.geometrical_alignment
        if geo_config.reference_path:
            targets["aruco_ref"] = {"label": "ArUco Reference Image", "abs_path": self.project_path / geo_config.reference_path, "rel_path": geo_config.reference_path}
        return targets

    def refresh_all_statuses(self):
        # Refresh config file statuses
        for key, info in self.target_files.items():
            widget = self.file_widgets.get(key)
            if not widget:
                continue
            abs_path = info.get("abs_path")
            if abs_path and abs_path.exists() and abs_path.is_file():
                widget.config(text="Found", foreground="green")
                if key == 'ideal_checker':
                    self._validate_color_checker(abs_path)
            else:
                widget.config(text="Missing", foreground="red")
                if key == 'ideal_checker' and key in self.validation_widgets:
                    self.validation_widgets[key].config(text="Pending file...", foreground="gray")
        
        # Refresh training images list
        self._refresh_training_images_list()

    def _validate_color_checker(self, image_path: Path):
        validation_label = self.validation_widgets.get("ideal_checker")
        if not validation_label:
            return
        try:
            validation_label.config(text="Validating...", foreground="orange")
            self.update_idletasks()
            image = cv2.imread(str(image_path))
            if image is None:
                validation_label.config(text="Validation failed: Could not read image file.", foreground="red")
                return
            pipeline = ColorCorrectionPipeline(reference_color_checker_path=str(image_path), debug_mode=False)
            result = pipeline.run_patch_detection_on_image(image)
            num_patches = len(result.get("patches", []))
            method = result.get("detection_method", "unknown")
            if num_patches >= 18:
                validation_label.config(text=f"Success: Found {num_patches} patches via '{method}'.", foreground="green")
            else:
                validation_label.config(text=f"Warning: Only {num_patches} patches found. Correction may fail.", foreground="orange")
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
            messagebox.showwarning("No Action Needed", "The selected source file is already the destination file.")
            if key == 'ideal_checker':
                self._validate_color_checker(destination_path)
            return
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, destination_path)
            messagebox.showinfo("Success", f"File copied successfully to:\n{destination_path}")
        except shutil.SameFileError:
            messagebox.showwarning("No Action Needed", "The selected source file is the same as the destination file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy file: {e}")
        self.refresh_all_statuses()