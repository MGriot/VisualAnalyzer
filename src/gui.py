import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import argparse
import os
import json
from pathlib import Path
import cv2
from PIL import Image, ImageTk

from src.pipeline import run_analysis
from src.project_manager import ProjectManager
from src.create_project import create_project
from src.sample_manager.dataset_gui import DatasetManagerGUI

class VisualAnalyzerGUI(tk.Tk):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.title("Visual Analyzer")
        self.geometry("700x800")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.analysis_tab = ttk.Frame(self.notebook)
        self.create_project_tab = ttk.Frame(self.notebook)
        self.manage_dataset_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.analysis_tab, text='Run Analysis')
        self.notebook.add(self.create_project_tab, text='Create Project')
        self.notebook.add(self.manage_dataset_tab, text='Manage Dataset')

        self.project_manager = ProjectManager()
        self.setup_variables()
        self.populate_analysis_tab()
        self.populate_create_project_tab()
        self.populate_manage_dataset_tab()

    def setup_variables(self):
        self.available_projects = self.project_manager.list_projects()
        self.project_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.color_checker_path_var = tk.StringVar()
        self.report_type_var = tk.StringVar(value="reportlab")
        self.masking_order_var = tk.StringVar(value="1-2-3")
        self.blur_kernel_var = tk.StringVar(value="5 5")
        self.agg_kernel_size_var = tk.StringVar(value="7")
        self.agg_min_area_var = tk.StringVar(value="0.0005")
        self.agg_density_thresh_var = tk.StringVar(value="0.5")
        self.color_alignment_var = tk.BooleanVar(value=True)
        self.symmetry_var = tk.BooleanVar(value=True)
        self.aggregate_var = tk.BooleanVar(value=True)
        self.debug_var = tk.BooleanVar()
        self.alignment_var = tk.BooleanVar(value=True)
        self.object_alignment_var = tk.BooleanVar(value=True)
        self.apply_mask_var = tk.BooleanVar(value=True)
        self.mask_bg_is_white_var = tk.BooleanVar(value=False)
        self.blur_var = tk.BooleanVar(value=True)
        self.new_project_name_var = tk.StringVar()
        self.manage_project_var = tk.StringVar()

    def populate_analysis_tab(self):
        main_frame = self.analysis_tab
        project_frame = tk.LabelFrame(main_frame, text="Project")
        project_frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(project_frame, text="Project Name:").pack(side=tk.LEFT, padx=5)
        self.project_combobox = ttk.Combobox(project_frame, textvariable=self.project_var, values=self.available_projects)
        self.project_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        if self.available_projects:
            self.project_combobox.set(self.available_projects[0])

        file_frame = tk.LabelFrame(main_frame, text="Files")
        file_frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Button(file_frame, text="Select Image", command=self.select_image).pack(fill=tk.X, pady=2)
        tk.Label(file_frame, textvariable=self.image_path_var, wraplength=500, anchor=tk.W).pack(fill=tk.X, padx=5)
        tk.Button(file_frame, text="Select Color Checker", command=self.select_color_checker).pack(fill=tk.X, pady=2)
        tk.Label(file_frame, textvariable=self.color_checker_path_var, wraplength=500, anchor=tk.W).pack(fill=tk.X, padx=5)

        if self.debug_mode:
            options_frame = tk.LabelFrame(main_frame, text="Analysis Steps & Options")
            options_frame.pack(fill=tk.X, pady=5, padx=5)

            grid_frame = ttk.Frame(options_frame)
            grid_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
            grid_frame.columnconfigure(0, weight=1)
            grid_frame.columnconfigure(1, weight=1)

            # Column 1: Boolean flags & sub-options
            bool_frame = ttk.Frame(grid_frame)
            bool_frame.grid(row=0, column=0, sticky="new")
            
            tk.Checkbutton(bool_frame, text="Color Alignment", variable=self.color_alignment_var).pack(anchor=tk.W)
            tk.Checkbutton(bool_frame, text="Geometrical Alignment", variable=self.alignment_var).pack(anchor=tk.W)
            tk.Checkbutton(bool_frame, text="Object Alignment", variable=self.object_alignment_var).pack(anchor=tk.W)
            
            tk.Checkbutton(bool_frame, text="Apply Mask", variable=self.apply_mask_var).pack(anchor=tk.W)
            mask_sub_frame = ttk.Frame(bool_frame)
            mask_sub_frame.pack(anchor=tk.W, fill=tk.X, padx=(20, 0))
            tk.Checkbutton(mask_sub_frame, text="Mask BG is White", variable=self.mask_bg_is_white_var).pack(anchor=tk.W)

            tk.Checkbutton(bool_frame, text="Symmetry Analysis", variable=self.symmetry_var).pack(anchor=tk.W)

            # Column 2: Parameters
            param_frame = ttk.Frame(grid_frame)
            param_frame.grid(row=0, column=1, sticky="new")

            tk.Label(param_frame, text="Report Type:").pack(anchor=tk.W)
            ttk.Combobox(param_frame, textvariable=self.report_type_var, values=["html", "reportlab", "all"]).pack(anchor=tk.W, fill=tk.X)

            tk.Label(param_frame, text="Masking Order (e.g., 1-2-3):").pack(anchor=tk.W, pady=(5,0))
            tk.Entry(param_frame, textvariable=self.masking_order_var).pack(anchor=tk.W, fill=tk.X)

            # Blur Options
            blur_frame = ttk.Frame(param_frame)
            blur_frame.pack(fill=tk.X, pady=(5,0))
            tk.Checkbutton(blur_frame, text="Blur", variable=self.blur_var).pack(anchor=tk.W)
            blur_options_frame = ttk.Frame(blur_frame)
            blur_options_frame.pack(fill=tk.X, padx=(20,0))
            tk.Label(blur_options_frame, text="Blur Kernel (W H, e.g., 5 5):").pack(anchor=tk.W)
            tk.Entry(blur_options_frame, textvariable=self.blur_kernel_var).pack(anchor=tk.W, fill=tk.X)

            # Aggregation Options
            agg_frame = ttk.Frame(param_frame)
            agg_frame.pack(fill=tk.X, pady=(5,0))
            tk.Checkbutton(agg_frame, text="Aggregate", variable=self.aggregate_var).pack(anchor=tk.W)
            agg_options_frame = ttk.Frame(agg_frame)
            agg_options_frame.pack(fill=tk.X, padx=(20,0))
            tk.Label(agg_options_frame, text="Agg. Kernel Size:").pack(anchor=tk.W)
            tk.Entry(agg_options_frame, textvariable=self.agg_kernel_size_var).pack(anchor=tk.W, fill=tk.X)
            tk.Label(agg_options_frame, text="Agg. Min Area:").pack(anchor=tk.W)
            tk.Entry(agg_options_frame, textvariable=self.agg_min_area_var).pack(anchor=tk.W, fill=tk.X)
            tk.Label(agg_options_frame, text="Agg. Density Thresh:").pack(anchor=tk.W)
            tk.Entry(agg_options_frame, textvariable=self.agg_density_thresh_var).pack(anchor=tk.W, fill=tk.X)

        tk.Button(main_frame, text="Run Analysis", command=self.run_analysis_dialog).pack(fill=tk.X, pady=10, padx=5)

    def populate_create_project_tab(self):
        main_frame = self.create_project_tab
        create_frame = tk.LabelFrame(main_frame, text="Create a New Project")
        create_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(create_frame, text="New Project Name:").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Entry(create_frame, textvariable=self.new_project_name_var, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        tk.Button(create_frame, text="Create", command=self.handle_create_project).pack(side=tk.LEFT, padx=5, pady=5)

        self.create_project_output = scrolledtext.ScrolledText(main_frame, height=20)
        self.create_project_output.pack(pady=5, padx=10, fill="both", expand=True)
        self.create_project_output.config(state=tk.DISABLED)

    def populate_manage_dataset_tab(self):
        main_frame = self.manage_dataset_tab
        manage_frame = tk.LabelFrame(main_frame, text="Launch Dataset Manager")
        manage_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(manage_frame, text="Select Project:").pack(side=tk.LEFT, padx=5, pady=5)
        self.manage_project_combobox = ttk.Combobox(manage_frame, textvariable=self.manage_project_var, values=self.available_projects)
        self.manage_project_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        if self.available_projects:
            self.manage_project_combobox.set(self.available_projects[0])
        
        tk.Button(manage_frame, text="Launch Manager", command=self.launch_dataset_manager).pack(side=tk.LEFT, padx=5, pady=5)

    def handle_create_project(self):
        project_name = self.new_project_name_var.get()
        if not project_name:
            messagebox.showerror("Error", "Please enter a project name.")
            return
        
        messages = create_project(project_name)
        
        self.create_project_output.config(state=tk.NORMAL)
        self.create_project_output.delete('1.0', tk.END)
        self.create_project_output.insert(tk.END, "\n".join(messages))
        self.create_project_output.config(state=tk.DISABLED)

        self.available_projects = self.project_manager.list_projects()
        self.project_combobox['values'] = self.available_projects
        self.manage_project_combobox['values'] = self.available_projects

    def launch_dataset_manager(self):
        project_name = self.manage_project_var.get()
        if not project_name:
            messagebox.showerror("Error", "Please select a project.")
            return
        
        try:
            project_files = self.project_manager.get_project_file_paths(project_name, debug_mode=self.debug_mode)
            dataset_image_configs = project_files.get("training_image_configs", [])
            dataset_image_paths = [cfg['path'] for cfg in dataset_image_configs]

            if not dataset_image_paths:
                messagebox.showinfo("Info", rf"No training images found in project '{project_name}'. Please add images to the 'training' folder and ensure the 'training_path' in project_config.json is correct.")
                return

            config_file_path = self.project_manager.projects_root / project_name / "dataset_item_processing_config.json"

            dataset_manager_window = tk.Toplevel(self)
            DatasetManagerGUI(dataset_manager_window, dataset_image_paths, str(config_file_path))
            self.wait_window(dataset_manager_window)
            
            messagebox.showinfo("Dataset Manager", "Dataset management complete. Remember to re-run analysis to use updated color space.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while trying to open the dataset manager: {e}")

    def select_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_path_var.set(path)

    def select_color_checker(self):
        path = filedialog.askopenfilename()
        if path:
            self.color_checker_path_var.set(path)

    def run_analysis_dialog(self):
        if self.debug_mode:
            debug_choice = messagebox.askyesno("Run Mode", "Do you want to run the pipeline in Debug Mode?")
            self.debug_var.set(debug_choice)
        else:
            self.debug_var.set(False)
        self.run_analysis()

    def run_analysis(self):
        args = argparse.Namespace()
        args.project = self.project_var.get()
        if not args.project:
            messagebox.showerror("Error", "Please select a project.")
            return
        args.image = self.image_path_var.get()
        if not args.image:
            messagebox.showerror("Error", "Please select an image.")
            return
        args.video = None
        args.camera = False
        args.drawing = None
        args.color_alignment = self.color_alignment_var.get()
        args.sample_color_checker = self.color_checker_path_var.get() if args.color_alignment else None
        if args.color_alignment and not args.sample_color_checker:
            messagebox.showerror("Error", "Please select a color checker image for Color Alignment.")
            return
        args.debug = self.debug_var.get()
        args.aggregate = self.aggregate_var.get()
        args.blur = self.blur_var.get()
        args.alignment = self.alignment_var.get()
        args.object_alignment = self.object_alignment_var.get()
        args.apply_mask = self.apply_mask_var.get()
        args.mask_bg_is_white = self.mask_bg_is_white_var.get()
        args.symmetry = self.symmetry_var.get()
        args.report_type = self.report_type_var.get()
        args.masking_order = self.masking_order_var.get()

        # Add missing arguments from pipeline refactoring
        args.load_state_from = None
        args.save_state_to = None
        args.skip_color_analysis = False
        args.skip_report_generation = False

        # Add missing arguments from pipeline refactoring
        args.load_state_from = None
        args.save_state_to = None
        args.skip_color_analysis = False
        args.skip_report_generation = False
        try:
            args.agg_kernel_size = int(self.agg_kernel_size_var.get()) if self.agg_kernel_size_var.get() else 7
            args.agg_min_area = float(self.agg_min_area_var.get()) if self.agg_min_area_var.get() else 0.0005
            args.agg_density_thresh = float(self.agg_density_thresh_var.get()) if self.agg_density_thresh_var.get() else 0.5
            if self.blur_kernel_var.get():
                w, h = map(int, self.blur_kernel_var.get().split())
                if w % 2 == 0 or h % 2 == 0:
                    messagebox.showerror("Error", "Blur kernel dimensions must be odd.")
                    return
                args.blur_kernel = [w, h]
            else:
                args.blur_kernel = None
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid numeric value for an option: {e}")
            return
        try:
            messagebox.showinfo("Running Analysis", "Analysis is starting. This may take a moment.")
            run_analysis(args)
            messagebox.showinfo("Success", "Analysis completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis:{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Analyzer GUI.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for GUI layout.")
    gui_args = parser.parse_args()
    app = VisualAnalyzerGUI(debug_mode=gui_args.debug)
    app.mainloop()