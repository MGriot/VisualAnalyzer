import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import argparse
import os

from src.pipeline import run_analysis
from src.color_analysis.project_manager import ProjectManager

class VisualAnalyzerGUI(tk.Tk):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.title("Visual Analyzer")
        self.geometry("600x750" if self.debug_mode else "400x300")

        # --- Variables ---
        self.project_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.color_checker_path_var = tk.StringVar()
        self.report_type_var = tk.StringVar(value="reportlab")
        
        # Step-specific options
        self.masking_order_var = tk.StringVar(value="1-2-3")
        self.blur_kernel_var = tk.StringVar()
        self.agg_kernel_size_var = tk.StringVar(value="7")
        self.agg_min_area_var = tk.StringVar(value="0.0005")
        self.agg_density_thresh_var = tk.StringVar(value="0.5")

        # Checkbox variables
        self.color_alignment_var = tk.BooleanVar(value=True)
        self.symmetry_var = tk.BooleanVar(value=True)
        self.aggregate_var = tk.BooleanVar(value=True)
        self.debug_var = tk.BooleanVar()
        self.alignment_var = tk.BooleanVar(value=True)
        self.object_alignment_var = tk.BooleanVar(value=True)
        self.apply_mask_var = tk.BooleanVar(value=True)
        self.mask_bg_is_white_var = tk.BooleanVar(value=False)
        self.blur_var = tk.BooleanVar(value=True)

        self.project_manager = ProjectManager()
        self.available_projects = self.project_manager.list_projects()

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Project Selection
        project_frame = tk.LabelFrame(main_frame, text="Project")
        project_frame.pack(fill=tk.X, pady=5)
        tk.Label(project_frame, text="Project Name:").pack(side=tk.LEFT, padx=5)
        self.project_combobox = ttk.Combobox(project_frame, textvariable=self.project_var, values=self.available_projects)
        self.project_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        if self.available_projects:
            self.project_combobox.set(self.available_projects[0])

        # File Selection
        file_frame = tk.LabelFrame(main_frame, text="Files")
        file_frame.pack(fill=tk.X, pady=5)
        tk.Button(file_frame, text="Select Image", command=self.select_image).pack(fill=tk.X, pady=2)
        tk.Label(file_frame, textvariable=self.image_path_var, wraplength=500, anchor=tk.W).pack(fill=tk.X, padx=5)
        tk.Button(file_frame, text="Select Color Checker", command=self.select_color_checker).pack(fill=tk.X, pady=2)
        tk.Label(file_frame, textvariable=self.color_checker_path_var, wraplength=500, anchor=tk.W).pack(fill=tk.X, padx=5)

        if self.debug_mode:
            options_frame = tk.LabelFrame(main_frame, text="Analysis Steps & Options")
            options_frame.pack(fill=tk.X, pady=5)
            tk.Checkbutton(options_frame, text="Color Alignment", variable=self.color_alignment_var).grid(row=0, column=0, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Geometrical Alignment (ArUco)", variable=self.alignment_var).grid(row=1, column=0, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Object Alignment", variable=self.object_alignment_var).grid(row=2, column=0, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Apply Mask", variable=self.apply_mask_var).grid(row=3, column=0, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Treat White as BG", variable=self.mask_bg_is_white_var).grid(row=3, column=1, sticky=tk.W)
            tk.Label(options_frame, text="Masking Order (e.g., 1-2-3):").grid(row=4, column=0, sticky=tk.W)
            tk.Entry(options_frame, textvariable=self.masking_order_var, width=15).grid(row=4, column=1, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Blur Image", variable=self.blur_var).grid(row=5, column=0, sticky=tk.W)
            tk.Label(options_frame, text="Blur Kernel (W H, odd):").grid(row=6, column=0, sticky=tk.W)
            tk.Entry(options_frame, textvariable=self.blur_kernel_var, width=15).grid(row=6, column=1, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Aggregate Matched Pixels", variable=self.aggregate_var).grid(row=7, column=0, sticky=tk.W)
            tk.Label(options_frame, text="Agg Kernel Size:").grid(row=8, column=0, sticky=tk.W)
            tk.Entry(options_frame, textvariable=self.agg_kernel_size_var, width=10).grid(row=8, column=1, sticky=tk.W)
            tk.Label(options_frame, text="Agg Min Area:").grid(row=9, column=0, sticky=tk.W)
            tk.Entry(options_frame, textvariable=self.agg_min_area_var, width=10).grid(row=9, column=1, sticky=tk.W)
            tk.Label(options_frame, text="Agg Density Thresh:").grid(row=10, column=0, sticky=tk.W)
            tk.Entry(options_frame, textvariable=self.agg_density_thresh_var, width=10).grid(row=10, column=1, sticky=tk.W)
            tk.Checkbutton(options_frame, text="Symmetry Analysis", variable=self.symmetry_var).grid(row=11, column=0, sticky=tk.W)
            
            report_frame = tk.LabelFrame(main_frame, text="Report")
            report_frame.pack(fill=tk.X, pady=5)
            tk.Label(report_frame, text="Report Type:").pack(side=tk.LEFT, padx=5)
            tk.OptionMenu(report_frame, self.report_type_var, "all", "html", "reportlab").pack(side=tk.LEFT, padx=5)
        else:
            pass

        tk.Button(main_frame, text="Run Analysis", command=self.run_analysis_dialog).pack(fill=tk.X, pady=10)

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
            messagebox.showerror("Error", f"An error occurred during analysis:\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Analyzer GUI.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for GUI layout.")
    gui_args = parser.parse_args()

    app = VisualAnalyzerGUI(debug_mode=gui_args.debug)
    app.mainloop()