import tkinter as tk
from tkinter import filedialog
import pandas as pd
from PIL import Image, ImageTk
import io
import webbrowser
from datetime import date
import os
from image_analysis import ColorFinder, get_color_limits_from_dataset
from report import ImageReportGenerator, ReportConfig

# Global variables for folder and file paths
TEMPLATES_DIR = "path/to/your/templates"
OUTPUT_DIR = "output/report"
SUGGESTED_DATASET_PATH = "path/to/your/dataset"
SUGGESTED_SAMPLE_PATH = "path/to/your/sample"

class ReportGeneratorGUI:

    def __init__(self, master):
        self.master = master
        master.title("HTML Report Generator")

        self.dataset_path = SUGGESTED_DATASET_PATH
        self.image_path = SUGGESTED_SAMPLE_PATH

        self.dataset_label = tk.Label(master, text=f"Dataset: {self.dataset_path}")
        self.dataset_label.grid(row=0, column=0)

        self.dataset_button = tk.Button(
            master, text="Browse", command=self.browse_dataset
        )
        self.dataset_button.grid(row=0, column=1)

        self.image_label = tk.Label(master, text=f"Image: {self.image_path}")
        self.image_label.grid(row=1, column=0)

        self.image_button = tk.Button(master, text="Browse", command=self.browse_image)
        self.image_button.grid(row=1, column=1)

        self.apply_pixelation_var = tk.BooleanVar()
        self.pixelation_size_var = tk.IntVar(value=10)

        self.pixelation_checkbutton = tk.Checkbutton(
            master, text="Apply Pixelation", variable=self.apply_pixelation_var
        )
        self.pixelation_checkbutton.grid(row=2, column=0, columnspan=2)

        self.pixelation_size_label = tk.Label(master, text="Pixelation Size:")
        self.pixelation_size_label.grid(row=3, column=0)

        self.pixelation_size_entry = tk.Entry(master, textvariable=self.pixelation_size_var)
        self.pixelation_size_entry.grid(row=3, column=1)

        self.analyze_button = tk.Button(
            master, text="Analyze Image", command=self.analyze_image
        )
        self.analyze_button.grid(row=4, column=0, columnspan=2)

        self.generate_button = tk.Button(
            master, text="Generate Report", command=self.generate_report
        )
        self.generate_button.grid(row=5, column=0, columnspan=2)

        self.result_label = tk.Label(master, text="")
        self.result_label.grid(row=6, column=0, columnspan=2)

        self.color_finder = ColorFinder()

        # Set initial paths
        self.dataset_label.config(text=f"Dataset: {self.dataset_path}")
        self.image_label.config(text=f"Image: {self.image_path}")

    def browse_dataset(self):
        self.dataset_path = filedialog.askdirectory(initialdir=SUGGESTED_DATASET_PATH)
        self.dataset_label.config(text=f"Dataset: {self.dataset_path}")

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir=SUGGESTED_SAMPLE_PATH,
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        self.image_label.config(text=f"Image: {self.image_path}")

        try:
            image = Image.open(self.image_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            image_label = tk.Label(self.master, image=photo)
            image_label.image = photo
            image_label.grid(row=7, column=0, columnspan=2)
        except Exception as e:
            self.result_label.config(text=f"Error loading image: {e}")

    def analyze_image(self):
        if not self.image_path:
            self.result_label.config(text="Please select an image first.")
            return

        lower_limit_main, upper_limit_main, center_main = get_color_limits_from_dataset(
            self.dataset_path, show_plot=True, outlier_removal_method="zscore"
        )
        self.color_finder.lower_limit = lower_limit_main
        self.color_finder.upper_limit = upper_limit_main
        self.color_finder.center = center_main
        results_main = self.color_finder.find_color_and_percentage(
            self.image_path,
            exclude_transparent=True,
            adaptive_thresholding=False,
            apply_morphology=False,
            apply_blur=True,
            blur_radius=2,
        )
        if results_main:
            self.color_finder.plot_and_save_results(
                output_dir=OUTPUT_DIR,
                save=False,
                show=True,
            )

    def generate_report(self):
        print("Report generation...")
        if not self.dataset_path or not self.image_path:
            self.result_label.config(text="Please select both a dataset and an image.")
            return

        try:
            # Ensure the templates directory path is correct
            config = ReportConfig(
                database_path=self.dataset_path,
                image_dir=os.path.dirname(self.image_path),
                output_dir=OUTPUT_DIR,
                templates_dir=TEMPLATES_DIR  # Use global variable for templates directory
            )
            report_generator = ImageReportGenerator(config)
            analysis_results = self.color_finder.find_color_and_percentage(
                self.image_path,
                exclude_transparent=True,
                adaptive_thresholding=False,
                apply_morphology=False,
                apply_blur=True,
                blur_radius=2,
            )
            report_path = report_generator.generate_report(analysis_results, self.image_path)
            self.result_label.config(text="Report generated successfully!")

            # Open the report.html in the default web browser
            webbrowser.open(report_path)

        except Exception as e:
            self.result_label.config(text=f"Error generating report: {e}")

root = tk.Tk()
gui = ReportGeneratorGUI(root)
root.mainloop()
