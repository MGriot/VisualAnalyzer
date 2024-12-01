import tkinter as tk
from tkinter import filedialog
import pandas as pd
from PIL import Image, ImageTk  # Try importing ImageTk directly
import io
import webbrowser
import report
from datetime import date
import os
from VisualAnalyzer.NewColorFinder import *


class ReportGeneratorGUI:

    def __init__(self, master):
        self.master = master
        master.title("HTML Report Generator")

        self.dataset_path = ""
        self.image_path = ""

        self.dataset_label = tk.Label(master, text="Dataset:")
        self.dataset_label.grid(row=0, column=0)

        self.dataset_button = tk.Button(
            master, text="Browse", command=self.browse_dataset
        )
        self.dataset_button.grid(row=0, column=1)

        self.image_label = tk.Label(master, text="Image:")
        self.image_label.grid(row=1, column=0)

        self.image_button = tk.Button(master, text="Browse", command=self.browse_image)
        self.image_button.grid(row=1, column=1)

        self.analyze_button = tk.Button(
            master, text="Analyze Image", command=self.analyze_image
        )
        self.analyze_button.grid(row=2, column=0, columnspan=2)

        self.generate_button = tk.Button(
            master, text="Generate Report", command=self.generate_report
        )
        self.generate_button.grid(row=3, column=0, columnspan=2)

        self.result_label = tk.Label(master, text="")
        self.result_label.grid(row=4, column=0, columnspan=2)

    def browse_dataset(self):
        self.dataset_path = filedialog.askdirectory()
        self.dataset_label.config(text=f"Dataset: {self.dataset_path}")

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        self.image_label.config(text=f"Image: {self.image_path}")

        try:
            image = Image.open(self.image_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            image_label = tk.Label(self.master, image=photo)
            image_label.image = photo
            image_label.grid(row=5, column=0, columnspan=2)
        except Exception as e:
            self.result_label.config(text=f"Error loading image: {e}")

    def analyze_image(self):
        if not self.image_path:
            self.result_label.config(text="Please select an image first.")
            return

        try:
            color_finder = ColorFinder()
            lower_limit_main, upper_limit_main, center_main = (
                get_color_limits_from_dataset(
                    self.dataset_path, show_plot=True, outlier_removal_method="grubbs"
                )
            )
            color_finder.lower_limit = lower_limit_main
            color_finder.upper_limit = upper_limit_main
            color_finder.center = center_main
            results_main = color_finder.find_color_and_percentage(
                self.image_path,
                exclude_transparent=True,
                adaptive_thresholding=False,
                apply_morphology=True,
                apply_blur=True,
            )
            if results_main:
                (
                    processed_image_main,
                    selected_colors_main,
                    percentage_main,
                    matched_pixels_main,
                    total_pixels_main,
                    average_non_selected_color_main,
                    non_selected_pixel_count_main,
                ) = results_main

                original_image_main = cv2.imread(self.image_path)
                hsv_image_main = cv2.cvtColor(processed_image_main, cv2.COLOR_BGR2HSV)
                mask_main = cv2.inRange(
                    hsv_image_main, color_finder.lower_limit, color_finder.upper_limit
                )

                # ... (print statements)

                color_finder.plot_and_save_results(
                    original_image_main,
                    processed_image_main,
                    mask_main,
                    percentage_main,
                    matched_pixels_main,
                    total_pixels_main,
                    output_dir="output",
                    save=False,
                    show=True,
                )
        except Exception as e:
            self.result_label.config(text=f"Error analyzing image: {e}")

    def generate_report(self):
        if not self.dataset_path or not self.image_path:
            self.result_label.config(text="Please select both a dataset and an image.")
            return

        try:
            for filename in os.listdir(self.dataset_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(self.dataset_path, filename)
                    df = pd.read_csv(filepath)
                    today = date.today()
                    report.generate_report(
                        df,
                        self.image_path,
                        "mask_path",
                        "pie_chart",
                        "color_space_plot",
                        1,
                        10,
                        "logo",
                        today,
                        "author",
                        "department",
                        "report_title",
                        "report.html",
                    )
                    self.result_label.config(text="Report generated successfully!")

                    # Open the report.html in the default web browser
                    webbrowser.open("report.html")

        except Exception as e:
            self.result_label.config(text=f"Error generating report: {e}")


root = tk.Tk()
gui = ReportGeneratorGUI(root)
root.mainloop()
