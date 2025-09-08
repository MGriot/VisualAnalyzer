print("---" + " Starting gui_main.py ---")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading

# Assuming main_analysis_logic is a function in main.py that takes an args-like object
# We will refactor main.py to expose this function
from src.pipeline import run_analysis # This will be created in main.py

print("---" + " Before VisualAnalyzerGUI class definition ---")
class VisualAnalyzerGUI:
    def __init__(self, master):
        print("---" + " Inside VisualAnalyzerGUI __init__ ---")
        self.master = master
        master.title("Visual Analyzer")

        self.project_manager = None # Will be initialized later
        self.available_projects = []

        self.create_widgets()
        self.load_projects()

    def create_widgets(self):
        # Project Selection
        project_frame = ttk.LabelFrame(self.master, text="Project Selection")
        project_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(project_frame, text="Select Project:").pack(side="left", padx=5, pady=5)
        self.project_var = tk.StringVar(self.master)
        self.project_dropdown = ttk.OptionMenu(project_frame, self.project_var, "", *self.available_projects)
        self.project_dropdown.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        # Input Selection
        input_frame = ttk.LabelFrame(self.master, text="Input Selection")
        input_frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(input_frame, text="Image/Video Path:").pack(side="left", padx=5, pady=5)
        self.input_path_var = tk.StringVar(self.master)
        self.input_path_entry = ttk.Entry(input_frame, textvariable=self.input_path_var)
        self.input_path_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_file).pack(side="left", padx=5, pady=5)
        ttk.Button(input_frame, text="Camera", command=self.use_camera).pack(side="left", padx=5, pady=5)

        # Options
        options_frame = ttk.LabelFrame(self.master, text="Options")
        options_frame.pack(padx=10, pady=5, fill="x")

        self.debug_var = tk.BooleanVar(self.master)
        ttk.Checkbutton(options_frame, text="Debug Mode", variable=self.debug_var).pack(anchor="w", padx=5, pady=2)

        self.aggregate_var = tk.BooleanVar(self.master)
        ttk.Checkbutton(options_frame, text="Aggregate Pixels", variable=self.aggregate_var).pack(anchor="w", padx=5, pady=2)

        self.blur_var = tk.BooleanVar(self.master)
        ttk.Checkbutton(options_frame, text="Apply Blur", variable=self.blur_var).pack(anchor="w", padx=5, pady=2)

        self.alignment_var = tk.BooleanVar(self.master)
        ttk.Checkbutton(options_frame, text="Enable Alignment", variable=self.alignment_var).pack(anchor="w", padx=5, pady=2)

        self.color_alignment_var = tk.BooleanVar(self.master)
        ttk.Checkbutton(options_frame, text="Enable Color Alignment", variable=self.color_alignment_var).pack(anchor="w", padx=5, pady=2)

        # Run Button
        ttk.Button(self.master, text="Run Analysis", command=self.run_analysis).pack(pady=10)

        # Output Console
        self.output_console = tk.Text(self.master, height=10, state="disabled")
        self.output_console.pack(padx=10, pady=5, fill="both", expand=True)

    def load_projects(self):
        try:
            from src.color_analysis.project_manager import ProjectManager
            self.project_manager = ProjectManager()
            self.available_projects = self.project_manager.list_projects()
            if self.available_projects:
                self.project_var.set(self.available_projects[0]) # Set default
                self.project_dropdown["menu"].delete(0, "end") # Clear existing menu
                for project in self.available_projects:
                    self.project_dropdown["menu"].add_command(label=project, command=tk._setit(self.project_var, project))
            else:
                self.log_message("No projects found. Please create a project in the 'data/projects' directory.")
        except Exception as e:
            self.log_message(f"Error loading projects: {e}")

    def browse_file(self):
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.input_path_var.set(file_path)

    def use_camera(self):
        self.input_path_var.set("camera") # Special keyword for camera input
        self.log_message("Camera input selected. Ensure camera is connected.")

    def log_message(self, message):
        self.output_console.config(state="normal")
        self.output_console.insert(tk.END, message + "\n")
        self.output_console.see(tk.END) # Scroll to end
        self.output_console.config(state="disabled")

    def run_analysis(self):
        project_name = self.project_var.get()
        input_path = self.input_path_var.get()

        if not project_name:
            messagebox.showerror("Error", "Please select a project.")
            return
        if not input_path:
            messagebox.showerror("Error", "Please provide an image/video path or select camera.")
            return

        # Create a dummy args object to pass to the analysis logic
        class Args:
            pass
        args = Args()
        args.project = project_name
        args.image = input_path if input_path != "camera" else None
        args.video = input_path if input_path.endswith((".mp4", ".avi", ".mov")) else None # Basic video check
        args.camera = True if input_path == "camera" else False
        args.debug = self.debug_var.get()
        args.aggregate = self.aggregate_var.get()
        args.blur = self.blur_var.get()
        args.alignment = self.alignment_var.get()
        args.color_alignment = self.color_alignment_var.get()

        self.log_message(f"Running analysis for project: {args.project}")
        self.log_message(f"Input: {args.image or args.video or 'Camera'}")
        self.log_message("Options: " +
                         f"Debug={args.debug}, " +
                         f"Aggregate={args.aggregate}, " +
                         f"Blur={args.blur}, " +
                         f"Alignment={args.alignment}, " +
                         f"Color Alignment={args.color_alignment}")
        
        # Run analysis in a separate thread to keep GUI responsive
        analysis_thread = threading.Thread(target=self._run_analysis_thread, args=(args,))
        analysis_thread.start()

    def _run_analysis_thread(self, args):
        try:
            # Redirect stdout to the console widget
            import sys
            original_stdout = sys.stdout
            sys.stdout = self.OutputRedirector(self.output_console)

            run_analysis(args) # Call the refactored analysis logic

        except Exception as e:
            self.log_message(f"Analysis Error: {e}")
            messagebox.showerror("Analysis Error", str(e))
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            self.log_message("Analysis complete.")

    class OutputRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, text):
            self.widget.config(state="normal")
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END) # Scroll to end
            self.widget.config(state="disabled")

        def flush(self):
            pass # Required for file-like objects

print("---" + " Before start_gui() ---")
def start_gui():
    print("---" + " Inside start_gui() ---")
    try:
        root = tk.Tk()
        app = VisualAnalyzerGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")

if __name__ == "__main__":
    start_gui()
