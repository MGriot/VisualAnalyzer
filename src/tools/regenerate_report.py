import argparse
import zipfile
import tempfile
import shutil
import json
from pathlib import Path
import sys

# Add src to path to allow for imports from the src directory
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.reporting.generator import ReportGenerator

def regenerate_report_from_archive(archive_path: Path):
    """
    Extracts a report archive and regenerates the HTML and PDF files.
    """
    if not archive_path.is_file() or archive_path.suffix != '.zip':
        print(f"Error: Archive not found or is not a .zip file at {archive_path}")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting archive to {temp_dir}...")
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile:
            print(f"Error: The file at {archive_path} is not a valid zip file.")
            return

        report_data_path = Path(temp_dir) / 'report_data.json'
        if not report_data_path.is_file():
            print(f"Error: 'report_data.json' not found in the archive.")
            return

        with open(report_data_path, 'r') as f:
            report_data = json.load(f)

        # --- Instantiate the generator with the correct context ---
        project_name = report_data.get('project_name')
        sample_name = report_data.get('part_number')
        # Determine if it was a debug report by checking for debug-specific data
        debug_mode = report_data.get('debug_data') is not None

        if not project_name or not sample_name:
            print("Error: Archive is missing 'project_name' or 'part_number' for regeneration.")
            return

        print(f"Regenerating report for project '{project_name}', sample '{sample_name}'...")
        report_generator = ReportGenerator(project_name, sample_name, debug_mode)
        
        # Use the dedicated method to generate from the extracted data
        report_generator.generate_from_archived_data(report_data, Path(temp_dir))

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate a Visual Analyzer report from a .zip archive.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--archive", 
        required=True, 
        type=Path, 
        help="Path to the report archive (.zip file) to regenerate."
    )
    args = parser.parse_args()
    regenerate_report_from_archive(args.archive)

if __name__ == "__main__":
    main()
