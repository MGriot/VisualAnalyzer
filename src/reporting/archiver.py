import os
import json
import zipfile
import shutil
from pathlib import Path
import datetime

class ReportArchiver:
    """Handles the archiving of report data and assets into a zip file."""

    def __init__(self, project_output_dir: Path):
        """
        Initializes the archiver.

        Args:
            project_output_dir (Path): The main output directory for the sample,
                                     where the 'archives' folder will be created.
        """
        self.archive_root_dir = project_output_dir / "archives"
        os.makedirs(self.archive_root_dir, exist_ok=True)

    def _copy_and_update_paths(self, item, source_dir: Path, dest_dir: Path):
        """
        Recursively traverses a data structure, copies file paths from source_dir
        to dest_dir, and returns a new data structure with updated, relative paths.
        """
        if isinstance(item, dict):
            return {k: self._copy_and_update_paths(v, source_dir, dest_dir) for k, v in item.items()}
        if isinstance(item, list):
            return [self._copy_and_update_paths(v, source_dir, dest_dir) for v in item]
        
        # Check if the item is a string that looks like a relative file path inside the source_dir
        if isinstance(item, str):
            # A simple heuristic: does it have a file extension and does it exist?
            try:
                source_path = source_dir / item
                if source_path.is_file():
                    shutil.copy(source_path, dest_dir)
                    return Path(item).name  # Return just the filename for the new path
            except (TypeError, ValueError):
                # Not a valid path, return as is
                return item
        
        return item

    def archive_report(self, report_data: dict, source_dir: Path):
        """
        Archives all report assets and data into a single zip file.

        Args:
            report_data (dict): The dictionary of data passed to the report template.
            source_dir (Path): The directory where the original report and assets were generated.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        part_number = report_data.get("part_number", "report")
        archive_name = f"archive_{part_number}_{timestamp}"
        temp_build_dir = self.archive_root_dir / f"__build_{archive_name}"
        
        if temp_build_dir.exists():
            shutil.rmtree(temp_build_dir)
        os.makedirs(temp_build_dir)

        try:
            # Create a new dict with updated paths and copy files simultaneously
            archived_data = self._copy_and_update_paths(report_data, source_dir, temp_build_dir)
            
            # Save the modified data dictionary as JSON
            with open(temp_build_dir / 'report_data.json', 'w') as f:
                json.dump(archived_data, f, indent=4)

            # Create the zip file
            zip_path_base = self.archive_root_dir / archive_name
            shutil.make_archive(base_name=str(zip_path_base), format='zip', root_dir=str(temp_build_dir))
            
            zip_path = f"{zip_path_base}.zip"
            print(f"Report successfully archived to {zip_path}")
            return zip_path
        finally:
            # Clean up the temporary build directory
            if temp_build_dir.exists():
                shutil.rmtree(temp_build_dir)
