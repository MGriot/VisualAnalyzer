"""
This module provides the `ReportArchiver` class, which is responsible for
serializing and archiving analysis report data.

It uses Python's `pickle` module to store report data, including images and
metadata, into a single file for later regeneration or review.
"""

import os
import pickle
from pathlib import Path
import datetime

class ReportArchiver:
    """
    Handles the archiving of report data by serializing it to a pickle file.

    This class is responsible for taking a dictionary of report data (which can
    include NumPy arrays for images) and saving it to a `.gri` file using `pickle`.
    This allows for the regeneration of reports at a later time.
    """

    def __init__(self, project_output_dir: Path, debug_mode: bool = False):
        """
        Initializes the ReportArchiver.

        Args:
            project_output_dir (Path): The root directory where project-specific
                                       output (including archives) will be stored.
            debug_mode (bool, optional): If True, enables debug print statements.
                                         Defaults to False.
        """
        self.archive_root_dir = project_output_dir / "archives"
        os.makedirs(self.archive_root_dir, exist_ok=True)
        self.debug_mode = debug_mode

    def archive_report(self, serializable_data: dict):
        """
        Serializes the provided report data into a `.gri` (Griot Report Information) file.

        The filename is generated based on the current timestamp and a part number
        extracted from the `serializable_data`.

        Args:
            serializable_data (dict): A dictionary containing all the data to be archived.
                                      This dictionary should be serializable by `pickle`.

        Returns:
            Path or None: The path to the created archive file if successful, otherwise None.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        part_number = serializable_data.get("part_number", "report")
        archive_name = f"analysis_{part_number}_{timestamp}.gri"
        archive_path = self.archive_root_dir / archive_name
        
        try:
            with open(archive_path, 'wb') as f:
                pickle.dump(serializable_data, f)
            
            if self.debug_mode:
                print(f"Analysis results successfully serialized to {archive_path}")
            return archive_path
        except Exception as e:
            print(f"[ERROR] Failed to serialize report data: {e}")
            return None
