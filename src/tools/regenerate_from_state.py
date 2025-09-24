import argparse
from pathlib import Path
import sys

# Add src to path to allow for imports from the src directory
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline import Pipeline

def regenerate_report_from_state(state_path: Path, args):
    """
    Loads a saved pipeline state and regenerates the report.
    """
    if not state_path.is_file():
        print(f"Error: State file not found at {state_path}")
        return

    print(f"Loading pipeline state from {state_path}...")
    pipeline = Pipeline.load_state(state_path)
    pipeline.args = args # Update args to allow for changes, e.g., report type

    print(f"Regenerating report for project '{pipeline.args.project}', sample '{pipeline.metadata['part_number']}'...")
    pipeline.generate_report()

    print("Report regeneration complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate a Visual Analyzer report from a saved pipeline state.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--state-file", 
        required=True, 
        type=Path, 
        help="Path to the saved pipeline state file."
    )
    # Add other arguments that might be useful to override during regeneration
    parser.add_argument("--report-type", type=str, default="all", choices=["html", "reportlab", "all"], help="Specify the type of PDF report to generate (default: all).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for regeneration.")


    args = parser.parse_args()
    
    # We need to provide the full args namespace to the pipeline object
    # So we will create a dummy args object with all the possible arguments
    from src.main import main as main_cli
    cli_parser = main_cli.__closure__[0].cell_contents
    dummy_args = cli_parser.parse_args([]) # Get default values
    
    # Update with the regeneration-specific args
    for key, value in vars(args).items():
        setattr(dummy_args, key, value)

    regenerate_report_from_state(args.state_file, dummy_args)

if __name__ == "__main__":
    main()
