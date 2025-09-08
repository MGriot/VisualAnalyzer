import argparse
from src.pipeline import run_analysis

def main():
    """
    Main function to parse command-line arguments and initiate the analysis.
    """
    parser = argparse.ArgumentParser(description="Visual Analyzer for Color Correction and Analysis.")
    parser.add_argument("--project", type=str, help="Name of the project to use.")
    parser.add_argument("--image", type=str, help="Path to a single image file or a directory of images for analysis.")
    parser.add_argument("--video", type=str, help="Path to a single video file or a directory of videos for analysis.")
    parser.add_argument("--camera", action="store_true", help="Use live camera stream for analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--aggregate", action="store_true", help="Enable aggregation of matched pixel areas.")
    parser.add_argument("--blur", action="store_true", help="Enable blurring of the input image.")
    parser.add_argument("--alignment", action="store_true", help="Enable geometrical alignment.")
    parser.add_argument("--drawing", type=str, help="Path to a technical drawing for masking.")
    parser.add_argument("--color-alignment", action="store_true", help="Enable color correction.")
    parser.add_argument("--sample-color-checker", type=str, help="Path to the color checker used as a sample for color alignment.")
    parser.add_argument("--symmetry", action="store_true", help="Enable symmetry analysis.")

    args = parser.parse_args()
    
    if args.color_alignment and not args.sample_color_checker:
        parser.error("--sample-color-checker is required when --color-alignment is enabled.")

    run_analysis(args)

if __name__ == "__main__":
    main()
