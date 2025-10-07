import argparse
from html import parser
from src.pipeline import run_analysis

def main():
    """
    This script serves as the command-line interface (CLI) for the Visual Analyzer application.
    
    It parses command-line arguments to configure and run the image and video analysis pipeline.
    The script supports various analysis features, including color correction, color analysis,
    image alignment, background removal, and symmetry analysis.
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
    parser.add_argument("--object-alignment", action="store_true", help="Enable object alignment.")
    parser.add_argument("--apply-mask", action="store_true", help="Enable masking to remove background.")
    parser.add_argument("--mask-bg-is-white", action="store_true", help="Treat white as background during masking.")
    parser.add_argument("--drawing", type=str, help="Path to a technical drawing for masking.")
    parser.add_argument("--color-alignment", action="store_true", help="Enable color correction.")
    parser.add_argument("--sample-color-checker", type=str, help="Path to the color checker used as a sample for color alignment.")
    parser.add_argument("--color-correction-method", type=str, default="linear", choices=["linear", "polynomial", "hsv", "histogram"], help="Specify the color correction algorithm.")
    parser.add_argument("--symmetry", action="store_true", help="Enable symmetry analysis.")
    parser.add_argument("--masking-order", type=str, default="1-2-3", help="Specify the order of masking layers (e.g., '1-2-3', '3-1-2').")

    parser.add_argument("--agg-kernel-size", type=int, default=7, help="Kernel size for the aggregation dilation step.")
    parser.add_argument("--agg-min-area", type=float, default=0.0005, help="Minimum area ratio for keeping a component during aggregation.")
    parser.add_argument("--agg-density-thresh", type=float, default=0.5, help="Minimum density of original pixels for an aggregated area to be kept.")
    parser.add_argument("--blur-kernel", type=int, nargs=2, metavar=('W', 'H'), help="Specify a custom kernel size (width height) for blurring. Both values must be odd.")

    parser.add_argument("--skip-color-analysis", action="store_true", help="Skip the color analysis step.")
    parser.add_argument("--skip-report-generation", action="store_true", help="Skip the report generation step.")
    parser.add_argument("--save-state-to", type=str, help="Path to save the pipeline state.")
    parser.add_argument("--load-state-from", type=str, help="Path to load a previously saved pipeline state.")
    parser.add_argument(
        '--object-alignment-shadow-removal', 
        type=str, 
        default='clahe', 
        choices=['clahe', 'gamma', 'none'], 
        help='Shadow removal method for object alignment. "clahe" is advanced, "gamma" is simple, "none" disables it.'
    )
    args = parser.parse_args()
    
    if args.color_alignment and not args.sample_color_checker:
        parser.error("--sample-color-checker is required when --color-alignment is enabled.")

    run_analysis(args)

if __name__ == "__main__":
    main()
