"""
main.py
Entry point and simple usage examples for the ColorCheckerGenerator package.
Creates a default classic ColorChecker (200 mm wide) in PNG and PDF.
"""

from colorchecker.generator import ColorCheckerGenerator

def example():
    """Generates a classic ColorChecker with ArUco markers and saves it to the project root."""
    gen = ColorCheckerGenerator(
        size="20cm",
        dpi=300,
        checker_type="classic",
        include_aruco=True,
        logo_text="Reference Checker"
    )
    gen.build()
    # Save to the project root directory, which is two levels up from this script's location.
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_path = os.path.join(project_root, "ColorChecker_with_ArUco.png")
    gen.save(output_path)
    print(f"Done: created {output_path}")

if __name__ == "__main__":
    example()
