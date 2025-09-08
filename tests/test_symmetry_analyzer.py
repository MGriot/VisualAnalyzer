import unittest
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.symmetry_analysis.symmetry import SymmetryAnalyzer

class TestSymmetryAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up test images for symmetry analysis and create output directory."""
        self.size = 200
        self.center_x, self.center_y = self.size // 2, self.size // 2

        self.output_dir = os.path.join('output', 'test_symmetry_visualizations')
        os.makedirs(self.output_dir, exist_ok=True)

        # Helper to save images
        def save_image(name, img):
            cv2.imwrite(os.path.join(self.output_dir, f'{name}.png'), img)

        # 1. Asymmetric Image
        self.img_asym = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.rectangle(self.img_asym, (10, 10), (self.center_x - 10, self.center_y - 10), 255, -1)
        save_image('sample_asymmetric', self.img_asym)

        # 2. Vertically Symmetric Image
        self.img_vert_sym = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.ellipse(self.img_vert_sym, (self.center_x, self.center_y), (self.size//4, self.size//2 - 20), 0, 0, 360, 255, -1)
        save_image('sample_vertical_symmetric', self.img_vert_sym)

        # 3. Horizontally Symmetric Image
        self.img_horiz_sym = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.ellipse(self.img_horiz_sym, (self.center_x, self.center_y), (self.size//2 - 20, self.size//4), 0, 0, 360, 255, -1)
        save_image('sample_horizontal_symmetric', self.img_horiz_sym)

        # 4. Four-Quadrant & 180-degree Rotational Symmetric Image
        self.img_quad_sym = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.rectangle(self.img_quad_sym, (30, 30), (self.size - 30, self.size - 30), 255, -1)
        save_image('sample_four_quadrant_symmetric', self.img_quad_sym)

        # 5. 90-degree Rotational Symmetric Image
        self.img_rot_sym_90 = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.line(self.img_rot_sym_90, (30, self.center_y), (self.size - 30, self.center_y), 255, 20)
        cv2.line(self.img_rot_sym_90, (self.center_x, 30), (self.center_x, self.size - 30), 255, 20)
        save_image('sample_rotational_90_symmetric', self.img_rot_sym_90)

        # 6. Horizontal Translational Symmetry
        self.img_trans_horiz = np.zeros((self.size, self.size), dtype=np.uint8)
        for i in range(10, self.size, 20):
            cv2.line(self.img_trans_horiz, (i, 0), (i, self.size), 255, 5)
        save_image('sample_translational_horizontal', self.img_trans_horiz)

        # 7. Vertical Translational Symmetry
        self.img_trans_vert = np.zeros((self.size, self.size), dtype=np.uint8)
        for i in range(10, self.size, 20):
            cv2.line(self.img_trans_vert, (0, i), (self.size, i), 255, 5)
        save_image('sample_translational_vertical', self.img_trans_vert)

        # 8. Horizontal Glide-Reflection Symmetry
        self.img_glide_horiz = np.zeros((self.size, self.size), dtype=np.uint8)
        # Top part: series of slanted lines
        for i in range(20, self.size - 40, 40):
            cv2.line(self.img_glide_horiz, (i, 20), (i + 20, self.center_y - 20), 255, 3)
        # Bottom part: reflected and translated
        for i in range(40, self.size - 20, 40):
            cv2.line(self.img_glide_horiz, (i, self.center_y + 20), (i - 20, self.size - 20), 255, 3)
        save_image('sample_glide_reflection_horizontal', self.img_glide_horiz)

    def test_vertical_reflection(self):
        analyzer_sym = SymmetryAnalyzer(self.img_vert_sym)
        analyzer_sym.analyze_vertical_reflection()
        self.assertGreater(analyzer_sym.results['vertical_reflection']['score'], 0.99)
        
        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_sym.processed_image, cmap='gray')
        axes[0].set_title('Original Processed')
        # Add vertical symmetry axis
        h, w = analyzer_sym.processed_image.shape
        axes[0].axvline(x=w // 2, color='r', linestyle='--', linewidth=1)
        axes[1].imshow(analyzer_sym.results['vertical_reflection']['chunks']['right (flipped)'], cmap='gray')
        axes[1].set_title('Right Half (Flipped)')
        plt.suptitle(f'Vertical Reflection (Score: {analyzer_sym.results['vertical_reflection']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'vertical_reflection_sym.png'))
        plt.close(fig)

        analyzer_asym = SymmetryAnalyzer(self.img_asym)
        analyzer_asym.analyze_vertical_reflection()
        self.assertLess(analyzer_asym.results['vertical_reflection']['score'], 0.8)

    def test_horizontal_reflection(self):
        analyzer_sym = SymmetryAnalyzer(self.img_horiz_sym)
        analyzer_sym.analyze_horizontal_reflection()
        self.assertGreater(analyzer_sym.results['horizontal_reflection']['score'], 0.99)

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_sym.processed_image, cmap='gray')
        axes[0].set_title('Original Processed')
        # Add horizontal symmetry axis
        h, w = analyzer_sym.processed_image.shape
        axes[0].axhline(y=h // 2, color='r', linestyle='--', linewidth=1)
        axes[1].imshow(analyzer_sym.results['horizontal_reflection']['chunks']['bottom (flipped)'], cmap='gray')
        axes[1].set_title('Bottom Half (Flipped)')
        plt.suptitle(f'Horizontal Reflection (Score: {analyzer_sym.results['horizontal_reflection']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'horizontal_reflection_sym.png'))
        plt.close(fig)

        analyzer_asym = SymmetryAnalyzer(self.img_asym)
        analyzer_asym.analyze_horizontal_reflection()
        self.assertLess(analyzer_asym.results['horizontal_reflection']['score'], 0.8)

    def test_four_quadrant(self):
        analyzer_sym = SymmetryAnalyzer(self.img_quad_sym)
        analyzer_sym.analyze_four_quadrant()
        self.assertGreater(analyzer_sym.results['four_quadrant']['score'], 0.99)

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_sym.processed_image, cmap='gray')
        axes[0].set_title('Original Processed')
        # Add horizontal and vertical symmetry axes
        h, w = analyzer_sym.processed_image.shape
        axes[0].axvline(x=w // 2, color='r', linestyle='--', linewidth=1)
        axes[0].axhline(y=h // 2, color='r', linestyle='--', linewidth=1)
        axes[1].imshow(analyzer_sym.results['four_quadrant']['reconstruction'], cmap='gray')
        axes[1].set_title('Ideal Reconstruction')
        plt.suptitle(f'Four Quadrant Symmetry (Score: {analyzer_sym.results['four_quadrant']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'four_quadrant_sym.png'))
        plt.close(fig)

        analyzer_asym = SymmetryAnalyzer(self.img_asym)
        analyzer_asym.analyze_four_quadrant()
        self.assertLess(analyzer_asym.results['four_quadrant']['score'], 0.7)

    def test_rotational(self):
        # Test 90-degree rotation
        analyzer_90 = SymmetryAnalyzer(self.img_rot_sym_90)
        analyzer_90.analyze_rotational(90)
        self.assertGreater(analyzer_90.results['rotational_90deg']['score'], 0.95) # Rotation can have artifacts

        # Save visualization for 90-deg
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_90.processed_image, cmap='gray')
        axes[0].set_title('Original Processed')
        # Mark center of rotation
        h, w = analyzer_90.processed_image.shape
        axes[0].plot(w // 2, h // 2, 'rx', markersize=8, markeredgewidth=2) # Red 'x' at center
        axes[1].imshow(analyzer_90.results['rotational_90deg']['chunks']['rotated'], cmap='gray')
        axes[1].set_title('Rotated 90deg')
        plt.suptitle(f'Rotational Symmetry 90deg (Score: {analyzer_90.results['rotational_90deg']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'rotational_90deg_sym.png'))
        plt.close(fig)

        # Test 180-degree rotation
        analyzer_180 = SymmetryAnalyzer(self.img_quad_sym)
        analyzer_180.analyze_rotational(180)
        self.assertGreater(analyzer_180.results['rotational_180deg']['score'], 0.99)

        # Save visualization for 180-deg
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_180.processed_image, cmap='gray')
        axes[0].set_title('Original Processed')
        # Mark center of rotation
        h, w = analyzer_180.processed_image.shape
        axes[0].plot(w // 2, h // 2, 'rx', markersize=8, markeredgewidth=2) # Red 'x' at center
        axes[1].imshow(analyzer_180.results['rotational_180deg']['chunks']['rotated'], cmap='gray')
        axes[1].set_title('Rotated 180deg')
        plt.suptitle(f'Rotational Symmetry 180deg (Score: {analyzer_180.results['rotational_180deg']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'rotational_180deg_sym.png'))
        plt.close(fig)

        # Test asymmetric
        analyzer_asym = SymmetryAnalyzer(self.img_asym)
        analyzer_asym.analyze_rotational(90)
        self.assertLess(analyzer_asym.results['rotational_90deg']['score'], 0.7)

    def test_translational(self):
        # Test horizontal translation
        analyzer_horiz = SymmetryAnalyzer(self.img_trans_horiz)
        analyzer_horiz.analyze_translational(direction='horizontal', template_frac=0.1)
        self.assertGreater(analyzer_horiz.results['horizontal_translation']['score'], 0.9)

        # Save visualization for horizontal translation
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_horiz.results['horizontal_translation']['chunks']['template'], cmap='gray')
        axes[0].set_title('Template')
        axes[1].imshow(analyzer_horiz.results['horizontal_translation']['chunks']['search_area'], cmap='gray')
        axes[1].set_title('Search Area')
        plt.suptitle(f'Horizontal Translational Symmetry (Score: {analyzer_horiz.results['horizontal_translation']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'translational_horizontal_sym.png'))
        plt.close(fig)

        # Test vertical translation
        analyzer_vert = SymmetryAnalyzer(self.img_trans_vert)
        analyzer_vert.analyze_translational(direction='vertical', template_frac=0.1)
        self.assertGreater(analyzer_vert.results['vertical_translation']['score'], 0.9)

        # Save visualization for vertical translation
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_vert.results['vertical_translation']['chunks']['template'], cmap='gray')
        axes[0].set_title('Template')
        axes[1].imshow(analyzer_vert.results['vertical_translation']['chunks']['search_area'], cmap='gray')
        axes[1].set_title('Search Area')
        plt.suptitle(f'Vertical Translational Symmetry (Score: {analyzer_vert.results['vertical_translation']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'translational_vertical_sym.png'))
        plt.close(fig)

    def test_glide_reflection(self):
        analyzer_glide = SymmetryAnalyzer(self.img_glide_horiz)
        analyzer_glide.analyze_glide_reflection(axis='horizontal')
        # This is a complex symmetry, score might not be perfect
        self.assertGreater(analyzer_glide.results['horizontal_glide_reflection']['score'], 0.8)

        # Save visualization for glide reflection
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(analyzer_glide.results['horizontal_glide_reflection']['chunks']['original_part'], cmap='gray')
        axes[0].set_title('Original Part')
        axes[1].imshow(analyzer_glide.results['horizontal_glide_reflection']['chunks']['reflected_part'], cmap='gray')
        axes[1].set_title('Reflected Part')
        plt.suptitle(f'Horizontal Glide Reflection (Score: {analyzer_glide.results['horizontal_glide_reflection']['score']:.4f})')
        plt.savefig(os.path.join(self.output_dir, 'glide_reflection_horizontal_sym.png'))
        plt.close(fig)

    def test_analyze_all(self):
        analyzer = SymmetryAnalyzer(self.img_quad_sym)
        analyzer.analyze_all()
        expected_keys = [
            'vertical_reflection', 'horizontal_reflection', 'four_quadrant',
            'rotational_90deg', 'rotational_180deg', 'horizontal_translation',
            'horizontal_glide_reflection'
        ]
        for key in expected_keys:
            self.assertIn(key, analyzer.results)

    def test_report(self):
        """Test that reporting runs without errors."""
        analyzer = SymmetryAnalyzer(self.img_quad_sym)
        analyzer.analyze_all()
        try:
            # Suppress print output for report
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            analyzer.report()
            sys.stdout.close()
            sys.stdout = original_stdout
        except Exception as e:
            self.fail(f"report() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
