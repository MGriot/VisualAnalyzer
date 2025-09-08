import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================================
#  SymmetryAnalyzer Class
# ====================================================================================
#  THEORY: In computational geometry and computer vision, symmetry detection involves
#  identifying invariances in an image under a set of geometric transformations.
#  These transformations form a mathematical structure known as a group. The fundamental
#  isometries (rigid transformations) of the Euclidean plane are reflection, rotation,
#  translation, and glide-reflection. This class provides methods to quantify an
#  image's invariance with respect to these fundamental operations on a discrete grid.
# ====================================================================================
class SymmetryAnalyzer:
    """
    A comprehensive class to perform symmetry analysis on an image.

    It quantifies the image's invariance under various geometric transformations
    (isometries) and provides detailed results and visualizations.
    """
    def __init__(self, image):
        self.original_image = image
        self.results = {}
        self._preprocess_image()

    def _preprocess_image(self):
        """Prepares the image for analysis (grayscale, even dimensions)."""
        if len(self.original_image.shape) > 2:
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR_GRAY)
        else:
            self.gray_image = self.original_image.copy()
        h, w = self.gray_image.shape
        h, w = h - (h % 2), w - (w % 2)
        self.processed_image = self.gray_image[0:h, 0:w]

    def _calculate_similarity(self, part1, part2, mask=None):
        """
        Calculates a normalized similarity score using the L1 norm (Mean Absolute Difference).
        This metric measures the average difference between pixel intensities. A score of 1.0
        signifies perfect identity between the two parts.
        """
        if part1.shape != part2.shape: return 0
        if mask is not None:
            active_pixels = np.sum(mask > 0)
            if active_pixels == 0: return 1.0
            abs_diff = np.sum(np.abs(part1.astype("float") - part2.astype("float")) * (mask > 0))
            mean_diff = abs_diff / active_pixels
        else:
            mean_diff = np.mean(np.abs(part1.astype("float") - part2.astype("float")))
        return 1 - (mean_diff / 255.0)
    
    # ------------------------------------------------------------------------------------
    #  I. REFLECTION SYMMETRY (Simmetria di riflessione)
    # ------------------------------------------------------------------------------------
    #  THEORY: Reflection is an isometry that maps an object onto its mirror image
    #  across a hyperplane (an axis in 2D). This analysis tests for invariance when
    #  the image is reflected across its central vertical and horizontal axes.
    # ------------------------------------------------------------------------------------
    def analyze_vertical_reflection(self):
        h, w = self.processed_image.shape
        left_half = self.processed_image[:, 0:w//2]
        right_half = self.processed_image[:, w//2:]
        flipped_right = cv2.flip(right_half, 1) # Flip operation is a discrete matrix reflection.
        self.results['vertical_reflection'] = {
            'score': self._calculate_similarity(left_half, flipped_right),
            'chunks': {'left': left_half, 'right (flipped)': flipped_right},
        }

    def analyze_horizontal_reflection(self):
        h, w = self.processed_image.shape
        top_half = self.processed_image[0:h//2, :]
        bottom_half = self.processed_image[h//2:, :]
        flipped_bottom = cv2.flip(bottom_half, 0)
        self.results['horizontal_reflection'] = {
            'score': self._calculate_similarity(top_half, flipped_bottom),
            'chunks': {'top': top_half, 'bottom (flipped)': flipped_bottom},
        }

    # ------------------------------------------------------------------------------------
    #  II. FOUR-QUADRANT SYMMETRY (Dihedral Group Symmetry)
    # ------------------------------------------------------------------------------------
    #  THEORY: This tests for invariance under a more complex symmetry group, specifically
    #  the Dihedral group D2, which includes reflections across two orthogonal axes.
    #  A robust analysis must not be biased by the choice of a single "master" quadrant.
    #  By computing the symmetry score from the perspective of each of the four
    #  quadrants and averaging the results, we obtain a more stable and reliable
    #  metric of the image's overall dihedral symmetry.
    # ------------------------------------------------------------------------------------
    def analyze_four_quadrant(self):
        h, w = self.processed_image.shape
        # q1 (top-left), q2 (top-right), q3 (bottom-left), q4 (bottom-right)
        q1 = self.processed_image[0:h//2, 0:w//2]
        q2 = self.processed_image[0:h//2, w//2:]
        q3 = self.processed_image[h//2:, 0:w//2]
        q4 = self.processed_image[h//2:, w//2:]

        # --- Refined Four-Quadrant Symmetry Score ---
        # To obtain a robust, unbiased score, we calculate four sets of comparisons,
        # each assuming a different quadrant as the "master" reference.
        # This approach mitigates any bias that might arise from choosing a single
        # quadrant (e.g., q1) as the sole basis for the entire symmetry evaluation.
        # The final score is the average of these four perspectives, providing a
        # more holistic measure of the image's dihedral symmetry.

        # Perspective 1: q1 is master
        s1 = self._calculate_similarity(q1, cv2.flip(q2, 1)) # Compare q1 with flipped q2
        s2 = self._calculate_similarity(q1, cv2.flip(q3, 0)) # Compare q1 with flipped q3
        s3 = self._calculate_similarity(q1, cv2.flip(q4, -1))# Compare q1 with flipped q4
        score_from_q1 = (s1 + s2 + s3) / 3.0

        # Perspective 2: q2 is master
        s1 = self._calculate_similarity(q2, cv2.flip(q1, 1)) # Compare q2 with flipped q1
        s2 = self._calculate_similarity(q2, cv2.flip(q4, 0)) # Compare q2 with flipped q4
        s3 = self._calculate_similarity(q2, cv2.flip(q3, -1))# Compare q2 with flipped q3
        score_from_q2 = (s1 + s2 + s3) / 3.0

        # Perspective 3: q3 is master
        s1 = self._calculate_similarity(q3, cv2.flip(q1, 0)) # Compare q3 with flipped q1
        s2 = self._calculate_similarity(q3, cv2.flip(q4, 1)) # Compare q3 with flipped q4
        s3 = self._calculate_similarity(q3, cv2.flip(q2, -1))# Compare q3 with flipped q2
        score_from_q3 = (s1 + s2 + s3) / 3.0

        # Perspective 4: q4 is master
        s1 = self._calculate_similarity(q4, cv2.flip(q2, 0)) # Compare q4 with flipped q2
        s2 = self._calculate_similarity(q4, cv2.flip(q3, 1)) # Compare q4 with flipped q3
        s3 = self._calculate_similarity(q4, cv2.flip(q1, -1))# Compare q4 with flipped q1
        score_from_q4 = (s1 + s2 + s3) / 3.0
        
        final_score = np.mean([score_from_q1, score_from_q2, score_from_q3, score_from_q4])

        # The reconstruction is based on a single quadrant (q1) for visualization purposes.
        reconstruction = np.vstack([
            np.hstack([q1, cv2.flip(q1, 1)]),
            np.hstack([cv2.flip(q1, 0), cv2.flip(q1, -1)])
        ])
        
        self.results['four_quadrant'] = {
            'score': final_score,
            'reconstruction': reconstruction,
            'diff_map': np.abs(self.processed_image.astype("float") - reconstruction.astype("float"))
        }

    # ------------------------------------------------------------------------------------
    #  III. ROTATIONAL SYMMETRY (Simmetria di rotazione)
    # ------------------------------------------------------------------------------------
    #  THEORY: Rotational symmetry exists if the image is invariant under a rotation
    #  around a central point. This is an application of affine transformations from
    #  linear algebra. A 2D rotation is represented by a 2x3 transformation matrix.
    #  The comparison must be masked to the overlapping region post-rotation to
    #  avoid penalizing for boundary artifacts (no-data areas).
    # ------------------------------------------------------------------------------------
    def analyze_rotational(self, angle):
        h, w = self.processed_image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(self.processed_image, M, (w, h))
        mask = cv2.warpAffine(np.ones_like(self.processed_image) * 255, M, (w, h))
        score = self._calculate_similarity(self.processed_image, rotated_img, mask=mask)
        self.results[f'rotational_{angle}deg'] = {'score': score, 'chunks': {'original': self.processed_image, 'rotated': rotated_img}}

    # ------------------------------------------------------------------------------------
    #  IV. TRANSLATIONAL SYMMETRY (Simmetria di traslazione)
    # ------------------------------------------------------------------------------------
    #  THEORY: This tests for discrete translational symmetry, characteristic of
    #  periodic patterns (frieze and wallpaper groups). The method employed is
    #  template matching, a fundamental technique in signal processing which is
    #  effectively a 2D cross-correlation. A template (a sub-image) is chosen and
    #  slid across a search area. The quality of the best match, measured by the
    #  Normalized Cross-Correlation Coefficient (TM_CCOEFF_NORMED), indicates the
    #  presence of a repeating motif.
    # ------------------------------------------------------------------------------------
    def analyze_translational(self, direction='horizontal', template_frac=0.25):
        h, w = self.processed_image.shape
        if direction == 'horizontal':
            tw = int(w * template_frac)
            template, search_area = self.processed_image[:, 0:tw], self.processed_image[:, tw:]
        else:
            th = int(h * template_frac)
            template, search_area = self.processed_image[0:th, :], self.processed_image[th:, :]
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        self.results[f'{direction}_translation'] = {'score': max_val, 'chunks': {'template': template, 'search_area': search_area}}

    # ------------------------------------------------------------------------------------
    #  V. GLIDE-REFLECTION SYMMETRY (Simmetria di glissoriflessione)
    # ------------------------------------------------------------------------------------
    #  THEORY: Glide-reflection is a composite isometry consisting of a reflection
    #  followed by a translation parallel to the axis of reflection. This analysis
    #  operationalizes this definition by first reflecting a sub-image (e.g., the top
    #  half) and then using template matching to find the best translational match
    #  for this reflected template in the other half of the image. A high score
    #  indicates the presence of this complex but common form of symmetry.
    # ------------------------------------------------------------------------------------
    def analyze_glide_reflection(self, axis='horizontal'):
        h, w = self.processed_image.shape
        if axis == 'horizontal':
            template_orig, search_area = self.processed_image[0:h//2, :], self.processed_image[h//2:, :]
            template_flipped = cv2.flip(template_orig, 0)
        else:
            template_orig, search_area = self.processed_image[:, 0:w//2], self.processed_image[:, w//2:]
            template_flipped = cv2.flip(template_orig, 1)
        res = cv2.matchTemplate(search_area, template_flipped, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        self.results[f'{axis}_glide_reflection'] = {'score': max_val, 'chunks': {'original_part': template_orig, 'reflected_part': template_flipped}}

    # --- Utility Methods ---
    def analyze_all(self):
        """Runs a standard suite of analyses."""
        self.analyze_vertical_reflection()
        self.analyze_horizontal_reflection()
        self.analyze_four_quadrant()
        self.analyze_rotational(90)
        self.analyze_rotational(180)
        self.analyze_translational()
        self.analyze_glide_reflection()

    def report(self):
        """Prints a formatted report of all analysis scores."""
        print("\n--- Symmetry Analysis Report ---")
        for key, value in self.results.items():
            print(f"{key.replace('_', ' ').title():<25}: {value['score']:.4f}")
        print("---------------------------------")
        
    def visualize(self, analysis_type):
        """Generates a detailed plot for a specific analysis type."""
        if analysis_type not in self.results: return
        res = self.results[analysis_type]
        title = f"{analysis_type.replace('_', ' ').title()} (Score: {res['score']:.4f})"
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(title, fontsize=16)
        axes[0].imshow(self.processed_image, cmap='gray')
        axes[0].set_title('Original Processed')
        if 'reconstruction' in res:
            axes[1].imshow(res['reconstruction'], cmap='gray')
            axes[1].set_title('Ideal Reconstruction')
        else:
            chunks = list(res['chunks'].values())
            axes[1].imshow(chunks[1], cmap='gray')
            axes[1].set_title(list(res['chunks'].keys())[1].title())
        for ax in axes: ax.axis('off')
        plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define a function to create a sample image for testing
    def create_sample_image(size=200):
        img = np.zeros((size, size), dtype=np.uint8)
        # Create a shape that is vertically symmetric but not horizontally
        cv2.ellipse(img, (size//2, size//2), (size//4, size//2 - 20), 0, 0, 360, 255, -1)
        cv2.circle(img, (size//2, size//4), 10, 150, -1)
        return img

    print("--- Analyzing a sample image for all symmetry types ---")
    sample_img = create_sample_image()
    
    # Instantiate the analyzer and run all tests
    analyzer = SymmetryAnalyzer(sample_img)
    analyzer.analyze_all()
    
    # Print the final report
    analyzer.report()
    
    # Visualize some of the key results
    print("\n--- Visualizing Key Analyses ---")
    analyzer.visualize('vertical_reflection')
    analyzer.visualize('horizontal_reflection')
    analyzer.visualize('four_quadrant')
    analyzer.visualize('rotational_180deg')