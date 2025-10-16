import cv2
import numpy as np
import os
import math
from typing import Tuple, List, Dict, Optional

from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab, deltaE_cie76

from src.geometric_alignment.geometric_aligner import ArucoAligner
from src.utils.image_utils import load_image, save_image

def _sort_patches(
    patches: List[np.ndarray],
    patch_coords: List[Tuple[int, int, int, int]]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Sorts patches based on their y and then x coordinates."""
    if not patches:
        return [], []
    combined = sorted(zip(patches, patch_coords), key=lambda item: (item[1][1], item[1][0]))
    sorted_patches, sorted_coords = zip(*combined)
    return list(sorted_patches), list(sorted_coords)

class ColorCorrectionPipeline:
    """
    A self-contained pipeline for performing color correction using a photo of a color checker.
    It uses ArUco markers for robust alignment and color similarity for patch matching.
    """

    def __init__(self, reference_color_checker_path: str, output_dir: Optional[str] = None, debug_mode: bool = False):
        self.reference_color_checker_path = reference_color_checker_path
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.aruco_aligner = ArucoAligner(debug_mode=self.debug_mode, output_dir=self.output_dir)
        self.reference_checker_img, _ = load_image(self.reference_color_checker_path)
        if self.reference_checker_img is None:
            raise FileNotFoundError(f"Could not load reference color checker at: {self.reference_color_checker_path}")
        self.debug_paths = {}
        self._step_counter = 0

    def _save_debug_image(self, name: str, image: np.ndarray):
        if self.debug_mode and self.output_dir and image is not None:
            path = os.path.join(self.output_dir, f"{self._step_counter:02d}_{name}.png")
            save_image(path, image)
            self.debug_paths[name] = path
            self._step_counter += 1

    def run(self, image_to_correct: np.ndarray, photo_of_checker_path: str, correction_method: str = "linear") -> Dict:
        if photo_of_checker_path and os.path.exists(photo_of_checker_path):
            photo_of_checker, _ = load_image(photo_of_checker_path)
            if self.debug_mode: print(f"[INFO] Loaded separate color checker photo from: {photo_of_checker_path}")
        else:
            if self.debug_mode: print(f"[INFO] No valid checker photo path provided. Assuming checker is in the main image.")
            photo_of_checker = image_to_correct

        if photo_of_checker is None:
            print(f"[ERROR] Could not load or find a color checker image.")
            return {"corrected_image": None, "debug_paths": self.debug_paths}
        self._save_debug_image("input_photo_of_checker", photo_of_checker)

        # --- Step 2: Align Checker (Auto with Manual Fallback) ---
        if self.debug_mode: print("[INFO] Step 2: Aligning checker photo to reference...")
        
        from . import alignment, manual_aligner_gui # Import new modules

        aligner = alignment.Aligner(photo_of_checker)
        
        # Try automatic ArUco alignment first
        aligned_checker_photo = aligner.align_with_aruco(self.reference_checker_img, self.debug_mode, self.output_dir)

        # If automatic alignment fails, fall back to manual GUI
        if aligned_checker_photo is None:
            print("[WARNING] Automatic ArUco alignment failed. Please select corners manually.")
            try:
                manual_corners = manual_aligner_gui.get_corners_from_user(photo_of_checker)
                if manual_corners and len(manual_corners) == 4:
                    print(f"[INFO] Using manually selected corners: {manual_corners}")
                    aligned_checker_photo = aligner.align_with_manual_points(manual_corners)
                    if aligned_checker_photo is not None:
                        self._save_debug_image("aligned_checker_photo_manual", aligned_checker_photo)
                else:
                    print("[ERROR] Manual alignment was cancelled or did not provide 4 points. Cannot proceed.")
                    return {"corrected_image": None, "debug_paths": self.debug_paths}
            except Exception as e:
                print(f"[ERROR] Manual alignment GUI failed with an error: {e}")
                return {"corrected_image": None, "debug_paths": self.debug_paths}

        if aligned_checker_photo is None:
            print("[ERROR] Alignment failed. Cannot proceed.")
            return {"corrected_image": None, "debug_paths": self.debug_paths}
        self._save_debug_image("aligned_checker_photo", aligned_checker_photo)

        if self.debug_mode: print("[INFO] Step 3: Cropping images to checker area using marker geometry...")
        source_cropped, ref_cropped = self._crop_checkers_geometrically(aligned_checker_photo)
        self._save_debug_image("geocrop_source_checker", source_cropped)
        self._save_debug_image("geocrop_reference_checker", ref_cropped)

        if source_cropped is None or ref_cropped is None:
            print("[ERROR] Geometric cropping failed. Cannot proceed.")
            return {"corrected_image": None, "debug_paths": self.debug_paths}

        if self.debug_mode: print("[INFO] Step 4: Detecting patches on cropped images...")
        source_patches, source_coords = self._detect_patches_from_image(source_cropped, "source")
        ref_patches, _ = self._detect_patches_from_image(ref_cropped, "reference")

        if not source_patches or not ref_patches:
            print("[ERROR] Patch detection failed on source or reference. Cannot proceed.")
            return {"corrected_image": None, "debug_paths": self.debug_paths}

        if self.debug_mode: print("[INFO] Step 5: Calculating average colors...")
        color_space = "hsv" if correction_method == "hsv" else "bgr"
        source_colors = self._calculate_average_colors(source_patches, color_space)
        self._visualize_average_colors(source_patches, source_colors, "source")

        ref_colors = self._calculate_average_colors(ref_patches, color_space)
        self._visualize_average_colors(ref_patches, ref_colors, "reference")

        if self.debug_mode: print("[INFO] Step 6: Aligning color patches by similarity...")
        aligned_source_colors, aligned_ref_colors, valid_pairs = self._align_color_patches_by_color(source_colors, ref_colors)

        self._visualize_patch_pairs(source_patches, ref_patches, valid_pairs)

        if self.debug_mode and source_coords:
            match_debug_img = source_cropped.copy()
            for detected_idx, ref_idx in valid_pairs:
                if detected_idx < len(source_coords):
                    x1, y1, x2, y2 = source_coords[detected_idx]
                    cv2.rectangle(match_debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(match_debug_img, f"->{ref_idx}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            self._save_debug_image("debug_color_matches_on_crop", match_debug_img)

        if len(aligned_source_colors) < 6:
            print(f"[ERROR] Not enough valid patch pairs found ({len(aligned_source_colors)}). Need at least 6.")
            return {"corrected_image": None, "debug_paths": self.debug_paths}

        if self.debug_mode: print(f"[INFO] Step 7: Calculating '{correction_method}' correction model...")
        from src.color_correction.corrector import ColorCorrector
        temp_corrector = ColorCorrector()
        correction_model = temp_corrector.calculate_correction_model(aligned_source_colors, aligned_ref_colors, method=correction_method)

        if self.debug_mode: print("[INFO] Step 8: Applying correction to main image...")
        corrected_image = temp_corrector.apply_correction_model(image_to_correct, correction_model, method=correction_method)
        self._save_debug_image("final_corrected_image", corrected_image)

        if self.debug_mode: print("--- Color Correction Pipeline Finished ---\n")
        return {"corrected_image": corrected_image, "correction_model": correction_model, "debug_paths": self.debug_paths}

    def _crop_checkers_geometrically(self, aligned_source_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Crops both images by finding the area defined by markers, then using morphological
        operations to isolate the central color checker grid as a single white rectangle.
        """
        # Step 1: Get the outer area defined by the markers
        ref_gray = cv2.cvtColor(self.reference_checker_img, cv2.COLOR_BGR2GRAY)
        ref_corners, _, _ = self.aruco_aligner.detector.detectMarkers(ref_gray)

        if ref_corners is None or len(ref_corners) == 0:
            print("[ERROR] Could not find markers on the ideal reference image. Cannot perform geometric crop.")
            return None, None

        all_ref_corners = np.concatenate(ref_corners).reshape(-1, 2)
        x_outer, y_outer, w_outer, h_outer = cv2.boundingRect(all_ref_corners)
        outer_crop_ref = self.reference_checker_img[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer]

        # Step 2: Create a binary image where patches and markers are white
        gray_outer_crop = cv2.cvtColor(outer_crop_ref, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray_outer_crop)
        _, thresh = cv2.threshold(inverted, 10, 255, cv2.THRESH_BINARY)
        self._save_debug_image("geocrop_threshold_initial", thresh)

        # Step 3: Use morphological closing to connect the individual patch blobs into one
        kernel = np.ones((7, 7), np.uint8)
        closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        self._save_debug_image("geocrop_threshold_closed", closed_thresh)

        # Step 4: Find the largest contour in the closed image, which should be the checker grid
        contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("[WARNING] No contours found after morphological closing. Falling back to marker bounds.")
            return aligned_source_img[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer], outer_crop_ref

        grid_contour = max(contours, key=cv2.contourArea)

        # Step 5: Get the bounding box of this grid contour
        x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(grid_contour)

        # Step 6: Use these inner coordinates to perform the final crop on both images
        final_x = x_outer + x_inner
        final_y = y_outer + y_inner
        
        final_source_crop = aligned_source_img[final_y : final_y + h_inner, final_x : final_x + w_inner]
        final_ref_crop = self.reference_checker_img[final_y : final_y + h_inner, final_x : final_x + w_inner]

        return final_source_crop, final_ref_crop

    def _detect_patches_from_image(self, image: np.ndarray, name: str) -> Tuple[List, List[Tuple[int, int, int, int]]]:
        """
        Detects patches using a multi-stage approach with fallbacks.
        1. Tries a simple hardcoded grid sampling method.
        2. Falls back to the robust grid-based ColorCheckerAligner.
        3. Falls back to a precise mask-based contour method.
        4. Finally, falls back to a simple bounding-box-based contour method.
        """
        if image is None or image.size == 0:
            print(f"[ERROR] Input image for patch detection ('{name}') is empty.")
            return [], []

        # --- Method 1: Simple Grid Sampling (Primary) ---
        patches, final_coords = self._detect_patches_simple_grid(image, name)
        if patches and len(patches) == 24:
            return patches, final_coords

        # --- Method 2: Grid-based detection (Fallback 1) ---
        patches, final_coords = self._detect_patches_grid_based(image, name)
        if patches and len(patches) >= 18:
            return patches, final_coords

        # --- Method 3: Mask-based contour detection (Fallback 2) ---
        patches, final_coords = self._detect_patches_mask_based(image, name)
        if patches and len(patches) >= 18:
            return patches, final_coords

        # --- Method 4: Bounding-box-based contour detection (Fallback 3) ---
        patches, final_coords = self._detect_patches_bbox_based(image, name)
        if patches and len(patches) >= 18:
            return patches, final_coords

        print(f"[ERROR] All patch detection methods failed for '{name}'.")
        return [], []

    def _detect_patches_simple_grid(self, image: np.ndarray, name: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extracts patches using a simple, hardcoded 4x6 grid and sampling the center of each cell.
        """
        if self.debug_mode:
            print(f".....Attempt 1: Detecting patches for '{name}' using simple grid sampling...")

        h, w, _ = image.shape
        ROWS, COLS = 4, 6
        patch_h, patch_w = h / ROWS, w / COLS

        patches = []
        final_coords = []
        
        if patch_h <= 1 or patch_w <= 1:
            if self.debug_mode: print(f".....Simple grid method failed: Image dimensions ({w}x{h}) too small for 4x6 grid.")
            return [], []

        for r in range(ROWS):
            for c in range(COLS):
                center_x = int(patch_w * (c + 0.5))
                center_y = int(patch_h * (r + 0.5))
                
                # Sample the central 50% of the patch area to avoid edges
                sample_h = int(patch_h * 0.5)
                sample_w = int(patch_w * 0.5)
                
                y1 = center_y - sample_h // 2
                y2 = center_y + sample_h // 2
                x1 = center_x - sample_w // 2
                x2 = center_x + sample_w // 2
                
                patch_roi = image[y1:y2, x1:x2]
                
                if patch_roi.size > 0:
                    patches.append(patch_roi)
                    final_coords.append((x1, y1, x2, y2))

        if len(patches) == 24:
            if self.debug_mode:
                print(f".....Success: Simple grid method found {len(patches)} patches.")
                output_vis_grid = image.copy()
                for (x1, y1, x2, y2) in final_coords:
                    cv2.rectangle(output_vis_grid, (x1, y1), (x2, y2), (255, 255, 255), 2)
                self._save_debug_image(f"patch_detect_simple_grid_success_{name}", output_vis_grid)
            return patches, final_coords
        else:
            if self.debug_mode:
                print(f".....Simple grid method failed, found {len(patches)} patches. Falling back.")
            return [], []

    def _detect_patches_grid_based(self, image: np.ndarray, name: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        if self.debug_mode:
            print(f".....Attempt 2: Detecting patches for '{name}' using grid-based 'ColorCheckerAligner'...")
        try:
            from .patch_detector import PatchExtractor
            extractor = PatchExtractor(image)
            patch_infos = extractor.detect_patches(adaptive=True)
            if patch_infos and len(patch_infos) >= 18:
                patches, final_coords = [], []
                for info in patch_infos:
                    x, y, w, h = info.bounding_box
                    patch_img = image[y : y + h, x : x + w]
                    if patch_img.size > 0:
                        patches.append(patch_img)
                        final_coords.append((x, y, x + w, y + h))
                if len(patches) >= 18:
                    if self.debug_mode:
                        print(f".....Success: Grid-based method found {len(patches)} patches.")
                        vis_img = extractor.visualize_patches(patch_infos, show_numbers=True)
                        self._save_debug_image(f"patch_detect_grid_based_success_{name}", vis_img)
                    return patches, final_coords
            if self.debug_mode:
                print(f".....Grid-based method found only {len(patch_infos) if patch_infos else 0} patches. Falling back.")
        except Exception as e:
            if self.debug_mode:
                print(f"[ERROR] Grid-based detection threw an exception: {e}. Falling back.")
        return [], []

    def _detect_patches_mask_based(self, image: np.ndarray, name: str) -> Tuple[List, List[Tuple[int, int, int, int]]]:
        if self.debug_mode:
            print(f".....Attempt 3: Detecting patches for '{name}' using mask-based contour method...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            patch_candidates = []
            min_area = (image.shape[0] * image.shape[1]) / 600
            max_area = (image.shape[0] * image.shape[1]) / 40
            for c in contours:
                area = cv2.contourArea(c)
                if not (min_area < area < max_area): continue
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / float(h)
                if not (0.7 < aspect_ratio < 1.4): continue
                patch_candidates.append(c)

            if len(patch_candidates) < 18: return [], []

            if len(patch_candidates) > 24:
                patch_candidates.sort(key=cv2.contourArea, reverse=True)
                patch_candidates = patch_candidates[:24]

            bounding_boxes = [cv2.boundingRect(c) for c in patch_candidates]
            sorted_contours = [c for c, b in sorted(zip(patch_candidates, bounding_boxes), key=lambda i: (i[1][1], i[1][0]))]
            
            patches, final_coords = [], []
            for c in sorted_contours:
                x, y, w, h = cv2.boundingRect(c)
                crop = image[y:y+h, x:x+w]
                mask = np.zeros(crop.shape[:2], dtype="uint8")
                shifted_contour = c - (x, y)
                cv2.drawContours(mask, [shifted_contour], -1, 255, -1)
                erosion_kernel = np.ones((3, 3), np.uint8)
                eroded_mask = cv2.erode(mask, erosion_kernel, iterations=1)
                if np.any(eroded_mask):
                    patches.append((crop, eroded_mask))
                    final_coords.append((x, y, x + w, y + h))

            if len(patches) >= 18:
                if self.debug_mode:
                    print(f".....Success: Mask-based method found {len(patches)} patches.")
                    vis_img = image.copy()
                    cv2.drawContours(vis_img, sorted_contours, -1, (0, 255, 0), 2)
                    self._save_debug_image(f"patch_detect_mask_based_success_{name}", vis_img)
                return patches, final_coords
        except Exception as e:
            if self.debug_mode:
                print(f"[ERROR] Mask-based detection threw an exception: {e}. Falling back.")
        return [], []

    def _detect_patches_bbox_based(self, image: np.ndarray, name: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        if self.debug_mode:
            print(f".....Attempt 4: Detecting patches for '{name}' using bbox-based contour method...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            patch_candidates = []
            min_area = (image.shape[0] * image.shape[1]) / 600
            max_area = (image.shape[0] * image.shape[1]) / 40
            for c in contours:
                area = cv2.contourArea(c)
                if not (min_area < area < max_area): continue
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / float(h)
                if not (0.75 < aspect_ratio < 1.25): continue
                patch_candidates.append(c)

            if len(patch_candidates) < 18: return [], []

            if len(patch_candidates) > 24:
                patch_candidates.sort(key=cv2.contourArea, reverse=True)
                patch_candidates = patch_candidates[:24]

            patch_coords = [cv2.boundingRect(c) for c in patch_candidates]
            patch_coords.sort(key=lambda r: (r[1], r[0]))

            patches, final_coords = [], []
            for (x, y, w, h) in patch_coords:
                inset = int(min(w, h) * 0.15)
                patch = image[y+inset:y+h-inset, x+inset:x+w-inset]
                if patch.size > 0:
                    patches.append(patch)
                    final_coords.append((x, y, x+w, y+h))

            if len(patches) >= 18:
                if self.debug_mode:
                    print(f".....Success: Bbox-based fallback found {len(patches)} patches.")
                    vis_img_final = image.copy()
                    for i, (x, y, w, h) in enumerate(patch_coords):
                        cv2.rectangle(vis_img_final, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    self._save_debug_image(f"patch_detect_bbox_fallback_success_{name}", vis_img_final)
                return patches, final_coords
        except Exception as e:
            if self.debug_mode:
                print(f"[ERROR] Bbox-based detection threw an exception: {e}.")
        return [], []

    def _visualize_average_colors(self, original_patches: List, avg_colors: List[np.ndarray], name: str):
        """
        Creates a debug image showing original patches and their calculated average colors.
        """
        if not self.debug_mode or not original_patches or not avg_colors:
            return

        # Determine a consistent patch size for visualization
        try:
            if isinstance(original_patches[0], tuple):
                h, w, _ = original_patches[0][0].shape
            else:
                h, w, _ = original_patches[0].shape
            patch_size = max(h, w, 50)
        except IndexError:
            return # No patches to visualize

        # Create a canvas to hold pairs of (original_patch, average_color_swatch)
        num_patches = len(original_patches)
        canvas_h = num_patches * (patch_size + 5) + 5
        canvas_w = 2 * patch_size + 15
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 50  # Dark gray background

        y_offset = 5
        for i, (patch_data, avg_bgr) in enumerate(zip(original_patches, avg_colors)):
            if isinstance(patch_data, tuple):
                original_img, _ = patch_data
            else:
                original_img = patch_data
            
            if original_img.size == 0: continue

            display_patch = cv2.resize(original_img, (patch_size, patch_size))
            avg_color_swatch = np.full((patch_size, patch_size, 3), avg_bgr, dtype=np.uint8)

            canvas[y_offset:y_offset + patch_size, 5:5 + patch_size] = display_patch
            canvas[y_offset:y_offset + patch_size, 10 + patch_size:10 + 2 * patch_size] = avg_color_swatch
            
            y_offset += patch_size + 5

        self._save_debug_image(f"average_colors_visualization_{name}", canvas)

    def _calculate_average_colors(self, patches: List, color_space="bgr") -> List[np.ndarray]:
        """Calculates the average color of patches, supporting both standard images and (image, mask) tuples."""
        average_colors = []
        for i, patch_data in enumerate(patches):
            mask = None
            if isinstance(patch_data, tuple):
                img, mask = patch_data
            else:
                img = patch_data
            
            if img is None or img.size == 0:
                if self.debug_mode: print(f"[WARNING] Patch {i} is empty, skipping.")
                continue

            img = img[:, :, :3]
            if color_space == "hsv":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Use mask if available for a more precise average, otherwise use numpy.mean for rectangular patches
            if mask is not None and mask.size > 0 and np.any(mask):
                avg_color = cv2.mean(img, mask=mask)[:3]
            else:
                avg_color = np.mean(img, axis=(0, 1))
            
            average_colors.append(np.array(avg_color).astype(np.uint8))
        return average_colors

    def _visualize_patch_pairs(self, source_patches: List, ref_patches: List, valid_pairs: List[Tuple[int, int]]):
        """
        Creates a debug image showing the matched pairs of source and reference patches.
        """
        if not self.debug_mode or not valid_pairs:
            return

        patch_size = 100  # A fixed size for consistent visualization
        num_pairs = len(valid_pairs)
        canvas_h = num_pairs * (patch_size + 10) + 10
        canvas_w = 2 * patch_size + 20
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 50

        y_offset = 10
        for i, (source_idx, ref_idx) in enumerate(valid_pairs):
            # Extract source patch image, handling (image, mask) tuples
            source_patch_data = source_patches[source_idx]
            source_img = source_patch_data[0] if isinstance(source_patch_data, tuple) else source_patch_data

            # Extract reference patch image
            ref_patch_data = ref_patches[ref_idx]
            ref_img = ref_patch_data[0] if isinstance(ref_patch_data, tuple) else ref_patch_data

            if source_img.size == 0 or ref_img.size == 0:
                continue

            # Resize for consistent display
            display_source = cv2.resize(source_img, (patch_size, patch_size))
            display_ref = cv2.resize(ref_img, (patch_size, patch_size))

            # Place patches side-by-side on the canvas
            x_source = 10
            x_ref = x_source + patch_size + 10
            canvas[y_offset:y_offset + patch_size, x_source:x_source + patch_size] = display_source
            canvas[y_offset:y_offset + patch_size, x_ref:x_ref + patch_size] = display_ref
            
            # Add text labels to indicate the pairing
            cv2.putText(canvas, f"S:{source_idx} -> R:{ref_idx}", (x_source, y_offset + patch_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset += patch_size + 10

        self._save_debug_image("patch_pairing_visualization", canvas)

    def _align_color_patches_by_color(self, detected_colors_bgr: List[np.ndarray], reference_colors_bgr: List[np.ndarray]) -> Tuple[List, List, List]:
        detected_colors_rgb = [c[::-1] for c in detected_colors_bgr]
        reference_colors_rgb = [c[::-1] for c in reference_colors_bgr]
        reference_lab = rgb2lab(np.array(reference_colors_rgb, dtype=np.uint8).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
        detected_lab = rgb2lab(np.array(detected_colors_rgb, dtype=np.uint8).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

        REJECTION_THRESHOLD = 50.0
        num_detected, num_reference = len(detected_lab), len(reference_lab)
        cost_matrix = np.zeros((num_detected, num_reference))
        for i in range(num_detected):
            for j in range(num_reference):
                cost_matrix[i, j] = deltaE_cie76(detected_lab[i], reference_lab[j])
        cost_matrix[cost_matrix > REJECTION_THRESHOLD] = np.inf

        DUMMY_COST = REJECTION_THRESHOLD - 1.0
        size = max(num_detected, num_reference)
        padded_cost_matrix = np.full((size, size), DUMMY_COST, dtype=float)
        padded_cost_matrix[:num_detected, :num_reference] = cost_matrix

        row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
        valid_pairs = []
        if self.debug_mode: print("\n.....Filtering Color Patch Assignments.....")
        for r, c in zip(row_ind, col_ind):
            cost = padded_cost_matrix[r, c]
            if r < num_detected and c < num_reference and cost != np.inf:
                valid_pairs.append((r, c))
                if self.debug_mode: print(f".....✔️ Valid: Detected patch {r} -> Reference patch {c} (Cost: {cost:.2f})")
            elif self.debug_mode: print(f".....❌ Discarded: Assignment of row {r} to col {c} (Cost: {cost:.2f})")

        valid_detected_indices = [pair[0] for pair in valid_pairs]
        valid_reference_indices = [pair[1] for pair in valid_pairs]
        aligned_detected_colors = [detected_colors_bgr[i] for i in valid_detected_indices]
        aligned_reference_colors = [reference_colors_bgr[i] for i in valid_reference_indices]
        return aligned_detected_colors, aligned_reference_colors, valid_pairs
